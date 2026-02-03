import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mulifractal.device import get_device
from mulifractal.data import build_loaders
from mulifractal.model import CIFARResNet18

from mulifractal.utils.repro import set_seed, temp_seed
from mulifractal.utils.modes import measurement_eval_mode, train_mode_with_bn_frozen
from mulifractal.utils.params import get_param_list, get_param_shapes_and_numels, get_flat_param_vector
from mulifractal.utils.math import safe_normalize, fit_alpha_loglog, alpha_local_adjacent, collapse_score_quantiles

from mulifractal.probes.fd import fd_probe_rich, sample_random_direction_filter_norm, sample_random_direction_raw
from mulifractal.probes.hessian import estimate_hessian_metrics
from mulifractal.probes.curves import loss_on_batch, margin_stats, grad_vector, directional_1d_curve
from mulifractal.probes.sam import sam_sharpness_probe
from mulifractal.probes.barrier import interpolation_barrier
from mulifractal.probes.grad_noise import grad_noise_proxy
from mulifractal.probes.plane import plane_2d_map
from mulifractal.probes.nested_zoom import nested_zoom_probe


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def build_scheduler(optimizer, scheduler_type, num_epochs, steps_per_epoch, lr_max, warmup_epochs=5):
    scheduler_type = str(scheduler_type)

    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if scheduler_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_max,
            total_steps=num_epochs * steps_per_epoch,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4
        )

    if scheduler_type == "cosine_warmup":
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        total_steps = int(num_epochs * steps_per_epoch)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler_type={scheduler_type}")


def train_ultra_all_saved(cfg: dict) -> str:
    device = get_device()
    print("Using device:", device)

    seed = int(cfg["seed"])
    out_path = cfg["out_path"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    set_seed(seed)

    train_loader, test_loader, probe_batch = build_loaders(
        batch_size=cfg["batch_size"],
        use_randaugment=cfg["use_randaugment"],
        ra_num_ops=cfg["ra_num_ops"],
        ra_magnitude=cfg["ra_magnitude"],
        num_workers=cfg.get("num_workers", 2),
    )
    probe_batch = (probe_batch[0].to(device), probe_batch[1].to(device))

    model = CIFARResNet18().to(device)
    ls_eps = float(cfg["label_smoothing"])
    criterion = nn.CrossEntropyLoss(label_smoothing=ls_eps)

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=0.9,
        weight_decay=float(cfg["weight_decay"]),
        nesterov=bool(cfg.get("nesterov", False))
    )

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=cfg["scheduler"],
        num_epochs=int(cfg["num_epochs"]),
        steps_per_epoch=steps_per_epoch,
        lr_max=float(cfg["lr"]),
        warmup_epochs=int(cfg.get("warmup_epochs", 5))
    )

    params = get_param_list(model)
    shapes, numels = get_param_shapes_and_numels(params)

    # epoch logs
    epochs, train_loss, train_acc, test_acc = [], [], [], []

    # measurement logs
    measure_steps, measure_epochs, measure_lr = [], [], []
    base_probe_loss, probe_margin = [], []

    epsilons = [float(e) for e in cfg["epsilons"]]
    num_fd_directions = int(cfg["num_fd_directions"])

    fd_mean_list, fd_var_list, fd_deltas_list = [], [], []
    alpha_fit_list, alpha_local_list = [], []
    collapse_score_list, collapse_stdq_list = [], []

    lambda_max_list, traceH_list, eta_lambda_list = [], [], []
    vtop_saved = []

    curves_pack = {"curve_eps": np.asarray(cfg["curve_eps"], dtype=np.float32),
                   "grad": [], "update": [], "top": [], "rand": []}
    sam_list, barrier_list, grad_noise_list, plane_maps_list = [], [], [], []

    do_nested_zoom = bool(cfg.get("do_nested_zoom", False))
    nested_mean_list = [] if do_nested_zoom else None

    prev_measure_theta = None

    measure_every = int(cfg["measure_every"])
    bn_eval = bool(cfg["bn_eval_for_measure"])
    direction_mode = str(cfg["direction_mode"])

    do_directional_curves = bool(cfg["do_directional_curves"])
    do_sam = bool(cfg["do_sam_sharpness"])
    do_barrier = bool(cfg["do_barrier"])
    do_grad_noise = bool(cfg["do_grad_noise"])
    do_plane_map = bool(cfg["do_plane_map"])
    plane_every = int(cfg["plane_every"])

    do_hessian = bool(cfg["do_hessian"])
    hessian_stride = int(cfg["hessian_stride"])
    num_power_iters = int(cfg["num_power_iters"])
    num_trace_samples = int(cfg["num_trace_samples"])
    save_vtop = bool(cfg.get("save_vtop", False))

    global_step = 0
    measure_idx = 0
    noise_iter = iter(train_loader)

    for ep in range(1, int(cfg["num_epochs"]) + 1):
        model.train()
        run_loss, run_total, run_correct = 0.0, 0, 0

        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if cfg["scheduler"] in ["onecycle", "cosine_warmup"]:
                scheduler.step()

            run_loss += loss.item() * y.size(0)
            run_total += y.size(0)
            run_correct += (logits.argmax(dim=1) == y).sum().item()
            global_step += 1

            if global_step % measure_every == 0:
                cur_lr = float(optimizer.param_groups[0]["lr"])
                epoch_frac = ep + bi / max(1, steps_per_epoch)
                print(f"[Measure #{measure_idx}] epoch≈{epoch_frac:.3f} step={global_step} lr={cur_lr:.5g}")

                with temp_seed(12345 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                    probeL = loss_on_batch(model, probe_batch, criterion)
                    marg = margin_stats(model, probe_batch)
                    theta_now = get_flat_param_vector(model).detach().to(device)

                with temp_seed(55555 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                    base_fd, fd_mean, fd_var, deltas = fd_probe_rich(
                        model, probe_batch, criterion,
                        epsilons=epsilons,
                        num_directions=num_fd_directions,
                        shapes=shapes, numels=numels,
                        direction_mode=direction_mode,
                    )

                alpha_fit = fit_alpha_loglog(epsilons, fd_mean)
                alpha_loc = alpha_local_adjacent(epsilons, fd_mean)
                if np.isfinite(alpha_fit):
                    cscore, cstdq, _ = collapse_score_quantiles(epsilons, deltas, alpha_fit)
                else:
                    cscore, cstdq = float("nan"), np.full(5, np.nan, np.float32)

                if do_hessian and ((measure_idx % hessian_stride) == 0):
                    try:
                        with temp_seed(77777 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                            lam, trH, vtop = estimate_hessian_metrics(
                                model, probe_batch, criterion,
                                shapes, numels,
                                device=device,
                                num_power_iters=num_power_iters,
                                num_trace_samples=num_trace_samples,
                            )
                    except Exception as e:
                        print("  [Hessian] failed:", repr(e))
                        lam, trH, vtop = float("nan"), float("nan"), None
                else:
                    lam, trH, vtop = float("nan"), float("nan"), None

                eta_lam = cur_lr * lam if np.isfinite(lam) else float("nan")

                if do_directional_curves:
                    with temp_seed(88888 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        gflat, _ = grad_vector(model, probe_batch, criterion, create_graph=False)
                        ghat = safe_normalize(gflat.detach())

                        if prev_measure_theta is not None:
                            upd = (theta_now - prev_measure_theta).detach()
                            uhat = safe_normalize(upd) if upd.norm().item() > 1e-12 else ghat
                        else:
                            uhat = ghat

                        if vtop is not None:
                            vhat = safe_normalize(vtop)
                        else:
                            vhat = sample_random_direction_filter_norm(model) if direction_mode == "filter_norm" \
                                else sample_random_direction_raw(theta_now)

                        rhat = sample_random_direction_filter_norm(model) if direction_mode == "filter_norm" \
                            else sample_random_direction_raw(theta_now)

                    with temp_seed(100001 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        curves_pack["grad"].append(directional_1d_curve(
                            model, probe_batch, criterion, ghat, cfg["curve_eps"], shapes, numels
                        ))
                    with temp_seed(100101 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        curves_pack["update"].append(directional_1d_curve(
                            model, probe_batch, criterion, uhat, cfg["curve_eps"], shapes, numels
                        ))
                    with temp_seed(100201 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        curves_pack["top"].append(directional_1d_curve(
                            model, probe_batch, criterion, vhat, cfg["curve_eps"], shapes, numels
                        ))
                    with temp_seed(100301 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        curves_pack["rand"].append(directional_1d_curve(
                            model, probe_batch, criterion, rhat, cfg["curve_eps"], shapes, numels
                        ))

                if do_sam:
                    with temp_seed(200000 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        sam_list.append(sam_sharpness_probe(
                            model, probe_batch, criterion,
                            rho_list=cfg["sam_rhos"],
                            shapes=shapes, numels=numels,
                            steps=int(cfg["sam_steps"]),
                            lr_ascent=float(cfg["sam_lr_ascent"]),
                        ))

                if do_barrier and prev_measure_theta is not None:
                    with measurement_eval_mode(model, bn_eval=bn_eval):
                        barrier_list.append(interpolation_barrier(
                            model, probe_batch, criterion,
                            prev_measure_theta, theta_now,
                            shapes, numels,
                            alphas=cfg["barrier_alphas"],
                        ))
                else:
                    barrier_list.append(None)

                if do_grad_noise and ((measure_idx % int(cfg["grad_noise_stride"])) == 0):
                    try:
                        with train_mode_with_bn_frozen(model, bn_eval=bn_eval):
                            gn = grad_noise_proxy(model, noise_iter, int(cfg["grad_noise_k"]), criterion, device=device)
                    except StopIteration:
                        noise_iter = iter(train_loader)
                        with train_mode_with_bn_frozen(model, bn_eval=bn_eval):
                            gn = grad_noise_proxy(model, noise_iter, int(cfg["grad_noise_k"]), criterion, device=device)
                    grad_noise_list.append(gn)
                else:
                    grad_noise_list.append(None)

                if do_plane_map and ((measure_idx % plane_every) == 0):
                    with temp_seed(300000 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        if prev_measure_theta is not None:
                            u = (theta_now - prev_measure_theta).detach()
                            u = safe_normalize(u) if u.norm().item() > 1e-12 else sample_random_direction_raw(theta_now)
                        else:
                            u = sample_random_direction_raw(theta_now)
                        v = sample_random_direction_raw(theta_now)
                        v = v - (torch.dot(v, u) * u)
                        v = safe_normalize(v)

                    gv = [float(cfg["plane_scale"]) * float(t) for t in cfg["plane_grid"]]
                    with temp_seed(300123 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        pm = plane_2d_map(model, probe_batch, criterion, u, v, gv, shapes, numels)
                    plane_maps_list.append(pm)
                else:
                    plane_maps_list.append(None)

                if do_nested_zoom:
                    with temp_seed(99999 + measure_idx), measurement_eval_mode(model, bn_eval=bn_eval):
                        nm = nested_zoom_probe(
                            model, probe_batch, criterion,
                            eps_outer=float(cfg["eps_outer"]),
                            eps_inner_list=[float(e) for e in cfg["eps_inner_list"]],
                            num_outer=int(cfg["num_outer"]),
                            num_inner=int(cfg["num_inner"]),
                            shapes=shapes, numels=numels,
                            direction_mode=direction_mode,
                        )
                    nested_mean_list.append(nm)

                measure_steps.append(int(global_step))
                measure_epochs.append(float(epoch_frac))
                measure_lr.append(float(cur_lr))

                base_probe_loss.append(float(probeL))
                probe_margin.append(marg)

                fd_mean_list.append(fd_mean.astype(np.float32))
                fd_var_list.append(fd_var.astype(np.float32))
                fd_deltas_list.append(deltas.astype(np.float32))
                alpha_fit_list.append(float(alpha_fit))
                alpha_local_list.append(alpha_loc.astype(np.float32))
                collapse_score_list.append(float(cscore))
                collapse_stdq_list.append(np.asarray(cstdq, dtype=np.float32))

                lambda_max_list.append(float(lam))
                traceH_list.append(float(trH))
                eta_lambda_list.append(float(eta_lam))

                if save_vtop:
                    if vtop is None:
                        vtop_saved.append(None)
                    else:
                        vtop_saved.append(vtop.detach().cpu().to(torch.float16).numpy())
                else:
                    vtop_saved.append(None)

                prev_measure_theta = theta_now.detach().clone()
                measure_idx += 1

        ep_loss = run_loss / max(1, run_total)
        ep_acc = run_correct / max(1, run_total)
        te_acc = accuracy(model, test_loader, device=device)

        if cfg["scheduler"] == "cosine":
            scheduler.step()

        print(f"[Epoch {ep}/{cfg['num_epochs']}] train_loss={ep_loss:.4f} train_acc={ep_acc:.4f} test_acc={te_acc:.4f}")

        epochs.append(ep)
        train_loss.append(float(ep_loss))
        train_acc.append(float(ep_acc))
        test_acc.append(float(te_acc))

    # ---- pack & save (schema preserved)
    eps_arr = np.asarray(epsilons, dtype=np.float32)
    curve_eps_arr = np.asarray(cfg["curve_eps"], dtype=np.float32)

    fd_mean_arr = np.stack(fd_mean_list, axis=0).astype(np.float32) if len(fd_mean_list) else np.zeros((0, len(epsilons)), np.float32)
    fd_var_arr  = np.stack(fd_var_list, axis=0).astype(np.float32) if len(fd_var_list) else np.zeros((0, len(epsilons)), np.float32)
    fd_deltas_arr = np.stack(fd_deltas_list, axis=0).astype(np.float32) if len(fd_deltas_list) else np.zeros((0, len(epsilons), num_fd_directions), np.float32)

    alpha_fit_arr = np.asarray(alpha_fit_list, dtype=np.float32) if len(alpha_fit_list) else np.zeros((0,), np.float32)
    alpha_local_arr = np.stack(alpha_local_list, axis=0).astype(np.float32) if len(alpha_local_list) else np.zeros((0, max(0, len(epsilons) - 1)), np.float32)
    collapse_score_arr = np.asarray(collapse_score_list, dtype=np.float32) if len(collapse_score_list) else np.zeros((0,), np.float32)
    collapse_stdq_arr  = np.stack(collapse_stdq_list, axis=0).astype(np.float32) if len(collapse_stdq_list) else np.zeros((0, 5), np.float32)

    lambda_max_arr = np.asarray(lambda_max_list, dtype=np.float32) if len(lambda_max_list) else np.zeros((0,), np.float32)
    traceH_arr     = np.asarray(traceH_list, dtype=np.float32) if len(traceH_list) else np.zeros((0,), np.float32)
    eta_lambda_arr = np.asarray(eta_lambda_list, dtype=np.float32) if len(eta_lambda_list) else np.zeros((0,), np.float32)

    def mcol(k):
        return np.asarray([m[k] for m in probe_margin], dtype=np.float32) if len(probe_margin) else np.zeros((0,), np.float32)

    margin_pack = {
        "acc": mcol("acc"), "mean": mcol("mean"), "median": mcol("median"),
        "p10": mcol("p10"), "p25": mcol("p25"), "p75": mcol("p75"), "p90": mcol("p90")
    }

    curves_obj = {
        "curve_eps": curve_eps_arr,
        "grad": np.asarray(curves_pack["grad"], dtype=object),
        "update": np.asarray(curves_pack["update"], dtype=object),
        "top": np.asarray(curves_pack["top"], dtype=object),
        "rand": np.asarray(curves_pack["rand"], dtype=object),
    }
    sam_obj = np.asarray(sam_list, dtype=object)
    barrier_obj = {"alphas": np.asarray(cfg["barrier_alphas"], dtype=np.float32),
                   "items": np.asarray(barrier_list, dtype=object)}
    grad_noise_obj = np.asarray(grad_noise_list, dtype=object)
    plane_maps_obj = np.asarray(plane_maps_list, dtype=object)
    vtop_obj = np.asarray(vtop_saved, dtype=object)

    meta = dict(cfg)
    meta["device"] = str(device)
    meta["epsilons"] = eps_arr
    meta["curve_eps"] = curve_eps_arr
    meta["measurement_protocol"] = "eval+bn_frozen (resnet50-style)"

    if do_nested_zoom:
        nested_mean_arr = np.stack(nested_mean_list, axis=0).astype(np.float32) if len(nested_mean_list) else np.zeros((0, len(cfg["eps_inner_list"])), np.float32)
        meta["do_nested_zoom"] = True
    else:
        nested_mean_arr = np.zeros((0, 1), np.float32)
        meta["do_nested_zoom"] = False

    np.savez(
        out_path,
        epochs=np.asarray(epochs, dtype=np.int32),
        train_loss=np.asarray(train_loss, dtype=np.float32),
        train_acc=np.asarray(train_acc, dtype=np.float32),
        test_acc=np.asarray(test_acc, dtype=np.float32),

        measure_steps=np.asarray(measure_steps, dtype=np.int64),
        measure_epochs=np.asarray(measure_epochs, dtype=np.float32),
        measure_lr=np.asarray(measure_lr, dtype=np.float32),

        base_loss=np.asarray(base_probe_loss, dtype=np.float32),
        margin=margin_pack,

        epsilons=eps_arr,
        fd_mean=fd_mean_arr,
        fd_var=fd_var_arr,
        fd_deltas=fd_deltas_arr,
        alpha_fit=alpha_fit_arr,
        alpha_local=alpha_local_arr,
        collapse_score=collapse_score_arr,
        collapse_stdq=collapse_stdq_arr,

        lambda_max=lambda_max_arr,
        trace_H=traceH_arr,
        eta_lambda=eta_lambda_arr,
        v_top=vtop_obj,

        curves=curves_obj,
        sam=sam_obj,
        barrier=barrier_obj,
        grad_noise=grad_noise_obj,
        plane_maps=plane_maps_obj,

        nested_mean=nested_mean_arr,
        meta=meta
    )

    print("\nSaved:", out_path)
    return out_path
