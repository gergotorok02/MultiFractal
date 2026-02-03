import numpy as np

from multifractal.utils.params import get_flat_param_vector, set_flat_param_vector
from multifractal.probes.curves import loss_on_batch, margin_stats

def interpolation_barrier(
    model, batch, criterion,
    theta_a, theta_b, shapes, numels,
    alphas,
):
    theta_a = theta_a.detach()
    theta_b = theta_b.detach()

    losses, accs = [], []
    backup = get_flat_param_vector(model).detach()

    for a in alphas:
        th = (1.0 - float(a)) * theta_a + float(a) * theta_b
        set_flat_param_vector(model, th, shapes, numels)
        l = loss_on_batch(model, batch, criterion)
        ms = margin_stats(model, batch)
        losses.append(l)
        accs.append(ms["acc"])

    set_flat_param_vector(model, backup, shapes, numels)

    losses = np.asarray(losses, dtype=np.float32)
    alphas = np.asarray(list(alphas), dtype=np.float32)
    barrier = float(losses.max() - max(losses[0], losses[-1]))
    return {
        "alphas": alphas,
        "losses": losses,
        "accs": np.asarray(accs, dtype=np.float32),
        "barrier": barrier,
        "max_loss": float(losses.max()),
    }
