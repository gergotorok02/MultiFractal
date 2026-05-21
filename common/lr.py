def infer_lr(payload: dict, step_index: dict) -> float:
    step = int(payload.get("global_step", -1))
    if step in step_index:
        lr = step_index[step].get("lr")
        if isinstance(lr, list) and lr: return float(lr[0])
        if isinstance(lr, (float, int)): return float(lr)
    summary = payload.get("train_summary", {})
    if isinstance(summary, dict):
        lr = summary.get("latest_lr")
        if isinstance(lr, list) and lr: return float(lr[0])
        if isinstance(lr, (float, int)): return float(lr)
    opt = payload.get("optimizer_state")
    if opt and opt.get("param_groups"):
        return float(opt["param_groups"][0]["lr"])
    return float("nan")
