import numpy as np

from multifractal.utils.params import get_flat_param_vector
from multifractal.probes.curves import loss_on_batch, loss_for_vector

def plane_2d_map(
    model, batch, criterion,
    u_vec, v_vec, grid_vals, shapes, numels
):
    base_vec = get_flat_param_vector(model).detach()
    base_loss = loss_on_batch(model, batch, criterion)

    gv = np.asarray(grid_vals, dtype=np.float64)
    G = len(gv)
    grid = np.zeros((G, G), dtype=np.float32)

    for i, a in enumerate(gv):
        for j, b in enumerate(gv):
            th = base_vec + float(a) * u_vec + float(b) * v_vec
            grid[i, j] = float(loss_for_vector(model, th, batch, criterion, shapes, numels))

    return {"grid_vals": gv.astype(np.float32), "loss_grid": grid, "base_loss": float(base_loss)}
