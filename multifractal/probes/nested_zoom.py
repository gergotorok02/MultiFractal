import numpy as np

from multifractal.probes.fd import sample_random_direction_filter_norm, sample_random_direction_raw
from multifractal.utils.params import get_flat_param_vector, set_flat_param_vector
from multifractal.probes.curves import loss_on_batch, loss_for_vector

def nested_zoom_probe(
    model, batch, criterion,
    eps_outer, eps_inner_list,
    num_outer, num_inner,
    shapes, numels,
    direction_mode="filter_norm",
):
    base_vec = get_flat_param_vector(model).detach()
    backup = base_vec.detach().clone()

    nested_abs = np.zeros(len(eps_inner_list), dtype=np.float64)

    for _ in range(int(num_outer)):
        v_out = sample_random_direction_filter_norm(model) if direction_mode == "filter_norm" \
                else sample_random_direction_raw(base_vec)

        theta_out = backup + float(eps_outer) * v_out
        set_flat_param_vector(model, theta_out, shapes, numels)
        loss_out = loss_on_batch(model, batch, criterion)

        for k, eps_in in enumerate(eps_inner_list):
            for _ in range(int(num_inner)):
                v_in = sample_random_direction_filter_norm(model) if direction_mode == "filter_norm" \
                       else sample_random_direction_raw(theta_out)

                theta_in = theta_out + float(eps_in) * v_in
                l = loss_for_vector(model, theta_in, batch, criterion, shapes, numels)
                nested_abs[k] += abs(l - loss_out)

        set_flat_param_vector(model, backup, shapes, numels)

    nested_mean = nested_abs / max(1, int(num_outer) * int(num_inner))
    return nested_mean.astype(np.float32)
