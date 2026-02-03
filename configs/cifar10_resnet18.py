BASE_CFG = {
    "seed": 0,
    "num_epochs": 50,
    "batch_size": 128,
    "lr": 0.1,
    "weight_decay": 5e-4,
    "nesterov": False,

    "use_randaugment": False,
    "ra_num_ops": 2,
    "ra_magnitude": 7,
    "label_smoothing": 0.2,
    "scheduler": "cosine",
    "warmup_epochs": 5,

    "measure_every": 100,
    "bn_eval_for_measure": True,
    "direction_mode": "filter_norm",

    "epsilons": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "num_fd_directions": 32,

    "do_directional_curves": True,
    "curve_eps": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],

    "do_sam_sharpness": True,
    "sam_rhos": [5e-4, 1e-3, 2e-3, 5e-3],
    "sam_steps": 1,
    "sam_lr_ascent": 1.0,

    "do_barrier": True,
    "barrier_alphas": [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],

    "do_grad_noise": True,
    "grad_noise_k": 3,
    "grad_noise_stride": 3,

    "do_plane_map": True,
    "plane_every": 10,
    "plane_grid": [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2],
    "plane_scale": 3e-3,

    "do_hessian": True,
    "hessian_stride": 5,
    "num_power_iters": 5,
    "num_trace_samples": 2,
    "save_vtop": True,

    "do_nested_zoom": True,
    "eps_outer": 3e-3,
    "eps_inner_list": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "num_outer": 8,
    "num_inner": 8,
}

EXPERIMENTS = [
    ("none_baseline", dict(use_randaugment=False, label_smoothing=0.0, scheduler="cosine")),
    ("randaugment_only", dict(use_randaugment=True,  label_smoothing=0.0, scheduler="cosine")),
    ("labelsmoothing_only", dict(use_randaugment=False, label_smoothing=0.2, scheduler="cosine")),
    ("scheduler_only_onecycle", dict(use_randaugment=False, label_smoothing=0.0, scheduler="onecycle")),
    ("scheduler_only_cosine_warmup", dict(use_randaugment=False, label_smoothing=0.0, scheduler="cosine_warmup")),
    ("all_three_onecycle", dict(use_randaugment=True, label_smoothing=0.2, scheduler="onecycle")),
    ("all_three_cosine_warmup", dict(use_randaugment=True, label_smoothing=0.2, scheduler="cosine_warmup")),
]
