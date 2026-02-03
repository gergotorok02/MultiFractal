import os
from multifractal.train import train_ultra_all_saved
from configs.ultra_cifar10_resnet18 import BASE_CFG, EXPERIMENTS

RUN_MODE = "all"   # "single" or "all"
OUT_DIR = "./runs_ultra_all"
os.makedirs(OUT_DIR, exist_ok=True)

if RUN_MODE == "single":
    cfg = dict(BASE_CFG)
    cfg.update({
        "use_randaugment": True,
        "label_smoothing": 0.2,
        "scheduler": "onecycle",
    })
    cfg["out_path"] = os.path.join(
        OUT_DIR,
        f"single_long_seed{cfg['seed']}_RA{int(cfg['use_randaugment'])}_LS{cfg['label_smoothing']}_SCH{cfg['scheduler']}.npz"
    )
    train_ultra_all_saved(cfg)

else:
    for name, upd in EXPERIMENTS:
        cfg = dict(BASE_CFG)
        cfg.update(upd)
        cfg["out_path"] = os.path.join(
            OUT_DIR,
            f"{name}_seed{cfg['seed']}_RA{int(cfg['use_randaugment'])}_LS{cfg['label_smoothing']}_SCH{cfg['scheduler']}.npz"
        )
        train_ultra_all_saved(cfg)
