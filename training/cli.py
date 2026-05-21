from common.config import load_dataclass
from .config import TrainConfig
from training.core.engine import run_training

def main() -> None:
    run_training(load_dataclass(TrainConfig))

if __name__ == "__main__":
    main()
