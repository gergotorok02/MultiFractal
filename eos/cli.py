from common.config import load_dataclass
from .config import EosConfig
from .runner import eos_probe

def main() -> None:
    eos_probe(load_dataclass(EosConfig))

if __name__ == "__main__":
    main()
