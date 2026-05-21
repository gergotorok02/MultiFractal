from common.config import load_dataclass
from .config import ProbeConfig
from .runner import offline_probe

def main() -> None:
    offline_probe(load_dataclass(ProbeConfig))

if __name__ == "__main__":
    main()
