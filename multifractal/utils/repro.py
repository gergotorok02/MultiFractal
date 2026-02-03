from contextlib import contextmanager
import numpy as np
import torch

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@contextmanager
def temp_seed(seed: int):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
