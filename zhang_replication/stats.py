"""Block-bootstrap inference (identical to the research programme's implementation)."""
import numpy as np


def one_sided_p(x: np.ndarray, block: int = 5, n_boot: int = 10_000, seed: int = 0) -> float:
    """Block-bootstrap p-value for H0: mean <= 0 (share of centred bootstrap means >= the observed mean)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    centred = x - x.mean()
    boot = centred[idx].mean(axis=1)
    return float((boot >= x.mean()).mean())
