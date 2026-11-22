"""Bootstrap: could this backtest result just be luck?

A backtest is one history. Reshuffle it into thousands of alternative histories
and see how much the result moves:

  * block bootstrap: draw random blocks of consecutive days (about a month), so
    calm stretches and crashes keep their clustering; stitch blocks into a
    history as long as the original
  * paired: each alternative history uses the SAME days for the strategy and
    for buy & hold, so comparing them is fair

Fast because Sharpe and growth only need per-block sums (of returns, squared
returns and log returns), computed once for every possible block.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR = 252


@dataclass
class Bootstrap:
    sharpe: np.ndarray      # one Sharpe ratio per alternative history
    cagr: np.ndarray        # one yearly return per alternative history
    actual_sharpe: float
    actual_cagr: float

    def summary(self) -> dict:
        p5, p50, p95 = np.percentile(self.sharpe, [5, 50, 95])
        c5, c95 = np.percentile(self.cagr, [5, 95])
        return {"actual": self.actual_sharpe, "p5": p5, "p50": p50, "p95": p95,
                "prob_positive": float((self.sharpe > 0).mean()),
                "cagr": self.actual_cagr, "cagr_p5": c5, "cagr_p95": c95}


def _block_sums(r: np.ndarray, block: int):
    """Sums over every block of `block` consecutive days (wrapping around the end)."""
    ext = np.concatenate([r, r[:block - 1]])
    c = np.concatenate([[0.0], np.cumsum(ext)])
    return c[block:block + len(r)] - c[:len(r)]


def _stats(r: np.ndarray, starts: np.ndarray, block: int, ppy: float = YEAR):
    """Sharpe and CAGR of each alternative history (rows of block start indices)."""
    n = starts.shape[1] * block
    s1 = _block_sums(r, block)[starts].sum(axis=1)
    s2 = _block_sums(r * r, block)[starts].sum(axis=1)
    lg = _block_sums(np.log1p(r), block)[starts].sum(axis=1)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0) * n / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(var > 0, mean / np.sqrt(var) * np.sqrt(ppy), 0.0)
    cagr = np.exp(lg * ppy / n) - 1
    return sharpe, cagr


def _actual(r: np.ndarray, ppy: float = YEAR):
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    return sharpe, float(np.exp(np.log1p(r).sum() * ppy / len(r)) - 1)


def bootstrap(returns, benchmark=None, samples: int = 5000, block: int = 21, seed: int = 0,
              ppy: float = YEAR):
    """Reshuffle per-bar returns into `samples` alternative histories.

    block: bars per block (21 = about a month of daily bars; for intraday data
    use about a day of bars). ppy: bars per year, to express results per year.

    Returns (strategy Bootstrap, benchmark Bootstrap or None, chance the strategy's
    Sharpe beats the benchmark's in the same reshuffled history, or None).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 3 * block:
        raise ValueError(f"need at least {3 * block} days of returns to bootstrap")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(r), size=(samples, len(r) // block))
    s_sh, s_cg = _stats(r, starts, block, ppy)
    strat = Bootstrap(s_sh, s_cg, *_actual(r, ppy))
    if benchmark is None:
        return strat, None, None
    b = np.asarray(benchmark, dtype=float)
    if len(b) != len(r):
        raise ValueError("strategy and benchmark returns must cover the same days")
    b_sh, b_cg = _stats(b, starts, block, ppy)
    bench = Bootstrap(b_sh, b_cg, *_actual(b, ppy))
    return strat, bench, float((s_sh > b_sh).mean())


def histogram(strat: Bootstrap, bench: Bootstrap = None, bins: int = 40) -> dict:
    """Shared bins for drawing both distributions side by side."""
    allv = strat.sharpe if bench is None else np.concatenate([strat.sharpe, bench.sharpe])
    lo, hi = np.percentile(allv, [0.25, 99.75])
    edges = np.linspace(lo, hi, bins + 1)
    clip = lambda x: np.clip(x, lo, hi)
    out = {"edges": edges.round(4).tolist(),
           "strategy": np.histogram(clip(strat.sharpe), edges)[0].tolist()}
    if bench is not None:
        out["benchmark"] = np.histogram(clip(bench.sharpe), edges)[0].tolist()
    return out
