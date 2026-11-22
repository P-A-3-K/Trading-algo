import numpy as np
import pytest

from quantlab import bootstrap as bs


def test_fast_block_stats_match_brute_force():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0004, 0.01, 300)
    block = 7
    starts = rng.integers(0, len(r), size=(50, len(r) // block))
    sharpe, cagr = bs._stats(r, starts, block)
    ext = np.concatenate([r, r])                       # wrap-around, like the blocks
    for k in range(50):
        sample = np.concatenate([ext[s:s + block] for s in starts[k]])
        sd = sample.std(ddof=1)
        assert sharpe[k] == pytest.approx(sample.mean() / sd * np.sqrt(252), rel=1e-9)
        assert cagr[k] == pytest.approx(np.prod(1 + sample) ** (252 / len(sample)) - 1, rel=1e-9)


def test_spread_matches_theory_for_random_returns():
    """For independent daily returns, Sharpe's standard error is about sqrt((1 + S^2/2) / years)."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, 252 * 20)             # 20 years, Sharpe ~0.79
    st, _, _ = bs.bootstrap(r, samples=4000, block=1, seed=3)
    S = st.actual_sharpe
    se = np.sqrt((1 + S ** 2 / 2) / 20)
    assert st.sharpe.std() == pytest.approx(se, rel=0.12)
    assert np.median(st.sharpe) == pytest.approx(S, abs=0.05)


def test_paired_comparison_and_reproducible():
    rng = np.random.default_rng(4)
    a = rng.normal(0.0005, 0.01, 2000)
    st, be, beat = bs.bootstrap(a, a, samples=500, seed=5)
    assert beat == 0.0                                  # identical returns never strictly beat
    st2, be2, beat2 = bs.bootstrap(a, a * 0.5 - 0.0002, samples=500, seed=5)
    assert beat2 > 0.9                                  # clearly better on the same days
    assert np.array_equal(bs.bootstrap(a, samples=100, seed=9)[0].sharpe,
                          bs.bootstrap(a, samples=100, seed=9)[0].sharpe)
    with pytest.raises(ValueError):
        bs.bootstrap(a[:10])


def test_luck_payload_shape():
    import pandas as pd
    from quantlab.chart import luck_payload
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0004, 0.01, 1500))
    p = luck_payload(r, r * 0.8, samples=300)
    assert p["samples"] == 300 and 0 <= p["beats"] <= 1
    assert p["strategy"]["p5"] < p["strategy"]["p50"] < p["strategy"]["p95"]
    assert len(p["hist"]["edges"]) == len(p["hist"]["strategy"]) + 1 == len(p["hist"]["benchmark"]) + 1
    assert sum(p["hist"]["strategy"]) == 300
