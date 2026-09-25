"""Phase 10: prediction #7 is settled exactly as committed (ratio >= 1.5 AND random > user > global),
and a settled verdict is never changed."""
import pandas as pd
import pytest

from src import split_comparison as sc


def frame(r, u, g):
    return pd.DataFrame({"scheme": ["user", "global", "random"], "ndcg@10": [u, g, r],
                         "ratio_ci_low": [None, None, 1.0], "ratio_ci_high": [None, None, 2.0]})


@pytest.mark.parametrize("r,u,g,verdict", [
    (0.30, 0.15, 0.10, "confirmed"),   # ratio 3.0, order random > user > global
    (0.14, 0.12, 0.10, "refuted"),     # order ok, ratio 1.4 < 1.5
    (0.30, 0.10, 0.15, "refuted"),     # ratio 2.0 ok, but global > user
])
def test_settle(tmp_path, monkeypatch, r, u, g, verdict):
    status = tmp_path / "status.csv"
    pd.DataFrame([{"id": 1, "verdict": "confirmed", "evidence": "x"}]).to_csv(status, index=False)
    monkeypatch.setattr(sc, "STATUS", status)
    sc.settle(frame(r, u, g))
    s = pd.read_csv(status).set_index("id")
    assert s.loc[7, "verdict"] == verdict and s.loc[1, "verdict"] == "confirmed"
    sc.settle(frame(0.9, 0.5, 0.1))  # sealed: a later run never changes it
    assert pd.read_csv(status).set_index("id").loc[7, "verdict"] == verdict
