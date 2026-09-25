"""Phase 10: the README is generated from results/*.csv and --check catches any drift."""
import pytest

from scripts import write_readme as wr


def test_committed_readme_is_up_to_date():
    assert wr.README.read_text() == wr.render(wr.TEMPLATE.read_text())


def test_unknown_placeholder_is_an_error():
    with pytest.raises(KeyError, match="no.such.value"):
        wr.render("x {{no.such.value}} y")
    with pytest.raises(KeyError, match="table:nope"):
        wr.render("{{table:nope}}")


def test_check_fails_on_stale_readme(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "README.md"
    stale.write_text(wr.render(wr.TEMPLATE.read_text()).replace("0.1679", "0.1700", 1))
    monkeypatch.setattr(wr, "README", stale)
    with pytest.raises(SystemExit) as e:
        wr.main(["--check"])
    assert e.value.code == 1 and "out of date" in capsys.readouterr().out


def test_values_track_the_csvs(tmp_path, monkeypatch):
    """Changing a result file changes the rendered README (numbers aren't hard-coded)."""
    import shutil
    res = tmp_path / "results"
    shutil.copytree(wr.RESULTS, res)
    monkeypatch.setattr(wr, "RESULTS", res)
    before = wr.build_values()["test.two_stage.ndcg"]
    import pandas as pd
    df = pd.read_csv(res / "test.csv")
    df.loc[df.model == "two_stage", "ndcg@10"] = 0.5
    df.to_csv(res / "test.csv", index=False)
    assert before != "0.5000" and wr.build_values()["test.two_stage.ndcg"] == "0.5000"


def test_false_claim_fails_rendering(tmp_path, monkeypatch):
    import shutil
    import pandas as pd
    res = tmp_path / "results"
    shutil.copytree(wr.RESULTS, res)
    monkeypatch.setattr(wr, "RESULTS", res)
    b = pd.read_csv(res / "benchmark.csv")
    b.loc[(b["size"] == "25M") & (b.engine == "pandas"), "with_startup_s"] = 1.0  # pandas "wins" 25M
    b.to_csv(res / "benchmark.csv", index=False)
    with pytest.raises(AssertionError, match="Spark wins at 25M"):
        wr.render(wr.TEMPLATE.read_text())
