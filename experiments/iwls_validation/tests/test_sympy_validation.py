from pathlib import Path

from iwls_validation.sympy_validation import run_sympy_checks


def test_sympy_report_written(tmp_path: Path) -> None:
    out = tmp_path / "sympy_report.txt"
    run_sympy_checks(out)
    text = out.read_text(encoding="utf-8")
    assert "SymPy validation report" in text
    assert "C1" in text and "C2" in text and "C3" in text
