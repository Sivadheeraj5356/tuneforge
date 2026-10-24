import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge import evals


def test_exact_match_normalizes():
    assert evals.exact_match("  Hello   World ", "hello world")


def test_sql_normalization():
    assert evals.sql_exact_match("SELECT  *\nFROM t;", "select * from t")
    assert not evals.sql_exact_match("select * from t", "select * from u")


def test_extract_after_marker():
    out = evals.extract_after_marker("SELECT 1 <END_SQL> trailing junk", "<END_SQL>")
    assert out == "SELECT 1"
    fenced = evals.extract_after_marker("```sql\nSELECT 2\n```", "<END_SQL>")
    assert fenced == "SELECT 2"


def test_compare_runs_delta():
    base = {"n": 10, "exact_match": 0.2, "execution_accuracy": 0.3}
    tuned = {"n": 10, "exact_match": 0.6, "execution_accuracy": 0.7}
    cmp = evals.compare_runs(base, tuned)
    assert abs(cmp["metrics"]["exact_match"]["delta"] - 0.4) < 1e-9
