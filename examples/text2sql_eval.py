"""Complete example: base-vs-tuned text-to-SQL eval on a schema-held-out set.

Dataset: gretelai/synthetic_text_to_sql (Apache-2.0). The split groups by
normalized schema signature so no table schema appears in both train and
eval - a naive row split leaks near-duplicates in this dataset.

Run on a GPU:
  python examples/text2sql_eval.py --adapter /path/to/adapter --n 300
"""
import argparse
import json
import sqlite3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tuneforge.evals import sql_exact_match, extract_after_marker, run_generations
from tuneforge.models import load_for_inference

END_OF_SQL = "<END_SQL>"
PROMPT = """### Task
Translate the natural-language question into a single SQL query that runs against the given SQLite database. Output only the SQL query, then {end}.

### Database schema
{context}

### Question
{question}

### SQL
"""


def exec_match(context: str, gold: str, pred: str, timeout_s: float = 5.0) -> bool:
    """Run gold and pred on a throwaway in-memory db built from the dataset's
    CREATE/INSERT context; pred executes under a SELECT-only authorizer."""
    try:
        db = sqlite3.connect(":memory:")
        db.executescript(context)
    except Exception:
        return False
    def run(sql, sandbox):
        try:
            if sandbox:
                db.set_authorizer(lambda a, *_, **__: (
                    sqlite3.SQLITE_OK if a in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                                               sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
                    else sqlite3.SQLITE_DENY))
            cur = db.execute(sql)
            rows = cur.fetchall()
            db.set_authorizer(None)
            return sorted([tuple(sorted(map(str, r))) for r in rows])
        except Exception:
            db.set_authorizer(None)
            return None
    g, p = run(gold, False), run(pred, True)
    db.close()
    return g is not None and p is not None and g == p


def score(model, tok, rows, n, max_new):
    exact = execok = 0
    for r in rows[:n]:
        raw = run_generations(model, tok, [PROMPT.format(end=END_OF_SQL,
                              context=r["context"], question=r["question"])], max_new)[0]
        pred = extract_after_marker(raw, END_OF_SQL, stops=("###",))
        exact += sql_exact_match(pred, r["gold_sql"])
        if pred.strip():
            execok += exec_match(r["context"], r["gold_sql"], pred)
    n = min(n, len(rows))
    return {"n": n, "exact_match": exact / n, "execution_accuracy": execok / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--held-out", default="held_out.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=192)
    args = ap.parse_args()
    rows = json.load(open(args.held_out))

    model, tok = load_for_inference(args.base)
    base_scores = score(model, tok, rows, args.n, args.max_new)
    del model
    model, tok = load_for_inference(args.base, args.adapter)
    tuned_scores = score(model, tok, rows, args.n, args.max_new)

    print(json.dumps({"base": base_scores, "finetuned": tuned_scores,
                      "delta_exact": tuned_scores["exact_match"] - base_scores["exact_match"],
                      "delta_exec": tuned_scores["execution_accuracy"] - base_scores["execution_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
