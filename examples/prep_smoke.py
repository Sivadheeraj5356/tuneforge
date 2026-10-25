"""Build a small SFT slice for the tuneforge smoke run.

Self-contained: pulls N rows from gretelai/synthetic_text_to_sql (Apache-2.0,
non-gated) and writes a jsonl with a single "text" field in the instruction
format used by the verified smoke run. Usage:

    python examples/prep_smoke.py --rows 200 --out smoke_rows.jsonl
"""
from __future__ import annotations

import argparse
import json

END_OF_SQL = "<END_SQL>"

PROMPT_TEMPLATE = """### Task
Translate the natural-language question into a single SQL query that runs against the given SQLite database. Output only the SQL query, then {end_marker}.

### Database schema
{context}

### Question
{question}

### SQL
"""


def build_text(question: str, context: str, gold_sql: str) -> str:
    prompt = PROMPT_TEMPLATE.format(
        question=question.strip(), context=context.strip(), end_marker=END_OF_SQL
    )
    return prompt + gold_sql.strip() + " " + END_OF_SQL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--out", default="smoke_rows.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset  # lazy: needs the training stack

    rows = load_dataset("gretelai/synthetic_text_to_sql", split="train").select(
        range(args.rows)
    )
    with open(args.out, "w") as f:
        for r in rows:
            f.write(
                json.dumps({"text": build_text(r["sql_prompt"], r["sql_context"], r["sql"])})
                + "\n"
            )
    print(f"wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
