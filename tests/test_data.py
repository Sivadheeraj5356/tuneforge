import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge import data


def test_pack_respects_budget():
    texts = ["x" * 100, "y" * 100, "z" * 100]
    packed = data.pack_sequences(texts, 55)  # 55 tokens * 4 chars = 220 budget
    assert all(len(p) <= 230 for p in packed)
    assert len(packed) == 2  # two fit together, third needs its own bin


def test_pack_never_drops_content():
    texts = [f"sample-{i} " * (i + 1) for i in range(50)]
    packed = data.pack_sequences(texts, 256)
    joined = "\n".join(packed)
    for t in texts:
        assert t in joined


def test_group_split_no_leakage():
    rows = [{"schema": f"s{i%7}", "q": i} for i in range(70)]
    tr, ev = data.train_eval_split(rows, 0.3, key="schema")
    assert not {r["schema"] for r in tr} & {r["schema"] for r in ev}


def test_split_deterministic():
    rows = [{"q": i} for i in range(30)]
    a = data.train_eval_split(rows, 0.2, seed=7)
    b = data.train_eval_split(rows, 0.2, seed=7)
    assert a == b


def test_render_chat_chatml():
    s = data.render_chat([{"role": "system", "content": "a"},
                          {"role": "user", "content": "b"}])
    assert s.startswith("<|im_start|>system") and s.count("<|im_end|>") == 2
