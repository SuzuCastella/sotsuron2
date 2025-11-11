# sipit/datasets/pile.py
from typing import Iterable, Optional
from .base import TextSource, normalize_text

class ThePile(TextSource):
    def __init__(self, local_jsonl: Optional[str] = None):
        self.local_jsonl = local_jsonl

    def iter_texts(self, split=None, seed=42, shuffle=True) -> Iterable[str]:
        """
        The Pile データセットをストリーミングで読み込む。
        優先順位:
            1. local_jsonl が指定されていればそれを読む
            2. EleutherAI/the_pile_deduped を使用（標準）
            3. monology/pile-10k にフォールバック（軽量テスト用）
        """
        # ---- ローカルJSONL ----
        if self.local_jsonl:
            import json
            with open(self.local_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    txt = obj.get("text") or obj.get("content") or ""
                    if txt:
                        yield normalize_text(txt)
            return

        # ---- HF datasets ----
        from datasets import load_dataset

        dataset_candidates = [
            ("EleutherAI/the_pile_deduped", "train"),
            ("monology/pile-10k", "train"),
        ]

        for name, split_name in dataset_candidates:
            try:
                print(f"[INFO] Trying dataset: {name} (streaming=True)")
                ds = load_dataset(name, split=split_name, streaming=True)
                if shuffle:
                    ds = ds.shuffle(seed=seed, buffer_size=10_000)
                for ex in ds:
                    txt = ex.get("text") or ex.get("content") or ""
                    if txt:
                        yield normalize_text(txt)
                return
            except Exception as e:
                print(f"[WARN] Failed to load {name}: {e}")
                continue

        raise RuntimeError(
            "No valid The Pile dataset found. "
            "Try installing `datasets` properly or specify a local JSONL."
        )
