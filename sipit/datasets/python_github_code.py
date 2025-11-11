# sipit/datasets/python_github_code.py
from typing import Iterable, Optional
from .base import TextSource, normalize_text

class PythonGithubCode(TextSource):
    def __init__(self, local_jsonl: Optional[str] = None):
        self.local_jsonl = local_jsonl

    def iter_texts(self, split=None, seed=42, shuffle=True) -> Iterable[str]:
        if self.local_jsonl:
            import json
            with open(self.local_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    txt = obj.get("content") or obj.get("text") or ""
                    if txt:
                        yield normalize_text(txt)
            return

        # 例: codeparrot/github-code (streaming可能)
        from datasets import load_dataset
        ds = load_dataset("codeparrot/github-code", split="train", streaming=True)
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for ex in ds:
            txt = ex.get("content", "")
            if txt and "python" in (ex.get("lang") or "").lower():
                yield normalize_text(txt)
