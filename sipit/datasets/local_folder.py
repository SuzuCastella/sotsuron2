# sipit/datasets/local_folder.py
import os
import glob
import json
from typing import Iterable, Optional
from .base import TextSource, normalize_text

class LocalFolderSource(TextSource):
    """
    指定フォルダ内のファイル群からテキストを読み出す簡易ソース。
    - デフォルトは *.txt をUTF-8で読み、1ファイル=1ドキュメント
    - JSONL(.gz)にも対応（--local_is_jsonl指定時）
    """

    def __init__(self,
                 folder: str,
                 pattern: str = "*.txt",
                 is_jsonl: bool = False,
                 field: str = "text"):
        self.folder = folder
        self.pattern = pattern
        self.is_jsonl = is_jsonl
        self.field = field

    def iter_texts(self, split=None, seed=42, shuffle=True) -> Iterable[str]:
        if not self.folder or not os.path.isdir(self.folder):
            raise RuntimeError(f"LocalFolderSource: folder not found: {self.folder}")

        paths = sorted(glob.glob(os.path.join(self.folder, self.pattern)))
        if not paths:
            raise RuntimeError(f"No files found in {self.folder} matching {self.pattern}")

        if shuffle:
            import random
            random.Random(seed).shuffle(paths)

        # JSONL(.gz) 読み込みモード
        if self.is_jsonl:
            import gzip
            for p in paths:
                opener = gzip.open if p.endswith(".gz") else open
                try:
                    with opener(p, "rt", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                txt = obj.get(self.field) or ""
                                if txt:
                                    yield normalize_text(txt)
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[WARN] skip {p}: {e}")
            return

        # テキストファイルモード
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
                    if txt:
                        yield normalize_text(txt)
            except Exception as e:
                print(f"[WARN] skip {p}: {e}")
