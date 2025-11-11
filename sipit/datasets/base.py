# sipit/datasets/base.py
from abc import ABC, abstractmethod
from typing import Iterable, Optional

class TextSource(ABC):
    """テキストのストリーミング供給IF。"""

    @abstractmethod
    def iter_texts(
        self,
        split: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> Iterable[str]:
        ...

def normalize_text(s: str) -> str:
    # 最低限の前処理（空白の正規化など）
    return " ".join(s.strip().split())
