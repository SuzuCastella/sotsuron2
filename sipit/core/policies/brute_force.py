# sipit/core/policies/brute_force.py
from typing import Iterable, Optional
import torch


class BruteForcePolicy:
    """
    語彙を逐次検証に回すベースライン探索ポリシー。
    - prefix が空 (t=1) のときは forward せず全語彙を返す。
    - top_k=None または 0 のときは常に全語彙探索。
    - top_k>0 のときは「Top-K優先 → 残り語彙を網羅」。
    """

    def __init__(self, tokenizer, model=None, top_k: Optional[int] = 256, debug: bool = False):
        self.tokenizer = tokenizer
        self.model = model
        if top_k == 0:  # 0をNone扱いにして全探索
            top_k = None
        self.top_k = top_k
        self.debug = debug

    @torch.inference_mode()
    def get_candidates(self, prefix_ids) -> Iterable[int]:
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("tokenizer.vocab_size が取得できません。")

        # ========== ① prefix空判定 ==========
        prefix_empty = (
            prefix_ids is None
            or prefix_ids.numel() == 0
            or (hasattr(prefix_ids, "shape") and prefix_ids.shape[-1] == 0)
        )

        if prefix_empty:
            if self.debug:
                print(f"[policy] prefix is empty → return full vocab ({vocab_size})")
            yield from range(vocab_size)
            return

        # ========== ② full探索モード ==========
        if self.model is None or self.top_k is None:
            if self.debug:
                print(f"[policy] full-vocab enumeration ({vocab_size})")
            yield from range(vocab_size)
            return

        # ========== ③ Top-K探索モード ==========
        device = next(self.model.parameters()).device
        outputs = self.model(input_ids=prefix_ids.to(device), return_dict=True)
        logits = outputs.logits[0, -1, :]

        k = int(self.top_k)
        if k <= 0 or k > logits.numel():
            k = logits.numel()

        topk = torch.topk(logits, k=k)
        top_ids = topk.indices.tolist()
        yielded = set(top_ids)

        if self.debug:
            print(
                f"[policy] prefix_len={prefix_ids.shape[-1]}, "
                f"top_k={self.top_k}, "
                f"top_first={len(top_ids)}, "
                f"rest={vocab_size - len(top_ids)}"
            )

        count = 0
        # 1) Top-K優先
        for tid in top_ids:
            yield tid
            count += 1
        # 2) 残り全探索
        for tid in range(vocab_size):
            if tid not in yielded:
                yield tid
                count += 1

        # ========== ④ 予期せぬゼロ件検知 ==========
        if count == 0:
            raise RuntimeError(
                "[policy-error] 候補が0件です。top_kやprefix処理を確認してください。"
            )
