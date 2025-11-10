# sipit/core/verify.py
from typing import Optional, Tuple
import torch

@torch.inference_mode()
def verify_batch_with_past(
    lm,
    past,
    past_length: int,
    candidate_ids: torch.LongTensor,  # [B]
    target_hidden: torch.Tensor,      # [d]
    layer_idx: int,
    eps: float,
) -> Tuple[Optional[int], float, Optional[int]]:
    """
    既存の past に対して候補トークンのバッチを1ステップ前進し、
    末尾の hidden を取得 → target_hidden と L2 距離を比較して閾値判定。
    戻り値: (matched_id or None, best_l2, best_id)
    """
    device = next(lm.model.parameters()).device
    if candidate_ids.ndim == 1:
        candidate_ids_batch = candidate_ids.view(-1, 1)
    else:
        candidate_ids_batch = candidate_ids

    hs_batch = lm.step_hidden_with_past(
        past=past,
        past_length=past_length,
        candidate_ids_batch=candidate_ids_batch.to(device),
        layer_idx=layer_idx,
    )  # [B, hidden]
    diffs = hs_batch - target_hidden.to(hs_batch.dtype).to(device).unsqueeze(0)  # [B,d]
    l2 = torch.norm(diffs, dim=1)  # [B]

    min_val, min_idx = torch.min(l2, dim=0)
    best_l2 = float(min_val.item())
    best_id = int(candidate_ids_batch[min_idx, 0].item())

    if best_l2 <= eps:
        return best_id, best_l2, best_id
    else:
        return None, best_l2, best_id


@torch.inference_mode()
def verify_full_vocab_with_past(
    lm,
    past,
    past_length: int,
    target_hidden: torch.Tensor,   # [d]
    layer_idx: int,
    eps: float,
    chunk_size: int = 8192,
) -> Tuple[Optional[int], float, Optional[int], int]:
    """
    ★ 高速ルート：語彙全体 V を GPU でチャンク分割しつつ一括検証。
    - Python での候補列挙ループを排除
    - 各チャンクごとに lm.step_hidden_with_past を呼び、L2 を GPU 上で計算
    戻り値: (matched_id or None, best_l2, best_id, tried_total)
    """
    device = next(lm.model.parameters()).device
    V = lm.model.get_input_embeddings().weight.shape[0]
    tried_total = 0

    best_l2 = float("inf")
    best_id: Optional[int] = None
    matched_id: Optional[int] = None

    # 連続領域の id をまとめて作成
    # メモリ節約のためにチャンクで流す
    for start in range(0, V, max(1, int(chunk_size))):
        end = min(V, start + max(1, int(chunk_size)))
        cand = torch.arange(start, end, dtype=torch.long, device=device)  # [B]
        hs_batch = lm.step_hidden_with_past(
            past=past,
            past_length=past_length,
            candidate_ids_batch=cand.view(-1, 1),
            layer_idx=layer_idx,
        )  # [B, hidden]

        # L2 を GPU 上で計算
        target = target_hidden.to(hs_batch.dtype).to(device).unsqueeze(0)  # [1,d]
        l2 = torch.norm(hs_batch - target, dim=1)  # [B]

        # 閾値一致の最初の要素をチェック（存在すれば即返す）
        ok_mask = (l2 <= eps)
        if torch.any(ok_mask):
            idx = int(torch.nonzero(ok_mask, as_tuple=False)[0].item())
            matched_id = int(cand[idx].item())
            best_l2 = float(l2[idx].item())
            best_id = matched_id
            tried_total += (end - start)
            return matched_id, best_l2, best_id, tried_total

        # まだ一致がないならベストを更新
        min_val, min_idx = torch.min(l2, dim=0)
        val = float(min_val.item())
        if val < best_l2:
            best_l2 = val
            best_id = int(cand[min_idx].item())

        tried_total += (end - start)

    return None, best_l2, best_id, tried_total
