# sipit/core/verify.py
from typing import Optional, Tuple
import torch

@torch.inference_mode()
def verify_candidate(
    lm,                      # LMWrapper
    prefix_ids: torch.LongTensor,
    candidate_id: int,
    target_hidden: torch.Tensor,  # shape [hidden_size] (CPU想定)
    layer_idx: int,
    eps: float = 1e-5,
    attention_mask: Optional[torch.LongTensor] = None,
) -> Tuple[bool, float]:
    """
    候補トークン cand_id を prefix の直後に追加して forward。
    layer_idx 層の hidden と target_hidden の L2距離を計算。
    ε 以下なら True, そうでなければ False を返す。
    """
    device = next(lm.model.parameters()).device
    cand = torch.tensor([[candidate_id]], dtype=torch.long, device=device)

    # prefix と結合
    ids = torch.cat([prefix_ids.to(device), cand], dim=1)
    am = None
    if attention_mask is not None and attention_mask.numel() > 0:
        am = torch.cat([attention_mask.to(device), torch.ones_like(cand)], dim=1)

    hidden_states, _ = lm.forward(input_ids=ids, attention_mask=am)
    h = hidden_states[layer_idx][0, -1, :].detach().cpu()
    diff = h - target_hidden
    l2 = float(torch.linalg.vector_norm(diff, ord=2))
    return (l2 <= eps), l2
