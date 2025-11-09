# sipit/core/sipit.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from .verify import verify_candidate

@dataclass
class SipItConfig:
    layer_idx: int = -1
    eps: float = 1e-5
    policy_top_k: Optional[int] = 256

class SipItInverter:
    """
    SipIt: 観測hidden列（各tのh_t^b）から逐次的にトークン列を復元する。
    """
    def __init__(self, lm_wrapper, tokenizer, policy):
        self.lm = lm_wrapper
        self.tokenizer = tokenizer
        self.policy = policy

    @torch.inference_mode()
    def invert_from_hidden(
        self,
        observed_hiddens: List[torch.Tensor],
        config: SipItConfig,
        bos_token_id: Optional[int] = None,
    ) -> List[int]:
        """
        各時刻tについて、候補トークンを列挙してverify_candidateで照合し、
        一致したトークンを逐次的に確定する。
        """
        device = next(self.lm.model.parameters()).device

        # prefix/attention_mask初期化
        if bos_token_id is not None:
            prefix = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            attn = torch.ones_like(prefix)
        else:
            prefix = torch.zeros((1, 0), dtype=torch.long, device=device)
            attn = torch.zeros_like(prefix)

        recovered: List[int] = []

        for t, target_h in enumerate(observed_hiddens, start=1):
            hit = False
            best_tid, best_l2 = None, float("inf")

            for cand_id in self.policy.get_candidates(prefix):
                ok, l2 = verify_candidate(
                    lm=self.lm,
                    prefix_ids=prefix,
                    candidate_id=cand_id,
                    target_hidden=target_h,
                    layer_idx=config.layer_idx,
                    eps=config.eps,
                    attention_mask=attn if attn.numel() > 0 else None,
                )

                # 最良候補を記録
                if l2 < best_l2:
                    best_tid, best_l2 = cand_id, l2

                if ok:
                    tok = torch.tensor([[cand_id]], dtype=torch.long, device=device)
                    prefix = torch.cat([prefix, tok], dim=1)
                    if attn.numel() > 0:
                        attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                    recovered.append(cand_id)
                    print(f"[t={t}] matched token_id={cand_id} "
                          f"'{self.tokenizer.decode([cand_id])}' (l2={l2:.4e})")
                    hit = True
                    break

            if not hit:
                tok_str = self.tokenizer.decode([best_tid]) if best_tid is not None else "<None>"
                raise RuntimeError(
                    f"[t={t}] No candidate matched (eps={config.eps}). "
                    f"best_l2={best_l2:.4e}, best_id={best_tid} ('{tok_str}')"
                )

        return recovered

    @torch.inference_mode()
    def get_observed_hiddens_for_text(
        self, text: str, layer_idx: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        真のテキストから観測hidden列を生成する。
        """
        # GPT-2 では add_special_tokens=False にしてBOS等を避ける
        inputs = self.lm.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        hs, _ = self.lm.forward(**inputs)
        seq_len = inputs["input_ids"].shape[1]
        observed = [hs[layer_idx][0, t, :].detach().cpu() for t in range(seq_len)]
        token_ids = inputs["input_ids"][0].tolist()
        return observed, token_ids
