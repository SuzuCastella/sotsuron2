# sipit/core/sipit.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import torch

from .verify import verify_batch_with_past, verify_full_vocab_with_past
from sipit.core.policies.gradient import GradientPolicy


@dataclass
class SipItConfig:
    # 共通
    layer_idx: int = -1
    eps: float = 1e-5

    # Brute-force 用
    policy_top_k: Optional[int] = 256   # None or 0 → 全語彙
    batch_size: int = 1024              # チャンクサイズとしても流用

    # Gradient policy 用
    use_gradient: bool = False
    grad_step_size: float = 1e-1
    grad_steps: int = 1
    grad_clip: Optional[float] = None
    grad_topk: int = 256
    grad_fallback_brute: bool = True


class SipItInverter:
    """
    観測 hidden 列からトークン列を逐次復元する。
    - BruteForcePolicy: 語彙候補を列挙し検証（★全語彙はGPU一括ルートで爆速化）
    - GradientPolicy  : Alg.3 に基づき連続埋め込み e を更新 → 近傍 top-K を検証
                        （外したら任意で Brute へフォールバック）
    """
    def __init__(self, lm_wrapper, tokenizer, policy):
        self.lm = lm_wrapper
        self.tokenizer = tokenizer
        self.policy = policy

    # 勾配を使うため inference_mode デコレータは付けない
    def invert_from_hidden(
        self,
        observed_hiddens: List[torch.Tensor],
        config: SipItConfig,
        bos_token_id: Optional[int] = None,
    ) -> List[int]:
        device = next(self.lm.model.parameters()).device

        # prefix/attention_mask 初期化
        if bos_token_id is not None:
            prefix = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            attn = torch.ones_like(prefix)
        else:
            prefix = torch.zeros((1, 0), dtype=torch.long, device=device)
            attn = torch.zeros_like(prefix)

        recovered: List[int] = []
        total_start = time.perf_counter()

        use_grad = isinstance(self.policy, GradientPolicy) or config.use_gradient
        if use_grad:
            E = self.lm.model.get_input_embeddings().weight.data  # [|V|, d]

        for t, target_h in enumerate(observed_hiddens, start=1):
            # 観測ベクトルは inference テンソルになっている可能性があるので通常テンソル化
            target_h = target_h.detach().clone()

            # prefix → past（prefix が空ならスキップ）
            if prefix.numel() == 0:
                past, past_len = None, 0
            else:
                # ここは勾配不要
                with torch.inference_mode():
                    past, past_len = self.lm.build_past(
                        prefix, attention_mask=(attn if attn.numel() > 0 else None)
                    )

            t_start = time.perf_counter()
            tried = 0
            hit = False
            best_tid, best_l2 = None, float("inf")

            if use_grad:
                # ===== Gradient-based policy (Alg.3) =====
                if prefix.numel() == 0:
                    e_init = E.mean(dim=0)
                else:
                    with torch.inference_mode():
                        logits = self.lm.model(prefix).logits[0, -1, :]
                        top1 = int(torch.argmax(logits).item())
                    e_init = E[top1]

                def F_embed(e: torch.Tensor) -> torch.Tensor:
                    return self.lm.hidden_from_continuous_last(
                        prefix_ids=prefix,
                        e_last=e,
                        layer_idx=config.layer_idx,
                        attention_mask=attn,
                    )

                from sipit.core.policies.gradient import GradPolicyConfig
                _ = GradPolicyConfig(
                    step_size=config.grad_step_size,
                    steps=config.grad_steps,
                    norm_clip=config.grad_clip,
                    device=str(device),
                )

                v_star, e_final = self.policy.step(
                    E=E,
                    e_init=e_init,
                    target_h=target_h,
                    F_embed=F_embed,
                    visited=(),
                )

                # 近傍 top-K を計算して検証
                with torch.no_grad():
                    dists = torch.norm(E.to(device) - e_final.to(device)[None, :], dim=1)
                    idx_sorted = torch.argsort(dists)  # 昇順
                    k = max(1, int(config.grad_topk))
                    cand_ids = idx_sorted[:k].to(torch.long)  # [K]

                batch: List[int] = []
                for cid in cand_ids.tolist():
                    batch.append(cid)
                    if len(batch) >= max(1, int(config.batch_size)):
                        with torch.inference_mode():
                            matched, l2, best = verify_batch_with_past(
                                lm=self.lm,
                                past=past,
                                past_length=past_len,
                                candidate_ids=torch.tensor(batch, dtype=torch.long),
                                target_hidden=target_h,
                                layer_idx=config.layer_idx,
                                eps=config.eps,
                            )
                        tried += len(batch)
                        if matched is not None:
                            tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                            prefix = torch.cat([prefix, tok], dim=1)
                            if attn.numel() > 0:
                                attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                            recovered.append(matched)
                            elapsed_ms = (time.perf_counter() - t_start) * 1e3
                            print(
                                f"[t={t}] matched (gradK) token_id={matched} "
                                f"'{self.tokenizer.decode([matched])}' "
                                f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                            )
                            hit = True
                            break
                        else:
                            if l2 < best_l2:
                                best_l2 = l2
                                best_tid = best
                        batch = []

                if not hit and len(batch) > 0:
                    with torch.inference_mode():
                        matched, l2, best = verify_batch_with_past(
                            lm=self.lm,
                            past=past,
                            past_length=past_len,
                            candidate_ids=torch.tensor(batch, dtype=torch.long),
                            target_hidden=target_h,
                            layer_idx=config.layer_idx,
                            eps=config.eps,
                        )
                    tried += len(batch)
                    if matched is not None:
                        tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                        prefix = torch.cat([prefix, tok], dim=1)
                        if attn.numel() > 0:
                            attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                        recovered.append(matched)
                        elapsed_ms = (time.perf_counter() - t_start) * 1e3
                        print(
                            f"[t={t}] matched (gradK) token_id={matched} "
                            f"'{self.tokenizer.decode([matched])}' "
                            f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                        )
                        hit = True
                    else:
                        if l2 < best_l2:
                            best_l2 = l2
                            best_tid = best

                # Grad でも外した場合、フォールバック
                if (not hit) and config.grad_fallback_brute:
                    # ★全語彙高速ルート（チャンク）で一気に検索
                    with torch.inference_mode():
                        matched, l2, best, tried_full = verify_full_vocab_with_past(
                            lm=self.lm,
                            past=past,
                            past_length=past_len,
                            target_hidden=target_h,
                            layer_idx=config.layer_idx,
                            eps=config.eps,
                            chunk_size=max(1, int(config.batch_size)),
                        )
                    tried += tried_full
                    if matched is not None:
                        tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                        prefix = torch.cat([prefix, tok], dim=1)
                        if attn.numel() > 0:
                            attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                        recovered.append(matched)
                        elapsed_ms = (time.perf_counter() - t_start) * 1e3
                        print(
                            f"[t={t}] matched (fallback-FULL) token_id={matched} "
                            f"'{self.tokenizer.decode([matched])}' "
                            f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                        )
                        hit = True
                    else:
                        if l2 < best_l2:
                            best_l2 = l2
                            best_tid = best

            else:
                # ===== Brute-force policy =====
                # ★ policy_top_k が None/0 → 語彙全体を一括検証の高速ルート
                if (config.policy_top_k is None) or (int(config.policy_top_k) == 0):
                    with torch.inference_mode():
                        matched, l2, best, tried_full = verify_full_vocab_with_past(
                            lm=self.lm,
                            past=past,
                            past_length=past_len,
                            target_hidden=target_h,
                            layer_idx=config.layer_idx,
                            eps=config.eps,
                            chunk_size=max(1, int(config.batch_size)),
                        )
                    tried += tried_full
                    if matched is not None:
                        tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                        prefix = torch.cat([prefix, tok], dim=1)
                        if attn.numel() > 0:
                            attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                        recovered.append(matched)
                        elapsed_ms = (time.perf_counter() - t_start) * 1e3
                        print(
                            f"[t={t}] matched (FULL) token_id={matched} "
                            f"'{self.tokenizer.decode([matched])}' "
                            f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                        )
                        hit = True
                    else:
                        if l2 < best_l2:
                            best_l2 = l2
                            best_tid = best

                else:
                    # 既存の部分バッチ列挙ルート
                    batch: List[int] = []
                    for cand_id in self.policy.get_candidates(prefix):
                        batch.append(cand_id)
                        if len(batch) >= max(1, int(config.batch_size)):
                            with torch.inference_mode():
                                matched, l2, best = verify_batch_with_past(
                                    lm=self.lm,
                                    past=past,
                                    past_length=past_len,
                                    candidate_ids=torch.tensor(batch, dtype=torch.long),
                                    target_hidden=target_h,
                                    layer_idx=config.layer_idx,
                                    eps=config.eps,
                                )
                            tried += len(batch)
                            if matched is not None:
                                tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                                prefix = torch.cat([prefix, tok], dim=1)
                                if attn.numel() > 0:
                                    attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                                recovered.append(matched)
                                elapsed_ms = (time.perf_counter() - t_start) * 1e3
                                print(
                                    f"[t={t}] matched token_id={matched} "
                                    f"'{self.tokenizer.decode([matched])}' "
                                    f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                                )
                                hit = True
                                break
                            else:
                                if l2 < best_l2:
                                    best_l2 = l2
                                    best_tid = best
                            batch = []

                    if not hit and len(batch) > 0:
                        with torch.inference_mode():
                            matched, l2, best = verify_batch_with_past(
                                lm=self.lm,
                                past=past,
                                past_length=past_len,
                                candidate_ids=torch.tensor(batch, dtype=torch.long),
                                target_hidden=target_h,
                                layer_idx=config.layer_idx,
                                eps=config.eps,
                            )
                        tried += len(batch)
                        if matched is not None:
                            tok = torch.tensor([[matched]], dtype=torch.long, device=device)
                            prefix = torch.cat([prefix, tok], dim=1)
                            if attn.numel() > 0:
                                attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
                            recovered.append(matched)
                            elapsed_ms = (time.perf_counter() - t_start) * 1e3
                            print(
                                f"[t={t}] matched token_id={matched} "
                                f"'{self.tokenizer.decode([matched])}' "
                                f"(l2={l2:.4e}, tried={tried}, time={elapsed_ms:.1f} ms)"
                            )
                            hit = True
                        else:
                            if l2 < best_l2:
                                best_l2 = l2
                                best_tid = best

            if not hit:
                elapsed_ms = (time.perf_counter() - t_start) * 1e3
                tok_str = self.tokenizer.decode([best_tid]) if best_tid is not None else "<None>"
                raise RuntimeError(
                    f"[t={t}] No candidate matched (eps={config.eps}). "
                    f"best_l2={best_l2:.4e}, best_id={best_tid} ('{tok_str}'), "
                    f"tried={tried}, time={elapsed_ms:.1f} ms"
                )

        total_elapsed = time.perf_counter() - total_start
        print(f"\n=== TIMING ===\nTotal inversion time: {total_elapsed:.3f} s")
        return recovered

    @torch.no_grad()
    def get_observed_hiddens_for_text(
        self, text: str, layer_idx: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        真のテキストから観測 hidden 列を作る。
        戻り値の各ベクトルは detach().clone() 済みで、勾配計算に安全。
        """
        inputs = self.lm.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        outputs = self.lm.forward(**inputs)  # ModelOutput
        hs = outputs.hidden_states
        seq_len = inputs["input_ids"].shape[1]
        observed = [hs[layer_idx][0, t, :].detach().clone() for t in range(seq_len)]
        token_ids = inputs["input_ids"][0].tolist()
        return observed, token_ids
