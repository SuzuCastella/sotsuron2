# sipit/cli/invert_prompt.py
import sys, os
from pathlib import Path

# repo ルートを import パスに
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import json
import torch
from typing import Tuple, Dict, Any

from sipit.models.hf_loader import load_model_and_tokenizer
from sipit.models.wrappers import LMWrapper
from sipit.core.sipit import SipItInverter, SipItConfig
from sipit.core.policies.brute_force import BruteForcePolicy
from sipit.core.policies.gradient import GradientPolicy, GradPolicyConfig


def _normalize_topk(topk_str: Any):
    """'none' / '0' / None を None に、数字文字列は int に正規化"""
    if topk_str is None:
        return None
    s = str(topk_str).strip().lower()
    if s in {"none", "0", ""}:
        return None
    return int(s)


def build_inverter_from_args(args) -> Tuple[SipItInverter, Any, SipItConfig]:
    # PyTorch の TF32 設定（警告抑制寄り）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # topk 正規化
    topk_arg = _normalize_topk(getattr(args, "topk", None))

    # ==== ここから 量子化/ dtype の正規化 ====
    def _norm_q(x):
        if x is None:
            return "none"
        s = str(x).strip().lower()
        return "none" if s in {"", "none", "null", "false", "0"} else s

    def _opt_str(x):
        return None if x is None else str(x).strip().lower()

    q_mode = _norm_q(getattr(args, "quantization", "none"))
    bnb_compute = _opt_str(getattr(args, "bnb_compute_dtype", None))
    bnb_qtype   = _opt_str(getattr(args, "bnb_quant_type", None))
    dtype_str   = str(getattr(args, "dtype", "float32")).strip().lower()
    device_str  = str(getattr(args, "device", "cuda")).strip().lower()
    model_name  = getattr(args, "model", "gpt2")

    # ログ（任意）
    print(f"🔹 Loading model: {model_name} on {device_str} "
          f"(dtype={dtype_str}, quantization={q_mode})")

    # ==== ここまで 量子化/ dtype の正規化 ====

    # モデル読み込み（量子化は文字列で渡す）
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device_str,
        dtype=dtype_str,
        quantization=q_mode,                 # ← None ではなく 'none'
        bnb_compute_dtype=bnb_compute,
        bnb_quant_type=bnb_qtype,
    )
    lm = LMWrapper(model, tokenizer)

    # ポリシー
    policy_name = getattr(args, "policy", "brute")
    if policy_name == "brute":
        policy = BruteForcePolicy(tokenizer, model=model, top_k=topk_arg, debug=False)
        use_gradient = False
    else:
        policy = GradientPolicy(
            GradPolicyConfig(
                step_size=getattr(args, "gamma", 1e-1),
                steps=getattr(args, "grad_steps", 1),
                norm_clip=getattr(args, "grad_clip", None),
            )
        )
        use_gradient = True

    inverter = SipItInverter(lm_wrapper=lm, tokenizer=tokenizer, policy=policy)

    cfg = SipItConfig(
        layer_idx=getattr(args, "layer", -1),
        eps=getattr(args, "eps", 1e-5),
        policy_top_k=(topk_arg if topk_arg is not None else None),
        batch_size=getattr(args, "batch_size", 1024),
        use_gradient=use_gradient,
        grad_step_size=getattr(args, "gamma", 1e-1),
        grad_steps=getattr(args, "grad_steps", 1),
        grad_clip=getattr(args, "grad_clip", None),
        grad_topk=getattr(args, "grad_topk", 256),
        grad_fallback_brute=getattr(args, "grad_fallback_brute", False),
    )
    return inverter, tokenizer, cfg



def invert_from_text(inverter: SipItInverter, tokenizer, text: str, cfg: SipItConfig,
                     max_tokens: int | None = None) -> Dict[str, Any]:
    # 観測hiddenと真ID
    observed_h, true_ids = inverter.get_observed_hiddens_for_text(text, layer_idx=cfg.layer_idx)

    # ★保険：上限でスライス
    if max_tokens is not None and len(true_ids) > max_tokens:
        observed_h = observed_h[:max_tokens]
        true_ids = true_ids[:max_tokens]

    # 反転
    recovered_ids = inverter.invert_from_hidden(observed_h, config=cfg, bos_token_id=None)

    # L2（埋め込み空間）簡易指標
    emb = inverter.lm_wrapper.model.get_input_embeddings().weight.detach()
    l2_each = torch.norm(emb[true_ids] - emb[recovered_ids], dim=-1)
    l2_min = float(torch.min(l2_each).item())

    return {
        "l2_min": l2_min,
        "layer": int(cfg.layer_idx),
        "n_tokens": int(len(true_ids)),
        "true_ids": true_ids,
        "recovered_ids": recovered_ids,
    }



def parse_args():
    ap = argparse.ArgumentParser(description="SipIt runner (brute/gradient, batching & quantization)")

    # モデル周り
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--quantization", default="none",
                    choices=["none", "int8", "int4", "gptq", "awq"])
    ap.add_argument("--bnb_compute_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bnb_quant_type", default="nf4", choices=["nf4", "fp4"])

    # ポリシー選択
    ap.add_argument("--policy", default="brute", choices=["brute", "grad"])

    # Gradient policy 用
    ap.add_argument("--gamma", type=float, default=1e-1, help="Gradient step size γ")
    ap.add_argument("--grad_steps", type=int, default=1, help="Gradient steps (Alg.3は1でも可)")
    ap.add_argument("--grad_clip", type=float, default=None, help="Gradient norm clip（任意）")
    ap.add_argument("--grad_topk", type=int, default=256, help="勾配更新後に検証する近傍候補数")
    ap.add_argument("--grad_fallback_brute", action="store_true",
                    help="grad候補が全て外れた場合にブルートフォースへフォールバック")

    # タスク設定
    ap.add_argument("--text", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--topk", type=str, default="256", help="brute の候補数。'0' or 'none' で全語彙")
    ap.add_argument("--batch_size", type=int, default=1024, help="候補検証のバッチサイズ（brute/grad共通）")
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--save_dir", default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    inverter, tokenizer, cfg = build_inverter_from_args(args)

    # 観測hidden列の生成（max_tokens の裁断はここで）
    observed_h, true_ids = inverter.get_observed_hiddens_for_text(args.text, layer_idx=cfg.layer_idx)
    if args.max_tokens is not None:
        observed_h = observed_h[:args.max_tokens]
        true_ids = true_ids[:args.max_tokens]

    # 反転実行
    recovered_ids = inverter.invert_from_hidden(observed_h, config=cfg, bos_token_id=None)

    # 結果表示
    rec_text = tokenizer.decode(recovered_ids, skip_special_tokens=True)
    true_text = tokenizer.decode(true_ids, skip_special_tokens=True)
    print("\n=== RESULT ===")
    print(f"True : {true_text}")
    print(f"Recov: {rec_text}")
    print(f"Match: {recovered_ids == true_ids}")

    # 保存（任意）
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        # 簡易 l2_min も保存
        emb = inverter.lm_wrapper.model.get_input_embeddings().weight.detach()
        true_vecs = emb[true_ids]
        rec_vecs = emb[recovered_ids]
        l2_each = torch.norm(true_vecs - rec_vecs, dim=-1)
        l2_min = float(torch.min(l2_each).item())

        out = {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "bnb_compute_dtype": args.bnb_compute_dtype,
            "bnb_quant_type": args.bnb_quant_type,
            "layer": args.layer,
            "eps": args.eps,
            "topk": args.topk,
            "batch_size": args.batch_size,
            "policy": args.policy,
            "gamma": args.gamma,
            "grad_steps": args.grad_steps,
            "grad_clip": args.grad_clip,
            "grad_topk": args.grad_topk,
            "grad_fallback_brute": args.grad_fallback_brute,
            "true_ids": true_ids.tolist() if hasattr(true_ids, "tolist") else list(true_ids),
            "recovered_ids": recovered_ids.tolist() if hasattr(recovered_ids, "tolist") else list(recovered_ids),
            "true_text": true_text,
            "recovered_text": rec_text,
            "l2_min_embed": l2_min,
        }
        with open(Path(args.save_dir) / "result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
