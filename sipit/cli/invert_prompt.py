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

from sipit.models.hf_loader import load_model_and_tokenizer
from sipit.models.wrappers import LMWrapper
from sipit.core.sipit import SipItInverter, SipItConfig
from sipit.core.policies.brute_force import BruteForcePolicy
from sipit.core.policies.gradient import GradientPolicy, GradPolicyConfig


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

    # PyTorch の TF32 設定（警告抑制気味の推奨設定）
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # topk の正規化
    topk_arg = None if str(args.topk).lower() in ["none", "0"] else int(args.topk)

    # モデル読み込み
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        bnb_compute_dtype=args.bnb_compute_dtype,
        bnb_quant_type=args.bnb_quant_type,
    )
    lm = LMWrapper(model, tokenizer)

    # ポリシー構築
    if args.policy == "brute":
        policy = BruteForcePolicy(tokenizer, model=model, top_k=topk_arg, debug=False)
        use_gradient = False
    else:
        policy = GradientPolicy(GradPolicyConfig(step_size=args.gamma,
                                                steps=args.grad_steps,
                                                norm_clip=args.grad_clip))
        use_gradient = True

    inverter = SipItInverter(lm_wrapper=lm, tokenizer=tokenizer, policy=policy)

    # 観測hidden列の生成
    observed_h, true_ids = inverter.get_observed_hiddens_for_text(args.text, layer_idx=args.layer)
    if args.max_tokens is not None:
        observed_h = observed_h[:args.max_tokens]
        true_ids = true_ids[:args.max_tokens]

    # 設定
    cfg = SipItConfig(
        layer_idx=args.layer,
        eps=args.eps,
        policy_top_k=(topk_arg if topk_arg is not None else None),
        batch_size=args.batch_size,
        use_gradient=use_gradient,
        grad_step_size=args.gamma,
        grad_steps=args.grad_steps,
        grad_clip=args.grad_clip,
        grad_topk=args.grad_topk,
        grad_fallback_brute=args.grad_fallback_brute,
    )

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
            "true_ids": true_ids,
            "recovered_ids": recovered_ids,
            "true_text": true_text,
            "recovered_text": rec_text,
        }
        with open(Path(args.save_dir) / "result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
