# sipit/cli/invert_prompt.py
import sys, os
from pathlib import Path
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

def main():
    ap = argparse.ArgumentParser(description="SipIt minimal runner")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--text", required=True, help="Ground-truth text to simulate observation")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype)
    lm = LMWrapper(model, tokenizer)
    policy = BruteForcePolicy(tokenizer, model=model, top_k=args.topk)
    inverter = SipItInverter(lm_wrapper=lm, tokenizer=tokenizer, policy=policy)

    # 観測hiddenを真のテキストから作成
    observed_h, true_ids = inverter.get_observed_hiddens_for_text(args.text, layer_idx=args.layer)
    if args.max_tokens is not None:
        observed_h = observed_h[:args.max_tokens]
        true_ids = true_ids[:args.max_tokens]

    cfg = SipItConfig(layer_idx=args.layer, eps=args.eps, policy_top_k=args.topk)
    recovered_ids = inverter.invert_from_hidden(observed_h, config=cfg, bos_token_id=None)

    rec_text = tokenizer.decode(recovered_ids, skip_special_tokens=True)
    true_text = tokenizer.decode(true_ids, skip_special_tokens=True)
    print("\n=== RESULT ===")
    print(f"True : {true_text}")
    print(f"Recov: {rec_text}")
    print(f"Match: {recovered_ids == true_ids}")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        out = {
            "model": args.model,
            "layer": args.layer,
            "eps": args.eps,
            "topk": args.topk,
            "true_ids": true_ids,
            "recovered_ids": recovered_ids,
            "true_text": true_text,
            "recovered_text": rec_text,
        }
        with open(Path(args.save_dir) / "result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
