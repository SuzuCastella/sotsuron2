# sipit/cli/run_dataset.py
import os
import sys
import csv
import time
import argparse
import random
from typing import Dict, Iterable, Tuple, Any

# ---- import path safety: project root ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ====== datasets wrappers ======
from sipit.datasets.wikipedia_en import WikipediaEn
from sipit.datasets.c4 import C4
from sipit.datasets.pile import ThePile
from sipit.datasets.python_github_code import PythonGithubCode
from sipit.datasets.local_folder import LocalFolderSource  # ★ローカル束読み

# ====== SIPIT entrypoints (from invert_prompt.py) ======
try:
    from sipit.cli.invert_prompt import build_inverter_from_args, invert_from_text
except Exception as e:
    print("[FATAL] invert_prompt のヘルパー読み込みに失敗しました。", file=sys.stderr)
    raise

# -------------------------
# helpers
# -------------------------
def _normalize_quant(q):
    """'none' 系は None にせず 'none' 文字列で統一（ローダ実装に依存するため）。"""
    if q is None:
        return "none"
    s = str(q).strip().lower()
    return "none" if s in {"", "none", "null", "false", "0"} else s

def _make_sources(args) -> Dict[str, Any]:
    """
    データセット名 -> 供給器 の辞書を構築。
    引数:
      - wikipedia-en: --wiki_local でローカルjsonl(.gz)を直接指定可能
      - local-folder: フォルダの *.txt / *.jsonl(.gz) を読む汎用
      - c4, pile, python-github-code: HF datasets (環境により未使用でもOK)
    """
    all_sources: Dict[str, Any] = {
        "wikipedia-en": WikipediaEn(local_jsonl=args.wiki_local),
        "c4": C4(),
        "pile": ThePile(),
        "python-github-code": PythonGithubCode(),
        "local-folder": LocalFolderSource(
            folder=args.local_folder_path,
            pattern=args.local_file_glob,
            is_jsonl=args.local_is_jsonl,
            field=args.local_json_field,
        ),
    }
    selected: Dict[str, Any] = {}
    for name in args.datasets:
        if name not in all_sources:
            raise ValueError(f"Unknown dataset: {name}")
        selected[name] = all_sources[name]
    return selected

def uniform_mix_sample(sources: Dict[str, Any], total: int, seed: int) -> Iterable[Tuple[str, str]]:
    """データセット間を一様サンプリングしつつテキストをストリーム供給。"""
    rng = random.Random(seed)
    iters = {k: iter(v.iter_texts(seed=seed)) for k, v in sources.items()}
    keys = list(sources.keys())
    for _ in range(total):
        k = rng.choice(keys)
        try:
            txt = next(iters[k])
        except StopIteration:
            iters[k] = iter(sources[k].iter_texts(seed=seed))
            txt = next(iters[k])
        yield (k, txt)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run SIPIT over mixed datasets and save metrics CSV.")

    # dataset / sampling
    p.add_argument("--datasets", nargs="+", default=["wikipedia-en", "c4", "pile", "python-github-code"],
                   help="使用するデータセット名の列（例: wikipedia-en c4）")
    p.add_argument("--limit", type=int, default=100000, help="処理するサンプル数（上限）")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--min_tokens", type=int, default=8, help="短すぎる文の除外しきい値")
    p.add_argument("--max_tokens", type=int, default=256, help="サンプル選別用の最大長（前段フィルタ）")

    # wikipedia-en のローカルJSONL読み
    p.add_argument("--wiki_local", type=str, default=None,
                   help="Wikipedia用のローカルjsonl(.gz)。指定時はオンライン不要")

    # local-folder 汎用読み
    p.add_argument("--local_folder_path", type=str, default=None,
                   help="local-folder ソースのフォルダパス")
    p.add_argument("--local_file_glob", type=str, default="*.txt",
                   help="local-folder で読むファイルパターン（例: *.jsonl.gz）")
    p.add_argument("--local_is_jsonl", action="store_true",
                   help="local-folder のファイルを JSONL として読む")
    p.add_argument("--local_json_field", type=str, default="text",
                   help="local-folder のJSONLで読むフィールド名")

    # output paths
    p.add_argument("--run_name", type=str, default=None,
                   help="結果フォルダ名。未指定なら <model>_<timestamp>")
    p.add_argument("--out_root", type=str, default=os.path.join("sipit", "results"),
                   help="results のルート（runs/, figures/ が作られる）")
    p.add_argument("--verbose", action="store_true")

    # model / SIPIT (pass-through; must match invert_prompt.py)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--quantization", type=str, default="none")
    p.add_argument("--bnb_compute_dtype", type=str, default=None)
    p.add_argument("--bnb_quant_type", type=str, default=None)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--eps", type=float, default=3e-3)
    p.add_argument("--policy", type=str, default="grad")
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--grad_steps", type=int, default=10)
    p.add_argument("--grad_topk", type=int, default=256)
    p.add_argument("--grad_fallback_brute", action="store_true")
    p.add_argument("--batch_size", type=int, default=1024)

    # repeats
    p.add_argument("--trials", type=int, default=3, help="1テキストあたりの繰り返し試行回数")

    # ★ 1試行あたりの最大トークン長（前段/後段の二重ガード）
    p.add_argument("--max_tokens_per_trial", type=int, default=20,
                   help="各試行で使う最大トークン数（先頭から）")
    p.add_argument("--truncate_long", action="store_true",
                   help="長文は先頭 max_tokens_per_trial トークンへ切り詰めて使用（未指定ならスキップ）")

    # 余剰をデータセット名とみなす救済（--datasets 付け忘れ対策）
    args, rest = p.parse_known_args()
    if rest:
        if args.verbose:
            print(f"[INFO] extra args detected; interpreting as datasets: {rest}")
        args.datasets = rest

    # 量子化の正規化（ローダ実装が 'none' 文字列を期待）
    args.quantization = _normalize_quant(args.quantization)

    return args

# -------------------------
# main
# -------------------------
def main():
    args = parse_args()

    # 出力先（先に作って絶対パスを表示）
    run_name = args.run_name or f"{args.model.replace('/', '_')}_{int(time.time())}"
    runs_dir = os.path.join(args.out_root, "runs", run_name)
    figures_dir = os.path.join(args.out_root, "figures")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    csv_path = os.path.join(runs_dir, "metrics.csv")

    print(f"[RUN] model={args.model} policy={args.policy} datasets={args.datasets}")
    print(f"[OUT] runs_dir: {os.path.abspath(runs_dir)}")
    print(f"[OUT] csv_path: {os.path.abspath(csv_path)}")

    try:
        # モデル/インバータ
        inverter, tokenizer, cfg = build_inverter_from_args(args)
        if args.verbose:
            print("[INFO] inverter built; starting sampling...")

        # データソース
        sources = _make_sources(args)

        # CSV を即オープン（ディレクトリは必ず生成される）
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index", "dataset", "trial", "text_len", "layer",
                    "l2_min", "model", "quantization", "dtype"
                ],
            )
            writer.writeheader()

            processed = 0
            for i, (ds_name, txt) in enumerate(uniform_mix_sample(sources, total=args.limit, seed=args.seed)):
                # ---- 前段フィルタ：トークン化 ----
                ids = tokenizer.encode(txt, add_special_tokens=False)

                # 最大長（試行の上限）に対する方針
                if len(ids) > args.max_tokens_per_trial:
                    if args.truncate_long:
                        ids = ids[:args.max_tokens_per_trial]
                        txt = tokenizer.decode(ids, skip_special_tokens=True)
                    else:
                        # スキップ運用
                        continue

                # 最小長チェック（例: 8〜）
                if len(ids) < args.min_tokens:
                    continue
                # （任意）前段 max_tokens（256）もチェックしたければここで
                if len(ids) > args.max_tokens:
                    continue

                for t in range(args.trials):
                    try:
                        # ---- 後段の保険：invert側にも max_tokens を渡す ----
                        metrics = invert_from_text(
                            inverter, tokenizer, txt, cfg,
                            max_tokens=args.max_tokens_per_trial
                        )
                        writer.writerow({
                            "index": i,
                            "dataset": ds_name,
                            "trial": t,
                            "text_len": len(ids),  # 実際に使った長さ（切り詰め後）
                            "layer": metrics.get("layer", args.layer),
                            "l2_min": metrics.get("l2_min", float("nan")),
                            "model": args.model,
                            "quantization": args.quantization or "",
                            "dtype": args.dtype,
                        })
                        processed += 1
                    except Exception as e:
                        print(f"[WARN] index={i} trial={t} failed: {e}", file=sys.stderr)

                if args.verbose and (i + 1) % 100 == 0:
                    print(f"[PROG] sampled={i + 1}  written={processed}")

        print(f"[DONE] CSV saved: {os.path.abspath(csv_path)}")
        print("[HINT] 可視化: py -m sipit.analysis.plot_l2 --csv " + csv_path)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] user canceled.")
    except Exception as e:
        print("[FATAL] Unhandled exception:", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        # 例外でも runs_dir は既に存在している想定

if __name__ == "__main__":
    main()
