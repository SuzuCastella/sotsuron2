# sipit/tools/make_wiki_cache.py
import os
import gzip
import json
import random
import argparse
from pathlib import Path

def normalize_text(s: str) -> str:
    # 余分な改行や空白を整形（復元性より一貫性重視）
    return " ".join((s or "").replace("\n", " ").strip().split())

def _flush_shard(buf, out_dir: Path, shard_idx: int) -> int:
    if not buf:
        return shard_idx
    path = out_dir / f"shard-{shard_idx:04d}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for obj in buf:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[SAVE] {path}  ({len(buf)} samples)")
    buf.clear()
    return shard_idx + 1

def main():
    ap = argparse.ArgumentParser(
        description="Make a small, token-windowed (segment) Wikipedia cache in JSONL shards."
    )
    # 出力仕様
    ap.add_argument("--out_dir", type=str, required=True,
                    help="出力ディレクトリ（例: data/wiki_cache_seg）")
    ap.add_argument("--total", type=int, default=10000,
                    help="書き出すサンプル数（合計）")
    ap.add_argument("--shard_size", type=int, default=2000,
                    help="1シャードあたりの行数（例: 2000）")

    # 乱択・長さ
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min_tokens", type=int, default=8,
                    help="短すぎるセグメントを除外（例: 8）")
    ap.add_argument("--max_tokens", type=int, default=20,
                    help="セグメント窓の長さ（例: 20）")
    ap.add_argument("--segment_stride", type=int, default=20,
                    help="セグメント窓のストライド。20で非重複、10で半重複など")

    # トークナイザ
    ap.add_argument("--model", type=str, default="gpt2",
                    help="トークン長計測に使うトークナイザ名")

    # データセット（新: wikimedia/wikipedia）
    ap.add_argument("--wiki_repo", type=str, default="wikimedia/wikipedia",
                    help="HF Hub の repo_id（例: wikimedia/wikipedia）")
    ap.add_argument("--wiki_config", type=str, default="20231101.en",
                    help="スナップショット/言語設定（例: 20231101.en）")

    args = ap.parse_args()

    # 事前準備
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("`datasets` が未インストールです。`pip install -U datasets`") from e
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("`transformers` が未インストールです。`pip install -U transformers`") from e

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # データセット読み込み（streaming=True）
    ds = None
    last_err = None
    print(f"[INFO] Trying dataset: {args.wiki_repo} / {args.wiki_config} (streaming=True)")
    try:
        ds = load_dataset(args.wiki_repo, args.wiki_config, split="train", streaming=True)
    except Exception as e:
        last_err = e
        ds = None

    if ds is None:
        raise RuntimeError(
            "Wikipediaの読み込みに失敗しました。ネットワーク/認証/プロキシなどをご確認ください。\n"
            "完全オフライン運用をしたい場合は、一度取得できる環境で本スクリプトを実行し、生成されたJSONLを配布してください。"
        ) from last_err

    # streaming でもバッファ付きシャッフルが可能
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    rng = random.Random(args.seed)
    written = 0
    shard_idx = 1
    buf = []

    win = max(1, int(args.max_tokens))
    stride = max(1, int(args.segment_stride))
    if stride > win:
        print(f"[WARN] segment_stride({stride}) > max_tokens({win}) です。非重複にするなら stride==win を推奨。")

    try:
        for ex in ds:
            # 1記事
            txt = normalize_text(ex.get("text", "") or "")
            if not txt:
                continue

            # トークン化
            ids = tok.encode(txt, add_special_tokens=False)
            n = len(ids)
            if n < args.min_tokens:
                continue  # 極短記事は捨て

            # --- 切り分け：固定長 win (既定=20) の窓でスライス ---
            # 最後の端数（win未満）は原則として捨てる。必要なら拾う処理に変えてOK。
            if n >= win:
                for s in range(0, n - win + 1, stride):
                    sub = ids[s:s + win]
                    # min_tokens を満たすものだけ採用（通常は win==max なので満たす）
                    if len(sub) < args.min_tokens:
                        continue
                    txt_sub = tok.decode(sub, skip_special_tokens=True)
                    buf.append({"text": txt_sub, "n_tokens": len(sub)})
                    written += 1

                    if written % args.shard_size == 0:
                        shard_idx = _flush_shard(buf, out_dir, shard_idx)
                    if written >= args.total:
                        break
            # n < win だが min_tokens を満たす場合：そのまま採用（窓より短いものも拾いたい場合）
            elif n >= args.min_tokens:
                # ここを採用する/しないは運用ポリシー次第。デフォは採用しておく。
                buf.append({"text": tok.decode(ids, skip_special_tokens=True), "n_tokens": n})
                written += 1
                if written % args.shard_size == 0:
                    shard_idx = _flush_shard(buf, out_dir, shard_idx)

            if written >= args.total:
                break

        # 最終フラッシュ
        shard_idx = _flush_shard(buf, out_dir, shard_idx)
        print(f"[DONE] total written: {written}, shards: {shard_idx-1}, out_dir={out_dir.resolve()}")

    except KeyboardInterrupt:
        shard_idx = _flush_shard(buf, out_dir, shard_idx)
        print(f"\n[INTERRUPTED] partial written: {written}, shards: {shard_idx-1}, out_dir={out_dir.resolve()}")
    except Exception as e:
        shard_idx = _flush_shard(buf, out_dir, shard_idx)
        raise
    # Windows の symlink 警告は無害。必要なら環境変数で黙らせ可能：
    # set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if __name__ == "__main__":
    main()
