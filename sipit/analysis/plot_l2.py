# sipit/analysis/plot_l2.py
import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="sipit/results/runs/.../metrics.csv")
    p.add_argument("--fig_out", type=str, default=None)
    p.add_argument("--collision_threshold", type=float, default=1e-5)
    p.add_argument("--title_suffix", type=str, default="")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # 左: モデル別散布（対数軸で個々の点を jitter）
    # 右: 層別の箱ひげ＋平均線

    fig, axes = plt.subplots(1, 2, figsize=(14,4), dpi=160)

    # --- 左 ---
    # モデル名をx軸、yはl2_minの対数
    group_key = "model"
    models = list(df[group_key].unique())
    xs, ys = [], []
    for m in models:
        sub = df[df[group_key]==m]
        y = sub["l2_min"].values
        x = np.full_like(y, fill_value=models.index(m), dtype=float)
        x = x + (np.random.rand(len(y)) - 0.5) * 0.25  # jitter
        xs.append(x); ys.append(y)
    for x,y in zip(xs,ys):
        axes[0].scatter(x, y, s=8, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("L2 Distance (min)")
    axes[0].axhline(args.collision_threshold, ls="--", color="brown")
    axes[0].text(0.05, args.collision_threshold*1.2, "Collision threshold", color="brown")

    # --- 右 ---
    # 層ごとに箱ひげ
    if "layer" in df.columns:
        layers = sorted(df["layer"].unique())
        data = [df[df["layer"]==l]["l2_min"].values for l in layers]
        axes[1].boxplot(data, positions=layers, showmeans=True, meanline=True)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("L2 Distance")
        axes[1].axhline(args.collision_threshold, ls="--", color="brown")
        axes[1].text(layers[0], args.collision_threshold*1.2, "Collision threshold", color="brown")

    fig.suptitle(f"L2 min distributions {args.title_suffix}")
    plt.tight_layout()

    fig_out = args.fig_out or os.path.join(os.path.dirname(args.csv), "..", "figures", "l2_plots.png")
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    plt.savefig(fig_out, bbox_inches="tight")
    print(f"Saved figure: {fig_out}")

if __name__ == "__main__":
    main()
