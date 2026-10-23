"""工程8 検証【4】距離の分布と novelty の分布を原論文と比べる。

原論文 Fig 2A（全参考文献ペアの d_ij）、Fig 2B と S1 Appendix S1.1（論文ごとの Novel_q）と
同じ形式の密度ヒストグラムを描く。原論文は分布の数値を報告していないので、比較は図の形に限る。
原論文のタイトル版の山の位置（図から目で読んだ値）を破線で重ねる。
README の「工程」8 を参照。

標本:
    all    計算できたすべての論文
    min10  n_refs_used >= 10 の論文（参考文献が少ない論文は Novel_q が不安定なため）

入力: data/derived/shibayama_novelty/pairs/reference_title_pairs.parquet, data/derived/shibayama_novelty/scores/novelty_shibayama.parquet
出力: data/derived/shibayama_novelty/validation/novelty_distribution_summary.json
      data/derived/shibayama_novelty/validation/figures/reference_pair_distance_{all,min10}.png
      data/derived/shibayama_novelty/validation/figures/novel_q_distribution_{all,min10}.png
      data/derived/shibayama_novelty/validation/figures/n_refs_used_distribution.png

実行:
    python src/shibayama_novelty/plot_novelty_distributions.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from build_title_distance_pairs import NOVELTY_PARQUET, PAIRS_DIR  # noqa: E402
from clean_reference_titles import file_sha, rel  # noqa: E402
from compute_shibayama_novelty import Q_VALUES  # noqa: E402
from validate_distance_shared_refs import COLOR, FIG_DIR, OUT_DIR  # noqa: E402

BIN_WIDTH = 0.02                 # 独自。原論文はビン幅を示していない
RANGE = (0.0, 1.2)
MIN_REFS_SUBSET = 10
PCTLS = (1, 5, 25, 50, 75, 95, 99)

# 原論文のタイトル版の山の位置（図から目で読んだ値）
SHIB_PEAK = {"pair_distance": 0.45, "novel_100": 0.85, "novel_50": 0.45}
SHIB_N = {"pair_distance": 231_041, "novel": 947}
SHIB_MEAN_REFS = 32              # 原論文 p.5「平均32本、約500ペア」
SHIB_LABEL = "Shibayama21 peak (read from figure)"


def summary(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    hist, edges = np.histogram(x, bins=np.arange(RANGE[0], RANGE[1] + BIN_WIDTH / 2, BIN_WIDTH))
    mode = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()),
            **{f"p{p}": float(v) for p, v in zip(PCTLS, np.percentile(x, PCTLS))},
            "mode_bin_center": mode, "n_above_range": int((x > RANGE[1]).sum())}


def hist(ax, x: np.ndarray) -> None:
    x = x[~np.isnan(x)]
    ax.hist(x, bins=np.arange(RANGE[0], RANGE[1] + BIN_WIDTH / 2, BIN_WIDTH),
            density=True, color=COLOR, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.set_xlim(*RANGE)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"])           # 原論文と同じ目盛り
    ax.text(0.02, 0.95, f"n={len(x):,d}", transform=ax.transAxes, fontsize=7, va="top")


def shib_line(ax, key: str) -> None:
    if key in SHIB_PEAK:
        ax.axvline(SHIB_PEAK[key], color="0.4", ls="--", lw=1)


def shib_legend(fig) -> None:
    """凡例は図の下に1つだけ置く（各パネルの n 表記と重ならないように）。"""
    fig.legend([Line2D([], [], color="0.4", ls="--", lw=1)], [SHIB_LABEL],
               loc="lower center", fontsize=7, frameon=False)


def fig_pairs(rp: pd.DataFrame, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    hist(ax, rp.d.to_numpy())
    shib_line(ax, "pair_distance")
    ax.set_xlabel("title distance $d_{ij}$ (Eq 1)")
    ax.set_ylabel("Density")
    shib_legend(fig)
    fig.suptitle(f"F4-1  Distance of cited-reference pairs (sample: {tag})\n"
                 f"[cf. Shibayama21 Fig 2A, title, n={SHIB_N['pair_distance']:,d}]", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG_DIR / f"reference_pair_distance_{tag}.png", dpi=300)
    plt.close(fig)


def fig_novel(nov: pd.DataFrame, tag: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), sharex=True)
    for ax, q in zip(axes.ravel(), Q_VALUES):
        hist(ax, nov[f"novel_{q}"].to_numpy())
        shib_line(ax, f"novel_{q}")
        ax.set_title(f"q = {q}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("$Novel_q$ (Eq 2)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=8)
    shib_legend(fig)
    fig.suptitle(f"F4-2  Novelty of focal papers (sample: {tag})\n"
                 f"[cf. Shibayama21 Fig 2B and S1 Appendix S1.1, title, n={SHIB_N['novel']}]",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_DIR / f"novel_q_distribution_{tag}.png", dpi=300)
    plt.close(fig)


def fig_nrefs(nov: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.logspace(np.log10(2), np.log10(500), 40)
    n = nov.n_refs_used.to_numpy()
    ax.hist(n, bins=bins, histtype="step", lw=1.5, color=COLOR,
            label=f"This study: mean {n.mean():.1f}, median {np.median(n):.0f}")
    ax.axvline(n.mean(), color=COLOR, lw=1, ls=":")
    ax.axvline(SHIB_MEAN_REFS, color="0.4", ls="--", lw=1,
               label=f"Shibayama21 mean ({SHIB_MEAN_REFS}, p.5)")
    ax.axvline(MIN_REFS_SUBSET, color="0.7", ls="-.", lw=1, label=f"min10 cut ({MIN_REFS_SUBSET})")
    ax.set_xscale("log")
    ax.set_xlabel("references used per focal paper (n_refs_used, log scale)")
    ax.set_ylabel("papers")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("F4-3  Number of references per focal paper (sample: all)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "n_refs_used_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rp = pd.read_parquet(PAIRS_DIR / "reference_title_pairs.parquet")
    nov = pd.read_parquet(NOVELTY_PARQUET)
    keep10 = set(nov.loc[nov.n_refs_used >= MIN_REFS_SUBSET, "paper_uid"])
    samples = {"all": (rp, nov),
               "min10": (rp[rp.paper_uid.isin(keep10)], nov[nov.paper_uid.isin(keep10)])}

    out: dict = {}
    for tag, (rp_s, nov_s) in samples.items():
        blk = {"n_papers": int(len(nov_s)), "pair_distance": summary(rp_s.d.to_numpy(np.float64))}
        for q in Q_VALUES:
            blk[f"novel_{q}"] = summary(nov_s[f"novel_{q}"].to_numpy(np.float64))
        blk["n_refs_used"] = {"mean": float(nov_s.n_refs_used.mean()),
                              "median": float(nov_s.n_refs_used.median())}
        out[tag] = blk
        print(f"── 標本 {tag}: 論文 {len(nov_s)} 本 / ペア {len(rp_s):,d} " + "─" * 16)
        print(f"  d_ij 平均 {blk['pair_distance']['mean']:.3f}"
              f"（最頻ビン {blk['pair_distance']['mode_bin_center']:.2f}）｜ Novel_q 最頻ビン "
              + " ".join(f"q{q}:{blk[f'novel_{q}']['mode_bin_center']:.2f}" for q in Q_VALUES))
        fig_pairs(rp_s, tag)
        fig_novel(nov_s, tag)
    fig_nrefs(nov)
    print("  原論文（目視）: d_ij 山 ≈ 0.45 / Novel_100 山 ≈ 0.85 / Novel_50 山 ≈ 0.45 / 参考文献 平均 32 本")

    out["shibayama21_read_from_figure"] = {"peaks": SHIB_PEAK, "n": SHIB_N,
                                           "mean_refs": SHIB_MEAN_REFS}
    out["params"] = {"bin_width": BIN_WIDTH, "range": list(RANGE),
                     "min_refs_subset": MIN_REFS_SUBSET, "percentiles": list(PCTLS)}
    out["inputs"] = {rel(p): file_sha(p) for p in [PAIRS_DIR / "reference_title_pairs.parquet", NOVELTY_PARQUET]}
    (OUT_DIR / "novelty_distribution_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_distribution_summary.json')}")
    print(f"      {rel(FIG_DIR)}/ reference_pair_distance_{{all,min10}}.png, "
          "novel_q_distribution_{all,min10}.png, n_refs_used_distribution.png")


if __name__ == "__main__":
    main()
