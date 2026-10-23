"""工程10 Novel_100 と被引用数の関係を、出版年を絞らずに全体で図にする。

工程9（検証【5】）は 2016年以降の article に絞り、上位10%入り（0/1）を従属変数にした。
被引用の回帰を改良する（目的変数の対数、ダミー変数の追加）前に、絞り込みも 0/1 化もしない
生の関係を確かめる。README の「工程」10 を参照。

図（独自。原論文に同じ図は無い）:
    横軸 Novel_100、縦軸 被引用数（OpenAlex の cited_by_count）。点の色は出版年
    左 縦軸そのまま / 右 縦軸を対数（0 を含むので symlog、1 未満は線形）
    黒線は Novel_100 の十分位ごとの被引用数の中央値

標本: 計算できたすべての論文（782本）のうち cited_by_count がある論文。出版年・文献種別で絞らない。
古い論文ほど Novel_100 が小さく被引用が多い（工程9）ので、色で出版年を見えるようにする。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/papers.csv
出力: data/derived/shibayama_novelty/citations/novelty_citations_summary.json
      data/derived/shibayama_novelty/citations/figs/novel_100_vs_cited_by.png

実行:
    python src/shibayama_novelty/plot_novelty_citations.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from build_title_distance_pairs import NOVELTY_PARQUET  # noqa: E402
from clean_reference_titles import PAPERS_CSV, ROOT, file_sha, rel  # noqa: E402
from validate_novelty_citations import YEAR_FROM  # noqa: E402

OUT_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "citations"
FIG_DIR = OUT_DIR / "figs"
N_BINS = 10          # 中央値の線を引く Novel_100 の分位の数
SYMLOG_LINTHRESH = 1


def spearman(df: pd.DataFrame) -> dict:
    r = stats.spearmanr(df.novel_100, df.cited_by_count)
    return {"n": int(len(df)), "rho": float(r.statistic), "p": float(r.pvalue)}


def binned_median(df: pd.DataFrame) -> pd.DataFrame:
    """Novel_100 の十分位ごとに、Novel_100 と被引用数の中央値を取る。"""
    b = pd.qcut(df.novel_100, N_BINS, labels=False, duplicates="drop")
    return df.groupby(b).agg(novel_100=("novel_100", "median"), cited_by=("cited_by_count", "median"),
                             n=("novel_100", "size")).reset_index(drop=True)


def plot(df: pd.DataFrame, med: pd.DataFrame, rho: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, scale in zip(axes, ("linear", "symlog")):
        sc = ax.scatter(df.novel_100, df.cited_by_count, c=df.year_openalex, cmap="viridis",
                        s=10, alpha=0.7, lw=0)
        ax.plot(med.novel_100, med.cited_by, color="black", lw=1.5, marker="o", ms=3,
                label="median by Novel_100 decile")
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Novel_100")
        ax.set_ylabel("cited_by_count" + (" (symlog)" if scale == "symlog" else ""))
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.colorbar(sc, ax=axes, label="publication year (year_openalex)")
    fig.suptitle(f"Novel_100 vs citations, all years (n={rho['n']}, "
                 f"Spearman rho={rho['rho']:+.3f})", fontsize=10)
    fig.savefig(FIG_DIR / "novel_100_vs_cited_by.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nov = pd.read_parquet(NOVELTY_PARQUET, columns=["paper_uid", "novel_100", "n_refs_used"])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "cited_by_count"])
    df = nov.merge(papers, on="paper_uid", how="left")
    n_all = len(df)
    df = df[df.cited_by_count.notna() & df.year_openalex.notna()].copy()

    rho = spearman(df)
    med = binned_median(df)
    out = {
        "sample_chain": {"computed": n_all, "with_cited_by_and_year": int(len(df))},
        "spearman_novel_100_cited_by": {
            "all": rho,
            # 工程9 と同じ年で分けた参考値。出版年の偏りがどれだけ効くかを見る
            f"before_{YEAR_FROM}": spearman(df[df.year_openalex < YEAR_FROM]),
            f"from_{YEAR_FROM}": spearman(df[df.year_openalex >= YEAR_FROM])},
        "cited_by": {"median": float(df.cited_by_count.median()), "max": int(df.cited_by_count.max()),
                     "n_zero": int((df.cited_by_count == 0).sum())},
        "year_range": [int(df.year_openalex.min()), int(df.year_openalex.max())],
        "median_by_novel_100_decile": med.round(4).to_dict(orient="records"),
        "params": {"n_bins": N_BINS, "symlog_linthresh": SYMLOG_LINTHRESH,
                   "filters": "none (all years, all document types)"},
        "inputs": {rel(p): file_sha(p) for p in (NOVELTY_PARQUET, PAPERS_CSV)},
    }
    plot(df, med, rho)

    s = out["spearman_novel_100_cited_by"]
    print(f"  N={rho['n']}（計算できた {n_all} 本のうち）  出版年 {out['year_range'][0]}〜{out['year_range'][1]}")
    for k, v in s.items():
        print(f"  {k:12s} N={v['n']:4d}  Spearman ρ {v['rho']:+.3f}  p={v['p']:.3g}")
    (OUT_DIR / "novelty_citations_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_citations_summary.json')}, {rel(FIG_DIR / 'novel_100_vs_cited_by.png')}")


if __name__ == "__main__":
    main()
