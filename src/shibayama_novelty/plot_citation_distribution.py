"""工程12 目的変数（被引用数）の分布を確かめ、工程11 で log(1+被引用数) にした理由を示す。

工程11 は目的変数を log(1+被引用数) にしたが、被引用数そのものの分布を確かめていなかった。
ここで分布を数値と図で示し、そのままの被引用数と log(1+被引用数) のどちらが線形回帰（OLS）に
向くかを、回帰の残差で比べる。README の「工程」12 を参照。指標と図の選び方は独自。

数値（novelty_citation_distribution.json）:
    分布の要約   平均・分散・中央値・分位点・最大・0件の数・歪度・尖度（そのまま / log1p）
    分散 / 平均  ポアソン分布なら 1。大きいほど裾が重い（過分散）
    集中度       上位1%・10% の論文が全被引用に占める割合、ジニ係数
    出版年別     出版年ごとの中央値（新しい論文ほど被引用が少ないか）
    残差         工程11 の M4（Novel_100 + 出版年 + 組織）を、目的変数そのまま / log1p で当てた残差の
                 歪度・尖度と、Shapiro-Wilk 検定。予測値と |残差| の Spearman ρ（不均一分散の目安）
図（figs/）:
    citation_distribution.png         ヒストグラム（そのまま / log1p）、相補累積分布（両対数）、出版年別の箱ひげ図
    citation_regression_residuals.png M4 の残差の Q-Q プロット（そのまま / log1p）

標本: 工程11 と同じ（計算できたすべての論文のうち、出版年・被引用数がある論文）。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/papers.csv
出力: data/derived/shibayama_novelty/citations/novelty_citation_distribution.json
      data/derived/shibayama_novelty/citations/figs/citation_distribution.png
      data/derived/shibayama_novelty/citations/figs/citation_regression_residuals.png

実行:
    python src/shibayama_novelty/plot_citation_distribution.py
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
from clean_reference_titles import PAPERS_CSV, file_sha, rel  # noqa: E402
from plot_novelty_citations import FIG_DIR, OUT_DIR  # noqa: E402
from regress_citations_on_novelty import MODELS, YEAR_POOL_UPTO, design  # noqa: E402
from validate_distance_shared_refs import COLOR  # noqa: E402

PCTLS = (10, 25, 50, 75, 90, 95, 99)
TOP_SHARES = (1, 10)            # 上位何% の論文が全被引用の何割を持つか
YEAR_BINS = [1900, 1999, 2005, 2010, 2015, 2018, 2020, 2022, 2024, 2026]   # 箱ひげ図の出版年の区切り
RESID_MODEL = "M4"              # 残差を比べるモデル（Novel_100 + 出版年 + 組織）


def describe(x: np.ndarray) -> dict:
    return {"n": int(len(x)), "mean": float(x.mean()), "var": float(x.var(ddof=1)),
            "sd": float(x.std(ddof=1)), "median": float(np.median(x)),
            **{f"p{p}": float(v) for p, v in zip(PCTLS, np.percentile(x, PCTLS))},
            "min": float(x.min()), "max": float(x.max()),
            "skewness": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x))}


def gini(x: np.ndarray) -> float:
    """ジニ係数。0 は全論文が同じ被引用、1 に近いほど少数の論文に集中する。"""
    s = np.sort(x)
    n = len(s)
    return float((2 * np.arange(1, n + 1) - n - 1) @ s / (n * s.sum()))


def top_share(x: np.ndarray, pct: int) -> float:
    s = np.sort(x)[::-1]
    k = max(1, int(np.ceil(len(s) * pct / 100)))
    return float(s[:k].sum() / s.sum())


def residual_check(df: pd.DataFrame, y: np.ndarray) -> tuple[dict, np.ndarray]:
    """M4 を最小二乗で当て、残差の形と不均一分散の目安を返す。"""
    X, _ = design(df, "novel_100", MODELS[RESID_MODEL])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ b
    resid = y - fitted
    return {"skewness": float(stats.skew(resid)), "excess_kurtosis": float(stats.kurtosis(resid)),
            "shapiro_w": float(stats.shapiro(resid).statistic),
            "shapiro_p": float(stats.shapiro(resid).pvalue),
            "rho_fitted_abs_resid": float(stats.spearmanr(fitted, np.abs(resid)).statistic)}, resid


def plot_distribution(df: pd.DataFrame) -> None:
    c = df.cited_by_count.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    ax.hist(c, bins=60, color=COLOR)
    ax.set_xlabel("cited_by_count")
    ax.set_ylabel("papers")
    ax.set_title(f"(a) raw: skewness {stats.skew(c):.1f}", fontsize=9)

    ax = axes[0, 1]
    ax.hist(np.log1p(c), bins=40, color=COLOR)
    ax.set_xlabel("log(1 + cited_by_count)")
    ax.set_ylabel("papers")
    ax.set_title(f"(b) log1p: skewness {stats.skew(np.log1p(c)):.2f}", fontsize=9)

    ax = axes[1, 0]
    pos = np.sort(c[c > 0])
    ccdf = 1 - np.arange(len(pos)) / len(pos)
    ax.loglog(pos, ccdf, marker=".", ls="none", ms=3, color="0.3")
    ax.set_xlabel("cited_by_count (papers with >= 1 citation)")
    ax.set_ylabel("P(X >= x)")
    ax.set_title(f"(c) complementary CDF, log-log (zeros: {int((c == 0).sum())})", fontsize=9)

    ax = axes[1, 1]
    bins = pd.cut(df.year_openalex, YEAR_BINS)
    groups = [(f"{int(k.left) + 1}-\n{int(k.right)}" if k.left > YEAR_BINS[0] else f"-{int(k.right)}",
               g.cited_by_count.to_numpy()) for k, g in df.groupby(bins, observed=True)]
    ax.boxplot([np.log1p(v) for _, v in groups], tick_labels=[f"{k}\nn={len(v)}" for k, v in groups])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("publication year")
    ax.set_ylabel("log(1 + cited_by_count)")
    ax.set_title("(d) citations by publication year", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "citation_distribution.png", dpi=300)
    plt.close(fig)


def plot_residuals(res_raw: np.ndarray, res_log: np.ndarray, out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, r, key, label in [(axes[0], res_raw, "raw", "cited_by_count"),
                              (axes[1], res_log, "log1p", "log(1 + cited_by_count)")]:
        stats.probplot(r, dist="norm", plot=ax)
        ax.get_lines()[0].set(marker=".", ms=3, color="0.3")
        ax.get_lines()[1].set(color=COLOR)
        s = out[key]
        ax.set_title(f"{RESID_MODEL} residuals, outcome = {label}\n"
                     f"skewness {s['skewness']:.2f}, excess kurtosis {s['excess_kurtosis']:.1f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "citation_regression_residuals.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nov = pd.read_parquet(NOVELTY_PARQUET, columns=["paper_uid", "novel_100"])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "cited_by_count", "source_org"])
    df = nov.merge(papers, on="paper_uid", how="left")
    n_all = len(df)
    df = df[df.cited_by_count.notna() & df.year_openalex.notna()].copy()
    year = df.year_openalex.astype(int)
    df["year_cat"] = np.where(year <= YEAR_POOL_UPTO, f"<= {YEAR_POOL_UPTO}", year.astype(str))

    c = df.cited_by_count.to_numpy(np.float64)
    raw, log = describe(c), describe(np.log1p(c))
    raw["var_mean_ratio"] = raw["var"] / raw["mean"]
    out: dict = {
        "sample": {"computed": n_all, "used": int(len(df))},
        "raw": raw, "log1p": log,
        "zeros": {"n": int((c == 0).sum()), "share": float((c == 0).mean())},
        "concentration": {"gini": gini(c), **{f"top{p}pct_share": top_share(c, p) for p in TOP_SHARES}},
        "median_by_year_bin": {str(k): {"n": int(len(g)), "median": float(g.cited_by_count.median())}
                               for k, g in df.groupby(pd.cut(df.year_openalex, YEAR_BINS), observed=True)},
        "rho_year_cited": float(stats.spearmanr(df.year_openalex, c).statistic),
    }
    out["residuals"] = {}
    out["residuals"]["raw"], res_raw = residual_check(df, c)
    out["residuals"]["log1p"], res_log = residual_check(df, np.log1p(c))

    plot_distribution(df)
    plot_residuals(res_raw, res_log, out["residuals"])

    print(f"  N={len(df)}（計算できた {n_all} 本のうち）")
    for k, d in (("そのまま", raw), ("log1p", log)):
        print(f"  {k:6s} 平均 {d['mean']:7.2f}  中央値 {d['median']:6.2f}  最大 {d['max']:7.1f}  "
              f"歪度 {d['skewness']:5.2f}  尖度 {d['excess_kurtosis']:6.2f}")
    print(f"  分散/平均 {raw['var_mean_ratio']:.1f}  0件 {out['zeros']['n']} 本  ジニ {out['concentration']['gini']:.3f}  "
          f"上位1% の占有 {out['concentration']['top1pct_share']:.1%}  上位10% {out['concentration']['top10pct_share']:.1%}")
    for k, r in out["residuals"].items():
        print(f"  {RESID_MODEL} の残差（{k}）歪度 {r['skewness']:5.2f}  尖度 {r['excess_kurtosis']:6.2f}  "
              f"Shapiro W {r['shapiro_w']:.3f}  予測値と|残差| の ρ {r['rho_fitted_abs_resid']:+.3f}")

    out["params"] = {"percentiles": list(PCTLS), "top_shares_pct": list(TOP_SHARES), "year_bins": YEAR_BINS,
                     "residual_model": RESID_MODEL, "residual_controls": MODELS[RESID_MODEL]}
    out["inputs"] = {rel(p): file_sha(p) for p in (NOVELTY_PARQUET, PAPERS_CSV)}
    (OUT_DIR / "novelty_citation_distribution.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_citation_distribution.json')}, "
          f"{rel(FIG_DIR / 'citation_distribution.png')}, {rel(FIG_DIR / 'citation_regression_residuals.png')}")


if __name__ == "__main__":
    main()
