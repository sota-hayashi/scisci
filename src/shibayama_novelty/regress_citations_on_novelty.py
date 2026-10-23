"""工程11 log(1+被引用数) を Novel_q で回帰し、出版年・組織・参考文献数・文献種別を統制する。

工程9 は原論文に合わせて、統制変数なしのロジスティック回帰（上位10%入り）を当てた。
工程10 で、Novel_100 と被引用の関係は出版年（期間の間のずれ）と参考文献数（期間の中の正の相関）に
歪められていることが分かった。そこで目的変数を log(1+被引用数) にした線形回帰（OLS）で、
統制変数を1つずつ足し、Novel_100 の係数がどう変わるかを見る。README の「工程」11 を参照。
2026-10-02 の打ち合わせの提案（目的変数の対数、ダミー変数）に従う。統制変数の選び方は独自。

モデル（すべて切片あり）:
    M1  Novel_100 のみ
    M2  M1 + 出版年のダミー（1999年以前は1つにまとめる。論文が少ないため。独自）
    M3  M1 + 組織のダミー（source_org の4組織）
    M4  M1 + 出版年 + 組織
    M5  M4 + log(参考文献数)（n_refs_used。Novel_q の計算に使った本数）
    M6  M5 + 文献種別のダミー（article / conference-paper / その他・不明）
M6 は q ごとにも当てる（Novel_99〜Novel_50）。

標準誤差は不均一分散に頑健な HC3。p 値と 95% 区間は t 分布（自由度 n − 説明変数の数）で出す。
係数 b は「Novel_q が 1 増えると log(1+被引用数) が b 増える」。Novel_q の範囲は 0.15〜1.2 程度なので、
0.1 増えたときの被引用の変化率 exp(0.1·b) − 1 も出す（被引用が多い論文での近似）。

標本: 計算できたすべての論文（782本）。出版年・被引用数が欠けた論文は除く。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/papers.csv
出力: data/derived/shibayama_novelty/citations/novelty_citation_regression.json

実行:
    python src/shibayama_novelty/regress_citations_on_novelty.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from build_title_distance_pairs import NOVELTY_PARQUET
from clean_reference_titles import PAPERS_CSV, file_sha, rel
from compute_shibayama_novelty import Q_VALUES
from plot_novelty_citations import OUT_DIR

YEAR_POOL_UPTO = 1999          # これ以前の出版年は1つのダミーにまとめる
MAIN_TYPES = ("article", "conference-paper")   # それ以外は「その他・不明」
STEP = 0.1                     # 変化率を出すときの Novel_q の増分

# モデル名 → 足していく統制変数のグループ
MODELS = {
    "M1": [],
    "M2": ["year"],
    "M3": ["org"],
    "M4": ["year", "org"],
    "M5": ["year", "org", "log_n_refs"],
    "M6": ["year", "org", "log_n_refs", "doc_type"],
}


def ols_hc3(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """OLS を解き、HC3（不均一分散に頑健な標準誤差）で係数ごとの t 検定をする。

    HC3 は各観測の残差を (1 − てこ比) で割って重みにする。てこ比が大きい点（外れた説明変数の値）の
    影響を抑えるので、標本が数百の回帰で推奨される。
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # てこ比（ハット行列の対角）
    meat = X.T @ (X * (resid / (1 - lev))[:, None] ** 2)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    dof = n - k
    t = b / se
    tcrit = stats.t.ppf(0.975, dof)
    ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    coefs = {nm: {"coef": float(b[i]), "se": float(se[i]),
                  "ci95": [float(b[i] - tcrit * se[i]), float(b[i] + tcrit * se[i])],
                  "p": float(2 * stats.t.sf(abs(t[i]), dof))} for i, nm in enumerate(names)}
    return {"n": int(n), "k": int(k), "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / dof, "coefs": coefs}


def design(df: pd.DataFrame, novel_col: str, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """切片 + Novel_q + 統制変数の計画行列を作る。ダミーは基準カテゴリを1つ落とす。"""
    parts = [pd.Series(1.0, index=df.index, name="const"), df[novel_col]]
    if "year" in groups:
        parts.append(pd.get_dummies(df.year_cat, prefix="year", drop_first=True, dtype=float))
    if "org" in groups:
        parts.append(pd.get_dummies(df.source_org, prefix="org", drop_first=True, dtype=float))
    if "log_n_refs" in groups:
        parts.append(df.log_n_refs)
    if "doc_type" in groups:
        parts.append(pd.get_dummies(df.type_cat, prefix="type", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    return X.to_numpy(np.float64), list(X.columns)


def summarize(res: dict, novel_col: str) -> dict:
    """出力 JSON に残す要約。Novel_q と log(参考文献数) の係数、ダミーは数だけ。"""
    c = res["coefs"]
    keep = {k: v for k, v in c.items() if k in (novel_col, "log_n_refs")}
    b = c[novel_col]["coef"]
    return {"n": res["n"], "k": res["k"], "r2": res["r2"], "adj_r2": res["adj_r2"], **keep,
            f"pct_change_per_{STEP}": float(np.exp(STEP * b) - 1),
            "n_dummies": {g: sum(k.startswith(f"{g}_") for k in c) for g in ("year", "org", "type")}}


def main() -> None:
    nov = pd.read_parquet(NOVELTY_PARQUET, columns=["paper_uid", "n_refs_used",
                                                    *[f"novel_{q}" for q in Q_VALUES]])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "cited_by_count",
                                              "source_org", "type"])
    df = nov.merge(papers, on="paper_uid", how="left")
    n_all = len(df)
    df = df[df.cited_by_count.notna() & df.year_openalex.notna()].copy()

    df["y"] = np.log1p(df.cited_by_count)
    df["log_n_refs"] = np.log(df.n_refs_used)
    year = df.year_openalex.astype(int)
    df["year_cat"] = np.where(year <= YEAR_POOL_UPTO, f"<= {YEAR_POOL_UPTO}", year.astype(str))
    df["type_cat"] = df["type"].where(df["type"].isin(MAIN_TYPES), "other_or_unknown")
    y = df.y.to_numpy(np.float64)

    out: dict = {"sample": {"computed": n_all, "used": int(len(df)),
                            "by_org": df.source_org.value_counts().to_dict(),
                            "by_type": df.type_cat.value_counts().to_dict(),
                            "n_year_categories": int(df.year_cat.nunique())},
                 "models_novel_100": {}, "m6_by_q": {}}

    print(f"── 目的変数 log(1+被引用数)、N={len(df)}（計算できた {n_all} 本のうち） " + "─" * 10)
    print(f"  {'':3s} {'Novel_100 の係数':>16s} {'95%区間':>18s} {'p':>7s} "
          f"{f'+{STEP} で被引用':>12s} {'log参考文献数':>12s} {'adj R2':>7s}")
    for name, groups in MODELS.items():
        X, names = design(df, "novel_100", groups)
        s = summarize(ols_hc3(X, y, names), "novel_100")
        s["controls"] = groups
        out["models_novel_100"][name] = s
        c = s["novel_100"]
        nr = f"{s['log_n_refs']['coef']:+.3f}" if "log_n_refs" in s else "—"
        print(f"  {name:3s} {c['coef']:+16.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
              f"{c['p']:7.3f} {s[f'pct_change_per_{STEP}']:+11.1%} {nr:>12s} {s['adj_r2']:7.3f}")

    print("── M6（全統制）を q ごとに " + "─" * 30)
    for q in Q_VALUES:
        col = f"novel_{q}"
        X, names = design(df, col, MODELS["M6"])
        s = summarize(ols_hc3(X, y, names), col)
        out["m6_by_q"][f"q{q}"] = s
        c = s[col]
        print(f"  q={q:3d} 係数 {c['coef']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p={c['p']:.3f}")

    out["params"] = {"outcome": "log1p(cited_by_count)", "model": "OLS", "se": "HC3",
                     "ci": "t, 95%", "year_pool_upto": YEAR_POOL_UPTO, "main_types": list(MAIN_TYPES),
                     "pct_change_step": STEP,
                     "baseline_categories": "first category of each dummy group (alphabetical) is dropped"}
    out["inputs"] = {rel(p): file_sha(p) for p in (NOVELTY_PARQUET, PAPERS_CSV)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "novelty_citation_regression.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_citation_regression.json')}")


if __name__ == "__main__":
    main()
