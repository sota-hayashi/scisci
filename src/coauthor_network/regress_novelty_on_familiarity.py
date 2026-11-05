"""工程3 論文の新規性 Novel_q を、チームのくり返しの組の割合・既知のメンバーの割合で回帰する。

仮説: 共著ネットワークが固定化している（くり返しの組・既知のメンバーが多い）論文ほど、新規性が小さい。
README の「工程」3 を参照。統制変数の選び方と、回帰に使う論文の絞り方は独自。

目的変数: Novel_100、Novel_50（shibayama_novelty の工程3 の出力）
説明変数（モデルごと）:
    R   くり返しの組の割合
    K   既知のメンバーの割合
    RK  R と K の両方
統制変数: log(著者数)（著者の行数。大きいチームほど既知のメンバーが入りやすい。Zeng21）、出版年のダミー
         （1999年以前は1つにまとめる。さらに、回帰に使う標本の中で論文が MIN_YEAR_N 本未満の年は、古い方から
         続く年とまとめて MIN_YEAR_N 本以上の区分にする。1本しかない年はてこ比が1になり HC3 が計算できないため。独自）、
         組織のダミー、log(参考文献数)（n_refs_used）、分野のダミー
         （primary_field。論文が MIN_FIELD 本未満の分野は other）
標準誤差は HC3。p 値と 95% 区間は t 分布。係数は「割合が 0 から 1 に変わったときの Novel_q の変化」。

回帰に使う論文: Novelty を計算できた論文のうち、所属が不明の著者を含む論文、誰か決められない著者を含む論文、
組織のデータの始まりから BURN_IN_YEARS 年以内の論文を除き、その版のチームが2人以上の論文。
主に見るのは版 all（全著者・記録は組織をまたぐ）、除く年数 2。次の場合でも結果が変わらないかを確かめる:
    版 toyota / all_org / toyota_org、除く年数 1 / 3

入力: data/derived/coauthor_network/team/team_familiarity.parquet,
      data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/papers.csv
出力: data/derived/coauthor_network/regression/novelty_familiarity_regression.json

実行:
    python src/coauthor_network/regress_novelty_on_familiarity.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from compute_team_familiarity import OUT_DIR as TEAM_DIR
from identify_authors import ROOT, file_sha, rel

TEAM_PARQUET = TEAM_DIR / "team_familiarity.parquet"
NOVELTY_PARQUET = ROOT / "data" / "derived" / "shibayama_novelty" / "scores" / "novelty_shibayama.parquet"
PAPERS_CSV = ROOT / "data" / "raw" / "papers.csv"
OUT_DIR = ROOT / "data" / "derived" / "coauthor_network" / "regression"
OUTCOMES = ("novel_100", "novel_50")
MODELS = {"R": ["repeat_pair_share"], "K": ["known_member_share"], "RK": ["repeat_pair_share", "known_member_share"]}
VERSIONS = ("all", "toyota", "all_org", "toyota_org")
MAIN_VERSION, MAIN_BURN_IN = "all", 2
BURN_IN_CHECKS = (1, 2, 3)
YEAR_POOL_UPTO = 1999
MIN_FIELD = 20
MIN_YEAR_N = 5             # 標本の中でこれ未満の論文しかない出版年は、続く年とまとめる


def ols_hc3(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """OLS を解き、HC3（不均一分散に頑健な標準誤差）で係数ごとの t 検定をする。

    shibayama_novelty の regress_citations_on_novelty.py と同じ計算。HC3 は各観測の残差を (1 − てこ比) で割って
    重みにし、てこ比が大きい点の影響を抑える。
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = X.T @ (X * (resid / (1 - lev))[:, None] ** 2)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    dof = n - k
    tcrit = stats.t.ppf(0.975, dof)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    coefs = {nm: {"coef": float(b[i]), "se": float(se[i]),
                  "ci95": [float(b[i] - tcrit * se[i]), float(b[i] + tcrit * se[i])],
                  "p": float(2 * stats.t.sf(abs(b[i] / se[i]), dof))} for i, nm in enumerate(names)}
    return {"n": int(n), "k": int(k), "adj_r2": 1 - (1 - r2) * (n - 1) / dof, "coefs": coefs}


def load() -> pd.DataFrame:
    team = pd.read_parquet(TEAM_PARQUET)
    nov = pd.read_parquet(NOVELTY_PARQUET, columns=["paper_uid", "n_refs_used", *OUTCOMES])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "primary_field"])
    df = nov.merge(team, on="paper_uid", how="left").merge(papers, on="paper_uid", how="left")
    year = df.year_openalex.astype(int)
    df["year_cat"] = np.where(year <= YEAR_POOL_UPTO, f"<= {YEAR_POOL_UPTO}", year.astype(str))
    field = df.primary_field.fillna("unknown")
    df["field_cat"] = field.where(field.map(field.value_counts()) >= MIN_FIELD, "other")
    df["log_n_authors"] = np.log(df.n_author_rows)
    df["log_n_refs"] = np.log(df.n_refs_used)
    return df


def sample(df: pd.DataFrame, version: str, burn_in: int) -> pd.DataFrame:
    keep = (~df.has_unknown_affil) & (~df.has_unassigned_author) & (df.years_since_org_start >= burn_in) \
        & (df[f"n_members_{version}"] >= 2)
    return df[keep].copy()


def year_bins(d: pd.DataFrame) -> pd.Series:
    """標本の中で、論文が MIN_YEAR_N 本以上になるまで、古い方から続く年をまとめた区分。"""
    counts = d.year_cat.value_counts()
    order = sorted(counts.index, key=lambda c: -1 if c.startswith("<=") else int(c))
    mapping, start, acc = {}, None, 0
    for c in order:
        start = c if start is None else start
        acc += counts[c]
        mapping[c] = start
        if acc >= MIN_YEAR_N:
            start, acc = None, 0
    if start is not None and len(set(mapping.values())) > 1:   # 最後に残った少ない年は、ひとつ前の区分に入れる
        prev = sorted(set(mapping.values()) - {start}, key=lambda c: -1 if c.startswith("<=") else int(c))[-1]
        mapping = {k: (prev if v == start else v) for k, v in mapping.items()}
    return d.year_cat.map(mapping)


def fit(d: pd.DataFrame, outcome: str, version: str, xs: list[str]) -> dict:
    cols = [f"{x}_{version}" for x in xs]
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), d[cols].set_axis(xs, axis=1),
                   d[["log_n_authors", "log_n_refs"]],
                   pd.get_dummies(year_bins(d), prefix="year", drop_first=True, dtype=float),
                   pd.get_dummies(d.source_org, prefix="org", drop_first=True, dtype=float),
                   pd.get_dummies(d.field_cat, prefix="field", drop_first=True, dtype=float)], axis=1)
    res = ols_hc3(X.to_numpy(float), d[outcome].to_numpy(float), list(X.columns))
    keep = {k: v for k, v in res["coefs"].items() if k in (*xs, "log_n_authors", "log_n_refs")}
    return {"n": res["n"], "adj_r2": res["adj_r2"], "coefs": keep}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    out: dict = {"sample_chain": {}, "results": {}}
    base = df
    chain = {"computed": len(base)}
    base = base[~base.has_unknown_affil]; chain["no_unknown_affil"] = len(base)
    base = base[~base.has_unassigned_author]; chain["no_unassigned_author"] = len(base)
    base = base[base.years_since_org_start >= MAIN_BURN_IN]; chain[f"after_burn_in_{MAIN_BURN_IN}"] = len(base)
    for v in VERSIONS:
        chain[f"members_2plus_{v}"] = int((base[f"n_members_{v}"] >= 2).sum())
    out["sample_chain"] = {k: int(x) for k, x in chain.items()}

    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    for v in VERSIONS:
        for b in (BURN_IN_CHECKS if v == MAIN_VERSION else (MAIN_BURN_IN,)):
            d = sample(df, v, b)
            key = f"{v}_burnin{b}"
            out["results"][key] = {}
            line = []
            for y in OUTCOMES:
                for mname, xs in MODELS.items():
                    r = fit(d, y, v, xs)
                    out["results"][key][f"{y}_{mname}"] = r
                    c = "  ".join(f"{x[:6]} {r['coefs'][x]['coef']:+.3f} (p={r['coefs'][x]['p']:.2f}){star(r['coefs'][x]['p']):<3s}" for x in xs)
                    line.append(f"    {y:9s} {mname:2s} N={r['n']:3d}  {c}")
            print(f"── 版 {v}、除く年数 {b}" + ("（主に見る）" if (v, b) == (MAIN_VERSION, MAIN_BURN_IN) else ""))
            print("\n".join(line))

    out["params"] = {"outcomes": list(OUTCOMES), "models": MODELS, "versions": list(VERSIONS),
                     "main": {"version": MAIN_VERSION, "burn_in": MAIN_BURN_IN}, "burn_in_checks": list(BURN_IN_CHECKS),
                     "controls": ["log(n_author_rows)", "log(n_refs_used)", "year dummies", "org dummies", "field dummies"],
                     "year_pool_upto": YEAR_POOL_UPTO, "min_year_n": MIN_YEAR_N, "min_field": MIN_FIELD, "se": "HC3", "ci": "t, 95%"}
    out["inputs"] = {rel(p): file_sha(p) for p in (TEAM_PARQUET, NOVELTY_PARQUET, PAPERS_CSV)}
    (OUT_DIR / "novelty_familiarity_regression.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_familiarity_regression.json')}")


if __name__ == "__main__":
    main()
