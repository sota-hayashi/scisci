"""工程9 検証【5】Novel_q は被引用を予測するか（原論文の2つ目の妥当性検証の再現）。

原論文は、分野内で被引用が上位1% に入るか（0/1）を従属変数、Novel_q を説明変数とする
説明変数1つのロジスティック回帰を q ごとに6本当てた（統制変数なし。2010年の生物医学論文）。
同じ形の回帰を本標本で当てる。README の「工程」9 を参照。

本標本に合わせて変えた点:
    出版年   2016年以降に限る。古い論文ほど Novel_q が小さく被引用が多いので、
             そのまま結ぶと novelty と被引用の関係とは無関係に負の関連が出る。
             2015年以前との差はこのスクリプトが記述統計と検定で示す
    文献種別 type == "article" に限る（会議録などを混ぜない）
    従属変数 上位10%（is_in_top_10_percent）。上位1% は該当が少なく判定できない（結果は参考に出す）
             どちらも OpenAlex の値（2026-10-01 取得）で、同じ出版年・同じ subfield・同じ文献種別の中での順位。
             citation_percentiles.csv から読む（作ったコードは fetch_citation_percentiles.py。工程には入れない）

Novel_q は 0〜1 程度の値なので、オッズ比は Novel_q が 1 増えたときの値になり、大きく出る。
原論文とは標本の作り方（上位1% を多めに抽出）が違うので、比べられるのは符号と有意性だけである。

参考文献数は Novel_100 とも被引用とも関係する（交絡）。原論文の式に合わせて統制変数は入れず、
その大きさを Spearman ρ で記録する。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/papers.csv,
      data/derived/shibayama_novelty/citations/citation_percentiles.csv
出力: data/derived/shibayama_novelty/validation/novelty_citation_validation.json

実行:
    python src/shibayama_novelty/validate_novelty_citations.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from build_title_distance_pairs import NOVELTY_PARQUET
from clean_reference_titles import PAPERS_CSV, file_sha, rel
from compute_shibayama_novelty import Q_VALUES
from fetch_citation_percentiles import PERCENTILES_CSV
from validate_distance_shared_refs import OUT_DIR

YEAR_FROM = 2016
DOC_TYPE = "article"
OUTCOMES = ("is_in_top_10_percent", "is_in_top_1_percent")   # 先頭が主、後ろは参考
Z95 = stats.norm.ppf(0.975)

# 原論文のオッズ比（上位1%、title 版）
SHIBAYAMA21_OR = {100: 154.02, 99: 53.26, 95: 29.27, 90: 20.30, 80: 14.97, 50: 5.87}


def to_bool(s: pd.Series) -> pd.Series:
    """CSV から読んだ True/False の文字列を bool にする（欠損は NaN のまま）。"""
    return s.map({True: True, False: False, "True": True, "False": False})


def logit(x: np.ndarray, y: np.ndarray) -> dict:
    """切片 + 説明変数1つのロジスティック回帰を最尤法（ニュートン法）で解く。

    標準誤差は情報行列の逆行列から取り、Wald の z で p 値と 95% 区間を出す。
    """
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        info = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(info, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    se = float(np.sqrt(np.linalg.inv(info)[1, 1]))
    coef = float(b[1])
    return {"coef": coef, "se": se, "odds_ratio": float(np.exp(coef)),
            "or_ci95": [float(np.exp(coef - Z95 * se)), float(np.exp(coef + Z95 * se))],
            "p": float(2 * stats.norm.sf(abs(coef / se)))}


def describe(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "novel_100_mean": float(df.novel_100.mean()),
            "top10_share": float(df.top10.mean()), "cited_by_median": float(df.cited_by_count.median())}


def main() -> None:
    nov = pd.read_parquet(NOVELTY_PARQUET)
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "type", "cited_by_count"])
    pct = pd.read_csv(PERCENTILES_CSV, usecols=["paper_uid", *OUTCOMES])
    df = nov.merge(papers, on="paper_uid", how="left").merge(pct, on="paper_uid", how="left")
    for c in OUTCOMES:
        df[c] = to_bool(df[c])
    n_all = len(df)
    df = df[df.is_in_top_10_percent.notna() & (df.type == DOC_TYPE)].copy()
    df["top10"] = df.is_in_top_10_percent.astype(bool)
    out: dict = {"sample_chain": {"computed": n_all, "with_percentile_and_article": int(len(df))}}

    # 出版年で分ける理由（古い論文ほど novelty が小さく被引用が多い）
    old, new = df[df.year_openalex < YEAR_FROM], df[df.year_openalex >= YEAR_FROM]
    tab = [[int(old.top10.sum()), int((~old.top10).sum())], [int(new.top10.sum()), int((~new.top10).sum())]]
    out["by_period"] = {
        f"before_{YEAR_FROM}": describe(old), f"from_{YEAR_FROM}": describe(new),
        # 検定は独自（原論文に無い）。分布の形を仮定しない順位の検定と、2×2 表の χ² 検定
        "p_novel_100_mannwhitney": float(stats.mannwhitneyu(old.novel_100, new.novel_100).pvalue),
        "p_cited_by_mannwhitney": float(stats.mannwhitneyu(old.cited_by_count, new.cited_by_count).pvalue),
        "p_top10_chi2": float(stats.chi2_contingency(tab)[1]),
    }
    for k in (f"before_{YEAR_FROM}", f"from_{YEAR_FROM}"):
        b = out["by_period"][k]
        print(f"  {k:12s} N={b['n']:4d}  Novel_100 平均 {b['novel_100_mean']:.3f}  "
              f"上位10%率 {b['top10_share']:.1%}  被引用の中央値 {b['cited_by_median']:.0f}")

    d = new
    out["sample_chain"][f"year_from_{YEAR_FROM}"] = int(len(d))
    out["regressions"] = {}
    for outcome in OUTCOMES:
        y = d[outcome].astype(bool).to_numpy(float)
        res = {"n": int(len(d)), "n_positive": int(y.sum())}
        print(f"── {outcome}: N={len(d)}、該当 {int(y.sum())} 件 " + "─" * 20)
        for q in Q_VALUES:
            r = logit(d[f"novel_{q}"].to_numpy(np.float64), y)
            res[f"q{q}"] = r
            print(f"  q={q:3d}  オッズ比 {r['odds_ratio']:7.2f}  95%区間 "
                  f"[{r['or_ci95'][0]:.2f}, {r['or_ci95'][1]:.2f}]  p={r['p']:.3f}"
                  f"  （原論文 {SHIBAYAMA21_OR[q]}）")
        out["regressions"][outcome] = res

    # 参考文献数による交絡の大きさ
    out["n_refs_confound"] = {
        "rho_n_refs_novel_100": float(stats.spearmanr(d.n_refs_used, d.novel_100).statistic),
        "rho_n_refs_top10": float(stats.spearmanr(d.n_refs_used, d.top10).statistic)}
    print(f"  参考文献数と Novel_100 の ρ {out['n_refs_confound']['rho_n_refs_novel_100']:+.3f} / "
          f"上位10%入りとの ρ {out['n_refs_confound']['rho_n_refs_top10']:+.3f}")

    out["shibayama21_odds_ratio_top1"] = SHIBAYAMA21_OR
    out["params"] = {"year_from": YEAR_FROM, "doc_type": DOC_TYPE, "outcomes": list(OUTCOMES),
                     "model": "logit, intercept + Novel_q, no controls", "ci": "Wald 95%"}
    out["inputs"] = {rel(p): file_sha(p) for p in (NOVELTY_PARQUET, PAPERS_CSV, PERCENTILES_CSV)}
    (OUT_DIR / "novelty_citation_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_citation_validation.json')}")


if __name__ == "__main__":
    main()
