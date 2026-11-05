"""工程3 語ベクトルの3つの版で計算した Novel_q を比べる。

版: original（shibayama_novelty の出力。scispaCy、ハイフンでつないだまま）、scispacy_split、mat2vec_split（工程2）。
README の「工程」3 を参照。比べ方は独自。辞書によって距離の大きさの目安が違う（mat2vec の Novel_100 の平均は 0.46、
scispaCy は 0.84）ので、値ではなく順位で比べる。

比べること:
    1. 版どうしの Novel_q の Spearman ρ（同じ論文の順位がどれだけ同じか）
    2. Novel_q と参考文献数（n_refs_used）の Spearman ρ（参考文献数に左右される度合い）
    3. 妥当性の確認: SciSciNet の Uzzi らの新規性・定石らしさとの偏相関（shibayama_novelty の工程13 と同じ計算。
       参考文献数を除いたもの（partial_own）と、さらに分野・出版年・組織を除いたもの（partial_all））。
       Uzzi らの新規性（ss_novelty）は小さいほど新しいので、同じものを測っていれば負の相関になる。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet、data/derived/embedding_comparison/scores/novelty_<版>.parquet、
      data/raw/sciscinet_metrics.csv、data/raw/papers.csv
出力: data/derived/embedding_comparison/comparison/embedding_comparison.json

実行:
    python src/embedding_comparison/compare_embeddings.py
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from compute_novelty import OUT_DIR as SCORES_DIR
from embed_titles import ROOT, VERSIONS, file_sha, rel
from compare_novelty_sciscinet import MIN_FIELD, SCISCINET_CSV, corr_ranks  # noqa: E402（shibayama_novelty の関数）
from clean_reference_titles import PAPERS_CSV  # noqa: E402
from regress_citations_on_novelty import YEAR_POOL_UPTO  # noqa: E402

ORIGINAL = ROOT / "data" / "derived" / "shibayama_novelty" / "scores" / "novelty_shibayama.parquet"
OUT_DIR = ROOT / "data" / "derived" / "embedding_comparison" / "comparison"
QS = (100, 50)
TARGETS = ("ss_novelty", "ss_conventionality")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["paper_uid", "n_refs_used", *[f"novel_{q}" for q in QS]]
    vers = {"original": pd.read_parquet(ORIGINAL, columns=cols)}
    for v in VERSIONS:
        vers[v] = pd.read_parquet(SCORES_DIR / f"novelty_{v}.parquet", columns=cols)
    names = list(vers)
    df = vers["original"][["paper_uid", "n_refs_used"]]
    for v, d in vers.items():
        df = df.merge(d[["paper_uid", *[f"novel_{q}" for q in QS]]].rename(columns={f"novel_{q}": f"{v}_{q}" for q in QS}),
                      on="paper_uid")
    ss = pd.read_csv(SCISCINET_CSV, usecols=["paper_uid", *TARGETS])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "primary_field", "source_org"])
    df = df.merge(ss, on="paper_uid", how="left").merge(papers, on="paper_uid", how="left")
    field = df.primary_field.fillna("unknown")
    df["field_cat"] = field.where(field.map(field.value_counts()) >= MIN_FIELD, "other")
    year = df.year_openalex.astype("Int64")
    df["year_cat"] = np.where(year <= YEAR_POOL_UPTO, f"<= {YEAR_POOL_UPTO}", year.astype(str))

    out: dict = {"n_papers": int(len(df)), "between_versions": {}, "with_n_refs": {}, "sciscinet": {}}
    print(f"── 版どうしの Spearman ρ（{len(df)} 本）")
    for q in QS:
        for a, b in combinations(names, 2):
            r = float(stats.spearmanr(df[f"{a}_{q}"], df[f"{b}_{q}"]).statistic)
            out["between_versions"][f"q{q}|{a}|{b}"] = r
            print(f"  q={q:3d} {a:15s} × {b:15s} ρ {r:+.3f}")
    print("── 参考文献数との Spearman ρ")
    for q in QS:
        for v in names:
            r = float(stats.spearmanr(df[f"{v}_{q}"], df.n_refs_used).statistic)
            out["with_n_refs"][f"q{q}|{v}"] = r
        print(f"  q={q:3d} " + "  ".join(f"{v} {out['with_n_refs'][f'q{q}|{v}']:+.3f}" for v in names))
    print("── SciSciNet との偏相関（参考文献数を除く / さらに分野・出版年・組織を除く）。負ほど一致")
    for t in TARGETS:
        d = df[df[t].notna() & df.year_openalex.notna()]
        own = [d.n_refs_used.to_numpy(float)]
        dummies = pd.concat([pd.get_dummies(d[c], drop_first=True, dtype=float)
                             for c in ("field_cat", "year_cat", "source_org")], axis=1).to_numpy(float)
        for q in QS:
            line = []
            for v in names:
                x, y = d[f"{v}_{q}"].to_numpy(float), d[t].to_numpy(float)
                res = {"raw": corr_ranks(x, y, []), "partial_own": corr_ranks(x, y, own),
                       "partial_all": corr_ranks(x, y, own, dummies)}
                out["sciscinet"][f"{t}|q{q}|{v}"] = res
                line.append(f"{v} {res['partial_own']['rho']:+.3f} / {res['partial_all']['rho']:+.3f}")
            print(f"  {t:18s} q={q:3d} N={len(d)}  " + "  ".join(line))
    out["params"] = {"qs": list(QS), "versions": names, "method": "Spearman; partial = shibayama_novelty.compare_novelty_sciscinet.corr_ranks"}
    out["inputs"] = {rel(p): file_sha(p) for p in (ORIGINAL, *[SCORES_DIR / f"novelty_{v}.parquet" for v in VERSIONS],
                                                    SCISCINET_CSV, PAPERS_CSV)}
    (OUT_DIR / "embedding_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'embedding_comparison.json')}")


if __name__ == "__main__":
    main()
