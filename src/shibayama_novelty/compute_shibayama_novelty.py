"""工程3 Shibayama semantic novelty の計算（原論文の Step 2〜4）。

原典: Shibayama, Yin & Matsumoto (2021) PLOS ONE 16(7):e0254034
README の「工程」3 と「指標の定義」を参照。

  Step 2  焦点論文ごとに、参考文献の全ペアのコサイン距離 d_ij = 1 - cos(v_i, v_j)   式(1)
  Step 3  距離を昇順に並べ、q% の位置の1点を取る = Novel_q                         式(2)
  Step 4  保存。あわせて分野 × 出版年帯の中でのパーセンタイルを付ける

式(2)の N は参考文献の本数ではなく**ペアの数**である（q=100 が最大値になるのはこの読み方のときだけ）。
分位点は線形補間ではなく nearest-rank（`method="inverted_cdf"`）で取る。

入力: data/derived/shibayama_novelty/input/{focal_papers.parquet, reference_titles.parquet}
      data/derived/shibayama_novelty/title_vectors/{reference_title_index.parquet, reference_title_vectors.npy}
出力: data/derived/shibayama_novelty/scores/
    novelty_shibayama.parquet          論文単位の Novel_q と品質フラグ
    novelty_shibayama_ranked.parquet   分野 × 出版年帯の中でのパーセンタイル
    novelty_shibayama_config.json      実行時の定数と入力の版

実行:
    python src/shibayama_novelty/compute_shibayama_novelty.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clean_reference_titles import INPUT_DIR, MIN_REFS, ROOT, file_sha, rel
from embed_titles_scispacy import VECTORS_DIR, MODEL_NAME

NOVELTY_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "scores"

Q_VALUES = (100, 99, 95, 90, 80, 50)      # 原論文と同じ6水準
FEW_REFS_THRESHOLD = 10                   # is_few_refs
HIGH_REFS_THRESHOLD = 100                 # is_high_refs（総説の疑い）
YEAR_BANDS = [1940, 2010, 2018, 2022, 2027]   # 分野内パーセンタイルの年帯
MIN_CELL_N = 10                           # パーセンタイルを付ける最小のセルの大きさ


def pair_distances(v: np.ndarray) -> np.ndarray:
    """原論文 式(1)。L2 正規化して内積 → 上三角を取る。値域 0〜2。"""
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = 1.0 - v @ v.T
    iu = np.triu_indices(len(v), k=1)
    return d[iu]


def novel_q(d: np.ndarray, q: int) -> float:
    """原論文 式(2)。R⁻¹(n_pairs * q / 100) の nearest-rank パーセンタイル。

    `inverted_cdf` が式(2)の切り上げ規約と一致する。
    既定の線形補間を使うと、とくに高い q でずれる。
    """
    return float(np.percentile(d, q, method="inverted_cdf"))


def novelty_table(focal: pd.DataFrame, refs: pd.DataFrame, index: pd.DataFrame,
                  vectors: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """焦点論文ごとにペア距離を出し、分位点に集約する（Step 2・3）。"""
    row_of = dict(zip(index.title_key, index.row))
    oov_of = dict(zip(index.title_key, index.oov_rate))
    zero_of = dict(zip(index.title_key, index.is_zero_norm))

    raw_counts = focal.set_index("paper_uid").referenced_works_count.to_dict()
    titled = refs.groupby("paper_uid").size().to_dict()

    records = []
    skipped: list[str] = []
    for uid, g in refs.groupby("paper_uid", sort=False):
        keys = [k for k in g.title_key if not zero_of[k]]
        if len(keys) < MIN_REFS:
            skipped.append(uid)          # 零ベクトルを除くと足りなくなった場合
            continue
        d = pair_distances(vectors[[row_of[k] for k in keys]])

        rec = {
            "paper_uid": uid,
            "text_source": "title",
            "embed_model": MODEL_NAME,
            "n_refs_raw": int(raw_counts[uid]),
            "n_refs_titled": int(titled[uid]),
            "n_refs_used": len(keys),
            "n_pairs": int(len(d)),
            "dist_mean": float(d.mean()),
            "dist_sd": float(d.std(ddof=0)),
            "dist_min": float(d.min()),
            "mean_oov_rate": float(np.mean([oov_of[k] for k in keys])),
            "has_nonascii_ref": bool(g.is_nonascii.any()),
        }
        for q in Q_VALUES:
            rec[f"novel_{q}"] = novel_q(d, q)
        records.append(rec)

    out = pd.DataFrame(records)
    out["title_coverage"] = out.n_refs_titled / out.n_refs_raw
    out["n_dup_title_removed"] = out.n_refs_titled - out.n_refs_used
    out["is_few_refs"] = out.n_refs_used < FEW_REFS_THRESHOLD
    out["is_high_refs"] = out.n_refs_used > HIGH_REFS_THRESHOLD

    cols = ["paper_uid", "text_source", "embed_model",
            "n_refs_raw", "n_refs_titled", "n_refs_used", "title_coverage",
            "n_dup_title_removed", "n_pairs"] \
        + [f"novel_{q}" for q in Q_VALUES] \
        + ["dist_mean", "dist_sd", "dist_min", "mean_oov_rate",
           "is_few_refs", "is_high_refs", "has_nonascii_ref"]
    out = out[cols]

    rep = {
        "n_computed": len(out),
        "n_pairs_total": int(out.n_pairs.sum()),
        "skipped_zero_norm": skipped,
        "n_few_refs": int(out.is_few_refs.sum()),
        "n_high_refs": int(out.is_high_refs.sum()),
        "n_has_nonascii_ref": int(out.has_nonascii_ref.sum()),
    }
    return out, rep


def ranked_table(out: pd.DataFrame, focal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """分野 × 年帯の中でのパーセンタイル。分野をまたいで比べるときはこちらを使う。

    セルが小さいとパーセンタイルが意味を持たないので、MIN_CELL_N 未満は NaN にする。
    """
    df = out.merge(focal[["paper_uid", "primary_field", "year_openalex", "source_org"]],
                   on="paper_uid", how="left")
    df["year_band"] = pd.cut(df.year_openalex, YEAR_BANDS).astype(str)
    grp = df.groupby(["primary_field", "year_band"], observed=True)
    df["rank_cell_n"] = grp.paper_uid.transform("size")

    for q in Q_VALUES:
        pct = grp[f"novel_{q}"].rank(pct=True)
        df[f"novel_{q}_pct"] = pct.where(df.rank_cell_n >= MIN_CELL_N)

    cols = ["paper_uid", "primary_field", "year_band", "source_org", "rank_cell_n"] \
        + [f"novel_{q}_pct" for q in Q_VALUES]
    df = df[cols]
    rep = {
        "n_ranked": int((df.rank_cell_n >= MIN_CELL_N).sum()),
        "n_cells": int(df.groupby(["primary_field", "year_band"], observed=True).ngroups),
    }
    return df, rep


def main() -> None:
    focal = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")
    refs = pd.read_parquet(INPUT_DIR / "reference_titles.parquet")
    index = pd.read_parquet(VECTORS_DIR / "reference_title_index.parquet")
    vectors = np.load(VECTORS_DIR / "reference_title_vectors.npy")

    out, rep = novelty_table(focal, refs, index, vectors)
    print(f"  計算完了 {rep['n_computed']} 件 / 総ペア {rep['n_pairs_total']:,d}")
    if rep["skipped_zero_norm"]:
        print(f"  零ベクトルを除くと参考文献が {MIN_REFS} 本未満になり計算不可 "
              f"{len(rep['skipped_zero_norm'])} 件")
    print(f"  is_few_refs {rep['n_few_refs']} / is_high_refs {rep['n_high_refs']} / "
          f"has_nonascii_ref {rep['n_has_nonascii_ref']}")
    print("  Novel_q の分布:")
    for q in Q_VALUES:
        c = out[f"novel_{q}"]
        print(f"    q={q:3d}  mean {c.mean():.3f}  sd {c.std():.3f}  "
              f"min {c.min():.3f}  p50 {c.median():.3f}  max {c.max():.3f}")

    ranked, rep_r = ranked_table(out, focal)

    NOVELTY_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(NOVELTY_DIR / "novelty_shibayama.parquet", index=False)
    ranked.to_parquet(NOVELTY_DIR / "novelty_shibayama_ranked.parquet", index=False)

    config = {
        "reference": "Shibayama, Yin & Matsumoto (2021) PLOS ONE 16:e0254034",
        "embed_model": MODEL_NAME,
        "text_source": "title",
        "params": {
            "q_values": list(Q_VALUES), "percentile_method": "inverted_cdf",
            "min_refs": MIN_REFS, "few_refs_threshold": FEW_REFS_THRESHOLD,
            "high_refs_threshold": HIGH_REFS_THRESHOLD, "year_bands": YEAR_BANDS,
            "min_cell_n": MIN_CELL_N,
        },
        "novelty": rep, "ranked": rep_r,
        "inputs": {rel(p): file_sha(p) for p in (
            INPUT_DIR / "focal_papers.parquet", INPUT_DIR / "reference_titles.parquet",
            VECTORS_DIR / "reference_title_index.parquet", VECTORS_DIR / "reference_title_vectors.npy")},
    }
    (NOVELTY_DIR / "novelty_shibayama_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    print(f"  ranked: パーセンタイル付与 {rep_r['n_ranked']} 件 / {rep_r['n_cells']} セル")
    print(f"保存: {rel(NOVELTY_DIR)}/ novelty_shibayama.parquet, novelty_shibayama_ranked.parquet, "
          "novelty_shibayama_config.json")


if __name__ == "__main__":
    main()
