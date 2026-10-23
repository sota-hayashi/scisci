"""工程4 検証用のペア距離のデータセット。

検証の工程6〜8 が読む入力をここで一度だけ作る。検証の工程はベクトル化もペア作成もしない。
README の「工程」4 を参照。

    reference_title_pairs.parquet          参考文献の全ペア距離（Novel_q の計算に使ったもの）  → 工程8
    focal_paper_pairs.parquet        焦点論文の全ペア距離 + 共通参照距離（式(3)）+ 同一著者 → 工程6, 7
    same_author_paper_pairs.parquet  ペア × 共有している著者                              → 工程6

reference_title_pairs は工程3と同じ関数を同じ入力に当てて作るので、値はビット単位で一致する。
reference_title_pairs から Novel_q を計算し直し、工程3の保存値と全件一致することを確かめる。一致しなければ停止する。

著者の同定は ORCID を優先し、無ければ name_key（姓＋名の頭文字）を使う。
openalex_author_id は別人が併合されているので使わない（README の「データの扱い」）。

入力: data/derived/shibayama_novelty/input/{focal_papers.parquet, reference_titles.parquet}
      data/derived/shibayama_novelty/title_vectors/{reference_title_index.parquet, reference_title_vectors.npy, focal_title_index.parquet, focal_title_vectors.npy}
      data/derived/shibayama_novelty/scores/novelty_shibayama.parquet（照合のみ）
      data/raw/{paper_references.csv, authorships.csv}
出力: data/derived/shibayama_novelty/pairs/{reference_title_pairs, focal_paper_pairs, same_author_paper_pairs}.parquet, build_title_distance_pairs_summary.json

実行:
    python src/shibayama_novelty/build_title_distance_pairs.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clean_reference_titles import AUTHORS_CSV, INPUT_DIR, MIN_REFS, REFS_CSV, ROOT, file_sha, rel
from compute_shibayama_novelty import NOVELTY_DIR, Q_VALUES, novel_q, pair_distances
from embed_titles_scispacy import VECTORS_DIR

PAIRS_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "pairs"
NOVELTY_PARQUET = NOVELTY_DIR / "novelty_shibayama.parquet"


def cocitation_distance(a: set, b: set) -> float:
    """原論文 式(3)。d^C = 1 - coref / sqrt(ref_i * ref_j)。参照集合は空でないこと。"""
    return 1.0 - len(a & b) / np.sqrt(len(a) * len(b))


def author_id(df: pd.DataFrame) -> pd.Series:
    """著者の同定。ORCID を最優先、無ければ name_key。openalex_author_id は使わない。"""
    return df.orcid.fillna("KEY:" + df.name_key.astype(str))


def normalize(v: np.ndarray) -> np.ndarray:
    """L2 正規化。零ベクトルの行は NaN になる（距離を定義できないことを伝播させる）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=1, keepdims=True)


# ────────────────────────────────────────────────────────────────
# 参考文献ペア
# ────────────────────────────────────────────────────────────────
def build_ref_pairs(refs: pd.DataFrame, index: pd.DataFrame, vecs: np.ndarray) -> pd.DataFrame:
    """参考文献の全ペア。零ベクトルを含むペアの距離は NaN にする。"""
    row_of = dict(zip(index.title_key, index.row))
    zero = np.linalg.norm(vecs, axis=1) == 0

    parts = []
    # 並び順は工程3と同じ（groupby sort=False、参考文献は reference_titles の行順）
    for uid, g in refs.groupby("paper_uid", sort=False):
        rows = np.array([row_of[k] for k in g.title_key], dtype=np.int32)
        n = len(rows)
        if n < 2:
            continue
        a, b = np.triu_indices(n, k=1)
        keep = ~zero[rows]
        d = np.full(len(a), np.nan, dtype=np.float32)
        if keep.sum() >= MIN_REFS:
            # 工程3と同じ入力（零ベクトルを除いた行列）に同じ関数を当てる。
            # 残した添字どうしの上三角は、全体の上三角から keep[a] & keep[b] を抜いた順と一致する
            d[keep[a] & keep[b]] = pair_distances(vecs[rows[keep]])
        parts.append(pd.DataFrame({"paper_uid": uid, "row_i": rows[a], "row_j": rows[b], "d": d}))

    out = pd.concat(parts, ignore_index=True)
    out["paper_uid"] = out.paper_uid.astype("category")
    return out


def check_ref_pairs(pairs: pd.DataFrame) -> dict:
    """reference_title_pairs から Novel_q を計算し直し、工程3の保存値と全件で完全に一致するか。"""
    saved = pd.read_parquet(NOVELTY_PARQUET).set_index("paper_uid")
    mism, n = 0, 0
    for uid, g in pairs.groupby("paper_uid", observed=True, sort=False):
        d = g.d.to_numpy()
        d = d[~np.isnan(d)]
        if uid not in saved.index:
            if len(d):                   # 工程3が計算しなかった論文に距離があってはいけない
                mism += 1
            continue
        n += 1
        if len(d) != saved.at[uid, "n_pairs"]:
            mism += 1
            continue
        if any(novel_q(d, q) != saved.at[uid, f"novel_{q}"] for q in Q_VALUES):
            mism += 1
    ok = mism == 0 and n == len(saved)
    print(f"  Novel_q 再計算 {n} 件 / 保存 {len(saved)} 件 / 不一致 {mism} 件 {'✅' if ok else '❌'}")
    if not ok:
        raise SystemExit("reference_title_pairs から計算した Novel_q が保存値と一致しない。停止する。")
    return {"n_papers_checked": n, "n_saved": int(len(saved)), "n_mismatch": mism}


# ────────────────────────────────────────────────────────────────
# 焦点論文ペア
# ────────────────────────────────────────────────────────────────
def build_focal_pairs(fidx: pd.DataFrame, vecs: np.ndarray, focal: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    uids = fidx.paper_uid.to_numpy()
    a, b = np.triu_indices(len(uids), k=1)
    out = pd.DataFrame({"uid_a": uids[a], "uid_b": uids[b]})
    vn = normalize(vecs[fidx.row.to_numpy()])     # 零ベクトルの行は NaN → 距離も NaN
    out["d"] = (1.0 - vn @ vn.T)[a, b].astype(np.float32)

    # 式(3)。参照集合はタイトル欠損の行も含めた ref_work_id
    refs = pd.read_csv(REFS_CSV)
    ref_sets = (refs[refs.paper_uid.isin(set(focal.paper_uid))]
                .groupby("paper_uid").ref_work_id.apply(set).to_dict())
    sa = [ref_sets[u] for u in out.uid_a]
    sb = [ref_sets[u] for u in out.uid_b]
    out["n_refs_a"] = np.array([len(s) for s in sa], dtype=np.int16)
    out["n_refs_b"] = np.array([len(s) for s in sb], dtype=np.int16)
    out["n_coref"] = np.array([len(x & y) for x, y in zip(sa, sb)], dtype=np.int16)
    out["d_cocite"] = np.array([cocitation_distance(x, y) for x, y in zip(sa, sb)],
                               dtype=np.float32)

    au = pd.read_csv(AUTHORS_CSV)
    au = au[au.paper_uid.isin(set(uids))].copy()
    au["aid"] = author_id(au)
    aids = au.groupby("paper_uid").aid.apply(set).to_dict()
    empty: set = set()
    shared = [aids.get(x, empty) & aids.get(y, empty) for x, y in zip(out.uid_a, out.uid_b)]
    out["is_same_author"] = np.array([bool(s) for s in shared])

    nw = dict(zip(fidx.paper_uid, fidx.n_words))
    out["n_words_a"] = out.uid_a.map(nw).astype(np.int16)
    out["n_words_b"] = out.uid_b.map(nw).astype(np.int16)

    idx = np.flatnonzero(out.is_same_author.to_numpy())
    sap = pd.DataFrame(
        [(out.uid_a.iat[i], out.uid_b.iat[i], aid) for i in idx for aid in sorted(shared[i])],
        columns=["uid_a", "uid_b", "aid"])
    return out, sap


def main() -> None:
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    focal = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")
    refs = pd.read_parquet(INPUT_DIR / "reference_titles.parquet")
    ridx = pd.read_parquet(VECTORS_DIR / "reference_title_index.parquet")
    rvec = np.load(VECTORS_DIR / "reference_title_vectors.npy")
    fidx = pd.read_parquet(VECTORS_DIR / "focal_title_index.parquet")
    fvec = np.load(VECTORS_DIR / "focal_title_vectors.npy")
    report: dict = {"checks": {}}

    print("── 参考文献の全ペア距離 " + "─" * 30)
    rp = build_ref_pairs(refs, ridx, rvec)
    g = refs.groupby("paper_uid").size()
    expected = int((g * (g - 1) // 2).sum())
    report["reference_title_pairs"] = {"n_rows": int(len(rp)), "n_rows_expected": expected,
                           "n_papers": int(rp.paper_uid.nunique()),
                           "n_nan": int(rp.d.isna().sum())}
    print(f"  {len(rp):,d} 行（期待値 {expected:,d}）/ 論文 {rp.paper_uid.nunique()} 本 / "
          f"NaN {report['reference_title_pairs']['n_nan']:,d}")
    report["checks"]["novel_q_recomputed"] = check_ref_pairs(rp)
    rp.to_parquet(PAIRS_DIR / "reference_title_pairs.parquet", index=False)

    print("── 焦点論文の全ペア（距離・式(3)・同一著者） " + "─" * 10)
    fp, sap = build_focal_pairs(fidx, fvec, focal)
    report["focal_paper_pairs"] = {
        "n_papers": int(len(fidx)), "n_rows": int(len(fp)),
        "n_same_author": int(fp.is_same_author.sum()),
        "n_same_author_rows": int(len(sap)), "n_authors_in_pairs": int(sap.aid.nunique()),
        "n_zero_norm_papers": int(fidx.is_zero_norm.sum()),
    }
    print(f"  {len(fp):,d} 行 / 同一著者 {fp.is_same_author.sum():,d} 組"
          f"（著者 {sap.aid.nunique():,d} 名、行 {len(sap):,d}）")
    fp.to_parquet(PAIRS_DIR / "focal_paper_pairs.parquet", index=False)
    sap.to_parquet(PAIRS_DIR / "same_author_paper_pairs.parquet", index=False)

    report["params"] = {"q_values": list(Q_VALUES), "author_id": "orcid, else name_key"}
    report["inputs"] = {rel(p): file_sha(p) for p in [
        INPUT_DIR / "focal_papers.parquet", INPUT_DIR / "reference_titles.parquet",
        VECTORS_DIR / "reference_title_index.parquet", VECTORS_DIR / "reference_title_vectors.npy",
        VECTORS_DIR / "focal_title_index.parquet", VECTORS_DIR / "focal_title_vectors.npy",
        NOVELTY_PARQUET, REFS_CSV, AUTHORS_CSV]}
    (PAIRS_DIR / "build_title_distance_pairs_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(PAIRS_DIR)}/ reference_title_pairs.parquet, focal_paper_pairs.parquet, "
          "same_author_paper_pairs.parquet, build_title_distance_pairs_summary.json")


if __name__ == "__main__":
    main()
