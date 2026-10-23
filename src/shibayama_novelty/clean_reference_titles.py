"""工程1 焦点論文の絞り込みと参考文献の整形（原論文の Step 0 に当たる前処理）。

原典: Shibayama, Yin & Matsumoto (2021) PLOS ONE 16(7):e0254034
README の「工程」1 と「データの扱い」を参照。

    焦点論文   papers.csv を is_work_primary → matched → referenced_works_count > 0 の順に絞る
    参考文献   タイトル欠損を除く → 正規化タイトルで重複を除く → 参考文献が2本未満の論文を除く

同じ論文の参考文献に同じタイトルが2回出ると、その組の距離は厳密に 0 になり、
低い q の Novel_q を人為的に下げる。実データでは重複は ref_work_id ではなくタイトルの側に
あるので（プレプリントと出版版が別 ID で登録されている）、正規化タイトルで除く。

入力: data/raw/papers.csv, data/raw/paper_references.csv
出力: data/derived/shibayama_novelty/input/{focal_papers.parquet, reference_titles.parquet, clean_reference_titles_summary.json}

実行:
    python src/shibayama_novelty/clean_reference_titles.py
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]    # リポジトリ直下。実行時のカレントに依存しない

RAW_DIR = ROOT / "data" / "raw"
PAPERS_CSV = RAW_DIR / "papers.csv"
REFS_CSV = RAW_DIR / "paper_references.csv"
AUTHORS_CSV = RAW_DIR / "authorships.csv"
INPUT_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "input"

MIN_REFS = 2                              # ペアが1組も作れない論文は計算できない
NONASCII_THRESHOLD = 0.2                  # これを超える非 ASCII 率のタイトルを非英語とみなす

HTML_TAG_RE = r"<[^>]{1,20}>"             # 実データでは2行のみ該当


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def file_sha(path: Path) -> str:
    """出力 JSON に入力ファイルの版を残すためのハッシュ（先頭16桁）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def clean_title(s: pd.Series) -> pd.Series:
    """埋め込みへ渡すタイトル。NFKC 正規化、HTML タグ除去、空白の整理。小文字化はしない。"""
    out = s.fillna("").map(lambda x: unicodedata.normalize("NFKC", x))
    out = out.str.replace(HTML_TAG_RE, " ", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def nonascii_rate(s: pd.Series) -> pd.Series:
    return s.map(lambda x: sum(ord(c) > 127 for c in x) / max(len(x), 1))


def main() -> None:
    papers = pd.read_csv(PAPERS_CSV)
    refs = pd.read_csv(REFS_CSV)
    rep: dict = {}

    # 焦点論文の絞り込み
    chain = [("papers_all", len(papers))]
    focal = papers[papers.is_work_primary == True]                    # noqa: E712
    chain.append(("is_work_primary", len(focal)))
    focal = focal[focal.matched == True]                              # noqa: E712
    chain.append(("matched", len(focal)))
    focal = focal[focal.referenced_works_count.fillna(0) > 0]
    chain.append(("referenced_works_count_gt0", len(focal)))
    rep["focal_chain"] = dict(chain)

    keep = ["paper_uid", "work_id", "source_org", "year_openalex", "type",
            "primary_field", "primary_subfield", "primary_domain",
            "referenced_works_count", "cited_by_count", "n_authors",
            "doi_norm", "matched_title"]
    focal = focal[keep].reset_index(drop=True)

    refs = refs[refs.paper_uid.isin(set(focal.paper_uid))].copy()
    rep["refs_rows_raw"] = len(refs)

    # タイトル欠損の除去
    refs["title_clean"] = clean_title(refs.ref_title)
    n_before = len(refs)
    refs = refs[refs.title_clean != ""]
    rep["refs_dropped_no_title"] = n_before - len(refs)
    rep["title_coverage_overall"] = round(len(refs) / n_before, 4)

    # 正規化タイトルで重複除去（ref_work_id で除いても1行も減らない）
    refs["title_key"] = refs.title_clean.str.lower()
    n_before = len(refs)
    refs = refs.sort_values(["paper_uid", "ref_seq"]).drop_duplicates(["paper_uid", "title_key"])
    rep["refs_dropped_dup_title"] = n_before - len(refs)

    refs["is_nonascii"] = nonascii_rate(refs.title_clean) > NONASCII_THRESHOLD
    rep["refs_nonascii_high"] = int(refs.is_nonascii.sum())

    # ペアが作れない焦点論文を落とす
    n_used = refs.groupby("paper_uid").size()
    computable = n_used[n_used >= MIN_REFS].index
    dropped = focal[~focal.paper_uid.isin(computable)]
    rep["focal_dropped_lt2_refs"] = len(dropped)
    rep["focal_dropped_lt2_by_org"] = dropped.source_org.value_counts().to_dict()
    focal = focal[focal.paper_uid.isin(computable)].reset_index(drop=True)
    refs = refs[refs.paper_uid.isin(computable)].reset_index(drop=True)

    rep["n_focal_computable"] = len(focal)
    rep["refs_rows_used"] = len(refs)
    rep["n_unique_titles"] = int(refs.title_key.nunique())

    refs = refs[["paper_uid", "ref_seq", "ref_work_id", "title_clean", "title_key", "is_nonascii"]]

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    focal.to_parquet(INPUT_DIR / "focal_papers.parquet", index=False)
    refs.to_parquet(INPUT_DIR / "reference_titles.parquet", index=False)

    rep["params"] = {"min_refs": MIN_REFS, "nonascii_threshold": NONASCII_THRESHOLD,
                     "dedup_by": "normalized_title"}
    rep["inputs"] = {rel(p): file_sha(p) for p in (PAPERS_CSV, REFS_CSV)}
    (INPUT_DIR / "clean_reference_titles_summary.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False) + "\n")

    for k, v in rep["focal_chain"].items():
        print(f"  {k:28s} {v:5d}")
    print(f"  タイトル欠損で除去        {rep['refs_dropped_no_title']:5d} 行 "
          f"(coverage {rep['title_coverage_overall']:.1%})")
    print(f"  タイトル重複で除去        {rep['refs_dropped_dup_title']:5d} 行")
    print(f"  参考文献 < {MIN_REFS} で除外    {rep['focal_dropped_lt2_refs']:5d} 件 "
          f"{rep['focal_dropped_lt2_by_org']}")
    print(f"  → 焦点論文 {rep['n_focal_computable']} 件 / 参考文献 {rep['refs_rows_used']:,d} 行 "
          f"/ unique タイトル {rep['n_unique_titles']:,d}")
    print(f"保存: {rel(INPUT_DIR)}/ focal_papers.parquet, reference_titles.parquet, clean_reference_titles_summary.json")


if __name__ == "__main__":
    main()
