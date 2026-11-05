"""工程1 著者を区別し、所属を3つの値に分ける。

共著ネットワークの指標（くり返しの組の割合、既知のメンバーの割合）は「以前に同じ人と共著したか」を数えるので、
社外の共著者も含めた全著者を、論文をまたいで同じ人かどうか区別する。README の「工程」1 を参照。

区別の規則（独自・暫定）:
    0. data/review/orcid_name_mismatches.csv で is_wrong_orcid が True の行は、ORCID を使わない
       （ORCID が論文の中の別の著者の行に付いている誤り）。
    1. 名（given）の最初の語を小文字の英字だけにし、3文字以上のものを「名のつづり」とする
       （"Martin E." → martin。"M. E." のような頭文字は名のつづりとみなさない）。
    2. 姓＋名の頭文字（name_key）ごとに、名のつづりの種類と ORCID の数を数える。
       名のつづりが1種類以下で ORCID も1つ以下なら、同じ人とみなす。
       そうでなければ、名のつづりごとに別人とする。名のつづりが無い行は、同じ ORCID の別の行の名のつづりで補う。
       補えない行（頭文字だけで、同じ姓・頭文字に別人がいる）は、誰か決められないので author_id を空にする。
       name_key が空の行も author_id を空にする。
    3. 2 で同じ人とした中に ORCID が2つ以上あれば、ORCID ごとに別人とする（同じ研究所に同姓同名が2人いた例がある）。
       ORCID の無い行は、誰か決められないので author_id を空にする。
    4. 同じ ORCID の行は同じ人にまとめる（例: "Philip St. J.Russell" の姓が J.Russell と読まれ、name_key が割れた）。
       ORCID は check_orcid_names.py で全行を名前と照らし合わせ済み。
    openalex_author_id は使わない（別人が併合されている）。

所属（affil_label）は行ごと（論文1本の著者1人ごと）に決める:
    toyota    is_toyota が True
    external  is_toyota が False で、所属の文字列（affil_openalex / affil_scopus / inst_openalex）のどれかがある
    unknown   所属の文字列がすべて空

入力: data/raw/authorships.csv, data/review/orcid_name_mismatches.csv
出力: data/derived/coauthor_network/authors/author_ids.parquet（authorships.csv と同じ行数。1行 = 論文1本の著者1人）
      data/derived/coauthor_network/authors/identify_authors_summary.json

実行:
    python src/coauthor_network/identify_authors.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]    # リポジトリ直下。実行時のカレントに依存しない
AUTHORS_CSV = ROOT / "data" / "raw" / "authorships.csv"
MISMATCH_CSV = ROOT / "data" / "review" / "orcid_name_mismatches.csv"
OUT_DIR = ROOT / "data" / "derived" / "coauthor_network" / "authors"
MIN_GIVEN_LEN = 3


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def file_sha(path: Path) -> str:
    """出力 JSON に入力ファイルの版を残すためのハッシュ（先頭16桁）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def first_given(given: pd.Series) -> pd.Series:
    """名の最初の語を小文字の英字だけにする。MIN_GIVEN_LEN 文字未満は頭文字とみなして欠損にする。"""
    g = (given.fillna("").str.lower().str.replace(r"[.\-]", " ", regex=True).str.split().str[0]
         .fillna("").str.replace(r"[^a-z]", "", regex=True))
    return g.where(g.str.len() >= MIN_GIVEN_LEN)


def assign_ids(a: pd.DataFrame) -> pd.Series:
    """規則 2 と 3 で author_id を決める。決められない行は欠損。"""
    orcid_given = a.dropna(subset=["orcid", "given_first"]).groupby("orcid").given_first.agg(lambda s: s.mode().iat[0])

    def by_key(df: pd.DataFrame) -> pd.Series:
        k = df.name_key.iat[0]
        givens = set(df.given_first.dropna()) | set(orcid_given.reindex(df.orcid.dropna().unique()).dropna())
        if len(givens) <= 1 and df.orcid.nunique() <= 1:
            return pd.Series(f"{k}|{next(iter(givens), '?')}", index=df.index)
        g = df.given_first.fillna(df.orcid.map(orcid_given))
        return g.map(lambda x: f"{k}|{x}" if isinstance(x, str) else None)

    # name_key が空の行は groupby で落ちるので、元の行に並べ直す（その行は author_id が欠損）
    aid = a.groupby("name_key", group_keys=False)[["name_key", "given_first", "orcid"]].apply(by_key).reindex(a.index)
    # 規則 3: 同じ人とした中に ORCID が2つ以上
    n_orcid = a.orcid.groupby(aid).transform("nunique")
    multi = (n_orcid >= 2).fillna(False).to_numpy()
    aid = aid.copy()
    aid[multi] = [f"{i}|{o.rsplit('/', 1)[-1]}" if isinstance(o, str) else None
                  for i, o in zip(aid[multi], a.orcid[multi])]
    # 規則 4: 同じ ORCID の行は同じ人にまとめる（姓・名の分け方の誤りで name_key が割れた例がある）。
    # ORCID は check_orcid_names.py で全行を名前と照らし合わせ済み。いちばん行の多い author_id にそろえる
    has = aid.notna() & a.orcid.notna()
    major = (pd.DataFrame({"orcid": a.orcid[has], "aid": aid[has]}).value_counts()
             .reset_index().drop_duplicates("orcid").set_index("orcid").aid)
    aid[has] = a.orcid[has].map(major)
    return aid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(AUTHORS_CSV, low_memory=False,
                    usecols=["paper_uid", "author_seq", "name_key", "given", "orcid", "is_toyota",
                             "affil_openalex", "affil_scopus", "inst_openalex"])
    rev = pd.read_csv(MISMATCH_CSV)
    wrong = set(map(tuple, rev.loc[rev.is_wrong_orcid.astype(str) == "True", ["paper_uid", "author_seq"]].to_numpy()))
    is_wrong = np.array([(u, s) in wrong for u, s in zip(a.paper_uid, a.author_seq)])
    n_orcid_rows_before = int(a.orcid.notna().sum())
    a.loc[is_wrong, "orcid"] = None

    a["given_first"] = first_given(a.given)
    a["author_id"] = assign_ids(a)

    toy = a.is_toyota.astype(str) == "True"
    has_affil = a[["affil_openalex", "affil_scopus", "inst_openalex"]].notna().any(axis=1)
    a["affil_label"] = np.where(toy, "toyota", np.where(has_affil, "external", "unknown"))

    out = a[["paper_uid", "author_seq", "author_id", "affil_label", "orcid"]].rename(columns={"orcid": "orcid_used"})
    out.to_parquet(OUT_DIR / "author_ids.parquet", index=False)

    per_id = out.dropna(subset=["author_id"]).groupby("author_id").paper_uid.nunique()
    summary = {
        "rows": int(len(out)),
        "rows_unassigned": int(out.author_id.isna().sum()),
        "papers_with_unassigned_author": int(out.loc[out.author_id.isna(), "paper_uid"].nunique()),
        "authors": int(per_id.size),
        "authors_with_orcid": int(out.dropna(subset=["author_id", "orcid_used"]).author_id.nunique()),
        "authors_by_n_papers": {f">={k}": int((per_id >= k).sum()) for k in (2, 5, 10)},
        "orcid_rows_before": n_orcid_rows_before, "orcid_rows_dropped_as_wrong": int(is_wrong.sum()),
        "affil_label_rows": out.affil_label.value_counts().to_dict(),
        "papers_with_unknown_affil": int(out.loc[out.affil_label == "unknown", "paper_uid"].nunique()),
        "params": {"min_given_len": MIN_GIVEN_LEN, "id_rule": "name_key + first given token; split by ORCID if >=2 ORCIDs"},
        "inputs": {rel(p): file_sha(p) for p in (AUTHORS_CSV, MISMATCH_CSV)},
    }
    (OUT_DIR / "identify_authors_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("params", "inputs")}, ensure_ascii=False, indent=1))
    print(f"保存: {rel(OUT_DIR / 'author_ids.parquet')}, {rel(OUT_DIR / 'identify_authors_summary.json')}")


if __name__ == "__main__":
    main()
