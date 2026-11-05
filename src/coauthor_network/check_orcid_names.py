"""authorships.csv の ORCID が、その行の著者本人のものかを、ORCID の公開記録の名前と照らし合わせる。工程には入れない。

ORCID が論文の中の別の著者の行に付いている誤りがある（2026-11-05 に、豊田中央研究所の論文2本・3行で見つけた）。
著者の区別（工程1 identify_authors.py）は ORCID を使うので、名前が合わない行を data/review/ に書き出し、
工程1 はその行の ORCID を使わない。

手順:
    data/raw/authorships.csv の ORCID（重複を除く）ごとに、ORCID の公開 API
    https://pub.orcid.org/v3.0/<id>/person から、名（given-names）・姓（family-name）・表示名（credit-name）・
    別名（other-names）を取る。行ごとに、authorships.csv の姓と名の最初の語が、記録の名前に含まれるかを見る。
    判定:
        一致          姓と名の両方が記録に含まれる
        記録の名前が不完全  記録に姓か名が無く、ある方は一致する（矛盾ではない）
        不一致        姓が記録に含まれない → data/review/orcid_name_mismatches.csv に書き出す
    「不一致」「名が合わない」は、書き出したあと人が確かめ、is_wrong_orcid（True = 本人の ORCID ではない）と
    note に判断を書く。流し直しても、既にある行の is_wrong_orcid と note は残す。工程1 は is_wrong_orcid が
    True の行の ORCID を使わない。

公開記録にはメールアドレスや住所が含まれることがあるので、キャッシュには名前の項目だけを保存する。
2回目からはキャッシュを使い、問い合わせない。

入力: data/raw/authorships.csv
出力: data/cache/orcid/<ORCID>.json（名前の項目だけ）
      data/review/orcid_name_mismatches.csv

実行:
    python src/coauthor_network/check_orcid_names.py
"""
from __future__ import annotations

import json
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]    # リポジトリ直下。実行時のカレントに依存しない
AUTHORS_CSV = ROOT / "data" / "raw" / "authorships.csv"
CACHE_DIR = ROOT / "data" / "cache" / "orcid"
OUT_CSV = ROOT / "data" / "review" / "orcid_name_mismatches.csv"
API = "https://pub.orcid.org/v3.0/{}/person"
SLEEP = 0.2                                     # 公開 API への負荷を抑える


def fetch_name(oid: str) -> dict:
    """ORCID の公開記録から名前の項目だけを取る（キャッシュがあればそれを使う）。"""
    f = CACHE_DIR / f"{oid}.json"
    if f.exists():
        return json.loads(f.read_text())
    req = urllib.request.Request(API.format(oid), headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"orcid": oid, "error": str(e)}
            time.sleep(2)
    n = d.get("name") or {}
    rec = {"orcid": oid,
           "given_names": (n.get("given-names") or {}).get("value"),
           "family_name": (n.get("family-name") or {}).get("value"),
           "credit_name": (n.get("credit-name") or {}).get("value"),
           "other_names": [x.get("content") for x in (d.get("other-names") or {}).get("other-name", [])]}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    time.sleep(SLEEP)
    return rec


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalpha() else " " for c in s).split())


def judge(family: str, given: str, rec: dict) -> str:
    if "error" in rec:
        return "取得できない"
    rg, rf = norm(rec["given_names"]), norm(rec["family_name"])
    pool = " ".join([rg, rf, norm(rec["credit_name"]), *[norm(x) for x in rec["other_names"]]])
    toks = set(pool.split())
    if not toks:
        return "記録の名前が不完全"
    fam, g1 = norm(family), (norm(given).split() or [""])[0]
    fam_ok = bool(fam) and all(x in toks for x in fam.split())
    giv_ok = (not g1) or any(x.startswith(g1) or g1.startswith(x) for x in toks)
    if fam_ok and giv_ok:
        return "一致"
    if fam_ok and not rg:
        return "記録の名前が不完全"
    if giv_ok and not (rf or rec["credit_name"] or rec["other_names"]):
        return "記録の名前が不完全"
    return "不一致" if not fam_ok else "名が合わない"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(AUTHORS_CSV, low_memory=False,
                    usecols=["paper_uid", "author_seq", "name_raw", "given", "family", "orcid"])
    a = a[a.orcid.notna()].copy()
    a["oid"] = a.orcid.str.rsplit("/", n=1).str[-1]
    ids = sorted(a.oid.unique())
    print(f"ORCID {len(ids)} 個（{len(a)} 行）。キャッシュ済み {sum((CACHE_DIR / f'{i}.json').exists() for i in ids)} 個")
    recs = {oid: fetch_name(oid) for oid in ids}
    a["status"] = [judge(f, g, recs[o]) for f, g, o in zip(a.family, a.given, a.oid)]
    a["orcid_record_name"] = a.oid.map(lambda o: " ".join(x for x in (recs[o].get("given_names"), recs[o].get("family_name")) if x)
                                       or recs[o].get("credit_name") or "")
    print(a.status.value_counts().to_string())
    bad = a[a.status.isin(["不一致", "名が合わない", "取得できない"])]
    out = bad[["paper_uid", "author_seq", "name_raw", "orcid", "orcid_record_name", "status"]].sort_values(["paper_uid", "author_seq"])
    # 人の判断（is_wrong_orcid, note）は、既にある行のものを残す
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)[["paper_uid", "author_seq", "is_wrong_orcid", "note"]]
        out = out.merge(prev, on=["paper_uid", "author_seq"], how="left")
    else:
        out["is_wrong_orcid"], out["note"] = pd.NA, pd.NA
    out.to_csv(OUT_CSV, index=False)
    print(f"保存: {OUT_CSV.relative_to(ROOT)}（{len(out)} 行）")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
