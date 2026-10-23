"""被引用の百分位（上位1%・上位10%）を OpenAlex から取る。工程には入れない。

工程9 が読む data/derived/shibayama_novelty/citations/citation_percentiles.csv を作ったときのコード。
表の値は 2026-10-01 に OpenAlex Works API の citation_normalized_percentile から取ったもので、
解析ではこの 2026-10-01 の値を使う（当時はその場で流しただけだったので、ここに書き直して残す）。

OpenAlex の値は日々更新されるので、今流すと 2026-10-01 とは違う値が返る。そのため工程に入れず、
reproduce の notebook でも流さない。既定では今の表と突き合わせて違いを表示するだけで、
--write を付けたときだけ表を書き換え、OpenAlex の応答を
data/cache/openalex/citation_normalized_percentile/<取得日>/ に1回の問い合わせ1ファイルで保存する
（2026-10-01 の応答は保存していない）。

手順:
    data/raw/papers.csv の work_id（重複を除いて並べたもの）を50件ずつ、
    /works?filter=ids.openalex:<id>|<id>|...&select=id,citation_normalized_percentile で引く。
    返ってきた citation_normalized_percentile の is_in_top_1_percent / is_in_top_10_percent を列にする。
    返らなかった work と、work_id が無い論文は欠損にする。

OpenAlex の推奨に従い、環境変数 OPENALEX_MAILTO があれば連絡先として送る（無くても動く）。

入力: data/raw/papers.csv（paper_uid, work_id）
出力（--write のときだけ）: data/derived/shibayama_novelty/citations/citation_percentiles.csv
      data/cache/openalex/citation_normalized_percentile/<取得日>/chunk_*.json

実行:
    python src/shibayama_novelty/fetch_citation_percentiles.py           # 突き合わせだけ
    python src/shibayama_novelty/fetch_citation_percentiles.py --write   # 表を書き換え、応答を保存する
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from clean_reference_titles import PAPERS_CSV, ROOT, rel

API = "https://api.openalex.org/works"
SELECT = "id,citation_normalized_percentile"
CHUNK = 50                                   # filter の ids.openalex に並べられる上限
COLUMNS = ("is_in_top_1_percent", "is_in_top_10_percent")
PERCENTILES_CSV = ROOT / "data" / "derived" / "shibayama_novelty" / "citations" / "citation_percentiles.csv"
CACHE_DIR = ROOT / "data" / "cache" / "openalex" / "citation_normalized_percentile"


def short_id(work_id: str) -> str:
    return str(work_id).rsplit("/", 1)[-1]


def fetch(ids: list[str]) -> dict:
    params = {"per_page": CHUNK, "select": SELECT, "filter": "ids.openalex:" + "|".join(ids)}
    if os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    url = API + "?" + urllib.parse.urlencode(params, safe=":|,")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                raise
            print(f"  再試行 {attempt + 1}: {e}")
            time.sleep(3)


def to_bool(s: pd.Series) -> pd.Series:
    """CSV から読んだ True/False の文字列を bool にする（欠損は None）。"""
    return s.map({True: True, False: False, "True": True, "False": False}).astype(object)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="citation_percentiles.csv を取得した値で書き換え、応答を data/cache/ に保存する")
    args = ap.parse_args()

    df = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "work_id"])
    ids = sorted({short_id(w) for w in df.work_id.dropna()})
    today = datetime.date.today().isoformat()
    got, responses = {}, []
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        res = fetch(chunk)
        # 送った連絡先（mailto）は残さない。問い合わせた work_id と応答だけを残す
        responses.append({"fetched_on": today, "endpoint": API, "select": SELECT, "ids": chunk,
                          "response": res})
        for w in res["results"]:
            got[short_id(w["id"])] = w.get("citation_normalized_percentile") or {}
        time.sleep(0.2)
    print(f"  work_id {len(ids)} 件のうち返ってきた work {len(got)} 件")

    key = df.work_id.map(lambda w: short_id(w) if pd.notna(w) else None)
    for c in COLUMNS:
        df[c] = key.map(lambda k: got[k].get(c) if k in got else None).astype(object)
    if PERCENTILES_CSV.exists():
        old = pd.read_csv(PERCENTILES_CSV).set_index("paper_uid").reindex(df.paper_uid)
        for c in COLUMNS:
            a, b = to_bool(old[c]).to_numpy(), df[c].to_numpy()
            same = sum((x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b))
            print(f"  {c:22s} 今の表と一致 {same}/{len(df)}")

    if args.write:
        PERCENTILES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PERCENTILES_CSV, index=False)
        out = CACHE_DIR / today
        out.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(responses):
            (out / f"chunk_{i:03d}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
        print(f"書き換え: {rel(PERCENTILES_CSV)}")
        print(f"保存: {rel(out)}/ chunk_000.json〜chunk_{len(responses) - 1:03d}.json")


if __name__ == "__main__":
    main()
