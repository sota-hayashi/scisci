"""工程2 論文ごとに、チームのくり返しの組の割合と既知のメンバーの割合を計算する。

README の「工程」2 と「指標」を参照。チームの人数を m とする。
    repeat_pair_share   それ以前に共著したことのある2人の組の数 ÷ m(m−1)/2（Guimerà05、Zeng21 の link freshness の裏返し）
    known_member_share  チームのほかの誰かと、それ以前に共著したことのある人の数 ÷ m（Zeng21 の node freshness の裏返し）
    mean_prior_copapers 2人の組ごとの、それ以前の共著論文の数の平均（Petersen15 の結びつきの強さ。参考）
「それ以前」は出版年（year_openalex）が前の論文。同じ年の論文は、どちらが先か分からないので数えない。
4つの版を作る（メンバー × 「以前の共著」の記録の範囲）:
    all         社外の共著者も含めた全著者（author_id が決まった人）。記録は4組織をまたぐ
    toyota      その論文でトヨタ側の所属の著者だけ（affil_label == toyota）。記録は4組織をまたぐ
    all_org     all と同じメンバー。記録は同じ組織（source_org）の論文だけ
    toyota_org  toyota と同じメンバー。記録は同じ組織の論文だけ
    _org の版は「その組織の中での固定化」を見る。組織を移った人の、前の組織での共著は数えない。

「以前の共著」の記録には、出版年が分かるすべての論文（917本のうち year_openalex がある論文）を、4組織をまたいで使う。
回帰から除く論文（データの始まりから2年以内など）も、記録には入る。
同じ論文が2サイトに載っている5件は、is_work_primary の方だけを記録に入れる。
誰か決められない著者（author_id が空）の行は、記録にも指標にも入らない。
回帰から除く論文は、ここでは除かずに印を付ける（工程3 で除く）:
    has_unassigned_author  誰か決められない著者を含む（チームの全員を判定できない）
    has_unknown_affil      所属が不明の著者を含む
    in_burn_in             組織のデータが始まってから BURN_IN_YEARS 年以内（以前の共著がほとんど記録されていない。独自・暫定）
                           「データの始まり」は、その組織の1年間の論文数が初めて MIN_PAPERS_START 本以上になった年（独自・暫定）。
                           出版年のおかしい論文が、たまたま同じ年に2本重なる場合も避けるため 3 とした。
                           最も古い年にしないのは、出版年が明らかにおかしい論文があるため（AISIN IMRA 米国に
                           year_openalex が 1946 と 1968 の論文、豊田中央研究所に公式サイトの年と11〜12年ずれる論文）
1人の論文と、m が2未満の版の指標は欠損。

入力: data/derived/coauthor_network/authors/author_ids.parquet, data/raw/papers.csv
出力: data/derived/coauthor_network/team/team_familiarity.parquet（917行、論文1本で1行）
      data/derived/coauthor_network/team/compute_team_familiarity_summary.json

実行:
    python src/coauthor_network/compute_team_familiarity.py
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from identify_authors import OUT_DIR as AUTHORS_DIR, ROOT, file_sha, rel

AUTHOR_IDS = AUTHORS_DIR / "author_ids.parquet"
PAPERS_CSV = ROOT / "data" / "raw" / "papers.csv"
OUT_DIR = ROOT / "data" / "derived" / "coauthor_network" / "team"
BURN_IN_YEARS = 2          # データの始まりの年から数えて、この年数の論文（例: 2018 開始なら 2018, 2019）に印を付ける
MIN_PAPERS_START = 3       # 組織の1年間の論文数が初めてこの本数以上になった年を「データの始まり」とする（累計ではない）
MEMBERS = ("all", "toyota")                       # メンバーの範囲
VERSIONS = ("all", "toyota", "all_org", "toyota_org")   # メンバー × 記録の範囲


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def team_measures(members: list[str], prior: Counter) -> dict:
    """チームの3つの指標。prior は2人の組 → それ以前の共著論文の数。"""
    m = len(members)
    if m < 2:
        return {"n_members": m, "n_pairs": 0, "repeat_pair_share": np.nan,
                "known_member_share": np.nan, "mean_prior_copapers": np.nan}
    pairs = [pair_key(a, b) for a, b in combinations(members, 2)]
    counts = [prior.get(p, 0) for p in pairs]
    knows = {x for p, c in zip(pairs, counts) if c > 0 for x in p}
    return {"n_members": m, "n_pairs": len(pairs),
            "repeat_pair_share": sum(c > 0 for c in counts) / len(pairs),
            "known_member_share": len(knows) / m,
            "mean_prior_copapers": float(np.mean(counts))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    au = pd.read_parquet(AUTHOR_IDS)
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "source_org", "is_work_primary"])
    flags = au.groupby("paper_uid").agg(n_author_rows=("author_seq", "size"),
                                        has_unassigned_author=("author_id", lambda s: s.isna().any()),
                                        has_unknown_affil=("affil_label", lambda s: (s == "unknown").any()))
    df = papers.merge(flags, on="paper_uid", how="left")
    # 著者の記録が無い論文（OpenAlex に対応づいていない論文）は、著者0人として印を False にする
    df["n_author_rows"] = df.n_author_rows.fillna(0).astype(int)
    for c in ("has_unassigned_author", "has_unknown_affil"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    per_year = df.dropna(subset=["year_openalex"]).groupby(["source_org", "year_openalex"]).size()
    org_start = per_year[per_year >= MIN_PAPERS_START].reset_index().groupby("source_org").year_openalex.min()
    df["years_since_org_start"] = df.year_openalex - df.source_org.map(org_start)
    df["in_burn_in"] = df.years_since_org_start < BURN_IN_YEARS

    # 論文ごとのメンバー（重複を除く）。版ごとに
    ids = au.dropna(subset=["author_id"])
    members = {m: (ids if m == "all" else ids[ids.affil_label == "toyota"])
               .groupby("paper_uid").author_id.agg(lambda s: sorted(set(s))).to_dict() for m in MEMBERS}
    org_of = df.set_index("paper_uid").source_org.to_dict()

    # 出版年の順に進み、その年より前の論文の共著だけを記録に入れる
    dated = df.dropna(subset=["year_openalex"]).sort_values("year_openalex")
    # 記録: 組織をまたぐもの（キー None）と、組織ごとのもの（キー = source_org）
    prior = {m: {None: Counter(), **{o: Counter() for o in df.source_org.unique()}} for m in MEMBERS}
    rows = []
    for year, g in dated.groupby("year_openalex", sort=True):
        for uid in g.paper_uid:
            row = {"paper_uid": uid}
            for m in MEMBERS:
                for v, key in ((m, None), (f"{m}_org", org_of[uid])):
                    r = team_measures(members[m].get(uid, []), prior[m][key])
                    row.update({f"{k}_{v}": x for k, x in r.items()})
            rows.append(row)
        # この年の論文を記録に足す（同じ年の論文どうしは「以前」に数えない）
        # 同じ論文が2サイトに載っている5件は、is_work_primary の方だけを記録に入れる（2回数えない）
        for uid in g[g.is_work_primary.astype(str) == "True"].paper_uid:
            for m in MEMBERS:
                for p in {pair_key(a, b) for a, b in combinations(members[m].get(uid, []), 2)}:
                    prior[m][None][p] += 1
                    prior[m][org_of[uid]][p] += 1
    out = df.merge(pd.DataFrame(rows), on="paper_uid", how="left")
    out.to_parquet(OUT_DIR / "team_familiarity.parquet", index=False)

    summary = {"papers": int(len(out)), "papers_without_year": int(out.year_openalex.isna().sum()),
               "flags": {c: int(out[c].sum()) for c in ("has_unassigned_author", "has_unknown_affil", "in_burn_in")},
               "org_start_year": {k: int(v) for k, v in org_start.items()}}
    for v in VERSIONS:
        ok = out[f"n_members_{v}"] >= 2
        summary[v] = {"papers_with_2plus_members": int(ok.sum()),
                      **{c: {"mean": float(out.loc[ok, f"{c}_{v}"].mean()), "median": float(out.loc[ok, f"{c}_{v}"].median()),
                             "share_zero": float((out.loc[ok, f"{c}_{v}"] == 0).mean())}
                         for c in ("repeat_pair_share", "known_member_share", "mean_prior_copapers")}}
    summary["params"] = {"burn_in_years": BURN_IN_YEARS, "min_papers_start": MIN_PAPERS_START, "prior": "papers with strictly earlier year_openalex",
                         "versions": list(VERSIONS)}
    summary["inputs"] = {rel(p): file_sha(p) for p in (AUTHOR_IDS, PAPERS_CSV)}
    (OUT_DIR / "compute_team_familiarity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("params", "inputs")}, ensure_ascii=False, indent=1))
    print(f"保存: {rel(OUT_DIR / 'team_familiarity.parquet')}, {rel(OUT_DIR / 'compute_team_familiarity_summary.json')}")


if __name__ == "__main__":
    main()
