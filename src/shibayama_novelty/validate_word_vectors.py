"""工程5 検証【1】単語ベクトルは意味を担っているか。

原論文は語ベクトル表を「確立された公開ライブラリ」として採用しただけで、単語の段階は確かめていない。
工学中心の本標本に生物医学向けの表を使うので、ここから確かめる。README の「工程」5 を参照。

手順 [Bommasani20 §4.1][Chiu16]:
    単語ペアのベンチマーク6種について、語ベクトルの cos 類似度と人の評定の Spearman ρ を出す。
    両方の語が語ベクトル表に直接ある1語どうしのペアだけを使う（[Chiu16] も表に無い語を除いている）。
    複数語の用語（UMNSRS に含まれる）は除く。

    一般語   RG65 / WS353 / SimLex-999 / SimVerb-3500   [Bommasani20 §4.1]
    生物医学 UMNSRS-Sim / UMNSRS-Rel                    [Chiu16]

ベンチマークは data/raw/word_similarity_benchmarks/<name>/ にある。原典と入手元・配布条件は各フォルダの
citation.txt と README の「単語ペアのベンチマーク」。結果を報告するときは原典を引用する。

語ベクトル表は工程2と同じく data/cache/scispacy/ から読む（無ければ落とす）。

入力: data/raw/word_similarity_benchmarks/、語ベクトル表

出力: data/derived/shibayama_novelty/validation/word_vector_validation.json

実行:
    python src/shibayama_novelty/validate_word_vectors.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_title_distance_pairs import normalize
from clean_reference_titles import RAW_DIR, file_sha, rel
from embed_titles_scispacy import MODEL_NAME, ScispacyEncoder, vocab_dir
from validate_distance_shared_refs import OUT_DIR

BENCH_RAW = RAW_DIR / "word_similarity_benchmarks"
# name → (ファイルの場所, 種別, 件数の期待値)
BENCHMARKS = {
    "rg65": (BENCH_RAW / "rg65" / "rg65_table1.json", "general", 65),
    "ws353": (BENCH_RAW / "ws353" / "combined.csv", "general", 353),
    "simlex999": (BENCH_RAW / "simlex999" / "simlex_999.txt", "general", 999),
    "simverb3500": (BENCH_RAW / "simverb3500" / "simverb_3500.txt", "general", 3500),
    "umnsrs_sim": (BENCH_RAW / "umnsrs" / "umnsrs_similarity.csv", "biomedical", 566),
    "umnsrs_rel": (BENCH_RAW / "umnsrs" / "umnsrs_relatedness.csv", "biomedical", 587),
}


def load(name: str) -> pd.DataFrame:
    """原本を読み、列 word1, word2, score に揃える。語は小文字にして前後の空白を除く。"""
    path = BENCHMARKS[name][0]
    if name == "rg65":
        rg = json.loads(path.read_text())
        df = pd.DataFrame({"word1": [x["term1"] for x in rg], "word2": [x["term2"] for x in rg],
                           "score": [x["value"] for x in rg]})
    elif name == "simlex999":
        sl = pd.read_csv(path, sep="\t")
        df = pd.DataFrame({"word1": sl.word1, "word2": sl.word2, "score": sl.SimLex999})
    elif name == "ws353":
        ws = pd.read_csv(path)
        df = pd.DataFrame({"word1": ws["Word 1"], "word2": ws["Word 2"], "score": ws["Human (mean)"]})
    elif name == "simverb3500":
        sv = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                         names=["word1", "word2", "pos", "score", "relation"])
        df = sv[["word1", "word2", "score"]]
    else:
        u = pd.read_csv(path)
        df = pd.DataFrame({"word1": u.Term1, "word2": u.Term2, "score": u.Mean})
    df = df.copy()
    for c in ("word1", "word2"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["score"] = df.score.astype(float)
    if len(df) != BENCHMARKS[name][2]:
        raise SystemExit(f"{name} の件数 {len(df)} が期待値 {BENCHMARKS[name][2]} と違う。停止する。")
    return df


def in_vocab(enc: ScispacyEncoder, w: str) -> bool:
    return " " not in w and enc._row(w) is not None


def main() -> None:
    enc = ScispacyEncoder(vocab_dir())
    out: dict = {"embed_model": MODEL_NAME, "benchmarks": {}}
    inputs = {}
    print(f"  {'data':12s} {'全体':>5s} {'使用':>5s} {'除外':>5s}  ρ")
    for name, (_, kind, _) in BENCHMARKS.items():
        df = load(name)
        inputs[rel(BENCHMARKS[name][0])] = file_sha(BENCHMARKS[name][0])
        keep = df.word1.map(lambda w: in_vocab(enc, w)) & df.word2.map(lambda w: in_vocab(enc, w))
        d = df[keep].reset_index(drop=True)
        words = sorted(set(d.word1) | set(d.word2))
        pos = {w: i for i, w in enumerate(words)}
        v = normalize(enc.encode(words)[0])
        cos = (v[[pos[w] for w in d.word1]] * v[[pos[w] for w in d.word2]]).sum(axis=1)
        sp = spearmanr(d.score.to_numpy(), cos)
        out["benchmarks"][name] = {
            "kind": kind, "n_pairs_total": int(len(df)), "n_pairs_used": int(len(d)),
            "n_dropped_oov": int(len(df) - len(d)),
            # H0: ρ=0 の両側 p（scipy の t 近似）。同じ語が複数のペアに出るので目安として読む
            "rho": float(sp.statistic), "rho_pvalue": float(sp.pvalue)}
        print(f"  {name:12s} {len(df):5d} {len(d):5d} {len(df) - len(d):5d}  "
              f"{sp.statistic:+.3f} (p={sp.pvalue:.1e})")

    general = [b["rho"] for b in out["benchmarks"].values() if b["kind"] == "general"]
    out["rho_mean_general"] = float(np.mean(general))
    print(f"  一般語4種の ρ の平均 {out['rho_mean_general']:+.3f}")
    out["params"] = {"pair_filter": "both words single-token and in the vector table",
                     "lowercase": True}
    out["inputs"] = inputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "word_vector_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'word_vector_validation.json')}")


if __name__ == "__main__":
    main()
