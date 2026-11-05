"""工程1 参考文献と焦点論文のタイトルを、2つの版の語ベクトルでベクトル化する。

柴山らの Novelty の語ベクトルを scispaCy から変えたとき、結果がどう変わるかを見るため。README の「工程」1 を参照。
入力のタイトルは shibayama_novelty の工程1（clean_reference_titles.py）の出力をそのまま使う。
タイトルの単語のベクトルを重み無しで平均する点は shibayama_novelty の工程2 と同じ。違うのは次の2点だけ。

版:
    mat2vec_split   mat2vec（材料科学の論文の要旨で学習した word2vec、200次元）。単語の切り分けを mat2vec の
                    学習時の前処理に合わせ、ハイフン・アポストロフィでつないだ語を分ける（lithium-ion → lithium, ion）。
                    数字だけの語は mat2vec と同じく "<nUm>" として引く。化学式の変換（LiCoO2 → CoLiO2）はしない
                    （化学式は参考文献のタイトルの語の出現の1.4%。2026-11-05 に確かめた）。
    scispacy_split  scispaCy（shibayama_novelty と同じ語ベクトル表）。単語の切り分けを mat2vec_split にそろえる。
                    辞書の違いと、切り分け方の違いを分けて見るための比較用。
    （元の版 = scispaCy・ハイフンでつないだまま は shibayama_novelty の出力をそのまま使う）
語の引き方は shibayama_novelty と同じく、表層形 → 小文字の順。

入力: data/derived/shibayama_novelty/input/{reference_titles.parquet, focal_papers.parquet}
      data/cache/mat2vec/{vocab.txt, pretrained_embeddings.wv.vectors.npy}（fetch_mat2vec.py が用意する）、data/cache/scispacy/（shibayama_novelty の工程2 が作る）
出力: data/derived/embedding_comparison/title_vectors/<版>/
          reference_title_index.parquet, reference_title_vectors.npy, focal_title_index.parquet, focal_title_vectors.npy
      data/derived/embedding_comparison/title_vectors/embed_titles_summary.json

実行:
    python src/embedding_comparison/embed_titles.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]    # リポジトリ直下。実行時のカレントに依存しない
# shibayama_novelty の関数をそのまま使う（同じ切り出し・整形・語ベクトル表の読み方にするため）
sys.path.insert(0, str(ROOT / "src" / "shibayama_novelty"))
from clean_reference_titles import INPUT_DIR, clean_title  # noqa: E402
from embed_titles_scispacy import TOKEN_RE, ScispacyEncoder, vocab_dir  # noqa: E402

MAT2VEC_DIR = ROOT / "data" / "cache" / "mat2vec"
MAT2VEC_VECTORS = MAT2VEC_DIR / "pretrained_embeddings.wv.vectors.npy"
OUT_DIR = ROOT / "data" / "derived" / "embedding_comparison" / "title_vectors"
SPLIT_RE = re.compile(r"[-']")
NUM_TOKEN = "<nUm>"
VERSIONS = ("mat2vec_split", "scispacy_split")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def file_sha(path: Path) -> str:
    """出力 JSON に入力ファイルの版を残すためのハッシュ（先頭16桁）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def split_words(text: str) -> list[str]:
    """工程2 と同じ切り出しのあと、ハイフン・アポストロフィでつないだ語を分ける。"""
    return [p for w in TOKEN_RE.findall(text) for p in SPLIT_RE.split(w) if p]


class Mat2vecEncoder:
    def __init__(self):
        words = (MAT2VEC_DIR / "vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]
        self.data = np.load(MAT2VEC_VECTORS, allow_pickle=False).astype(np.float32)
        assert len(words) == len(self.data)
        self.row = {w: i for i, w in enumerate(words)}
        self.dim = self.data.shape[1]

    def lookup(self, word: str) -> int | None:
        if word.isdigit():
            return self.row.get(NUM_TOKEN)
        for cand in (word, word.lower()):
            r = self.row.get(cand)
            if r is not None:
                return r
        return None


class ScispacySplitEncoder(ScispacyEncoder):
    def lookup(self, word: str) -> int | None:
        return self._row(word)


def encode(enc, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 200) float32 のベクトルと、各テキストの語彙外率。shibayama_novelty の工程2 と同じく重み無し平均。"""
    vecs = np.zeros((len(texts), enc.dim), dtype=np.float32)
    oovs = np.ones(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        words = split_words(t)
        if not words:
            continue
        rows = [r for r in (enc.lookup(w) for w in words) if r is not None]
        oovs[i] = 1.0 - len(rows) / len(words)
        if rows:
            vecs[i] = enc.data[rows].mean(axis=0)
    return vecs, oovs


def main() -> None:
    refs = pd.read_parquet(INPUT_DIR / "reference_titles.parquet")
    focal = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")
    ridx0 = refs.drop_duplicates("title_key")[["title_key", "title_clean"]].reset_index(drop=True)
    ridx0["row"] = np.arange(len(ridx0))
    fidx0 = focal[["paper_uid", "matched_title"]].copy()
    fidx0["title_clean"] = clean_title(fidx0.matched_title)
    fidx0 = fidx0[fidx0.title_clean != ""].sort_values("paper_uid").reset_index(drop=True)[["paper_uid", "title_clean"]]
    fidx0["row"] = np.arange(len(fidx0), dtype=np.int32)

    encoders = {"mat2vec_split": Mat2vecEncoder(), "scispacy_split": ScispacySplitEncoder(vocab_dir())}
    summary = {}
    for v in VERSIONS:
        enc = encoders[v]
        d = OUT_DIR / v
        d.mkdir(parents=True, exist_ok=True)
        ridx, fidx = ridx0.copy(), fidx0.copy()
        rvec, roov = encode(enc, ridx.title_clean.tolist())
        fvec, foov = encode(enc, fidx.title_clean.tolist())
        ridx["oov_rate"], ridx["is_zero_norm"] = roov, np.linalg.norm(rvec, axis=1) == 0
        fidx["oov_rate"], fidx["is_zero_norm"] = foov, np.linalg.norm(fvec, axis=1) == 0
        np.save(d / "reference_title_vectors.npy", rvec)
        ridx.to_parquet(d / "reference_title_index.parquet", index=False)
        np.save(d / "focal_title_vectors.npy", fvec)
        fidx.to_parquet(d / "focal_title_index.parquet", index=False)
        summary[v] = {"refs": {"n_titles": int(len(ridx)), "mean_oov_rate": round(float(roov.mean()), 4),
                               "n_zero_norm": int(ridx.is_zero_norm.sum())},
                      "focal": {"n_papers": int(len(fidx)), "mean_oov_rate": round(float(foov.mean()), 4),
                                "n_zero_norm": int(fidx.is_zero_norm.sum())}}
        print(f"  {v:15s} 参考文献 {len(ridx):,} タイトル / 平均の語彙外率 {roov.mean():.1%} / 零ベクトル {int(ridx.is_zero_norm.sum())}"
              f"  焦点論文 {len(fidx)} / 語彙外率 {foov.mean():.1%}")
    summary["params"] = {"token_pattern": TOKEN_RE.pattern, "split_on": SPLIT_RE.pattern, "num_token_mat2vec": NUM_TOKEN,
                         "lookup": "surface form, then lowercase", "pooling": "unweighted mean of in-vocabulary words",
                         "formula_normalization": False}
    summary["inputs"] = {rel(p): file_sha(p) for p in (INPUT_DIR / "reference_titles.parquet", INPUT_DIR / "focal_papers.parquet",
                                                       MAT2VEC_DIR / "vocab.txt", MAT2VEC_VECTORS)}
    (OUT_DIR / "embed_titles_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR)}/<版>/, {rel(OUT_DIR / 'embed_titles_summary.json')}")


if __name__ == "__main__":
    main()
