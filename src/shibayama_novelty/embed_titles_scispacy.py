"""工程2 タイトルのベクトル化（原論文の Step 1）。

原典: Shibayama, Yin & Matsumoto (2021) PLOS ONE 16(7):e0254034
README の「工程」2 を参照。

原論文と同じ scispaCy `en_core_sci_lg`（200次元 / 600,000ベクトル）の語ベクトル表を使い、
タイトルに含まれる語のベクトルを重み無しで平均する。ストップワードは除かない（原論文と同じ）。
語ベクトルを持たない語は平均に含めない。零ベクトルとして含めてもコサイン距離は変わらない。

scispacy パッケージは Python 3.13 に入らない（spacy<3.8 → thinc → Cython<3.0 が
NumPy 2.x と衝突する）。必要なのは語ベクトルの表だけなので、配布物の `vocab/` を
spaCy の Vectors クラスで直接読む。tagger / parser / ner は使わない。

語ベクトル表（約600MB）は 50MB を超えるのでコミットしない。初回に配布元（SCISPACY_URL）から
落として data/cache/scispacy/ に `vocab/` と `meta.json` だけを解凍し、2回目からはそれを使う。
data/cache/scispacy/ は .gitignore に入れてある。
正しいモデルかは2段で確かめる。meta.json の名前・版・表の大きさ（ここで確かめる）と、
reproduce_shibayama_novelty.ipynb での出力の一致（全タイトルのベクトルがコミットされたものとバイト単位で同じか）。
語ベクトル表を読むのはこの工程と工程5（単語の検証）だけ。工程3, 4, 7 はここで保存したベクトルを読む。

入力: data/derived/shibayama_novelty/input/{reference_titles.parquet, focal_papers.parquet}
出力: data/derived/shibayama_novelty/title_vectors/
    reference_title_index.parquet, reference_title_vectors.npy  参考文献タイトル（unique）→ 工程3, 4
    focal_title_index.parquet, focal_title_vectors.npy          焦点論文タイトル          → 工程4, 7
    embed_titles_scispacy_summary.json

実行:
    python src/shibayama_novelty/embed_titles_scispacy.py
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from clean_reference_titles import INPUT_DIR, ROOT, clean_title, file_sha, rel

VECTORS_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "title_vectors"
MODEL_NAME = "en_core_sci_lg-0.5.4"
SCISPACY_URL = ("https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
                "en_core_sci_lg-0.5.4.tar.gz")
SCISPACY_CACHE = ROOT / "data" / "cache" / "scispacy" / MODEL_NAME
# meta.json で確かめる値（名前、版、ベクトルの次元と本数）
SCISPACY_META = {"name": "core_sci_lg", "version": "0.5.4", "width": 200, "vectors": 600000}
# spaCy の Vectors.from_disk が読む vocab/ の中のファイル
VOCAB_FILES = ("vectors", "key2row", "strings.json", "vectors.cfg")

# 語の切り出し。ハイフン・アポストロフィで繋がる語は1語として扱う
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


class ScispacyEncoder:
    """scispaCy の語ベクトル表でテキストをベクトル化する。"""

    def __init__(self, vocab_dir: Path):
        # spaCy の Vectors は「文字列のハッシュ → 行番号」の表（key2row）と
        # ベクトルの行列（data）を持つ。語は hash_string で求めたキーで引く
        from spacy.strings import hash_string
        from spacy.vectors import Vectors

        self._hash = hash_string
        vec = Vectors().from_disk(vocab_dir)
        self.data = np.asarray(vec.data, dtype=np.float32)
        self.key2row = vec.key2row
        self.dim = self.data.shape[1]

    def _row(self, word: str) -> int | None:
        """表層形 → 小文字 の順に引く（spaCy の orth → norm の挙動に近い）。"""
        for cand in (word, word.lower()):
            r = self.key2row.get(self._hash(cand))
            if r is not None:
                return r
        return None

    def _encode_one(self, text: str) -> tuple[np.ndarray, float]:
        words = TOKEN_RE.findall(text)
        if not words:
            return np.zeros(self.dim, dtype=np.float32), 1.0
        rows = [r for r in (self._row(w) for w in words) if r is not None]
        oov = 1.0 - len(rows) / len(words)
        if not rows:
            return np.zeros(self.dim, dtype=np.float32), oov
        return self.data[rows].mean(axis=0), oov

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(n, 200) float32 のベクトルと、各テキストの OOV 率。L2 正規化はしない。"""
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        oovs = np.ones(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs[i], oovs[i] = self._encode_one(t)
        return vecs, oovs


def fetch_scispacy() -> None:
    """配布物を落とし、vocab/ と meta.json だけを SCISPACY_CACHE に解凍する。"""
    tmp = SCISPACY_CACHE.with_name(SCISPACY_CACHE.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(f"  語ベクトル表を落とす（約530MB）: {SCISPACY_URL}")
    # 配布物の中は en_core_sci_lg-0.5.4/en_core_sci_lg/en_core_sci_lg-0.5.4/{meta.json, vocab/...}。
    # mode="r|gz" はダウンロードしながら順に読むので、.tar.gz をディスクに置かずに済む
    inner = f"{MODEL_NAME}/en_core_sci_lg/{MODEL_NAME}/"
    with urllib.request.urlopen(SCISPACY_URL) as res, tarfile.open(fileobj=res, mode="r|gz") as tar:
        for m in tar:
            if not m.name.startswith(inner):
                continue
            name = m.name[len(inner):]
            if name == "meta.json" or name.startswith("vocab/"):
                m.name = name
                # filter="data" は外の場所への書き出しやリンクを拒む（tarfile の安全な展開）
                tar.extract(m, tmp, filter="data")
    if not (tmp / "meta.json").exists() or not (tmp / "vocab" / "vectors").exists():
        raise SystemExit("配布物に meta.json か vocab/vectors が無い。停止する。")
    tmp.rename(SCISPACY_CACHE)


def vocab_dir() -> Path:
    """語ベクトル表の vocab/ の場所。無ければ落とし、meta.json で正しいモデルか確かめる。"""
    files = ["meta.json", *(f"vocab/{n}" for n in VOCAB_FILES)]
    if not all((SCISPACY_CACHE / f).exists() for f in files):
        # 途中で欠けたもの（vocab/ だけ消えた場合など）は消して落とし直す
        shutil.rmtree(SCISPACY_CACHE, ignore_errors=True)
        fetch_scispacy()
    meta = json.loads((SCISPACY_CACHE / "meta.json").read_text())
    got = {"name": meta.get("name"), "version": meta.get("version"),
           "width": meta.get("vectors", {}).get("width"),
           "vectors": meta.get("vectors", {}).get("vectors")}
    if got != SCISPACY_META:
        raise SystemExit(f"{rel(SCISPACY_CACHE)} のモデルが違う: {got}（期待値 {SCISPACY_META}）。"
                         "フォルダを消して流し直すこと。")
    return SCISPACY_CACHE / "vocab"


def main() -> None:
    refs = pd.read_parquet(INPUT_DIR / "reference_titles.parquet")
    focal = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")

    enc = ScispacyEncoder(vocab_dir())
    print(f"  語ベクトル表 {len(enc.key2row):,d} キー / {enc.data.shape[0]:,d} ベクトル × {enc.dim} 次元")

    # 参考文献: title_key ごとに1本だけ（同じ文献を複数の焦点論文が引くため）
    ridx = (refs.drop_duplicates("title_key")[["title_key", "title_clean"]]
            .reset_index(drop=True))
    ridx["row"] = np.arange(len(ridx))
    t0 = time.time()
    rvec, roov = enc.encode(ridx.title_clean.tolist())
    ridx["oov_rate"] = roov
    ridx["is_zero_norm"] = np.linalg.norm(rvec, axis=1) == 0     # 距離を定義できない。工程3で除く
    print(f"  参考文献 {len(ridx):,d} タイトル（{time.time() - t0:.1f}秒）/ 平均 OOV 率 "
          f"{roov.mean():.1%} / 零ベクトル {int(ridx.is_zero_norm.sum())} 件")

    # 焦点論文: OpenAlex 側のタイトル。並びは paper_uid 順（工程4で uid_a < uid_b にするため）
    fidx = focal[["paper_uid", "matched_title"]].copy()
    fidx["title_clean"] = clean_title(fidx.matched_title)
    fidx = (fidx[fidx.title_clean != ""].sort_values("paper_uid")
            .reset_index(drop=True)[["paper_uid", "title_clean"]])
    fidx["row"] = np.arange(len(fidx), dtype=np.int32)
    fidx["n_words"] = fidx.title_clean.str.split().str.len().astype(np.int16)
    fvec, foov = enc.encode(fidx.title_clean.tolist())
    fidx["oov_rate"] = foov
    fidx["is_zero_norm"] = np.linalg.norm(fvec, axis=1) == 0
    print(f"  焦点論文 {len(focal)} 本中 タイトルあり {len(fidx)} 本 / 零ベクトル "
          f"{int(fidx.is_zero_norm.sum())} 本")

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_DIR / "reference_title_vectors.npy", rvec)
    ridx.to_parquet(VECTORS_DIR / "reference_title_index.parquet", index=False)
    np.save(VECTORS_DIR / "focal_title_vectors.npy", fvec)
    fidx.to_parquet(VECTORS_DIR / "focal_title_index.parquet", index=False)

    report = {
        "model": MODEL_NAME, "dim": int(enc.dim), "n_vectors_in_table": int(enc.data.shape[0]),
        "refs": {"n_titles": int(len(ridx)), "mean_oov_rate": round(float(roov.mean()), 4),
                 "n_zero_norm": int(ridx.is_zero_norm.sum())},
        "focal": {"n_papers": int(len(focal)), "n_titled": int(len(fidx)),
                  "mean_oov_rate": round(float(foov.mean()), 4),
                  "n_zero_norm": int(fidx.is_zero_norm.sum())},
        "params": {"token_pattern": TOKEN_RE.pattern, "lookup": "surface form, then lowercase",
                   "pooling": "unweighted mean of in-vocabulary words",
                   "stopwords_removed": False},
        "inputs": {rel(p): file_sha(p) for p in (INPUT_DIR / "reference_titles.parquet",
                                                 INPUT_DIR / "focal_papers.parquet")},
    }
    (VECTORS_DIR / "embed_titles_scispacy_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(VECTORS_DIR)}/ reference_title_index.parquet, reference_title_vectors.npy, focal_title_index.parquet, "
          "focal_title_vectors.npy, embed_titles_scispacy_summary.json")


if __name__ == "__main__":
    main()
