"""mat2vec の学習済み語ベクトルを落とし、語彙とベクトルを取り出す。工程には入れない。

mat2vec（Tshitoyan ら 2019, Nature 571:95。材料科学の論文の要旨で学習した word2vec、529,686語 × 200次元）を、
柴山らの Novelty の語ベクトルに使う（scispaCy と比べる）。配布元は github.com/materialsintelligence/mat2vec
（MIT ライセンス）。配布物は gensim 3.x の pickle（語彙）と numpy の .npy（ベクトル）。

pickle は読み込むと任意のコードを実行できる形式なので、gensim を入れずに、呼び出してよいモジュールを決めた
一覧（ALLOWED）だけを許す制限つきの読み方で読む。gensim のクラスは中身を持たない代わりのクラスで受ける。
語彙（index2word の順）を vocab.txt に保存する。ベクトルは配布された .npy をそのまま使う（写さない）。SHA-256 を info.json に残す。

出力: data/cache/mat2vec/pretrained_embeddings, pretrained_embeddings.wv.vectors.npy（配布物）, vocab.txt, info.json
     （data/cache/ は .gitignore 済み）

実行:
    python src/embedding_comparison/fetch_mat2vec.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]    # リポジトリ直下。実行時のカレントに依存しない
CACHE_DIR = ROOT / "data" / "cache" / "mat2vec"
BASE_URL = "https://storage.googleapis.com/mat2vec/"
FILES = {"model": "pretrained_embeddings", "vectors": "pretrained_embeddings.wv.vectors.npy"}
EXPECTED_SHAPE = (529686, 200)


class _Stub:
    """gensim のクラスの代わり。pickle の中身（属性）だけを受け取る。"""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
            state = state[0]                     # (state, slotstate) の形
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


def _stub_class(name: str) -> type:
    return type(name, (_Stub,), {})


# 呼び出してよいモジュールと名前。これ以外が出てきたら読み込みを止める
ALLOWED = {
    ("gensim.models.word2vec", "Word2Vec"), ("gensim.models.word2vec", "Word2VecVocab"),
    ("gensim.models.word2vec", "Word2VecTrainables"), ("gensim.models.keyedvectors", "Word2VecKeyedVectors"),
    ("gensim.models.keyedvectors", "Vocab"), ("gensim.utils", "call_on_class_only"),
    ("collections", "defaultdict"), ("__builtin__", "long"), ("builtins", "int"), ("__builtin__", "hash"),
    ("builtins", "hash"), ("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
    ("_codecs", "encode"), ("numpy.random", "__RandomState_ctor"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"許していない呼び出し: {module}.{name}")
        if module.startswith("gensim"):
            return _stub_class(name)
        if (module, name) in {("__builtin__", "long"), ("builtins", "int")}:
            return int
        if name == "hash":
            return hash
        if module == "numpy.random":
            return lambda *a, **k: _Stub()       # 乱数の状態は使わないので、中身を受け取るだけにする
        if module == "numpy.core.multiarray":
            return np._core.multiarray._reconstruct
        if module == "_codecs":
            import codecs
            return codecs.encode
        if module == "collections":
            import collections
            return collections.defaultdict
        return getattr(np, name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str) -> Path:
    path = CACHE_DIR / name
    if not path.exists():
        print(f"  落とす: {BASE_URL}{name}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        tmp.rename(path)
    return path


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path, vec_path = download(FILES["model"]), download(FILES["vectors"])
    with open(model_path, "rb") as f:
        model = SafeUnpickler(f, encoding="latin1").load()
    words = model.wv.__dict__["index2word"]
    vectors = np.load(vec_path, allow_pickle=False)
    if vectors.shape != EXPECTED_SHAPE or len(words) != EXPECTED_SHAPE[0]:
        raise SystemExit(f"大きさが想定と違う: 語彙 {len(words)}、ベクトル {vectors.shape}（期待値 {EXPECTED_SHAPE}）")
    (CACHE_DIR / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    (CACHE_DIR / "vectors.npy").unlink(missing_ok=True)      # 以前の版が写していたもの。配布物と同じ中身なので消す
    info = {"source": BASE_URL, "files": {k: {"name": v, "sha256": sha256(CACHE_DIR / v)} for k, v in FILES.items()},
            "n_words": len(words), "dim": int(vectors.shape[1]),
            "vocab_txt_sha256": sha256(CACHE_DIR / "vocab.txt")}
    (CACHE_DIR / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"  語彙 {len(words):,} 語 × {vectors.shape[1]} 次元。保存: {CACHE_DIR.relative_to(ROOT)}/vocab.txt, info.json")


if __name__ == "__main__":
    main()
