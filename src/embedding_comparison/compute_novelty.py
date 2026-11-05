"""工程2 2つの版のタイトルのベクトルから、柴山らの Novel_q を計算する。

計算は shibayama_novelty の工程3（compute_shibayama_novelty.py）の novelty_table をそのまま使う（ペア距離 = 1 − cos、
Novel_q = ペア距離の q パーセンタイル（nearest-rank）、零ベクトルの参考文献を除いて2本未満なら計算しない）。
違うのは、読み込むタイトルのベクトル（工程1 の版）だけ。README の「工程」2 を参照。

入力: data/derived/shibayama_novelty/input/{focal_papers.parquet, reference_titles.parquet}
      data/derived/embedding_comparison/title_vectors/<版>/{reference_title_index.parquet, reference_title_vectors.npy}
出力: data/derived/embedding_comparison/scores/novelty_<版>.parquet（列は shibayama_novelty の novelty_shibayama.parquet と同じ）
      data/derived/embedding_comparison/scores/compute_novelty_summary.json

実行:
    python src/embedding_comparison/compute_novelty.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from embed_titles import OUT_DIR as VECTORS_DIR, ROOT, VERSIONS, file_sha, rel
from clean_reference_titles import INPUT_DIR  # noqa: E402（embed_titles が shibayama_novelty を探す場所に足してある）
from compute_shibayama_novelty import Q_VALUES, novelty_table  # noqa: E402

OUT_DIR = ROOT / "data" / "derived" / "embedding_comparison" / "scores"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    focal = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")
    refs = pd.read_parquet(INPUT_DIR / "reference_titles.parquet")
    summary = {}
    inputs = [INPUT_DIR / "focal_papers.parquet", INPUT_DIR / "reference_titles.parquet"]
    for v in VERSIONS:
        d = VECTORS_DIR / v
        index = pd.read_parquet(d / "reference_title_index.parquet")
        vectors = np.load(d / "reference_title_vectors.npy")
        out, rep = novelty_table(focal, refs, index, vectors)
        out["embed_model"] = v
        out.to_parquet(OUT_DIR / f"novelty_{v}.parquet", index=False)
        summary[v] = {"n_computed": rep["n_computed"], "n_pairs_total": rep["n_pairs_total"],
                      "skipped_zero_norm": rep["skipped_zero_norm"],
                      "novel_q_mean": {f"q{q}": float(out[f"novel_{q}"].mean()) for q in Q_VALUES}}
        inputs += [d / "reference_title_index.parquet", d / "reference_title_vectors.npy"]
        print(f"  {v:15s} 計算できた {rep['n_computed']} 本 / Novel_100 平均 {out.novel_100.mean():.3f} / Novel_50 平均 {out.novel_50.mean():.3f}"
              f" / 除いた論文 {rep['skipped_zero_norm']}")
    summary["params"] = {"q_values": list(Q_VALUES), "method": "shibayama_novelty.compute_shibayama_novelty.novelty_table"}
    summary["inputs"] = {rel(p): file_sha(p) for p in inputs}
    (OUT_DIR / "compute_novelty_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR)}/novelty_<版>.parquet, {rel(OUT_DIR / 'compute_novelty_summary.json')}")


if __name__ == "__main__":
    main()
