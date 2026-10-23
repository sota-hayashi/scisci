"""工程6 検証【2】(a) タイトル距離と共通参照距離の相関（原論文 Table 3 の再現）。

原論文は、タイトルの埋め込み距離が意味の遠さを表しているかを、共通参照距離との相関で確かめた
（Shibayama et al. 2021, p.6–7 "Validation of distance", Table 3）。
原論文は生物医学の標本で検証した。工学が中心の本標本で同じことが言えるかを、同じ手順で確かめる。
README の「工程」6 を参照。

原論文 式(3):
    d^C_ij = 1 - coref_ij / sqrt(ref_i * ref_j)
        ref_i    : i が引用する参考文献の数
        coref_ij : i と j が共通して引用する参考文献の数

比べる相手は式(1)のタイトル距離 d^T。**焦点論文どうし**のタイトルで計算する。
原論文にならい、**同じ著者が書いた論文のペアだけ**を使う。著者が異なるペアは共通の参考文献が
ほとんど無いためである。その様子は、著者をまたいだ無作為ペアを対照として計算して示す（独自）。

区間推定（独自）: 同じ論文が多くのペアに入り、ペアどうしは独立でない。著者を単位に
復元抽出するクラスター・ブートストラップで r の 95% 区間を付ける。

入力: data/derived/shibayama_novelty/pairs/{focal_paper_pairs, same_author_paper_pairs}.parquet
      data/derived/shibayama_novelty/input/focal_papers.parquet, data/raw/papers.csv（対照ペアの抽出順）
出力: data/derived/shibayama_novelty/validation/distance_shared_refs_validation.json
      data/derived/shibayama_novelty/validation/figures/distance_shared_refs_correlation.png

実行:
    python src/shibayama_novelty/validate_distance_shared_refs.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from build_title_distance_pairs import PAIRS_DIR
from clean_reference_titles import INPUT_DIR, PAPERS_CSV, ROOT, file_sha, rel
from embed_titles_scispacy import MODEL_NAME

OUT_DIR = ROOT / "data" / "derived" / "shibayama_novelty" / "validation"
FIG_DIR = OUT_DIR / "figures"
COLOR = "#C9B458"

# 原論文 Table 3B と同じタイトル語数の区切り
WORD_BANDS = [(1, 10), (11, 13), (14, 17), (18, 999)]
N_RANDOM_PAIRS = 20_000       # 著者をまたいだ対照ペアの数
BOOTSTRAP_B = 2000
SEED = 0
MIN_BAND_N = 30

SHIBAYAMA21 = {"r": 0.231, "n": 21908,
               "by_title_words": {"1–10": 0.440, "11–13": 0.246, "14–17": 0.197, "18–": 0.259},
               "by_title_words_n": {"1–10": 1395, "11–13": 1151, "14–17": 1827, "18–": 1867},
               "source": "Shibayama, Yin & Matsumoto (2021) PLOS ONE, Table 3 (p.9)"}


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """相関係数と両側 p 値。"""
    from scipy import stats

    # float32 のままだと小さい p 値がアンダーフローして 0 になる
    r, p = stats.pearsonr(np.asarray(x, np.float64), np.asarray(y, np.float64))
    return float(r), float(p)


def corr_block(d: pd.DataFrame) -> dict:
    """全体・共引用ありのみ・語数帯別の r。"""
    out: dict = {}
    r, p = pearson(d.d.to_numpy(), d.d_cocite.to_numpy())
    out["same_author"] = {"r": round(r, 4), "p": p, "n": len(d),
                          "zero_coref_share": round(float((d.n_coref == 0).mean()), 4)}
    dz = d[d.n_coref > 0]
    if len(dz) > MIN_BAND_N:
        rz, pz = pearson(dz.d.to_numpy(), dz.d_cocite.to_numpy())
        out["same_author_coref_gt0"] = {"r": round(rz, 4), "p": pz, "n": len(dz)}
    bands = {}
    wmin = np.minimum(d.n_words_a, d.n_words_b)
    wmax = np.maximum(d.n_words_a, d.n_words_b)
    for lo, hi in WORD_BANDS:
        sub = d[(wmin >= lo) & (wmax <= hi)]          # ペアの両方がその帯に入る（Table 3B の注記）
        label = f"{lo}–{hi}" if hi < 999 else f"{lo}–"
        if len(sub) < MIN_BAND_N:
            bands[label] = {"r": None, "n": len(sub)}
            continue
        rb, pb = pearson(sub.d.to_numpy(), sub.d_cocite.to_numpy())
        bands[label] = {"r": round(rb, 4), "p": pb, "n": len(sub)}
    out["by_title_words"] = bands
    return out


def random_cross_author(fp: pd.DataFrame, same: pd.DataFrame) -> pd.DataFrame:
    """著者をまたいだ無作為ペア。同一著者ペアと重なるものは引き直す。"""
    papers = pd.read_csv(PAPERS_CSV)
    keep = set(pd.read_parquet(INPUT_DIR / "focal_papers.parquet").paper_uid)
    titled = set(fp.uid_a) | set(fp.uid_b)
    uids = [u for u in papers.loc[papers.paper_uid.isin(keep), "paper_uid"] if u in titled]
    same_set = set(zip(same.uid_a, same.uid_b))
    rng = np.random.default_rng(SEED)
    rand: set[tuple[str, str]] = set()
    while len(rand) < N_RANDOM_PAIRS:
        x, y = rng.choice(len(uids), 2, replace=False)
        pair = tuple(sorted((uids[x], uids[y])))
        if pair not in same_set:
            rand.add(pair)
    rp = pd.DataFrame(list(rand), columns=["uid_a", "uid_b"])
    return rp.merge(fp, on=["uid_a", "uid_b"], how="left", validate="one_to_one")


def cluster_bootstrap(d: pd.DataFrame, sap: pd.DataFrame) -> dict:
    """著者を復元抽出し、選ばれた著者のペアを集めて r を計算し直す（独自）。"""
    pos = {p: i for i, p in enumerate(zip(d.uid_a, d.uid_b))}
    sap = sap[[p in pos for p in zip(sap.uid_a, sap.uid_b)]]
    by_author = [np.array([pos[p] for p in zip(g.uid_a, g.uid_b)], dtype=np.int64)
                 for _, g in sap.groupby("aid")]
    x = d.d.to_numpy(np.float64)
    y = d.d_cocite.to_numpy(np.float64)
    rng = np.random.default_rng(SEED)
    rs = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        pick = rng.integers(0, len(by_author), len(by_author))
        idx = np.concatenate([by_author[i] for i in pick])
        rs[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "bootstrap_b": BOOTSTRAP_B,
            "n_authors": len(by_author), "unit": "author (aid)"}


def plot_bands(blk: dict) -> None:
    """全体と語数帯ごとの r を、原論文 Table 3 の値と並べる。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["all"] + list(blk["by_title_words"])
    x = np.arange(len(labels))
    w = 0.38
    vals = [blk["same_author"]["r"]] + [v["r"] if v["r"] is not None else np.nan
                                        for v in blk["by_title_words"].values()]
    shib = [SHIBAYAMA21["r"]] + [SHIBAYAMA21["by_title_words"][k] for k in labels[1:]]
    ns = [blk["same_author"]["n"]] + [v["n"] for v in blk["by_title_words"].values()]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - w / 2, vals, width=w, color=COLOR, label="This study (Toyota, scispaCy)")
    ax.bar(x + w / 2, shib, width=w, color="0.65", label="Shibayama21 (biomedicine, scispaCy)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(N={n:,d})" for l, n in zip(labels, ns)], fontsize=8)
    ax.set_xlabel("title word count of both papers in the pair")
    ax.set_ylabel("Pearson r  ($d^T$ vs $d^C$)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("F2  Title distance vs shared-reference distance, same-author pairs\n"
                 "[replication of Shibayama21 Table 3]", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_shared_refs_correlation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    fp = pd.read_parquet(PAIRS_DIR / "focal_paper_pairs.parquet")
    sap = pd.read_parquet(PAIRS_DIR / "same_author_paper_pairs.parquet")

    d = fp[fp.is_same_author].copy()
    n_before = len(d)
    d = d.dropna(subset=["d"]).reset_index(drop=True)
    print(f"同一著者ペア {n_before:,d} 組（零ベクトルで除外 {n_before - len(d)} 組）"
          f"／著者 {sap.aid.nunique():,d} 名")
    rp = random_cross_author(fp, d).dropna(subset=["d"])

    blk = corr_block(d)
    rr, pr = pearson(rp.d.to_numpy(), rp.d_cocite.to_numpy())
    rand = {"r": round(rr, 4), "p": pr, "n": len(rp),
            "zero_coref_share": round(float((rp.n_coref == 0).mean()), 4)}
    boot = cluster_bootstrap(d, sap)

    sa = blk["same_author"]
    print(f"  同一著者ペア r = {sa['r']:+.3f} (N={sa['n']:,d}, p={sa['p']:.1e}) "
          f"95%区間 [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]（著者 {boot['n_authors']} 名単位）")
    if "same_author_coref_gt0" in blk:
        print(f"  共引用ありのみ r = {blk['same_author_coref_gt0']['r']:+.3f}"
              f" (N={blk['same_author_coref_gt0']['n']:,d})")
    print("  語数帯 " + "  ".join(
        f"{k}: {v['r']:+.3f} (N={v['n']})" if v["r"] is not None else f"{k}: — (N={v['n']})"
        for k, v in blk["by_title_words"].items()))
    print(f"  著者をまたいだ対照 r = {rand['r']:+.3f} (N={rand['n']:,d}, "
          f"共引用なし {rand['zero_coref_share']:.1%})")
    print(f"  原論文（生物医学, scispaCy, title）: r = {SHIBAYAMA21['r']}, N={SHIBAYAMA21['n']:,d}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_bands(blk)

    out: dict = {"embed_model": MODEL_NAME, "n_pairs_same_author": len(d), **blk,
                 "same_author_r_bootstrap": boot, "random_cross_author": rand,
                 "shibayama21": SHIBAYAMA21}
    out["params"] = {"word_bands": WORD_BANDS, "n_random_pairs": N_RANDOM_PAIRS,
                     "bootstrap_b": BOOTSTRAP_B, "seed": SEED, "min_band_n": MIN_BAND_N}
    out["inputs"] = {rel(p): file_sha(p) for p in [
        PAIRS_DIR / "focal_paper_pairs.parquet", PAIRS_DIR / "same_author_paper_pairs.parquet",
        PAPERS_CSV, INPUT_DIR / "focal_papers.parquet"]}
    (OUT_DIR / "distance_shared_refs_validation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'distance_shared_refs_validation.json')}, {rel(FIG_DIR / 'distance_shared_refs_correlation.png')}")


if __name__ == "__main__":
    main()
