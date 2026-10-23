"""工程7 検証【2】(b) タイトル距離と OpenAlex の分野ラベルの対応、【3】遠い側。

README の「工程」7 を参照。

【2】(b) タイトルベクトルの距離が OpenAlex のラベル（field / subfield）と対応するか
    kNN 分類誤差           [Kusner15 §5.1]   k ∈ {1..19} を学習側の 80/20 で選ぶ。5 分割
    k-means + V-measure    [MTEB §3.2]       mini-batch, batch 32, k = ラベル数
    ペア AP                [MTEB §3.2]       同じラベルか否かを cos 類似度で当てる
    t-SNE + DBSCAN         [SPECTER §6]      homogeneity / completeness
【3】遠い側に絞っても距離が意味を担うか（独自の適用）
    Novel_q は距離の高い側を読むので、距離の上位 p% の帯ごとにペア AP を計算し直す
    距離の十分位ごとのラベル不一致率（記述図）

比較の基準は、kNN では「いつも最も多いラベルを答えた場合の誤分類率」、ペア AP では
「同じラベルのペアの割合」（ランダムなスコアで得られる AP の期待値）。
OpenAlex のラベルはタイトルも入力にした分類器で付いているので、一致は実際より高く出うる。
工程6（文章を使わない共通参照距離）と必ず併せて読む。

論文が10件未満のラベルは除く（独自）。零ベクトルになる論文は除く。

入力: data/derived/shibayama_novelty/title_vectors/{focal_title_index.parquet, focal_title_vectors.npy}
      data/derived/shibayama_novelty/pairs/focal_paper_pairs.parquet, data/derived/shibayama_novelty/input/focal_papers.parquet, data/raw/papers.csv
出力: data/derived/shibayama_novelty/validation/{distance_field_label_validation.json, tsne_title_coords.parquet}
      data/derived/shibayama_novelty/validation/figures/{tsne_by_field.png, field_disagreement_by_distance_decile.png}

実行:
    python src/shibayama_novelty/validate_distance_field_labels.py
"""
from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (average_precision_score, homogeneity_completeness_v_measure,
                             v_measure_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from build_title_distance_pairs import PAIRS_DIR, normalize
from clean_reference_titles import INPUT_DIR, PAPERS_CSV, file_sha, rel
from embed_titles_scispacy import VECTORS_DIR, MODEL_NAME
from validate_distance_shared_refs import FIG_DIR, OUT_DIR

MIN_LABEL_N = 10
KNN_K = range(1, 20)
N_FOLDS = 5
KMEANS_SEEDS = range(10)
KMEANS_BATCH = 32
TSNE_PERPLEXITY = 30
TSNE_SWEEP = (10, 50)
DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS_PCTL = 90            # 5番目に近い点までの距離の 90 パーセンタイル（独自の暫定値）
FAR_TOP_PCT = (50, 20, 10, 5, 1)
LEVELS_B = ("field", "subfield")
LEVELS_FAR = ("domain", "field")
LEVELS_DECILE = ("topic", "subfield", "field", "domain")
SEED = 0


def load() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, int]:
    fidx = pd.read_parquet(VECTORS_DIR / "focal_title_index.parquet")
    lab = pd.read_parquet(INPUT_DIR / "focal_papers.parquet")[
        ["paper_uid", "primary_domain", "primary_field", "primary_subfield"]]
    topic = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "primary_topic"])
    fidx = fidx.merge(lab, on="paper_uid", how="left").merge(topic, on="paper_uid", how="left")
    fidx = fidx.rename(columns={"primary_domain": "domain", "primary_field": "field",
                                "primary_subfield": "subfield", "primary_topic": "topic"})
    vecs = np.load(VECTORS_DIR / "focal_title_vectors.npy")
    zero = fidx.is_zero_norm.to_numpy()
    n_zero = int(zero.sum())
    fidx = fidx[~zero].reset_index(drop=True)
    vecs = vecs[fidx.row.to_numpy()]               # row は npy の行番号
    fp = pd.read_parquet(PAIRS_DIR / "focal_paper_pairs.parquet", columns=["uid_a", "uid_b", "d"])
    ok = set(fidx.paper_uid)
    fp = fp[fp.uid_a.isin(ok) & fp.uid_b.isin(ok)].reset_index(drop=True)
    return fidx, vecs, fp, n_zero


def target(fidx: pd.DataFrame, level: str) -> np.ndarray:
    """論文が MIN_LABEL_N 件以上あるラベルの論文だけ残すマスク。"""
    vc = fidx[level].value_counts()
    return fidx[level].isin(vc[vc >= MIN_LABEL_N].index).to_numpy()


# ────────────────────────────────────────────────────────────────
# 【2】(b)
# ────────────────────────────────────────────────────────────────
def knn_eval(X: np.ndarray, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    errs, ks, maj = [], [], []
    for tr, te in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        a, v = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=ytr,
                                random_state=SEED)
        best_k, best_e = None, np.inf
        for k in KNN_K:
            if k > len(a):
                break
            clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
            e = 1.0 - clf.fit(Xtr[a], ytr[a]).score(Xtr[v], ytr[v])
            if e < best_e:                         # 同点なら小さい k を残す
                best_k, best_e = k, e
        clf = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", algorithm="brute")
        errs.append(1.0 - clf.fit(Xtr, ytr).score(X[te], y[te]))
        ks.append(int(best_k))
        top = collections.Counter(ytr).most_common(1)[0][0]
        maj.append(float(np.mean(y[te] != top)))
    errs = np.array(errs)
    return {"error_mean": float(errs.mean()),
            "error_se": float(errs.std(ddof=1) / np.sqrt(len(errs))),
            "error_by_fold": errs.tolist(), "k_chosen": ks,
            "majority_error_mean": float(np.mean(maj))}


def kmeans_eval(X: np.ndarray, y: np.ndarray) -> dict:
    Xn = normalize(X)                              # 式(1)の cos 距離と整合させる
    k = len(set(y))
    vm = [v_measure_score(y, MiniBatchKMeans(n_clusters=k, batch_size=KMEANS_BATCH,
                                             random_state=s).fit_predict(Xn))
          for s in KMEANS_SEEDS]
    return {"v_measure_mean": float(np.mean(vm)), "v_measure_sd": float(np.std(vm, ddof=1)),
            "by_seed": [float(x) for x in vm]}


def pair_labels(fp: pd.DataFrame, fidx: pd.DataFrame, level: str, mask: np.ndarray
                ) -> tuple[pd.DataFrame, np.ndarray]:
    lab = dict(zip(fidx.paper_uid[mask], fidx[level][mask]))
    sub = fp[fp.uid_a.isin(lab) & fp.uid_b.isin(lab)]
    same = (sub.uid_a.map(lab).to_numpy() == sub.uid_b.map(lab).to_numpy())
    return sub, same


def ap(y: np.ndarray, d: np.ndarray) -> float | None:
    if y.sum() == 0 or y.all():
        return None
    return float(average_precision_score(y, 1.0 - d))


def tsne_dbscan(X: np.ndarray, y: np.ndarray, perplexity: int) -> tuple[dict, np.ndarray, np.ndarray]:
    xy = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
              random_state=SEED).fit_transform(X)
    dist, _ = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES + 1).fit(xy).kneighbors(xy)
    eps = float(np.percentile(dist[:, DBSCAN_MIN_SAMPLES], DBSCAN_EPS_PCTL))  # 列0は自分自身
    lab = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
    h, c, v = homogeneity_completeness_v_measure(y, lab)   # ノイズ(-1)は1つのクラスタ扱い
    return ({"homogeneity": float(h), "completeness": float(c), "v_measure": float(v),
             "n_clusters": int(len(set(lab) - {-1})), "noise_share": float(np.mean(lab == -1)),
             "eps": eps}, xy, lab)


def plot_tsne(xy: np.ndarray, y: np.ndarray, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cnt = collections.Counter(y)
    order = [lab for lab, _ in cnt.most_common()]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, lab in enumerate(order):
        m = y == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.8, color=cmap(i % 10),
                   label=f"{lab} ({cnt[lab]})")
    ax.set_title(f"t-SNE of focal-paper title vectors, coloured by OpenAlex field\n"
                 f"(perplexity={TSNE_PERPLEXITY}, n={len(y)})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")     # 外に出した凡例を切らない
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
# 【3】
# ────────────────────────────────────────────────────────────────
def far_bands(sub: pd.DataFrame, same: np.ndarray) -> list[dict]:
    d = sub.d.to_numpy()
    order = np.argsort(-d, kind="stable")
    out = []
    for p in FAR_TOP_PCT:
        n = int(np.ceil(len(d) * p / 100))
        idx = order[:n]
        y = same[idx]
        out.append({"top_pct": p, "d_threshold": float(d[idx].min()), "n_pairs": n,
                    "n_positive": int(y.sum()), "prevalence": float(y.mean()),
                    "ap": ap(y, d[idx])})
    return out


def decile_disagreement(fp: pd.DataFrame, fidx: pd.DataFrame) -> dict:
    d = fp.d.to_numpy()
    edges = np.quantile(d, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 9)
    out = {"edges": edges.astype(float).tolist()}
    for lv in LEVELS_DECILE:
        lab = dict(zip(fidx.paper_uid, fidx[lv]))
        diff = fp.uid_a.map(lab).to_numpy() != fp.uid_b.map(lab).to_numpy()
        out[lv] = [float(diff[bins == b].mean()) for b in range(10)]
    return out


def plot_deciles(dec: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(1, 11)
    for lv, mk in zip(LEVELS_DECILE, "osD^"):
        ax.plot(x, dec[lv], marker=mk, label=lv)
    ax.set_xticks(x)
    ax.set_xlabel("distance decile (1 = nearest, 10 = farthest)")
    ax.set_ylabel("share of pairs with different labels")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(title="OpenAlex level", fontsize=8)
    ax.set_title("Label disagreement by title-distance decile\n(focal papers, all pairs)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fidx, vecs, fp, n_zero = load()
    print(f"焦点論文 {len(fidx)} 本（零ベクトルで除外 {n_zero}）/ ペア {len(fp):,d} 組")

    report: dict = {"embed_model": MODEL_NAME, "n_zero_norm_excluded": n_zero}
    coords = []
    for level in LEVELS_B:
        mask = target(fidx, level)
        y = fidx[level].to_numpy()[mask]
        X = vecs[mask]
        sub, same = pair_labels(fp, fidx, level, mask)
        blk: dict = {"n_papers": int(mask.sum()), "n_labels": int(len(set(y))),
                     "knn": knn_eval(X, y), "kmeans": kmeans_eval(X, y),
                     "pair_ap": {"ap": ap(same, sub.d.to_numpy()),
                                 "prevalence": float(same.mean()), "n_pairs": int(len(sub))}}
        print(f"── 【2】(b) {level}: {mask.sum()} 本 / {len(set(y))} ラベル " + "─" * 20)
        print(f"  kNN 誤分類 {blk['knn']['error_mean']:.3f} ± {blk['knn']['error_se']:.3f}"
              f" (k={blk['knn']['k_chosen']}) / 基準（多数派）{blk['knn']['majority_error_mean']:.3f}")
        print(f"  V-measure {blk['kmeans']['v_measure_mean']:.3f} ± {blk['kmeans']['v_measure_sd']:.3f}")
        print(f"  ペア AP {blk['pair_ap']['ap']:.3f} / 基準（正例の割合）{blk['pair_ap']['prevalence']:.3f}")

        if level == "field":
            res, xy, lab = tsne_dbscan(X, y, TSNE_PERPLEXITY)
            blk["tsne_dbscan"] = res
            coords.append(pd.DataFrame({"paper_uid": fidx.paper_uid[mask].to_numpy(),
                                        "perplexity": TSNE_PERPLEXITY, "x": xy[:, 0],
                                        "y": xy[:, 1], "dbscan_label": lab}))
            plot_tsne(xy, y, FIG_DIR / "tsne_by_field.png")
            print(f"  t-SNE+DBSCAN homogeneity {res['homogeneity']:.3f} / completeness "
                  f"{res['completeness']:.3f}（クラスタ {res['n_clusters']}、ノイズ {res['noise_share']:.1%}）")
            blk["tsne_sweep"] = {}
            for p in TSNE_SWEEP:
                r2, xy2, lab2 = tsne_dbscan(X, y, p)
                blk["tsne_sweep"][str(p)] = r2
                coords.append(pd.DataFrame({"paper_uid": fidx.paper_uid[mask].to_numpy(),
                                            "perplexity": p, "x": xy2[:, 0], "y": xy2[:, 1],
                                            "dbscan_label": lab2}))
            print("  付録（perplexity 10 / 50）: " + "; ".join(
                f"p={p} h={blk['tsne_sweep'][str(p)]['homogeneity']:.2f}"
                f" c={blk['tsne_sweep'][str(p)]['completeness']:.2f}" for p in TSNE_SWEEP))
        report[level] = blk

    print("── 【3】遠い側: 距離の上位帯ごとのペア AP " + "─" * 14)
    report["far_bands"] = {}
    for level in LEVELS_FAR:
        mask = target(fidx, level)
        sub, same = pair_labels(fp, fidx, level, mask)
        report["far_bands"][level] = {"n_papers": int(mask.sum()), "n_pairs": int(len(sub)),
                                      "bands": far_bands(sub, same)}
        print(f"  {level:7s} " + "  ".join(
            f"上位{b['top_pct']}%: AP {b['ap'] if b['ap'] is None else round(b['ap'], 3)}"
            f"/基準 {b['prevalence']:.3f}" for b in report["far_bands"][level]["bands"]))

    print("── 【3】距離の十分位ごとのラベル不一致率 " + "─" * 14)
    dec = decile_disagreement(fp, fidx)
    report["decile_disagreement"] = dec
    plot_deciles(dec, FIG_DIR / "field_disagreement_by_distance_decile.png")
    print("  " + "  ".join(f"{lv}: {dec[lv][0]:.2f}→{dec[lv][-1]:.2f}" for lv in LEVELS_DECILE)
          + "（第1十分位→第10十分位）")

    pd.concat(coords, ignore_index=True).to_parquet(OUT_DIR / "tsne_title_coords.parquet", index=False)
    report["params"] = {
        "min_label_n": MIN_LABEL_N, "knn_k_range": [min(KNN_K), max(KNN_K)], "n_folds": N_FOLDS,
        "kmeans_seeds": [min(KMEANS_SEEDS), max(KMEANS_SEEDS)], "kmeans_batch_size": KMEANS_BATCH,
        "tsne_perplexity": TSNE_PERPLEXITY, "tsne_perplexity_sweep": list(TSNE_SWEEP),
        "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        "dbscan_eps_rule": f"p{DBSCAN_EPS_PCTL} of {DBSCAN_MIN_SAMPLES}th-NN distance",
        "far_top_pct": list(FAR_TOP_PCT), "seed": SEED}
    report["inputs"] = {rel(p): file_sha(p) for p in [
        VECTORS_DIR / "focal_title_index.parquet", VECTORS_DIR / "focal_title_vectors.npy",
        PAIRS_DIR / "focal_paper_pairs.parquet", INPUT_DIR / "focal_papers.parquet", PAPERS_CSV]}
    (OUT_DIR / "distance_field_label_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'distance_field_label_validation.json')}, {rel(OUT_DIR / 'tsne_title_coords.parquet')}")
    print(f"      {rel(FIG_DIR)}/ tsne_by_field.png, field_disagreement_by_distance_decile.png")


if __name__ == "__main__":
    main()
