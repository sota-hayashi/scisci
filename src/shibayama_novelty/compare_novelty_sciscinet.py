"""工程13 Novel_q を、SciSciNet が公開している別の新規性の指標と比べる（妥当性の確認）。

原理の違う新しさの指標と一致するかで、Novel_q の妥当性を確かめる。README の「工程」13 を参照。
比べる相手は Uzzi ら（2013）の新規性と定石らしさ（参考文献の学術誌の組み合わせのめずらしさ）。
Uzzi らの新規性は専門家の評価と整合したという報告がある（Bornmann ら 2019）。
あわせて disruption（新しさとは別の概念）とも比べる。比べ方（指標と統制）は独自。

符号: ss_novelty（Atyp_10pct_Z）は小さいほど新しい。Novel_q は大きいほど新しい。
2つが同じものを測っていれば、相関は負になる。ss_conventionality（Atyp_Median_Z）は大きいほど定石どおり。

相関:
    すべて Spearman の順位相関（ss_novelty は最大 1万を超える外れ値があるため）。
    raw          そのまま
    partial_own  n_refs_used（Novel_q の計算に使った参考文献数）を除いた偏相関
    partial_both n_refs_used と ss_reference_count（SciSciNet 側の参考文献数）を除いた偏相関
                 （Uzzi らの指標は SciSciNet 側の参考文献リストから計算されているため）
    partial_all  n_refs_used に加えて、分野・出版年・組織のダミーを除いた偏相関
                 （分野・年・組織という共通の背景だけで相関が生じていないかを見る）
    偏相関は、Novel_q と SciSciNet の指標を順位にし、統制変数で回帰した残差どうしの相関。
    参考文献数も順位にして回帰に入れる（順位にするので log をとっても結果は同じ）。
    分野は primary_field（論文が MIN_FIELD 本未満の分野は other）、出版年は YEAR_POOL_UPTO 年以前を
    1つにまとめ、組織は source_org。それぞれ基準を1つ落としたダミーにする（まとめ方は工程11 と同じ。独自）。
    p 値は t 分布（自由度 n − 2 − 統制変数の列数）、95% 区間は Fisher の z 変換で出す。

標本: 計算できた論文（782本）のうち、比べる列に値がある論文。

入力: data/derived/shibayama_novelty/scores/novelty_shibayama.parquet, data/raw/sciscinet_metrics.csv,
      data/raw/papers.csv（分野・出版年・組織）
出力: data/derived/shibayama_novelty/validation/novelty_sciscinet_comparison.json
      data/derived/shibayama_novelty/validation/figures/novelty_sciscinet_scatter.png

実行:
    python src/shibayama_novelty/compare_novelty_sciscinet.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from build_title_distance_pairs import NOVELTY_PARQUET  # noqa: E402
from clean_reference_titles import PAPERS_CSV, RAW_DIR, file_sha, rel  # noqa: E402
from compute_shibayama_novelty import Q_VALUES  # noqa: E402
from regress_citations_on_novelty import YEAR_POOL_UPTO  # noqa: E402
from validate_distance_shared_refs import COLOR, FIG_DIR, OUT_DIR  # noqa: E402

SCISCINET_CSV = RAW_DIR / "sciscinet_metrics.csv"
TARGETS = ("ss_novelty", "ss_conventionality", "ss_disruption")
SYMLOG_LINTHRESH = 10          # 図の縦軸（ss_novelty）で、これより小さい範囲は線形
MIN_FIELD = 20                 # これ未満の分野は other にまとめる
PARTIALS = ("raw", "partial_own", "partial_both", "partial_all")


def corr_ranks(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray],
               dummies: np.ndarray | None = None) -> dict:
    """順位の（偏）相関。統制変数があれば、回帰した残差どうしの相関を取る。

    controls は順位にしてから回帰に入れる連続量、dummies は 0/1 のダミーの列（そのまま入れる）。
    """
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    cols = [stats.rankdata(c) for c in controls]
    if dummies is not None:
        cols += list(dummies.T)
    k = len(cols)
    if k:
        Z = np.column_stack([np.ones(len(x)), *cols])
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    n = len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - k
    t = r * np.sqrt(dof / (1 - r ** 2))
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    return {"n": int(n), "rho": r, "p": float(2 * stats.t.sf(abs(t), dof)),
            "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]}


def plot(df: pd.DataFrame) -> None:
    d = df[df.ss_novelty.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, label in [(axes[0], "ss_novelty", "Uzzi novelty (Atyp_10pct_Z; lower = more novel)"),
                           (axes[1], "ss_conventionality", "Uzzi conventionality (Atyp_Median_Z)")]:
        ax.scatter(d.novel_100, d[col], s=8, alpha=0.6, lw=0, color="0.3")
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
        ax.axhline(0, color=COLOR, lw=1)
        rho = stats.spearmanr(d.novel_100, d[col]).statistic
        ax.set_xlabel("Novel_100 (this study; higher = more novel)")
        ax.set_ylabel(label + ", symlog")
        ax.set_title(f"n={len(d)}, Spearman rho={rho:+.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novelty_sciscinet_scatter.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nov = pd.read_parquet(NOVELTY_PARQUET, columns=["paper_uid", "n_refs_used", *[f"novel_{q}" for q in Q_VALUES]])
    ss = pd.read_csv(SCISCINET_CSV, usecols=["paper_uid", "ss_reference_count", *TARGETS])
    papers = pd.read_csv(PAPERS_CSV, usecols=["paper_uid", "year_openalex", "primary_field", "source_org"])
    df = nov.merge(ss, on="paper_uid", how="left").merge(papers, on="paper_uid", how="left")
    field = df.primary_field.fillna("unknown")
    df["field_cat"] = field.where(field.map(field.value_counts()) >= MIN_FIELD, "other")
    year = df.year_openalex.astype("Int64")
    df["year_cat"] = np.where(year <= YEAR_POOL_UPTO, f"<= {YEAR_POOL_UPTO}", year.astype(str))

    out: dict = {"sample": {"computed": int(len(df)), **{f"with_{c}": int(df[c].notna().sum()) for c in TARGETS}},
                 "correlations": {}}
    # 参考文献数どうし、参考文献数と各指標の関係（偏相関で何を除いているかの確認）
    d = df[df.ss_reference_count.notna()]
    out["n_refs"] = {"rho_n_refs_used_vs_ss_reference_count":
                     corr_ranks(d.n_refs_used.to_numpy(), d.ss_reference_count.to_numpy(), []),
                     **{f"rho_n_refs_used_vs_{c}": corr_ranks(df.loc[df[c].notna(), "n_refs_used"].to_numpy(),
                                                              df.loc[df[c].notna(), c].to_numpy(), [])
                        for c in TARGETS}}

    print("── Novel_q と SciSciNet の指標の Spearman ρ（そのまま / 自分の参考文献数を除く / 両方の参考文献数を除く"
          " / 参考文献数・分野・出版年・組織を除く）")
    for target in TARGETS:
        d = df[df[target].notna() & df.ss_reference_count.notna() & df.year_openalex.notna()]
        own = [d.n_refs_used.to_numpy(float)]
        both = own + [d.ss_reference_count.to_numpy(float)]
        dummies = pd.concat([pd.get_dummies(d[c], drop_first=True, dtype=float)
                             for c in ("field_cat", "year_cat", "source_org")], axis=1).to_numpy(float)
        res = {}
        for q in Q_VALUES:
            x, y = d[f"novel_{q}"].to_numpy(float), d[target].to_numpy(float)
            res[f"q{q}"] = {"raw": corr_ranks(x, y, []), "partial_own": corr_ranks(x, y, own),
                            "partial_both": corr_ranks(x, y, both),
                            "partial_all": corr_ranks(x, y, own, dummies)}
            r = res[f"q{q}"]
            print(f"  {target:18s} q={q:3d} N={r['raw']['n']}  "
                  + "  ".join(f"{k} {r[k]['rho']:+.3f} (p={r[k]['p']:.3g})" for k in PARTIALS))
        out["correlations"][target] = res
    print("── 参考文献数との関係")
    for k, v in out["n_refs"].items():
        print(f"  {k:45s} ρ {v['rho']:+.3f}  N={v['n']}")

    plot(df)
    out["params"] = {"method": "Spearman; partial = residuals of ranks on ranks of controls",
                     "controls_own": ["n_refs_used (rank)"],
                     "controls_both": ["n_refs_used (rank)", "ss_reference_count (rank)"],
                     "controls_all": ["n_refs_used (rank)", "field dummies", "year dummies", "org dummies"],
                     "min_field": MIN_FIELD, "year_pool_upto": YEAR_POOL_UPTO,
                     "ci": "Fisher z, 95%", "symlog_linthresh": SYMLOG_LINTHRESH,
                     "sign_note": "ss_novelty: lower = more novel; Novel_q: higher = more novel"}
    out["inputs"] = {rel(p): file_sha(p) for p in (NOVELTY_PARQUET, SCISCINET_CSV, PAPERS_CSV)}
    (OUT_DIR / "novelty_sciscinet_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"保存: {rel(OUT_DIR / 'novelty_sciscinet_comparison.json')}, {rel(FIG_DIR / 'novelty_sciscinet_scatter.png')}")


if __name__ == "__main__":
    main()
