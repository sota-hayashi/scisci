# embedding_comparison

柴山らの Novelty（shibayama_novelty の `Novel_q`）の語ベクトルを scispaCy から変えたとき、結果がどう変わるかを見る。2026-10-02 の打ち合わせで出た「工学系に向いた語ベクトルを使うと結果が変わるか」に答えるため。今の辞書（scispaCy）を主として残し、別の辞書は比較に使う。

## 比べる版

| 版 | 語ベクトル | 単語の切り分け |
|---|---|---|
| original | scispaCy `en_core_sci_lg` 0.5.4（200次元、60万語） | ハイフン・アポストロフィでつないだ語は1語（shibayama_novelty の出力をそのまま使う） |
| scispacy_split | 同上 | ハイフン・アポストロフィで分ける（mat2vec に合わせた比較用） |
| mat2vec_split | mat2vec（Tshitoyan ら 2019。材料科学の論文の要旨で学習した word2vec、529,686語・200次元） | ハイフン・アポストロフィで分ける。数字だけの語は `<nUm>` |

化学式の変換（mat2vec の LiCoO2 → CoLiO2）はしない。化学式は参考文献のタイトルの語の出現の1.4%（2026-11-05 に確かめた）。

## データの出どころ

| もの | 出どころ | 取得日・版 |
|---|---|---|
| mat2vec | https://storage.googleapis.com/mat2vec/ の `pretrained_embeddings`（語彙、gensim 3.x の pickle）と `pretrained_embeddings.wv.vectors.npy`（ベクトル）。配布元 github.com/materialsintelligence/mat2vec（MIT License）。Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G., & Jain, A. (2019). Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571, 95–98. doi:10.1038/s41586-019-1335-8。**使った結果を報告するときは、この論文を引用する**。約470MB のためコミットしない（`data/cache/` は `.gitignore` 済み） | 2026-11-05 取得（配布元の更新 2019-06-22）。SHA-256 は `data/cache/mat2vec/info.json` |

## 工程

上から順に流す。各工程の入出力はスクリプト先頭の docstring に書いてある。入力のタイトルは shibayama_novelty の工程1 の出力、scispaCy の語ベクトル表は shibayama_novelty の工程2 が作るものを使う。

| # | スクリプト | 内容 | 出力（`data/derived/embedding_comparison/`） |
|---:|---|---|---|
| 1 | `src/embedding_comparison/embed_titles.py` | 参考文献と焦点論文のタイトルを、mat2vec_split と scispacy_split でベクトル化する（重み無し平均。shibayama_novelty の工程2 と同じ） | `title_vectors/<版>/` |
| 2 | `src/embedding_comparison/compute_novelty.py` | 2つの版で `Novel_q` を計算する（shibayama_novelty の工程3 の関数をそのまま使う） | `scores/` |
| 3 | `src/embedding_comparison/compare_embeddings.py` | 3つの版の `Novel_q` を、版どうしの順位相関、参考文献数との相関、SciSciNet の Uzzi らの新規性との偏相関で比べる。比べ方は独自 | `comparison/` |

工程に入れないスクリプト:

- `src/embedding_comparison/fetch_mat2vec.py` — mat2vec を `data/cache/mat2vec/` に落とし、語彙を `vocab.txt` に取り出す（ベクトルは配布された .npy をそのまま使う）。配布物の pickle は、gensim を入れずに、呼び出してよいモジュールを決めた一覧だけを許す読み方で読む（pickle は読み込むと任意のコードを実行できるため）。

## 再現

```sh
.venv/bin/python src/embedding_comparison/fetch_mat2vec.py      # 初回だけ（約470MB を落とす）
for s in embed_titles compute_novelty compare_embeddings; do
  .venv/bin/python src/embedding_comparison/$s.py
done
```

`notebooks/reproduce/reproduce_embedding_comparison.ipynb` は工程1〜3 を流し、`data/derived/embedding_comparison/` の全ファイルがコミットされたものとバイト単位で一致するかを確かめる。

## 結果

| | original | scispacy_split | mat2vec_split |
|---|---:|---:|---:|
| 参考文献のタイトルの語彙外率（平均） | 2.1% | 0.9% | 2.1% |
| `Novel_100` の平均 / `Novel_50` の平均 | 0.841 / 0.447 | 0.845 / 0.452 | 0.460 / 0.189 |

辞書によって距離の大きさの目安が違うので、以下は順位で比べる。

版どうしの Spearman ρ（782本）:

| | q=100 | q=50 |
|---|---:|---:|
| original × scispacy_split（切り分け方の違い） | +0.967 | +0.984 |
| original × mat2vec_split（辞書の違い） | +0.758 | +0.558 |

参考文献数（`n_refs_used`）との Spearman ρ: q=100 で original +0.667、mat2vec_split +0.641。q=50 で original +0.427、mat2vec_split +0.152。

SciSciNet の Uzzi らの新規性との偏相関（580本。負ほど一致。参考文献数を除く / さらに分野・出版年・組織を除く）:

| | original | scispacy_split | mat2vec_split |
|---|---|---|---|
| 新規性 q=100 | −0.394 / −0.268 | −0.393 / −0.277 | −0.309 / −0.269 |
| 新規性 q=50 | −0.565 / −0.453 | −0.561 / −0.453 | −0.409 / −0.394 |
| 定石らしさ q=100 | −0.422 / −0.300 | −0.398 / −0.283 | −0.279 / −0.215 |
| 定石らしさ q=50 | −0.589 / −0.485 | −0.579 / −0.476 | −0.445 / −0.432 |

- 単語の切り分け方を変えても、`Novel_q` の順位はほぼ変わらない（ρ 0.97〜0.98）。
- 辞書を mat2vec に変えると、順位はかなり変わる（q=100 で ρ 0.76、q=50 で 0.56）。
- Uzzi らの新規性との一致は、mat2vec の方が強くはならない。条件をそろえた偏相関で、q=100 はほぼ同じ（−0.269 と −0.268）、q=50 は scispaCy の方が強い（−0.453 と −0.394）。この基準では、工学系に近い辞書に変える利点は見えない。
- mat2vec の `Novel_50` は参考文献数との相関が小さい（+0.152）。
- 確かめていないこと: 単語ベンチマーク・分野ラベルとの対応（shibayama_novelty の工程5・7 に当たる確認）を mat2vec ではまだしていない。

## 文献

| 略号 | 文献 | 借りたもの |
|---|---|---|
| Tshitoyan19 | Tshitoyan ら (2019) *Nature* 571:95 | mat2vec の語ベクトル |
| Shibayama21 | Shibayama, Yin & Matsumoto (2021) *PLOS ONE* 16(7):e0254034 | 指標の定義（shibayama_novelty と同じ） |
