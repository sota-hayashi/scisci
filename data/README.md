# データ

| フォルダ | 中身 | 作る工程 |
|---|---|---|
| `raw/` | 論文・著者・参考文献の3表。中身を手で編集しない | —（別の作業リポジトリの出力を写したもの） |
| `raw/word_similarity_benchmarks/` | 単語ペアのベンチマーク6種（RG65、WS353、SimLex-999、SimVerb-3500、UMNSRS 2種）。配布元のファイルのまま。各フォルダの `citation.txt` に原典・入手元・配布条件 | —（入手元は `src/shibayama_novelty/README.md` の「データの出どころ」） |
| `cache/scispacy/` | scispaCy の語ベクトル表（`vocab/` と `meta.json`）。約600MB のためコミットしない（`.gitignore` 済み） | shibayama_novelty の工程2 `embed_titles_scispacy.py` が初回に配布元から落とす |
| `derived/shibayama_novelty/input/` | 焦点論文と、整形した参考文献 | shibayama_novelty の工程1 `clean_reference_titles.py` |
| `derived/shibayama_novelty/title_vectors/` | タイトルのベクトル | shibayama_novelty の工程2 `embed_titles_scispacy.py` |
| `derived/shibayama_novelty/scores/` | `Novel_q`（主な成果物） | shibayama_novelty の工程3 `compute_shibayama_novelty.py` |
| `derived/shibayama_novelty/pairs/` | 検証用のペア距離 | shibayama_novelty の工程4 `build_title_distance_pairs.py` |
| `derived/shibayama_novelty/validation/` | 検証の結果と図 | shibayama_novelty の工程5 `validate_word_vectors.py`、6 `validate_distance_shared_refs.py`、7 `validate_distance_field_labels.py`、8 `plot_novelty_distributions.py`、9 `validate_novelty_citations.py` |
| `derived/shibayama_novelty/citations/` | 被引用の百分位（`citation_percentiles.csv`、2026-10-01 取得、工程9が読む）、`Novel_100` と被引用数の関係の要約（`novelty_citations_summary.json`）と図（`figs/`）、log(1+被引用数) の回帰（`novelty_citation_regression.json`）、被引用数の分布（`novelty_citation_distribution.json`） | shibayama_novelty の工程10 `plot_novelty_citations.py`、11 `regress_citations_on_novelty.py`、12 `plot_citation_distribution.py`（百分位の表は `fetch_citation_percentiles.py`。工程に入れない） |

各工程の JSON（`*_summary.json`、`*_config.json`、`validation/*.json`）には、件数、定数（`params`）、入力ファイルのハッシュ（`inputs`）が入っている。

---

# raw/

## word_similarity_benchmarks/（shibayama_novelty の工程5の入力）

単語ペアのベンチマーク。1行 = 単語ペア1組。配布元のファイルのままなので、列名と形式はデータごとに違う。工程5が使う列は次のとおり（語は工程5で小文字にする）。入手元と配布条件は `src/shibayama_novelty/README.md` の「単語ペアのベンチマーク」。

| ファイル | 行数 | 形式 | 1語目 | 2語目 | 人の評定（平均） |
|---|---:|---|---|---|---|
| `rg65/rg65_table1.json` | 65 | JSON、`{term1, term2, value}` の配列 | `term1` | `term2` | `value`（0〜4） |
| `ws353/combined.csv` | 353 | CSV、見出しあり | `Word 1` | `Word 2` | `Human (mean)`（0〜10） |
| `simlex999/simlex_999.txt` | 999 | タブ区切り、見出しあり。ほかに品詞・具体性・連想強度などの列がある | `word1` | `word2` | `SimLex999`（0〜10） |
| `simverb3500/simverb_3500.txt` | 3,500 | タブ区切り、見出しなし。列は word1, word2, POS, score, relation | 1列目 | 2列目 | 4列目（0〜10） |
| `umnsrs/umnsrs_similarity.csv` | 566 | CSV、見出しあり | `Term1` | `Term2` | `Mean`（類似度） |
| `umnsrs/umnsrs_relatedness.csv` | 587 | CSV、見出しあり | `Term1` | `Term2` | `Mean`（関連度） |

そのほかのファイル:

- `*/citation.txt`: 原典の論文（使うときに引用するもの）、入手元と取得日、元のファイル名、配布条件。このリポジトリで書いたもの
- `rg65/dataset.json`: 転記元（punuy-datasets）のメタ情報。評定者数や原論文の教示文を含む
- `ws353/instructions.txt`、`simlex999/README.txt`、`simverb3500/README.txt`、`umnsrs/readme.txt`: 配布物に付いていた説明


## paper_references.csv（27,247行 × 8列、790論文分）

1行 = 参考文献1件。

| カラム | 意味 |
|---|---|
| `paper_uid` | 引用元の論文。`papers.csv` と結合できる |
| `paper_title` | 引用元のタイトル（OpenAlex 側） |
| `paper_title_site` | 引用元のタイトル（サイト掲載のもの） |
| `ref_seq` | 参考文献の順番 |
| `ref_work_id` | 引用先の OpenAlex Work ID。NULL なし |
| `ref_doi` | 引用先の DOI |
| `ref_year` | 引用先の出版年 |
| `ref_title` | 引用先のタイトル。94.7% が埋まる。残りは OpenAlex 側で統合・削除された ID |

## papers.csv（917行 × 31列）

| カラム | 非NULL | ユニーク | 意味 |
|---|---:|---:|---|
| `paper_uid` | 100% | 917 | 主キー |
| `source_org` | 100% | 4 | 収集元サイト |
| `title_site` | 100% | 916 | サイト掲載のタイトル |
| `authors_site` | 66% | 571 | サイト掲載の著者名。imra.com は常に空 |
| `venue_site` | 100% | 721 | サイト掲載の掲載誌・巻号 |
| `year_site` | 70% | 23 | サイト掲載の出版年。**集計に使わない。** `year_openalex` を使う |
| `link` | 98% | 892 | サイトのリンク先 |
| `doi` / `doi_norm` | 70% | 641 | `doi_norm` が正規化済み。結合はこちら |
| `toyota_authors_site` | 17% | 106 | 未来創成センターのみ。サイトが太字で示した自社所属著者。所属判定の正解ラベル |
| `work_id` | 93% | 852 | OpenAlex Work ID。NULL は対応づけ失敗（60件） |
| `match_method` | 94% | 2 | `doi_exact` / `title_fuzzy` |
| `match_score` | 94% | 21 | タイトル類似度。`doi_exact` は 1.0 |
| `review_status` | 94% | 4 | `auto` / `review` / `accepted_by_review` / `rejected_by_review` |
| `review_decision` / `review_reason` | 1% | 2 | 目視判断10件のみ |
| `matched` | 100% | 2 | 対応づいたか |
| `is_work_primary` | 100% | 2 | **Work単位の集計はこれで絞る**（重複掲載5件の対策） |
| `matched_title` | 93% | 852 | OpenAlex 側のタイトル。目視確認用 |
| `year_openalex` | 93% | 38 | **出版年はこちら** |
| `type` | 93% | 5 | `article` / `conference-paper` / `book-chapter` / `preprint` / `editorial` |
| `cited_by_count` | 93% | 137 | 被引用数 |
| `referenced_works_count` | 93% | 102 | 参考文献数。0 の62件は参考文献系の指標を計算できない |
| `n_authors` | 93% | 19 | 著者数 |
| `institutions_distinct_count` | 93% | 16 | 関与機関数 |
| `primary_topic` | 93% | 280 | OpenAlex の分野分類（第4階層） |
| `primary_subfield` | 93% | 80 | 同（第3階層） |
| `primary_field` | 93% | 21 | 同（第2階層） |
| `primary_domain` | 93% | 4 | 同（第1階層） |
| `venue_openalex` | 80% | 307 | OpenAlex 側の掲載誌名 |
| `openalex_doi` | 93% | 848 | OpenAlex 側の DOI。`doi_norm` と食い違えば対応づけを疑う |

## authorships.csv（4,448行 × 24列、857論文分）

| カラム | 非NULL | ユニーク | 意味 |
|---|---:|---:|---|
| `paper_uid` | 100% | 857 | 主キー1。`papers.csv` への参照 |
| `author_seq` | 100% | 25 | 主キー2。著者順（1始まり） |
| `work_id` | 100% | 852 | `papers.csv` からの冗長列。JOIN を省くため |
| `doi` | 72% | 640 | 同上 |
| `source_org` | 100% | 4 | 同上 |
| `spine` | 100% | 1 | 著者リストの取得元。全行 `openalex` |
| `name_raw` | 100% | 1,997 | 著者名の生文字列 |
| `given` / `family` | 100% | 1,189 / 1,258 | 名 / 姓 |
| `name_key` | 100% | 1,639 | 名寄せキー（`姓\|名の頭文字`） |
| `orcid` | 21% | 501 | ORCID。**著者同定はこれが最優先** |
| `orcid_openalex_raw` / `orcid_crossref` | 21% / 17% | 501 / 378 | ORCID の出所別。OpenAlex は論文の登録データ由来のみ採用 |
| `affiliation` | 99% | 1,536 | 所属の代表値。含まれていたメールアドレスは `[email removed]` に置き換えた |
| `affil_openalex` | 98% | 1,529 | OpenAlex 由来。**最も網羅的だが表記がバラバラ**（豊田中研を含む文字列だけで311通り）。含まれていたメールアドレスは `[email removed]` に置き換えた |
| `affil_crossref` | 25% | 369 | Crossref 由来 |
| `affil_scopus` | 55% | 247 | Scopus 由来。**表記が正規化済み。組織単位の集計はこちら** |
| `inst_openalex` | 94% | 410 | OpenAlex が付けた機関名。**所属判定に使わない**（スイス法人への誤帰属を含む） |
| `openalex_author_id` | 98% | 1,910 | 保持のみ。**名寄せに使わない**（別人が併合されている） |
| `scopus_author_id` | 56% | 1,128 | 保持のみ |
| `is_corresponding` | 100% | 2 | 責任著者か |
| `is_toyota` | 100% | 2 | 所属がトヨタ系か。豊田工業大学・豊田高専・刈谷豊田総合病院は除外済み |
| `matched_crossref` / `matched_scopus` | 100% | 2 | 各DBと突き合わせできたか。DOI がなければ常に False |

---

---

# derived/shibayama_novelty/

## input/focal_papers.parquet（782行）

計算できた焦点論文。列は `papers.csv` の同名の列を写したもの（`paper_uid`, `work_id`, `source_org`, `year_openalex`, `type`, `primary_field`, `primary_subfield`, `primary_domain`, `referenced_works_count`, `cited_by_count`, `n_authors`, `doi_norm`, `matched_title`）。

## input/reference_titles.parquet（25,587行）

タイトル欠損と、同じ論文の中のタイトル重複を除いた参考文献。

| カラム | 意味 |
|---|---|
| `paper_uid` / `ref_seq` / `ref_work_id` | `paper_references.csv` と同じ |
| `title_clean` | NFKC 正規化・HTML タグ除去・空白整理をしたタイトル。ベクトル化の入力 |
| `title_key` | `title_clean` を小文字にしたもの。重複除去と突合のキー |
| `is_nonascii` | 非 ASCII 文字が 20% を超えるか（日本語・中国語のタイトル） |

## title_vectors/reference_title_index.parquet（20,317行）と reference_title_vectors.npy（20,317 × 200, float32）

unique な参考文献タイトルと、その行番号。`row` が `reference_title_vectors.npy` の行。

| カラム | 意味 |
|---|---|
| `title_key` / `title_clean` | 上と同じ |
| `row` | `reference_title_vectors.npy` の行番号 |
| `oov_rate` | 語ベクトル表に無い語の割合 |
| `is_zero_norm` | 語ベクトル表にある語が1つも無く、零ベクトルになったか（10件）。距離を定義できないので除く |

## title_vectors/focal_title_index.parquet（782行）と focal_title_vectors.npy（782 × 200, float32）

焦点論文のタイトル（OpenAlex 側の `matched_title`）。`paper_uid` 順。列は上に加えて `paper_uid`、`n_words`（空白区切りの語数）。

## scores/novelty_shibayama.parquet（782行）— 主な成果物

| カラム | 意味 |
|---|---|
| `paper_uid` | 主キー。`papers.csv` と結合できる |
| `text_source` | `title`（タイトルだけを使う） |
| `embed_model` | `en_core_sci_lg-0.5.4` |
| `n_refs_raw` | 参考文献の行数（= `referenced_works_count`） |
| `n_refs_titled` | タイトルのある参考文献の数（重複除去後） |
| `n_refs_used` | さらに零ベクトルを除いた、計算に使った数 |
| `title_coverage` | `n_refs_titled / n_refs_raw` |
| `n_dup_title_removed` | `n_refs_titled − n_refs_used` |
| `n_pairs` | `n_refs_used × (n_refs_used − 1) / 2` |
| `novel_100` … `novel_50` | 式(2)の `Novel_q`。q = 100, 99, 95, 90, 80, 50 |
| `dist_mean` / `dist_sd` / `dist_min` | ペア距離の平均・標準偏差・最小。`novel_100` だけが突出して `dist_mean` が平凡な論文は、外れたタイトルを疑う |
| `mean_oov_rate` | 参考文献タイトルの `oov_rate` の平均 |
| `is_few_refs` | `n_refs_used < 10`（111本） |
| `is_high_refs` | `n_refs_used > 100`（12本）。総説の疑い |
| `has_nonascii_ref` | 非英語のタイトルを含むか（4本） |

## scores/novelty_shibayama_ranked.parquet（782行）

分野をまたいで比べるための、`primary_field` × 出版年帯の中でのパーセンタイル（0〜1）。

| カラム | 意味 |
|---|---|
| `paper_uid` / `primary_field` / `source_org` | 上と同じ |
| `year_band` | `year_openalex` の帯（〜2010, 2011–18, 2019–22, 2023–） |
| `rank_cell_n` | 同じ分野 × 年帯の論文数 |
| `novel_100_pct` … `novel_50_pct` | セルの中での順位。`rank_cell_n < 10` のセルは NaN（103本） |

## pairs/reference_title_pairs.parquet（705,954行）

参考文献の全ペア。

| カラム | 意味 |
|---|---|
| `paper_uid` | 焦点論文 |
| `row_i` / `row_j` | `reference_title_vectors.npy` の行番号 |
| `d` | 式(1)の距離。零ベクトルを含むペアは NaN（442行） |

## pairs/focal_paper_pairs.parquet（305,371行）

焦点論文の全ペア（`uid_a < uid_b`）。

| カラム | 意味 |
|---|---|
| `uid_a` / `uid_b` | 焦点論文の `paper_uid` |
| `d` | タイトルの距離（式(1)） |
| `n_refs_a` / `n_refs_b` | 各論文の参考文献の数（`ref_work_id` の種類数。タイトル欠損も含む） |
| `n_coref` | 共通して引用している参考文献の数 |
| `d_cocite` | 共通参照距離（式(3)） |
| `is_same_author` | 著者を1人以上共有するか（ORCID、無ければ `name_key` で同定） |
| `n_words_a` / `n_words_b` | 各論文のタイトルの語数 |

## pairs/same_author_paper_pairs.parquet（16,118行）

`is_same_author` のペア × 共有している著者。`aid` は ORCID、無ければ `KEY:` + `name_key`。

## validation/

| ファイル | 中身 |
|---|---|
| `word_vector_validation.json` | 工程5。ベンチマークごとの使ったペア数、除いたペア数、Spearman ρ と p 値 |
| `distance_shared_refs_validation.json`, `figures/distance_shared_refs_correlation.png` | 工程6。r、語数帯別の r、著者単位のブートストラップ区間、著者をまたいだ対照、原論文 Table 3 の値 |
| `distance_field_label_validation.json`, `tsne_title_coords.parquet`, `figures/tsne_by_field.png`, `figures/field_disagreement_by_distance_decile.png` | 工程7。kNN・k-means・ペア AP・t-SNE+DBSCAN、距離の上位帯ごとの AP、十分位ごとのラベル不一致率 |
| `novelty_distribution_summary.json`, `figures/reference_pair_distance_{all,min10}.png`, `figures/novel_q_distribution_{all,min10}.png`, `figures/n_refs_used_distribution.png` | 工程8。距離と `Novel_q` の分布の要約値と図（標本 all / min10） |
| `novelty_citation_validation.json` | 工程9。2015年以前と2016年以降の比較、q ごとのロジスティック回帰（上位10%、参考に上位1%）のオッズ比・95%区間・p 値、参考文献数との ρ、原論文のオッズ比 |

`tsne_title_coords.parquet` の列: `paper_uid`, `perplexity`（30 が本体、10 と 50 は付録）, `x`, `y`, `dbscan_label`（-1 はノイズ）。

## citations/citation_percentiles.csv（917行）

`papers.csv` と同じ行・同じ並び。OpenAlex Works API の `citation_normalized_percentile` 由来（2026-10-01 取得）。shibayama_novelty の工程9はこの値を使う。
同API の `fwci`（同条件の期待被引用数に対する比。1.0 が世界平均）を百分位で言い換えたもの。
NULL は71件で、内訳は `work_id` なし60件と、OpenAlex 側に値がない11件。

| カラム | 非NULL | ユニーク | 意味 |
|---|---:|---:|---|
| `paper_uid` | 100% | 917 | 主キー。`papers.csv` への参照 |
| `work_id` | 93% | 852 | `papers.csv` の同名の列（OpenAlex の work） |
| `is_in_top_1_percent` | 92% | 2 | 被引用が上位1%に入るか。**同じ出版年・同じ `primary_subfield`・同じ `type` の中での順位**。True は30件 |
| `is_in_top_10_percent` | 92% | 2 | 同、上位10%。True は211件 |
