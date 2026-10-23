# shibayama_novelty

今井泰介先生の依頼による研究（トヨタ関連企業の研究活動を Science of Science の方法で分析する）の一部。

> **共有の範囲**: 入力の `data/raw/` には、公式サイトから集めた論文一覧と OpenAlex 等から取った著者・所属を含む。また、配布条件の記載が無い単語ベンチマーク2種（RG65、SimLex-999）を、原典の論文を明記して収録している。リポジトリは private にし、共有は研究の関係者に限る。

## データの出どころ

| ファイル | 出どころ | 取得日・版 |
|---|---|---|
| `data/raw/papers.csv` | トヨタ関連4サイト（トヨタ自動車未来創成センター、豊田中央研究所、AISIN IMRA 日本・米国）の論文一覧を OpenAlex に対応づけたもの | 取得日は未記入（ファイルの作成日は 2026-09-10）<!-- TODO: 取得日を確かめて書く --> |
| `data/raw/authorships.csv` | 上の論文の著者。OpenAlex を基に Crossref・Scopus で所属と ORCID を補ったもの | ファイルの作成日 2026-09-10 <!-- TODO: 取得日 --> |
| `data/raw/paper_references.csv` | 上の論文の参考文献（OpenAlex の `referenced_works` とそのタイトル） | ファイルの作成日 2026-09-10 <!-- TODO: 取得日 --> |
| `data/derived/shibayama_novelty/citations/citation_percentiles.csv` | 被引用が上位1%・上位10% に入るか。OpenAlex Works API の `citation_normalized_percentile`。取ったコードは `src/shibayama_novelty/fetch_citation_percentiles.py`（工程には入れない） | 2026-10-01 取得。工程9はこの値を使う |
| scispaCy `en_core_sci_lg` 0.5.4 | Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and robust models for biomedical natural language processing. *BioNLP 2019*, 319–327. doi:10.18653/v1/W19-5034。配布物 https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz （CC BY-SA 3.0）。約600MB のため**コミットしない**。shibayama_novelty の工程2が初回に落とし、`vocab/` と `meta.json` を `data/cache/scispacy/`（`.gitignore` 済み）に解凍する | 0.5.4（配布元の最終更新 2024-03-06、2026-10-14 確認） |

論文・著者・参考文献の3表は別の作業リポジトリ（収集と OpenAlex への対応づけ）の出力を写したもの。取り直すと値が変わるので、写したものを入力として固定する。手を加えたのは1点だけで、`authorships.csv` の所属文字列（`affiliation`、`affil_openalex`）に含まれていたメールアドレス 268 個を `[email removed]` に置き換えた（個人情報のため。どの工程もこの2列を使わない）。

被引用の百分位の表は、上の3表と違い、このリポジトリで OpenAlex から取ったもの。`fetch_citation_percentiles.py` を `--write` を付けて流すと表を書き換え、OpenAlex の応答を `data/cache/openalex/citation_normalized_percentile/<取得日>/` に保存する（2026-10-01 の応答は保存していない）。OpenAlex の値は日々更新され、今流すと 2026-10-01 とは違う値が返るので、工程には入れず、reproduce の notebook でも作り直さない。

### 単語ペアのベンチマーク（shibayama_novelty の工程5）

6種すべてを `data/raw/word_similarity_benchmarks/<名前>/` に置く。中身は配布元のファイルのままで、ファイル名だけ snake_case に直した。各フォルダの `citation.txt` に、原典の論文、入手元と取得日、元のファイル名、配布条件を書いてある。同梱の説明ファイル（SimLex-999 の `README.txt`、UMNSRS の `readme.txt`）にあった作成者のメールアドレスは `[email removed]` に置き換えた。**shibayama_novelty の工程5の結果を使うときは、下の原典を引用する。**

| 名前 | ペア数 | 原典（引用するもの） | 入手元 | 配布条件（2026-10-14 確認） |
|---|---:|---|---|---|
| RG65 | 65 | Rubenstein, H., & Goodenough, J. B. (1965). Contextual correlates of synonymy. *Communications of the ACM*, 8(10), 627–633. doi:10.1145/365628.365657 | 原論文の Table 1 を転記した github.com/andrefs/punuy-datasets（MIT License）の `profiles/rg65/` | データ自体の条件の記載なし。原著者の論文を明記して収録 |
| WS353 | 353 | Finkelstein, L., et al. (2002). Placing search in context: The concept revisited. *ACM TOIS*, 20(1), 116–131. doi:10.1145/503104.503110 | gabrilovich.com/resources/data/wordsim353/ の `wordsim353.zip` | CC BY 4.0（配布ページに明記） |
| SimLex-999 | 999 | Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating semantic models with (genuine) similarity estimation. *Computational Linguistics*, 41(4), 665–695. doi:10.1162/COLI_a_00237 | fh295.github.io/simlex.html の `SimLex-999.zip` | ライセンスの記載なし（配布元は論文の引用を求めている）。原著者の論文を明記して収録 |
| SimVerb-3500 | 3,500 | Gerz, D., Vulić, I., Hill, F., Reichart, R., & Korhonen, A. (2016). SimVerb-3500: A large-scale evaluation set of verb similarity. *EMNLP 2016*, 2173–2182. doi:10.18653/v1/D16-1235 | University of Cambridge Repository, doi:10.17863/CAM.9391 | CC BY 4.0（リポジトリに明記） |
| UMNSRS-Sim / -Rel | 566 / 587 | Pakhomov, S., et al. (2010). Semantic similarity and relatedness between clinical terms: An experimental study. *AMIA Annu Symp Proc*, 2010, 572–576. データ: hdl.handle.net/11299/196265 | University of Minnesota Digital Conservancy | CC0 1.0（同梱の readme に明記） |

RG65 は転記された値を使っており、原論文の表との照合はしていない。

## データの扱い

- 著者の同定は `orcid`、無ければ `name_key`（姓＋名の頭文字）を使う。**`openalex_author_id` は使わない**（別人が併合されている）。
- 出版年は `year_openalex` を使う。`year_site` は使わない。
- 同じ論文が2サイトに載っている5件は `is_work_primary == True` で除く。
- 参考文献タイトルの欠損（5.3%）は OpenAlex 側で統合・削除された ID で、取り直しても埋まらない。

## 再現の手順

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

各解析の工程の流し方は、その解析の節にある。

## 概要

トヨタ関連企業4組織の論文について、柴山らの方法（Shibayama, Yin & Matsumoto 2021, *PLOS ONE* 16(7):e0254034）で論文の新規性 `Novel_q` を計算し、その測り方が工学中心の標本でも働くかを確かめる。検証は、原論文が行ったもの（共通参照距離との相関、分布、被引用の予測）を同じ手順で再現し、原論文が行っていないもの（単語の段階、分野ラベルとの対応）を先行研究の手順で補う。

### 指標の定義

焦点論文1本につき、次の順に計算する。

1. 参考文献のタイトルを、含まれる語の語ベクトルの重み無し平均で200次元のベクトル `v_i` にする（原論文 Step 1）。
2. 参考文献の全ペアについて `d_ij = 1 − cos(v_i, v_j)` を計算する（式(1)）。
3. ペア距離を昇順に並べ、下から `n_pairs × q / 100` 番目（切り上げ）の値を `Novel_q` とする（式(2)）。q ∈ {100, 99, 95, 90, 80, 50}。q=100 は最大値。

語ベクトルは原論文と同じ scispaCy `en_core_sci_lg` 0.5.4（200次元、600,000ベクトル）を使う。

注意:

- 式(2)の `N` は参考文献の本数ではなく**ペアの数**である。q=100 が最大値になるのはこの読み方のときだけ。
- 分位点は線形補間ではなく nearest-rank（`np.percentile(..., method="inverted_cdf")`）で取る。
- 同じ論文の中で同じタイトルの参考文献（プレプリントと出版版など）は距離が 0 のペアを作るので、正規化タイトルで重複を除く。

### 工程

上から順に流す。各工程の入出力はスクリプト先頭の docstring に書いてある。

| # | スクリプト | 内容 | 出力（`data/derived/`） |
|---:|---|---|---|
| 1 | `src/shibayama_novelty/clean_reference_titles.py` | 焦点論文の絞り込み、参考文献のタイトル欠損・重複の除去 | `input/` |
| 2 | `src/shibayama_novelty/embed_titles_scispacy.py` | 参考文献と焦点論文のタイトルをベクトル化（原論文 Step 1）。語ベクトル表を読む | `title_vectors/` |
| 3 | `src/shibayama_novelty/compute_shibayama_novelty.py` | ペア距離と `Novel_q`（式(1)(2)）、分野 × 出版年帯の中でのパーセンタイル | `scores/` |
| 4 | `src/shibayama_novelty/build_title_distance_pairs.py` | 検証用のペア距離（参考文献ペア、焦点論文ペア、共通参照距離、同一著者）。工程3の `Novel_q` を計算し直して一致を確かめる | `pairs/` |
| 5 | `src/shibayama_novelty/validate_word_vectors.py` | 検証【1】単語ベクトルの cos 類似度と、単語ペアのベンチマーク6種の人の評定の Spearman ρ。語ベクトル表を読む | `validation/` |
| 6 | `src/shibayama_novelty/validate_distance_shared_refs.py` | 検証【2】(a) タイトル距離と共通参照距離の相関（原論文 Table 3 の再現） | `validation/` |
| 7 | `src/shibayama_novelty/validate_distance_field_labels.py` | 検証【2】(b) タイトル距離と OpenAlex の分野ラベルの対応、【3】距離の遠い側でも対応するか | `validation/` |
| 8 | `src/shibayama_novelty/plot_novelty_distributions.py` | 検証【4】距離と `Novel_q` の分布を原論文 Fig 2 と比べる | `validation/` |
| 9 | `src/shibayama_novelty/validate_novelty_citations.py` | 検証【5】`Novel_q` で被引用の上位10%入りを予測する（原論文の被引用予測の再現）。2016年以降の article に限る | `validation/` |
| 10 | `src/shibayama_novelty/plot_novelty_citations.py` | 被引用の回帰の改良に向けて、出版年・文献種別で絞らずに `Novel_100` と被引用数の関係を図にする（独自） | `citations/` |
| 11 | `src/shibayama_novelty/regress_citations_on_novelty.py` | log(1+被引用数) を `Novel_q` で OLS 回帰し、出版年・組織・log(参考文献数)・文献種別を統制した6つのモデルを当てる（HC3 の標準誤差。統制変数の選び方は独自） | `citations/` |
| 12 | `src/shibayama_novelty/plot_citation_distribution.py` | 工程11 の目的変数（被引用数）の分布を数値と図で示し、そのままと log(1+被引用数) で回帰の残差を比べる（独自） | `citations/` |

主な成果物は `data/derived/shibayama_novelty/scores/novelty_shibayama.parquet`（782行、論文1本で1行）。列の説明は `data/README.md`。

### 再現

```sh
for s in clean_reference_titles embed_titles_scispacy compute_shibayama_novelty build_title_distance_pairs validate_word_vectors validate_distance_shared_refs validate_distance_field_labels plot_novelty_distributions validate_novelty_citations plot_novelty_citations regress_citations_on_novelty plot_citation_distribution; do
  .venv/bin/python src/shibayama_novelty/$s.py
done
```

環境変数は要らない。初回だけ、工程2が scispaCy の配布物（約530MB）を落として `data/cache/scispacy/` に解凍する（数分）。2回目からは全工程で1分ほど。解凍したものが正しいモデルかは、`meta.json` の名前・版・表の大きさ（工程2と5が確かめる）と、`notebooks/reproduce/reproduce_shibayama_novelty.ipynb` での出力の一致で確かめる。

`notebooks/reproduce/reproduce_shibayama_novelty.ipynb` は同じ工程を流し、`data/derived/` の全ファイルがコミットされたものとバイト単位で一致するかを確かめる。

### 結果

#### 計算できた論文

| 段階 | 件数 |
|---|---:|
| `papers.csv` 全行 | 917 |
| 2サイト重複を除く | 912 |
| OpenAlex に対応づいた | 852 |
| 参考文献が1本以上ある | 790 |
| 参考文献（タイトルあり・重複除去後）が2本以上 = **計算できた** | **782** |

参考文献は 27,247 行のうち 25,587 行を使った（タイトル欠損 1,440 行、同一論文内のタイトル重複 212 行を除いた）。参考文献の数は平均 32.7 本、中央値 29 本で、原論文（平均32本）とほぼ同じ。ペアは計 705,512 組。

| q | 平均 | 標準偏差 | 中央値 |
|---:|---:|---:|---:|
| 100 | 0.841 | 0.198 | 0.894 |
| 95 | 0.717 | 0.156 | 0.751 |
| 50 | 0.447 | 0.118 | 0.465 |

#### 検証

| 検証 | 結果 | 原論文 |
|---|---|---|
| 【1】単語ペアの cos 類似度と人の評定の Spearman ρ（RG65 / WS353 / SimLex-999 / SimVerb-3500 / UMNSRS-Sim / UMNSRS-Rel） | +0.502 / +0.493 / +0.285 / +0.151 / +0.511 / +0.463。すべて p < .001。一般語4種の平均 +0.358 | （原論文は行っていない） |
| 【2】(a) 同じ著者の論文ペアで、タイトル距離と共通参照距離の Pearson r | **+0.100**（11,230組、著者744名単位のブートストラップ 95%区間 [+0.085, +0.231]）。共通の参考文献が1本以上あるペアに限ると +0.227 | +0.231 |
| 【2】(a) タイトルの語数帯別（1–10 / 11–13 / 14–17 / 18–語） | +0.103 / +0.107 / +0.113 / +0.387 | +0.440 / +0.246 / +0.197 / +0.259 |
| 【2】(b) field（9種、739本）の kNN 誤分類率 | 0.440 ± 0.005（多数派を答え続けた場合 0.591） | — |
| 【2】(b) field のペア AP | 0.360（基準 0.245） | — |
| 【3】距離が上位 10% のペアの中での、同じ domain かのペア AP | 0.634（基準 0.580）。上位1%でも 0.445（基準 0.404） | — |
| 【4】参考文献ペアの距離 / `Novel_100` / `Novel_50` の分布 | 平均 0.497 / 0.841 / 0.447。`Novel_100` は左に裾を引く | 山 約0.45 / 約0.85（左に裾）/ 約0.45 |
| 【5】被引用の上位10%入りを `Novel_q` で予測するロジスティック回帰（2016年以降の article 473本、該当93本） | q=100 でオッズ比 35.56 [3.20, 394.69]、p = .004。q=99〜50 は 1.69〜5.28 で有意でない | 上位1%を従属変数に、6本とも正。q=100 で 154.02 |

- タイトル距離は共通参照距離と正の相関を持つが、全体の r は原論文より弱い。同じ著者のペアでも 79.7% は共通の参考文献が0本で、相関が薄まる。
- 原論文は「短いタイトルほど相関が高い」だったが、本標本は長いタイトルの帯（18語以上、90組）で最も高い。
- 距離は分野ラベルと対応し、距離が遠い側に絞っても基準を上回る。
- 分布の形は原論文と似ている。
- 【5】は6本ともオッズ比が1を上回り、向きは原論文と同じ。有意なのは q=100 だけで、原論文の「q が下がるほどオッズ比が下がる」形は再現していない。原論文は上位1%を多めに抽出した標本なので、比べられるのは符号と有意性だけ。
- 【5】で2016年以降に絞るのは、古い論文ほど `Novel_100` が小さく被引用が多いため（2015年以前の156本は `Novel_100` 平均 0.700、上位10%率 46.8%、被引用の中央値 38。2016年以降の473本は 0.929、19.7%、10）。そのまま結ぶと novelty と関係なく負の関連が出る。上位1%は2016年以降の article で11本しかなく判定できない（`novelty_citation_validation.json` に参考として残す）。

#### 読むときの注意

- **`Novel_q` は参考文献の数と相関する**（`n_refs_used` との Spearman ρ: q=100 で +0.667、q=50 でも +0.427）。【5】の標本でも参考文献数は `Novel_100` と ρ = +0.408、上位10%入りと ρ = +0.212 の関係があり、【5】の結果の一部は参考文献数で説明できる可能性がある。原論文の式に合わせて統制していない。分析では参考文献数を必ず統制する。参考文献が100本を超える12本（`is_high_refs`）は総説の疑いがある。
- 分野をまたいで比べるときは、絶対値ではなく `novelty_shibayama_ranked.parquet` の分野 × 出版年帯の中でのパーセンタイルを使う。
- 語ベクトル表は生物医学の論文で学習されている。工学中心の本標本での妥当性は上の検証の範囲でしか言えない。
- 原論文が行った「自己申告の新規性との相関」は、調査データが無いため再現できない。
- 使うのはタイトルだけ。抄録版は作っていない。

### 文献

スクリプトの docstring で略号で引いているもの。

| 略号 | 文献 | 借りたもの |
|---|---|---|
| Shibayama21 | Shibayama, Yin & Matsumoto (2021) *PLOS ONE* 16(7):e0254034 + S1 Appendix | 指標の定義（式(1)(2)）、検証【5】の回帰の形、【2】(a) の手順（p.6–7、式(3)、Table 3）、【4】の図の形式（Fig 2、S1.1） |
| Bommasani20 | Bommasani, Davis & Cardie (2020) ACL | 【1】の一般語ベンチマーク4種と評価法（§4.1）。ベンチマーク自体の原典は「単語ペアのベンチマーク」の表 |
| Chiu16 | Chiu, Crichton, Korhonen & Pyysalo (2016) *Proc. BioNLP Workshop* | 【1】の生物医学ベンチマーク（UMNSRS）と、表に無い語の除外 |
| Kusner15 | Kusner, Sun, Kolkin & Weinberger (2015) *Proc. ICML* | 【2】(b) の kNN 分類誤差（§5.1） |
| MTEB | Muennighoff, Tazi, Magne & Reimers (2023) EACL | 【2】(b) の k-means と V-measure、ペア AP（§3.2） |
| SPECTER | Cohan, Feldman, Beltagy, Downey & Weld (2020) ACL | 【2】(b) の t-SNE、DBSCAN、homogeneity / completeness（§6） |
| OpenAlex | OpenAlex Help Center "Topics" | 【2】(b)【3】で正解に使うラベルの性質 |

【3】（距離の遠い側に絞った検証）は先行例が見つかっておらず、本研究独自の適用である。
