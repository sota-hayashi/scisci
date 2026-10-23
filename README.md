# scisci

今井泰介先生の依頼による研究（トヨタ関連企業の研究活動を Science of Science の方法で分析する）のうち、林奏汰の担当分。伊倉涼介さんが構築した論文データを入力に、novelty などの指標を計算・検証する。


## 解析

| 解析名 | 内容 | 説明 | 再現の確認 |
|---|---|---|---|
| shibayama_novelty | 柴山ら（2021）の方法で論文の新規性 `Novel_q` を計算し、工学中心の標本で働くかを検証する | [src/shibayama_novelty/README.md](src/shibayama_novelty/README.md) | [notebooks/reproduce/reproduce_shibayama_novelty.ipynb](notebooks/reproduce/reproduce_shibayama_novelty.ipynb) |

## 構成

```
README.md                このファイル
CONTRIBUTING.md          共通の決まり
AGENTS.md, CLAUDE.md     このリポジトリ固有の決まり（AI エージェントも読む）
requirements.txt, .env.example, .gitignore
src/<解析名>/             工程のスクリプトと、その解析の README.md
notebooks/reproduce/     各解析の全工程を流し直し、出力の一致を確かめる notebook
data/                    下の表
docs/                    報告書・メモ（手元のみ。コミットしない）
```

## データ

列の説明は [data/README.md](data/README.md)。

| フォルダ | 中身 | 手で編集してよいか |
|---|---|---|
| [data/raw/](data/raw/) | 論文・著者・参考文献の3表（伊倉さんの出力を写したもの）と、単語ペアのベンチマーク6種。出どころと取得日は [shibayama_novelty の「データの出どころ」](src/shibayama_novelty/README.md#データの出どころ) | しない |
| [data/cache/](data/cache/) | scispaCy の語ベクトル表（約600MB、コミットしない。shibayama_novelty の工程2が初回に落とす） | しない |
| [data/review/](data/review/) | OpenAlex から機関名で引いたトヨタ自動車の著者リスト。確認用で、母集団には使わない（[data/review/README.md](data/review/README.md)） | する |
| [data/derived/](data/derived/) | 各解析の工程の出力。`data/derived/<解析名>/<グループ>/` | しない（スクリプトを直して流し直す） |

## 環境

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

各解析の工程の流し方は、その解析の README にある。

