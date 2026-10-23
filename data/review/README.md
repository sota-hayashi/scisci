# data/review

OpenAlex から機関名で引いたトヨタ自動車の著者リスト。確認用に残すもので、**分析の母集団には使わない**（理由: `docs/memos/2026-08-25-/analysis_pipeline.md` §0。Scopus / WoS で照合した裏付率が対照群でも 63%）。

| ファイル | 中身 | 行数 | 作成日 |
|---|---|---:|---|
| `openalex_toyota_motor_authors_all.csv` | Toyota Motor Corporation（OpenAlex `I4210137853`）の lineage 配下に一度でも所属した著者すべて。組織別・現所属のフラグ、topics 上位5件、業績の最初と最後の年 | 9,483 | 2026-07-24 |
| `openalex_toyota_motor_authors_all_with_medical_flags.csv` | 上と同じ全件に、医学系ノイズの判定列（`med_share`、`excl_*`、`is_medical_noise` など）を足したもの | 9,483 | 2026-08-25 |
| `openalex_toyota_motor_authors_excl_medical.csv` | 医学系ノイズと判定した著者を除いたもの | 8,070 | 2026-08-25 |

豊田自動織機・豊田中央研究所・TRI は含まない。生成したスクリプト（旧 `src/fetch_toyota_authors.py`、`src/clean_toyota_authors.py`）はリポジトリの整理で外した。列の説明は `docs/reports/2026-07-30/toyota_authors_README.md` にある。
