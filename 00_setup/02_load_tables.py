# Databricks notebook source
# MAGIC %md
# MAGIC # 0-2. raw CSV → Delta 테이블 9종 + text_corpus
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC `RAW_VOL` 에 올린 CSV를 Unity Catalog Delta 테이블로 적재합니다. 모든 단계 노트북이 `{FQ}.*`
# MAGIC 테이블을 읽으므로 이 노트북이 데이터 계층의 단일 진입점입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC 모든 원천 컬럼을 **STRING으로 적재**합니다 — 주민번호 앞자리 0, 전화 하이픈, 카드 4-4-4-4 형식
# MAGIC 등을 보존해야 정규식·LLM 탐지가 실제 데이터와 동일하게 동작하기 때문입니다(타입 추론 OFF).
# MAGIC 텍스트 컬럼들은 `text_corpus` 라는 **long 포맷** 단일 테이블로 모읍니다 — `doc_id` 를 키로
# MAGIC 예측⇄정답을 조인하기 위함입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE(`00_data/02_load_tables.sql`)의 모든 하드코딩 literal(`<catalog>.<schema>`,
# MAGIC `/Volumes/.../raw`)을 `{FQ}`·`{RAW_VOL}` f-string으로 치환했습니다. 디버깅 단위로 셀당 1개
# MAGIC CREATE TABLE + 직후 count 확인 셀을 둡니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: 기대 산출물]
# MAGIC 구조화 6테이블(subscribers·billing·call_records·employees·cs_consultations·reviews) +
# MAGIC 정답/메타 3테이블(column_ground_truth·span_ground_truth·type_map) + `text_corpus`(약 4,860 docs).
# MAGIC 총 **9개 적재 테이블 + text_corpus**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC `multiLine => false` 를 씁니다 — 텍스트에 개행/임베디드 따옴표가 없음을 생성기가 보장하기
# MAGIC 때문입니다(multiLine=true는 일부 행이 NULL로 파싱되는 문제 실측). 실제 고객 텍스트엔 개행이
# MAGIC 있을 수 있어, 운영 적재 시 옵션 재검토가 필요합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC text_corpus가 준비되면 → `10_S1_pattern` 부터 단계별 탐지 아키텍처를 쌓습니다.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 내 데이터로 바꾸려면 (BYOD)
# MAGIC 이 노트북이 **데이터 진입점**입니다. 합성 CSV 대신 내 데이터를 쓰려면:
# MAGIC 1. `00_config` 위젯의 `catalog`/`schema` 를 내 작업공간으로 지정,
# MAGIC 2. 아래 `read_files('{RAW_VOL}/...')` 의 경로·컬럼을 내 테이블에 맞게 수정,
# MAGIC 3. `text_corpus` UNION 절에 내 **자유텍스트 컬럼**을 추가,
# MAGIC 4. 정답표(`column_ground_truth`·`span_ground_truth`)는 내 라벨로 교체(없으면 평가·파인튜닝 제약 → `../docs/learn/06_내_데이터에_적용.md`).

# COMMAND ----------

# MAGIC %md ## 구조화 6테이블 (read_files, 모두 STRING)

# COMMAND ----------

# MAGIC %md ### subscribers (+ memo 자유텍스트)

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.subscribers AS
SELECT subscriber_id, name_ko, rrn, phone, email, address, birth_date, gender, imei,
       x_contact, card_type, plan_code, status, region_code, join_date, memo
FROM read_files('{RAW_VOL}/subscribers.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.subscribers")[1])

# COMMAND ----------

# MAGIC %md ### billing

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.billing AS
SELECT bill_id, subscriber_id, card_number, card_expiry, bank_account, amount,
       billing_month, paid_yn, due_date
FROM read_files('{RAW_VOL}/billing.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.billing")[1])

# COMMAND ----------

# MAGIC %md ### call_records

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.call_records AS
SELECT call_id, caller_no, callee_no, duration_sec, cell_id, call_ts, call_type
FROM read_files('{RAW_VOL}/call_records.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.call_records")[1])

# COMMAND ----------

# MAGIC %md ### employees

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.employees AS
SELECT emp_id, name_ko, email, phone, salary, dept, position, hire_date
FROM read_files('{RAW_VOL}/employees.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.employees")[1])

# COMMAND ----------

# MAGIC %md ### cs_consultations

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.cs_consultations AS
SELECT consult_id, subscriber_id, channel, consult_ts, consult_note, complaint_body, category_code
FROM read_files('{RAW_VOL}/cs_consultations.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.cs_consultations")[1])

# COMMAND ----------

# MAGIC %md ### reviews

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.reviews AS
SELECT review_id, store_name, review_text, rating
FROM read_files('{RAW_VOL}/reviews.csv',
  format => 'csv', header => true, inferColumnTypes => false, multiLine => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.reviews")[1])

# COMMAND ----------

# MAGIC %md ## 정답/메타 3테이블 (캐스팅)

# COMMAND ----------

# MAGIC %md ### column_ground_truth (is_pii → BOOLEAN)

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.column_ground_truth AS
SELECT table_name, column_name, CAST(is_pii AS BOOLEAN) AS is_pii, category
FROM read_files('{RAW_VOL}/column_ground_truth.csv',
  format => 'csv', header => true, inferColumnTypes => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.column_ground_truth")[1])

# COMMAND ----------

# MAGIC %md ### span_ground_truth (start/end_char → INT)

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.span_ground_truth AS
SELECT doc_id, table_name, column_name, row_key,
       CAST(start_char AS INT) AS start_char, CAST(end_char AS INT) AS end_char,
       entity_type, pii_value, source_layer
FROM read_files('{RAW_VOL}/span_ground_truth.csv',
  format => 'csv', header => true, inferColumnTypes => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.span_ground_truth")[1])

# COMMAND ----------

# MAGIC %md ### type_map

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.type_map AS
SELECT entity_type, category, presidio_entity, mask_token, source_layer
FROM read_files('{RAW_VOL}/type_map.csv',
  format => 'csv', header => true, inferColumnTypes => false)""")
display(sql(f"SELECT count(*) AS n FROM {FQ}.type_map")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## text_corpus: 텍스트 컬럼 long 포맷
# MAGIC subscribers.memo + cs_consultations.(consult_note·complaint_body) + reviews.review_text 를
# MAGIC `doc_id` 키로 합칩니다(빈 문자열·NULL 제외). 예측⇄정답 조인의 기준 테이블입니다.

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.text_corpus AS
SELECT concat('subscribers.memo.', subscriber_id) AS doc_id,
       'subscribers' AS table_name, 'memo' AS column_name, subscriber_id AS row_key, memo AS text
FROM {FQ}.subscribers WHERE memo IS NOT NULL AND memo <> ''
UNION ALL
SELECT concat('cs_consultations.consult_note.', consult_id),
       'cs_consultations', 'consult_note', consult_id, consult_note
FROM {FQ}.cs_consultations WHERE consult_note IS NOT NULL AND consult_note <> ''
UNION ALL
SELECT concat('cs_consultations.complaint_body.', consult_id),
       'cs_consultations', 'complaint_body', consult_id, complaint_body
FROM {FQ}.cs_consultations WHERE complaint_body IS NOT NULL AND complaint_body <> ''
UNION ALL
SELECT concat('reviews.review_text.', review_id),
       'reviews', 'review_text', review_id, review_text
FROM {FQ}.reviews WHERE review_text IS NOT NULL AND review_text <> ''""")
display(sql(f"SELECT count(*) AS n_docs FROM {FQ}.text_corpus")[1])

# COMMAND ----------

# MAGIC %md ## 적재 요약 (테이블별 행수)

# COMMAND ----------

display(sql(f"""
SELECT 'subscribers' t, count(*) n FROM {FQ}.subscribers
UNION ALL SELECT 'billing', count(*) FROM {FQ}.billing
UNION ALL SELECT 'call_records', count(*) FROM {FQ}.call_records
UNION ALL SELECT 'employees', count(*) FROM {FQ}.employees
UNION ALL SELECT 'cs_consultations', count(*) FROM {FQ}.cs_consultations
UNION ALL SELECT 'reviews', count(*) FROM {FQ}.reviews
UNION ALL SELECT 'column_ground_truth', count(*) FROM {FQ}.column_ground_truth
UNION ALL SELECT 'span_ground_truth', count(*) FROM {FQ}.span_ground_truth
UNION ALL SELECT 'type_map', count(*) FROM {FQ}.type_map
UNION ALL SELECT 'text_corpus', count(*) FROM {FQ}.text_corpus
ORDER BY t""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`10_S1_pattern/S1_pattern_detection`](../10_S1_pattern/S1_pattern_detection)** — 정규식 기반 S1 패턴 탐지(컬럼+span).
