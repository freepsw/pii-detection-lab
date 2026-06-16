# Databricks notebook source
# MAGIC %md
# MAGIC # S1 — 패턴(정규식) 탐지
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC 가장 기본적인 PII 탐지: **정규식 + 컬럼명 키워드**. 전화·카드·주민번호·이메일·계좌·IMEI·여권처럼
# MAGIC **형식이 정해진 정형 PII**를 LLM 없이 결정적으로 잡습니다. 컬럼단위(거버넌스 태깅용)와
# MAGIC span단위(마스킹용) 두 트랙을 한 노트북에 담았습니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC 정규식은 '형식'을 봅니다. `010-1234-5678` 은 전화, `\d{4}-\d{4}-\d{4}-\d{4}` 는 카드.
# MAGIC 9종 정규식이 겹칠 때는 priority(EMAIL>RRN>ACCOUNT/CARD>...)로 해소합니다. 컬럼 트랙은
# MAGIC 값 패턴 히트율과 컬럼명 키워드를 결합해 컬럼 카테고리를 정합니다(`rule_conf >= 0.6` → PII).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC 컬럼 트랙은 SOURCE(`01_pattern/10_pattern_columns.sql`)의 RULE SQL을 `{FQ}` 치환해 그대로
# MAGIC 실행합니다. span 트랙은 `_common`의 검증된 `regex_spans()` (정규식 9종, 바이트 동일)를
# MAGIC text_corpus에 적용합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | 지표 | 값 |
# MAGIC |---|---|
# MAGIC | 정형 PII precision | **1.0** (형식 명확) |
# MAGIC | span exact F1 | **0.73** |
# MAGIC | span char recall | **0.78** |
# MAGIC | col F1 | **0.77** |
# MAGIC | PERSON / ADDRESS | **0%** (형식 없음 → 정규식 불가) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC 이름·주소는 **형식이 없어 정규식으로 잡을 수 없습니다(0%)**. 자유텍스트 속 비정형 PII가
# MAGIC 누락되므로, 의미를 이해하는 **LLM이 필요**합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ S2에서 정규식에 **LLM을 결합**해 이름·주소를 회수합니다(span F1 0.73 → 0.94).

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (1/3): 컬럼 샘플값 추출
# MAGIC 각 컬럼에서 DISTINCT 값 최대 20개를 long 포맷 `column_sample_values` 로 모읍니다 — RULE SQL과
# MAGIC (이후 S2/S5의) LLM 프롬프트가 공유하는 입력입니다. column_ground_truth의 (table, column) 목록을
# MAGIC 기준으로 합니다.

# COMMAND ----------

col_list = [(t, c) for t, c in sql(
    f"SELECT table_name, column_name FROM {FQ}.column_ground_truth")[1]]

parts = []
for t, c in col_list:
    # 컬럼당 DISTINCT 값 최대 20개로 제한: 이후 LLM 프롬프트(S2/S5)의 토큰 길이를 한정하고,
    # 표본 크기를 고정해 실행 간 재현성을 확보(샘플 폭주·비용 변동 방지).
    parts.append(
        f"SELECT '{t}' AS table_name, '{c}' AS column_name, CAST(`{c}` AS STRING) AS val "
        f"FROM (SELECT DISTINCT `{c}` FROM {FQ}.{t} WHERE `{c}` IS NOT NULL LIMIT 20)")
sql(f"CREATE OR REPLACE TABLE {FQ}.column_sample_values AS\n" + "\nUNION ALL ".join(parts))
print(f"  column_sample_values built ({len(col_list)} cols)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (2/3): RULE 계층 + col_predictions(S1)
# MAGIC `01_pattern/10_pattern_columns.sql` 이식. column_profile(LLM 프롬프트용 샘플 문자열) →
# MAGIC rule_results(값 정규식 ∪ 컬럼명 키워드) → col_predictions 마스터 테이블에 S1 적재.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 참고: 9종 정규식 → 탐지 PII 매핑
# MAGIC 아래 RULE SQL의 `CASE WHEN val RLIKE ...` 와 span 트랙의 `_common.regex_spans()` 가 공유하는
# MAGIC 정형 PII 9종 정규식입니다(겹칠 때는 priority EMAIL>RRN>ACCOUNT·CARD>PHONE>PASSPORT>IMEI 순으로 해소).
# MAGIC 예시는 모두 합성값입니다.
# MAGIC
# MAGIC | 정규식(요약) | 탐지 대상 | 예시 |
# MAGIC |---|---|---|
# MAGIC | `[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}` | EMAIL(이메일) | `a.b@kt.com` |
# MAGIC | `\d{6}-[1-4]\d{6}` | RRN(주민등록번호) | `901010-1234567` |
# MAGIC | `\d{3}-\d{2}-\d{6}` | ACCOUNT(계좌번호) | `123-45-678901` |
# MAGIC | `\d{4}-\d{4}-\d{4}-\d{4}` | CARD(카드, 하이픈) | `1234-5678-9012-3456` |
# MAGIC | `\d{16}` | CARD(카드, 16자리 연속) | `1234567890123456` |
# MAGIC | `01[016789]-\d{3,4}-\d{4}` | PHONE(휴대전화, 하이픈) | `010-1234-5678` |
# MAGIC | `01[016789]\d{7,8}` | PHONE(휴대전화, 연속) | `01098765432` |
# MAGIC | `[MSROD]\d{8}` | PASSPORT(여권번호) | `M12345678` |
# MAGIC | `\d{15}` | IMEI(단말 식별번호) | `356938035643809` |
# MAGIC
# MAGIC > 숫자 런은 경계(`\b` 류) 단언으로 독립된 길이만 매칭해 **카드16 · IMEI15 · 전화11** 을 구분합니다.
# MAGIC > 그래서 RULE SQL도 IMEI(정확히 15자리)를 카드(13~19자리)보다 **먼저** 판정합니다.

# COMMAND ----------

# (1) 컬럼 프로파일 — LLM 프롬프트용 샘플 문자열(최대 10개). S2/S5가 재사용.
sql(f"""CREATE OR REPLACE TABLE {FQ}.column_profile AS
SELECT table_name, column_name, count(*) AS n_samples,
       array_join(slice(collect_list(val), 1, 10), ' | ') AS samples_str
FROM {FQ}.column_sample_values
GROUP BY table_name, column_name""")

# (2) RULE 계층: 값 단위 한국형 정규식 + 컬럼명 키워드
sql(f"""CREATE OR REPLACE TABLE {FQ}.rule_results AS
WITH m AS (
  SELECT table_name, column_name, val,
    CASE
      WHEN val RLIKE '^[0-9]{{6}}-[1-4][0-9]{{6}}$' THEN 'IDENTIFICATION'
      WHEN val RLIKE '^01[016789]-?[0-9]{{3,4}}-?[0-9]{{4}}$' THEN 'PERSONAL_INFO'
      WHEN val RLIKE '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{{2,}}$' THEN 'PERSONAL_INFO'
      WHEN val RLIKE '^[0-9]{{4}}-[0-9]{{4}}-[0-9]{{4}}-[0-9]{{4}}$' THEN 'PAYMENT_INFO'
      WHEN val RLIKE '^[0-9]{{3}}-[0-9]{{2}}-[0-9]{{6}}$' THEN 'PAYMENT_INFO'
      -- IMEI(정확히 15자리)를 카드(13~19자리)보다 먼저 판정
      WHEN val RLIKE '^[0-9]{{15}}$' THEN 'IDENTIFICATION'
      WHEN val RLIKE '^[0-9]{{13,19}}$' THEN 'PAYMENT_INFO'
      WHEN val RLIKE '^[가-힣]{{2,4}}$' THEN 'PERSONAL_INFO'
      WHEN val RLIKE '.*(시|도).*(구|군).*' THEN 'PERSONAL_INFO'
      WHEN val RLIKE '^(19|20)[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}$' THEN 'PERSONAL_INFO'
      ELSE 'NON_PII'
    END AS cat
  FROM {FQ}.column_sample_values
),
agg AS (
  SELECT table_name, column_name, count(*) AS tot,
         sum(CASE WHEN cat <> 'NON_PII' THEN 1 ELSE 0 END) AS pii_hits
  FROM m GROUP BY table_name, column_name
),
catcnt AS (SELECT table_name, column_name, cat, count(*) AS c FROM m GROUP BY table_name, column_name, cat),
catpick AS (
  SELECT table_name, column_name, cat,
         row_number() OVER (PARTITION BY table_name, column_name ORDER BY (cat <> 'NON_PII') DESC, c DESC) AS rn
  FROM catcnt
),
best AS (SELECT table_name, column_name, cat AS value_cat FROM catpick WHERE rn = 1)
SELECT a.table_name, a.column_name,
  round(a.pii_hits / a.tot, 3) AS pii_frac,
  CASE
    WHEN b.value_cat <> 'NON_PII' THEN b.value_cat
    WHEN lower(a.column_name) RLIKE '(card|account|acct|bank)' THEN 'PAYMENT_INFO'
    WHEN lower(a.column_name) RLIKE '(salary)' THEN 'EMPLOYMENT_INFO'
    WHEN lower(a.column_name) RLIKE '(rrn|ssn|passport|imei)' THEN 'IDENTIFICATION'
    WHEN lower(a.column_name) RLIKE '(name|email|phone|mobile|tel|addr|address|birth|contact|memo)' THEN 'PERSONAL_INFO'
    ELSE 'NON_PII'
  END AS rule_category,
  greatest(
    CASE WHEN b.value_cat <> 'NON_PII' THEN round(a.pii_hits / a.tot, 3) ELSE 0 END,
    CASE WHEN lower(a.column_name) RLIKE '(name|email|phone|mobile|tel|rrn|ssn|passport|card|account|acct|bank|salary|addr|address|birth|contact|memo)' THEN 0.6 ELSE 0 END
  ) AS rule_conf,
  (greatest(
    CASE WHEN b.value_cat <> 'NON_PII' THEN round(a.pii_hits / a.tot, 3) ELSE 0 END,
    CASE WHEN lower(a.column_name) RLIKE '(name|email|phone|mobile|tel|rrn|ssn|passport|card|account|acct|bank|salary|addr|address|birth|contact|memo)' THEN 0.6 ELSE 0 END
  ) >= 0.6) AS rule_is_pii
FROM agg a JOIN best b USING (table_name, column_name)""")

# (3) 컬럼 예측 마스터 테이블 + S1 적재
sql(f"""CREATE OR REPLACE TABLE {FQ}.col_predictions (
  stage STRING, backend STRING, table_name STRING, column_name STRING,
  pred_is_pii BOOLEAN, pred_category STRING)""")
sql(f"""INSERT INTO {FQ}.col_predictions
SELECT 'S1' AS stage, 'NA' AS backend, table_name, column_name,
       rule_is_pii AS pred_is_pii, rule_category AS pred_category
FROM {FQ}.rule_results""")
print("  S1 rule + col_predictions(S1) done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (3/3): span 트랙 — regex_spans → span_predictions(S1)
# MAGIC `_common.regex_spans()` 를 text_corpus 전체에 적용해 정형 PII span을 추출하고 `span_predictions`
# MAGIC 마스터 테이블(stage='S1')에 적재합니다. S1은 전체 코퍼스를 처리하므로 span_coverage에 전 doc 기록.

# COMMAND ----------

# span_predictions / span_coverage 마스터 테이블 보장
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_predictions (
  stage STRING, backend STRING, doc_id STRING,
  start_char INT, end_char INT, entity_type STRING, pii_value STRING, score DOUBLE)""")
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_coverage
  (stage STRING, backend STRING, doc_id STRING)""")

corpus = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]
out = []
for doc_id, text in corpus:
    for s in regex_spans(text):
        out.append({"stage": "S1", "backend": "NA", "doc_id": doc_id,
                    "start_char": s["start"], "end_char": s["end"],
                    "entity_type": s["entity_type"], "pii_value": s["pii_value"],
                    "score": s["score"]})
print(f"S1 spans: {len(out)} over {len(corpus)} docs")
load_rows(f"{FQ}.span_predictions", out, where="stage='S1' AND backend='NA'")

# coverage: S1은 전체 코퍼스 처리
sql(f"DELETE FROM {FQ}.span_coverage WHERE stage='S1' AND backend='NA'")
sql(f"INSERT INTO {FQ}.span_coverage SELECT 'S1','NA', doc_id FROM {FQ}.text_corpus")
print("  span_coverage(S1) written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 결과확인 (1): span 유형 분포 — PERSON/ADDRESS = 0건
# MAGIC 정형 PII(PHONE·EMAIL·CARD·RRN·ACCOUNT·IMEI·PASSPORT)만 잡히고, **이름(PERSON)·주소(ADDRESS)는 0건**입니다 — 정규식의 구조적 한계를 데이터로 확인합니다.

# COMMAND ----------

display(sql(f"""SELECT entity_type, count(*) AS n
FROM {FQ}.span_predictions WHERE stage='S1'
GROUP BY entity_type ORDER BY n DESC""")[1])

# COMMAND ----------

# MAGIC %md ### NER 유형(PERSON/ADDRESS)이 정답엔 있으나 S1 예측엔 없음 — 명시 확인

# COMMAND ----------

display(sql(f"""
SELECT 'ground_truth' src, entity_type, count(*) n
FROM {FQ}.span_ground_truth WHERE entity_type IN ('PERSON','ADDRESS') GROUP BY entity_type
UNION ALL
SELECT 'S1_prediction', entity_type, count(*)
FROM {FQ}.span_predictions WHERE stage='S1' AND entity_type IN ('PERSON','ADDRESS') GROUP BY entity_type
ORDER BY src, entity_type""")[1])

# COMMAND ----------

# MAGIC %md ## 결과확인 (2): col_predictions(S1) 요약

# COMMAND ----------

display(sql(f"""SELECT pred_category, count(*) n, sum(CASE WHEN pred_is_pii THEN 1 ELSE 0 END) n_pii
FROM {FQ}.col_predictions WHERE stage='S1' GROUP BY pred_category ORDER BY n DESC""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 비교 (러닝 스코어보드)
# MAGIC | 단계 | span exact F1 | 라우팅 | 비용/1k | 한 줄 |
# MAGIC |---|---|---|---|---|
# MAGIC | **S1 ← 지금 여기** | **0.73** | — | **$0** | 정형만 — 이름·주소 0% |
# MAGIC | S2 (예정) | — | — | — | LLM으로 비정형(이름·주소) 회수 |
# MAGIC | S3 (예정) | — | — | — | NER cascade로 비용↓ |
# MAGIC | S4 (예정) | — | — | — | 파인튜닝 NER = 최적 |
# MAGIC | S5 (예정) | — | — | — | LLM 단독(대조군) |
# MAGIC
# MAGIC > 정형 PII는 정규식으로 충분하지만 **이름·주소가 0%** — 이 공백이 다음 단계(S2)에서 LLM을 쓰는 이유입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`20_S2_pattern_llm/S2_pattern_llm`](../20_S2_pattern_llm/S2_pattern_llm)** — 정규식에 LLM을 결합(hybrid 컬럼 + 정규식∪LLM span). 최종 점수는 90_eval에서 산출됩니다.
