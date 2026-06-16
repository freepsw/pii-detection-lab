# Databricks notebook source
# MAGIC %md
# MAGIC # 거버넌스 — 마스킹 UDF · 컬럼 태깅 · span 마스킹
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC 탐지한 PII를 **실제로 보호**합니다. 구조화 컬럼은 Unity Catalog 컬럼 마스크 + 태그로,
# MAGIC 자유텍스트는 span 단위 치환(유형 라벨)으로 가립니다. 권장 verdict는 컬럼=S2 hybrid(컬럼 F1 최고),
# MAGIC span 마스킹=S4 파인튜닝(span F1 최고)을 사용합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - **컬럼 마스크**: `pii_full_access` 그룹이면 원본, 아니면 마스킹된 값을 반환하는 UDF를 컬럼에
# MAGIC   바인딩(`SET MASK`). 정규식 미매치 값은 일반 마스킹으로 **fail-closed**(원본 노출 금지).
# MAGIC - **태그**: `pii`·`pii_category`·`pii_risk` 로 거버넌스 메타데이터 부착.
# MAGIC - **span 마스킹**: 자유텍스트는 예측 span을 유형 라벨(`[이름]`·`[전화번호]` 등)로 치환.
# MAGIC - **멱등성**: 적용 전 스키마 내 모든 마스크/태그를 해제(stale 마스크 방지).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `07_governance/70_masking_udfs.sql`(literal→{FQ}, fail-closed UDF 검증본) +
# MAGIC `_tools/build_governance.py`(태깅·span 마스킹·unapply). 마스크 해제 셀(`--unapply` 동등)도 포함.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: 기대 산출물]
# MAGIC 마스킹 UDF 7종 + 구조화 PII 컬럼 태깅/마스킹 + `text_corpus_masked`·`span_mask_audit`.
# MAGIC 검증: 마스킹 텍스트에 **잔존 gold PII 0건**(PERSON/ADDRESS는 2자부터 검사).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC span 마스킹 품질은 선택한 span verdict(S4)의 recall에 의존합니다. 미탐 span은 가려지지 않으므로,
# MAGIC 운영에선 가장 높은 recall 단계를 마스킹 기준으로 두는 것이 안전합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** 의 C6 거버넌스 검증이 잔존 PII 0을 독립 재확인합니다.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 권장 verdict 상수 + 헬퍼 (SOURCE: build_governance.py)
# MAGIC
# MAGIC **왜 두 가지 verdict인가** — 컬럼 태깅과 텍스트 마스킹은 *다른 트랙*이라 각 트랙에서 가장 정확한 단계를
# MAGIC 씁니다: 컬럼 분류는 **컬럼 F1 최고 = S2(0.9091)**, 텍스트 span 마스킹은 **span F1 최고 = S4(0.9838)**.
# MAGIC (트랙별 최강 단계 선택 → 90_eval의 `col_eval`·`span_eval`. 자세히는 `../docs/learn/05_거버넌스_마스킹.md`.)

# COMMAND ----------

from collections import defaultdict

CATALOG, SCHEMA = FQ.split(".", 1)
# 컬럼 태깅은 컬럼 F1 최고(S2 hybrid), span 마스킹은 span F1 최고(S4 파인튜닝 NER)
REC_STAGE, REC_BACKEND = "S2", "gpt-oss-120b"            # 컬럼 verdict
REC_SPAN_STAGE, REC_SPAN_BACKEND = "S4", "gpt-oss-120b"  # span 마스킹(최고 정확도)
FREE_TEXT = {"memo", "consult_note", "complaint_body", "review_text"}


def udf_for(col, cat):
    c = col.lower()
    if c == "name_ko":
        return "mask_name"
    if "rrn" in c:
        return "mask_rrn"
    if c in ("phone", "x_contact", "caller_no", "callee_no") or "phone" in c:
        return "mask_phone"
    if "email" in c:
        return "mask_email"
    if "card_number" in c:
        return "mask_card"
    if "account" in c or "bank" in c:
        return "mask_account"
    return "mask_default"


def risk_for(cat):
    return "HIGH" if cat in ("IDENTIFICATION", "PAYMENT_INFO") else "MEDIUM"


def unapply(verbose=True):
    """스키마 내 모든 컬럼 마스크 제거 + pii* 태그 해제(멱등성·오염 복구 수단)."""
    rows = sql(f"""SELECT table_name, column_name FROM {CATALOG}.information_schema.column_masks
      WHERE table_schema='{SCHEMA}'""")[1]
    for t, c in rows:
        sql(f"ALTER TABLE {FQ}.{t} ALTER COLUMN {c} DROP MASK")
        sql(f"ALTER TABLE {FQ}.{t} ALTER COLUMN {c} UNSET TAGS "
            f"('pii','pii_category','pii_risk')")
    if verbose:
        print(f"  unapply: {len(rows)} masks dropped (+tags unset)")
    return len(rows)


print("거버넌스 헬퍼 로드 완료 (REC_STAGE=S2 컬럼 / REC_SPAN_STAGE=S4 span)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (선택) 마스크 해제 셀 — build_governance.py `--unapply` 동등
# MAGIC 재실행 전 stale 마스크/태그를 정리하거나, S2/S5 빌드의 샘플 오염(마스킹된 원천) 시 사용합니다.
# MAGIC 거버넌스를 처음 적용하는 경우 이 셀은 생략해도 됩니다(아래 column_governance가 선행 해제 포함).

# COMMAND ----------

# 필요 시 주석 해제하여 실행:
# unapply()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (1/3): 마스킹 UDF 7종 생성
# MAGIC SOURCE(`70_masking_udfs.sql`)를 {FQ} f-string으로 이식. fail-closed: 포맷 미매치 값은
# MAGIC 일반 마스킹(`첫 글자 + ***`)으로 처리해 원본을 노출하지 않습니다.

# COMMAND ----------

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_phone(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  WHEN s RLIKE '^(01[0-9])-?([0-9]{{3,4}})-?([0-9]{{4}})$'
    THEN regexp_replace(s, '^(01[0-9])-?([0-9]{{3,4}})-?([0-9]{{4}})$', '$1-****-$3')
  ELSE concat(substr(s,1,1), repeat('*', greatest(char_length(s)-1, 3))) END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_rrn(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  ELSE concat(substr(s,1,6), '-*******') END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_email(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  WHEN s RLIKE '^.+@.+$' THEN regexp_replace(s, '(^.).*(@.*$)', '$1***$2')
  ELSE concat(substr(s,1,1), repeat('*', greatest(char_length(s)-1, 3))) END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_name(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  ELSE concat(substr(s,1,1), repeat('*', greatest(char_length(s)-1, 1))) END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_card(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  ELSE concat('****-****-****-', right(regexp_replace(s,'[^0-9]',''),4)) END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_account(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  ELSE concat(substr(s,1,3), '-**-******') END""")

sql(f"""CREATE OR REPLACE FUNCTION {FQ}.mask_default(s STRING) RETURNS STRING
RETURN CASE WHEN is_account_group_member('pii_full_access') THEN s WHEN s IS NULL THEN NULL
  ELSE concat(substr(s,1,1), '***') END""")

print("  masking UDFs created (7종)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (2/3): 컬럼 태깅 + 마스크 바인딩 (S2 verdict 기준)
# MAGIC 적용 전 stale 마스크/태그를 해제하고, S2/gpt-oss PII 컬럼(자유텍스트 제외)에 태그 + 마스크를
# MAGIC **테이블별 직렬**로 바인딩합니다(동일 테이블 SET MASK는 병렬 불가).

# COMMAND ----------

unapply(verbose=False)  # stale 마스크/태그 정리 후 fresh 적용

rows = sql(f"""SELECT c.table_name, c.column_name, c.pred_category
  FROM {FQ}.col_predictions c
  WHERE c.stage='{REC_STAGE}' AND c.backend='{REC_BACKEND}' AND c.pred_is_pii""")[1]
by_table = defaultdict(list)
for t, col, cat in rows:
    if col in FREE_TEXT:
        continue  # 자유텍스트는 span 마스킹으로 처리
    by_table[t].append((col, cat))

for t, cols in by_table.items():
    for col, cat in cols:  # 직렬 실행
        sql(f"ALTER TABLE {FQ}.{t} ALTER COLUMN {col} SET TAGS "
            f"('pii'='true','pii_category'='{cat}','pii_risk'='{risk_for(cat)}')")
        sql(f"ALTER TABLE {FQ}.{t} ALTER COLUMN {col} SET MASK {FQ}.{udf_for(col, cat)}")
    print(f"  governance: {t} ({len(cols)} cols tagged+masked)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (3/3): 자유텍스트 span 마스킹 (S4 verdict 기준)
# MAGIC type_map의 mask_token으로 예측 span을 **내림차순 치환**(offset 안전)해 `text_corpus_masked` 와
# MAGIC `span_mask_audit` 를 생성합니다.

# COMMAND ----------

tok = {et: mt for et, mt in sql(f"SELECT entity_type, mask_token FROM {FQ}.type_map")[1]}
text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]}

spans = defaultdict(list)
for doc, s, e, et in sql(
        f"""SELECT doc_id, start_char, end_char, entity_type FROM {FQ}.span_predictions
            WHERE stage='{REC_SPAN_STAGE}' AND backend='{REC_SPAN_BACKEND}'""")[1]:
    spans[doc].append((int(s), int(e), et))

masked_rows, audit = [], []
for doc, text in text_map.items():
    sp = sorted(spans.get(doc, []), key=lambda x: -x[0])  # 내림차순 치환
    m = text
    for s, e, et in sp:
        m = m[:s] + tok.get(et, "[PII]") + m[e:]
        audit.append({"doc_id": doc, "start_char": s, "end_char": e,
                      "entity_type": et, "masked_token": tok.get(et, "[PII]")})
    tbl, colpart = doc.split(".", 2)[0], doc.split(".", 2)[1]
    masked_rows.append({"doc_id": doc, "table_name": tbl, "column_name": colpart,
                        "original": text, "masked": m})

load_rows(f"{FQ}.text_corpus_masked", masked_rows, mode="replace")
load_rows(f"{FQ}.span_mask_audit", audit, mode="replace")
print(f"  span masking: {len(masked_rows)} docs, {len(audit)} spans masked")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 검증 셀: 마스킹 텍스트에 잔존 gold PII 0건 (PERSON/ADDRESS는 2자부터)
# MAGIC `length>=4` 필터가 한국 이름(대부분 3자)을 전부 제외하던 사각지대를 해소 — PERSON/ADDRESS는
# MAGIC 2자부터 검사합니다. 0이 이상적입니다.

# COMMAND ----------

resid = sql(f"""SELECT count(*) FROM {FQ}.text_corpus_masked m
  JOIN {FQ}.span_ground_truth g ON m.doc_id=g.doc_id
  WHERE locate(g.pii_value, m.masked) > 0
    AND (length(g.pii_value) >= 4
         OR (g.entity_type IN ('PERSON','ADDRESS') AND length(g.pii_value) >= 2))""")[1][0][0]
print(f"  잔존 gold PII 노출 건수(PERSON 2자+ 포함): {resid} (0이 이상적)")
assert int(resid) == 0, f"마스킹 후에도 gold PII 잔존 {resid}건"

# COMMAND ----------

# MAGIC %md ## 결과확인: 마스킹 샘플 (원문 → 마스킹)

# COMMAND ----------

display(sql(f"""SELECT doc_id, original, masked FROM {FQ}.text_corpus_masked
  WHERE original <> masked LIMIT 10""")[1])

# COMMAND ----------

# MAGIC %md ### 구조화 컬럼 마스크 바인딩 현황

# COMMAND ----------

display(sql(f"""SELECT table_name, column_name FROM {CATALOG}.information_schema.column_masks
  WHERE table_schema='{SCHEMA}' ORDER BY table_name, column_name""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** — 전 단계 비교 그리드 + 검증(거버넌스 C6 포함).
