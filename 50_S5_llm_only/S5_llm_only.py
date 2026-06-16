# Databricks notebook source
# MAGIC %md
# MAGIC # S5 — LLM 단독 (LLM-only)
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC 정규식·NER 없이 **LLM만으로** PII를 탐지하면 어떨까? S5는 패턴 레이어를 빼고 LLM 판정만
# MAGIC 사용합니다. 'LLM이 만능'이라는 가정을 검증하고, S4(파인튜닝 NER+cascade) 대비 우위를 드러내는
# MAGIC 대조군입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - **컬럼**: LLM의 `is_pii` 판정만 사용(rule 결합 없음).
# MAGIC - **span**: LLM이 추출한 부분문자열만 사용(정규식 병합 없음). `recover_spans()` 로 offset 복원.
# MAGIC - LLM 원천(raw)은 S2와 동일한 프롬프트라 S2 노트북을 이미 실행했다면 재사용 가능하지만, 이
# MAGIC   노트북은 **자체포함**으로 raw를 직접 추출합니다(S2 미실행에도 단독 동작).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `_tools/build_columns.py`(S5 컬럼: `llm_is_pii` only) + `_tools/build_span_llm.py`
# MAGIC (S5 span: LLM-only) + `_common.span_llm.recover_spans`. gpt-oss는 ai_query 전체, qwen은
# MAGIC `S5_QWEN_SAMPLE_DOCS`(=300) 표본. fail-closed 정책 동일.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | 지표 | 값 |
# MAGIC |---|---|
# MAGIC | span exact F1 | **0.9441** (S2 0.9448과 유사) |
# MAGIC | col F1 | **0.8889** |
# MAGIC | 비용 | **$0.132 / 1k docs** (전수 LLM) |
# MAGIC | LLM 라우팅율 | **100%** |

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC LLM 단독은 S2(패턴+LLM)보다 나을 게 없고(정형 PII는 정규식이 더 정밀), S4 대비 **비용은 13배, 정확도는 낮습니다**. 정형 패턴을 LLM에 맡기는 것은 비효율입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ 모든 단계를 **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** 에서 한 그리드로 비교합니다 — S4가 정확도·비용 동시 최적임을 확인.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## ============ 컬럼 트랙 (LLM-only) ============
# MAGIC ## 이 단계가 하는 일 (컬럼): LLM raw 확보 → llm_col → S5 예측
# MAGIC `llm_col` 이 이미 있으면(=S2 실행됨) 재사용하고, 없으면 컬럼 분류 raw를 추출해 만듭니다.
# MAGIC 그다음 **LLM `is_pii` 만으로** col_predictions(S5)를 적재합니다.

# COMMAND ----------

import time
from column_prompt import PROMPT_HEAD, PROMPT_TAIL, prompt_text

GPTOSS_EP = BACKENDS["gpt-oss-120b"]


def _llm_col_exists():
    try:
        sql(f"SELECT 1 FROM {FQ}.llm_col LIMIT 1")
        return True
    except Exception:
        return False


if _llm_col_exists():
    print("  llm_col 재사용 (S2에서 생성됨)")
else:
    print("  llm_col 없음 — 컬럼 분류 raw 추출")
    sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.llm_col_raw
      (backend STRING, table_name STRING, column_name STRING, llm_raw STRING)""")
    # gpt-oss: ai_query 배치
    sql(f"DELETE FROM {FQ}.llm_col_raw WHERE backend='gpt-oss-120b'")
    _ce = (f"concat('{rt.sqllit(PROMPT_HEAD)}', '컬럼명: ', column_name, '\\n테이블: ', table_name, "
           f"'\\n샘플값(최대 10개): ', samples_str, '{rt.sqllit(PROMPT_TAIL)}')")
    sql(f"""INSERT INTO {FQ}.llm_col_raw
      SELECT 'gpt-oss-120b', table_name, column_name, ai_query('{GPTOSS_EP}', {_ce})
      FROM {FQ}.column_profile""")
    # qwen: Python 직렬(선택)
    for backend in [b for b in LLM_BACKENDS if b != "gpt-oss-120b"]:
        rows = sql(f"SELECT table_name, column_name, samples_str FROM {FQ}.column_profile")[1]
        out, failed, t0 = [], [], time.time()
        for i, (t, c, samples) in enumerate(rows):
            p = prompt_text(t, c, samples or "")
            try:
                out.append([t, c, llm_client(p, backend=backend, max_tokens=200)])
            except Exception as e:
                failed.append((t, c, p, str(e)[:120]))
        for t, c, p, _ in failed:
            try:
                out.append([t, c, llm_client(p, backend=backend, max_tokens=200)])
            except Exception as e:
                raise RuntimeError(f"{backend} 컬럼 호출 최종 실패 — 적재 중단. 예: {(t, c, str(e)[:120])}")
        load_rows(f"{FQ}.llm_col_raw",
                  [{"backend": backend, "table_name": t, "column_name": c, "llm_raw": raw} for t, c, raw in out],
                  where=f"backend='{backend}'")
        print(f"  {backend} column LLM done ({len(out)} cols)")
    # 파싱
    sql(f"""CREATE OR REPLACE TABLE {FQ}.llm_col AS
      SELECT backend, table_name, column_name,
        coalesce(j.is_pii, false) AS llm_is_pii,
        coalesce(nullif(upper(j.category), ''), 'NON_PII') AS llm_category,
        coalesce(j.confidence, 0.5) AS llm_conf, llm_raw
      FROM {FQ}.llm_col_raw
      LATERAL VIEW OUTER explode(array(
        from_json(regexp_extract(llm_raw, '(?s)\\\\{{.*\\\\}}', 0),
          'is_pii BOOLEAN, category STRING, confidence DOUBLE, reason STRING'))) t AS j""")

# S5 LLM-only 컬럼 예측
sql(f"DELETE FROM {FQ}.col_predictions WHERE stage='S5'")
sql(f"""INSERT INTO {FQ}.col_predictions
  SELECT 'S5' AS stage, backend, table_name, column_name,
    llm_is_pii AS pred_is_pii,
    CASE WHEN llm_is_pii THEN llm_category ELSE 'NON_PII' END AS pred_category
  FROM {FQ}.llm_col""")
print("  col_predictions(S5) written")
display(sql(f"SELECT backend, count(*) n FROM {FQ}.col_predictions WHERE stage='S5' GROUP BY backend")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## ============ span 트랙 (LLM-only) ============
# MAGIC ## 이 단계가 하는 일 (span 1/2): span LLM raw 확보
# MAGIC `span_llm_raw` 가 이미 있으면(=S2 실행됨) 재사용, 없으면 추출합니다. gpt-oss는 ai_query 전체,
# MAGIC qwen은 300 표본(`xxhash64` 결정적 표집). fail-closed.

# COMMAND ----------

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_llm_raw
  (backend STRING, doc_id STRING, llm_raw STRING)""")
text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]}

# backend -> coverage doc 목록
covered = {}


def _raw_count(backend):
    return int(sql(f"SELECT count(*) FROM {FQ}.span_llm_raw WHERE backend='{backend}'")[1][0][0])


# gpt-oss (전체)
if _raw_count("gpt-oss-120b") > 0:
    print("  gpt-oss span raw 재사용")
else:
    sql(f"DELETE FROM {FQ}.span_llm_raw WHERE backend='gpt-oss-120b'")
    _expr = f"ai_query('{GPTOSS_EP}', concat('{rt.sqllit(SPAN_PROMPT)}', text))"
    sql(f"INSERT INTO {FQ}.span_llm_raw SELECT 'gpt-oss-120b', doc_id, {_expr} FROM {FQ}.text_corpus")
    print("  gpt-oss span raw done")
covered["gpt-oss-120b"] = [d for (d,) in sql(
    f"SELECT doc_id FROM {FQ}.span_llm_raw WHERE backend='gpt-oss-120b'")[1]]

# qwen (표본 300, 선택)
for backend in [b for b in LLM_BACKENDS if b != "gpt-oss-120b"]:
    if _raw_count(backend) > 0:
        print(f"  {backend} span raw 재사용")
    else:
        docs = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT {S5_QWEN_SAMPLE_DOCS}")[1]
        rows, failed, t0 = [], [], time.time()
        for i, (doc, text) in enumerate(docs):
            try:
                rows.append({"backend": backend, "doc_id": doc,
                             "llm_raw": llm_client(SPAN_PROMPT + text, backend=backend, max_tokens=400)})
            except Exception as e:
                failed.append((doc, text, str(e)[:120]))
            if (i + 1) % 50 == 0:
                print(f"    {backend} span {i+1}/{len(docs)} ({time.time()-t0:.0f}s, fail={len(failed)})")
        still = []
        for doc, text, _ in failed:
            try:
                rows.append({"backend": backend, "doc_id": doc,
                             "llm_raw": llm_client(SPAN_PROMPT + text, backend=backend, max_tokens=400)})
            except Exception as e:
                still.append((doc, str(e)[:120]))
        if len(still) > len(docs) * 0.05:
            raise RuntimeError(f"{backend} span 최종 실패 {len(still)}/{len(docs)} — 5% 초과, 중단. 예: {still[0]}")
        load_rows(f"{FQ}.span_llm_raw", rows, where=f"backend='{backend}'")
        print(f"  {backend} span raw done ({len(rows)}/{len(docs)} docs)")
    covered[backend] = [d for (d,) in sql(
        f"SELECT doc_id FROM {FQ}.span_llm_raw WHERE backend='{backend}'")[1]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (span 2/2): raw → S5 span 예측 (LLM-only)
# MAGIC `recover_spans()` 로 LLM 부분문자열의 offset만 복원해 `span_predictions`(stage='S5')에 적재합니다
# MAGIC (정규식 병합 없음 — S2와의 핵심 차이). coverage는 backend별 처리 doc만 기록.

# COMMAND ----------


def _row(s):
    return {"start_char": s["start"], "end_char": s["end"], "entity_type": s["entity_type"],
            "pii_value": s["pii_value"], "score": s["score"]}


for backend in LLM_BACKENDS:
    raw = sql(f"SELECT doc_id, llm_raw FROM {FQ}.span_llm_raw WHERE backend='{backend}'")[1]
    s5, n_parse_fail = [], 0
    for doc, llm_raw in raw:
        text = text_map.get(doc, "")
        parsed = extract_json(llm_raw)
        items = parsed if isinstance(parsed, list) else (parsed.get("spans") if isinstance(parsed, dict) else None)
        if items is None:
            n_parse_fail += 1
            items = []
        for s in recover_spans(text, items):
            s5.append({"stage": "S5", "backend": backend, "doc_id": doc, **_row(s)})
    sql(f"DELETE FROM {FQ}.span_predictions WHERE stage='S5' AND backend='{backend}'")
    if s5:
        load_rows(f"{FQ}.span_predictions", s5)
    sql(f"DELETE FROM {FQ}.span_coverage WHERE stage='S5' AND backend='{backend}'")
    load_rows(f"{FQ}.span_coverage",
              [{"stage": "S5", "backend": backend, "doc_id": d} for d in covered[backend]])
    print(f"  {backend}: S5={len(s5)} spans (parse_fail={n_parse_fail}/{len(raw)})")
    if raw and n_parse_fail > len(raw) * 0.05:
        raise RuntimeError(f"{backend} span 응답 파싱 실패 {n_parse_fail}/{len(raw)} — 5% 초과, 중단")

# COMMAND ----------

# MAGIC %md ## 결과확인: S5 span_predictions 셀별 분포

# COMMAND ----------

display(sql(f"""SELECT stage, backend, count(*) n
FROM {FQ}.span_predictions WHERE stage='S5' GROUP BY stage, backend ORDER BY backend""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 비교 (러닝 스코어보드 — 전 단계 완료)
# MAGIC | 단계 | span exact F1 | 라우팅 | 비용/1k | 한 줄 |
# MAGIC |---|---|---|---|---|
# MAGIC | S1 | 0.73 | — | $0 | 정형만 — 이름·주소 0% |
# MAGIC | S2 | 0.9448 | 100% | $0.132 | 전수 LLM(정확하나 비쌈) |
# MAGIC | S3 | 0.8966 | 10.9% | $0.0244 | 범용 NER cascade(과탐) |
# MAGIC | **S4** | **0.9838** | **0%** | **$0.01** | **최적**(파인튜닝 NER) |
# MAGIC | **S5 ← 지금 여기** | **0.9441** | **100%** | **$0.132** | LLM 단독: S2와 사실상 동률·col은 열위 → 정확도/비용 **우위 없음**. 유일 장점 = 파이프라인 단순 |
# MAGIC
# MAGIC > 결론: **S4가 최적**. S5(LLM 단독)는 가장 단순하지만 S2 대비 정확도·비용 우위가 없습니다 — 그래서 권장 기본은 S4입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** — 전 단계를 한 그리드로 비교하고 검증합니다.
