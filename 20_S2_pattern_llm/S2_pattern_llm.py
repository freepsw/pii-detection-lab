# Databricks notebook source
# MAGIC %md
# MAGIC # S2 — 패턴 + LLM (hybrid)
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC S1 정규식이 못 잡는 **이름·주소 같은 비정형 PII**를 LLM으로 회수합니다. 컬럼 트랙은
# MAGIC rule·LLM 점수를 가중 결합(hybrid)하고, span 트랙은 정규식 span ∪ LLM span을 합칩니다.
# MAGIC S2/span의 LLM 원천(raw)은 S5(LLM-only)와 공유하므로 이 노트북에서 한 번만 추출합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - **컬럼 hybrid**: `0.4*rule_score + 0.6*llm_score >= 0.5` → PII. 정규식의 정밀함과 LLM의
# MAGIC   의미 이해를 결합합니다.
# MAGIC - **span 병합**: 정규식 span을 우선 점유하고, 겹치지 않는 LLM span(이름·주소 등)을 추가합니다.
# MAGIC - **백엔드 2종**: gpt-oss는 `ai_query` SQL 배치(전체 코퍼스), qwen은 Python 직렬 호출
# MAGIC   (throughput 제약). qwen 위젯 미설정 시 자동으로 gpt-oss 단독.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `_tools/build_columns.py`(assert_no_masks·gptoss_llm·qwen_llm·parse_and_predict의 S2) +
# MAGIC `_tools/build_span_llm.py`(gptoss_raw·qwen_raw·derive의 S2) + `_common`의 `column_prompt`·`span_llm`.
# MAGIC **fail-closed** 정책(검증 수정본 그대로): LLM 호출/파싱 실패를 NON_PII로 위장하지 않고,
# MAGIC 1회 재시도 후 실패율 5% 초과 시 hard fail합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | 지표 | 값 |
# MAGIC |---|---|
# MAGIC | span exact F1 | **0.9448** (S1 0.73 → 대폭 상승) |
# MAGIC | span char F1 | **0.998** |
# MAGIC | col F1 | **0.9091** |
# MAGIC | 비용(전수 LLM) | **$0.132 / 1k docs** |
# MAGIC | LLM 라우팅율 | **100%** (전 문서 LLM) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC 모든 문서를 LLM에 보내므로 **대량 처리 시 비쌉니다**($0.132/1k). 비용을 줄이려면 LLM을
# MAGIC '필요한 문서에만' 보내는 라우팅이 필요합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ S3/S4에서 NER을 1차 필터로 두고 **저신뢰 span만 LLM으로 확인(cascade)** 해 비용을 10분의 1로 낮춥니다.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 선행 셀: 마스크 오염 방지 (assert_no_masks)
# MAGIC 원천 테이블에 컬럼 마스크가 남아 있고 실행자가 `pii_full_access` 비멤버면, 샘플이 마스킹된 값으로
# MAGIC 추출되어 LLM 예측이 오염됩니다. 그런 경우 **시작 전에 명시적으로 실패**시킵니다
# MAGIC (먼저 80_governance의 마스크 해제 셀 실행 필요). 마스크가 없으면 통과합니다.

# COMMAND ----------

CATALOG, SCHEMA = FQ.split(".", 1)
_tables = "', '".join(sorted({t for t, _ in sql(
    f"SELECT table_name, column_name FROM {FQ}.column_ground_truth")[1]}))
_masked = sql(f"""SELECT table_name, column_name FROM {CATALOG}.information_schema.column_masks
  WHERE table_schema='{SCHEMA}' AND table_name IN ('{_tables}')""")[1]
if _masked:
    _member = sql("SELECT is_account_group_member('pii_full_access')")[1]
    if str(_member[0][0]).lower() != "true":
        raise RuntimeError(
            f"원천 테이블에 컬럼 마스크 {len(_masked)}개 잔존 + 실행자가 pii_full_access 비멤버 "
            f"→ 샘플 오염. 먼저 80_governance의 '마스크 해제' 셀을 실행 후 재시도. 예: {_masked[:3]}")
    print(f"  (마스크 {len(_masked)}개 잔존하나 실행자가 pii_full_access 멤버 — 원본 샘플 추출 가능)")
else:
    print("  마스크 없음 — 원본 샘플 추출 가능")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ============ 컬럼 트랙 ============
# MAGIC ## 이 단계가 하는 일 (컬럼 1/3): gpt-oss 컬럼 분류 — ai_query 배치
# MAGIC `column_profile` 의 각 컬럼에 대해 `ai_query`로 PII 판정 JSON을 받아 `llm_col_raw` 에 적재합니다.
# MAGIC 프롬프트(`column_prompt.PROMPT_HEAD/TAIL`)는 `rt.sqllit()` 로 이스케이프해 SQL에 인라인합니다.
# MAGIC (S1에서 column_profile/rule_results가 이미 생성돼 있어야 합니다.)

# COMMAND ----------

from column_prompt import PROMPT_HEAD, PROMPT_TAIL

GPTOSS_EP = BACKENDS["gpt-oss-120b"]

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.llm_col_raw
  (backend STRING, table_name STRING, column_name STRING, llm_raw STRING)""")
sql(f"DELETE FROM {FQ}.llm_col_raw WHERE backend='gpt-oss-120b'")

concat_expr = (
    f"concat('{rt.sqllit(PROMPT_HEAD)}', '컬럼명: ', column_name, '\\n테이블: ', table_name, "
    f"'\\n샘플값(최대 10개): ', samples_str, '{rt.sqllit(PROMPT_TAIL)}')")
sql(f"""INSERT INTO {FQ}.llm_col_raw
  SELECT 'gpt-oss-120b', table_name, column_name,
         ai_query('{GPTOSS_EP}', {concat_expr}) AS llm_raw
  FROM {FQ}.column_profile""")
print("  gpt-oss column LLM done:",
      sql(f"SELECT count(*) FROM {FQ}.llm_col_raw WHERE backend='gpt-oss-120b'")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (컬럼 2/3): qwen 컬럼 분류 — Python 직렬 호출 (선택)
# MAGIC `LLM_BACKENDS` 에 qwen이 있을 때만 실행됩니다(없으면 자동 스킵 → gpt-oss 단독). 실패는 수집 후
# MAGIC 1회 일괄 재시도, 그래도 남으면 hard fail(오염 적재 금지 — fail-closed).

# COMMAND ----------

import time
from column_prompt import prompt_text

for backend in [b for b in LLM_BACKENDS if b != "gpt-oss-120b"]:
    rows = sql(f"SELECT table_name, column_name, samples_str FROM {FQ}.column_profile")[1]
    out, failed, t0 = [], [], time.time()
    for i, (t, c, samples) in enumerate(rows):
        p = prompt_text(t, c, samples or "")
        try:
            out.append([t, c, llm_client(p, backend=backend, max_tokens=200)])
        except Exception as e:
            failed.append((t, c, p, str(e)[:120]))
        if (i + 1) % 10 == 0:
            print(f"    {backend} {i+1}/{len(rows)} ({time.time()-t0:.0f}s, fail={len(failed)})")
    if failed:
        print(f"    {backend} 1차 실패 {len(failed)}건 → 일괄 재시도")
        still = []
        for t, c, p, _ in failed:
            try:
                out.append([t, c, llm_client(p, backend=backend, max_tokens=200)])
            except Exception as e:
                still.append((t, c, str(e)[:120]))
        if still:
            raise RuntimeError(
                f"{backend} 컬럼 호출 {len(still)}/{len(rows)}건 최종 실패 — 적재 중단(silent 오염 방지). "
                f"예: {still[0]}")
    rows_out = [{"backend": backend, "table_name": t, "column_name": c, "llm_raw": raw}
                for t, c, raw in out]
    load_rows(f"{FQ}.llm_col_raw", rows_out, where=f"backend='{backend}'")
    print(f"  {backend} column LLM done ({len(out)} cols, {time.time()-t0:.0f}s)")
print(f"  (qwen 미설정 시 위 루프는 스킵됨 — LLM_BACKENDS={LLM_BACKENDS})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (컬럼 3/3): 파싱 + S2 hybrid 예측 적재
# MAGIC `llm_col_raw` → `llm_col`(is_pii/category/conf 파싱). 파싱 불능 응답은 가시화하고 5% 초과 시 hard
# MAGIC fail. 그다음 **S2 hybrid**(`0.4*rule + 0.6*llm`)를 col_predictions에 적재합니다. (S5 LLM-only
# MAGIC 컬럼 예측은 동일 `llm_col` 에서 50_S5 노트북이 적재합니다 — 원천은 공유, 단계만 분리.)

# COMMAND ----------

# 통합 파싱: llm_col_raw → llm_col
sql(f"""CREATE OR REPLACE TABLE {FQ}.llm_col AS
  SELECT backend, table_name, column_name,
    coalesce(j.is_pii, false) AS llm_is_pii,
    coalesce(nullif(upper(j.category), ''), 'NON_PII') AS llm_category,
    coalesce(j.confidence, 0.5) AS llm_conf, llm_raw
  FROM {FQ}.llm_col_raw
  LATERAL VIEW OUTER explode(array(
    from_json(regexp_extract(llm_raw, '(?s)\\\\{{.*\\\\}}', 0),
      'is_pii BOOLEAN, category STRING, confidence DOUBLE, reason STRING'))) t AS j""")

# parse_fail 카운터: coalesce(false)로 조용히 NON_PII 처리되는 것을 가시화 + 과다 시 hard fail
_pf = sql(f"""SELECT backend, count(*) AS n_unparsed FROM {FQ}.llm_col_raw
  WHERE llm_raw IS NULL OR from_json(regexp_extract(llm_raw, '(?s)\\\\{{.*\\\\}}', 0),
    'is_pii BOOLEAN, category STRING, confidence DOUBLE, reason STRING') IS NULL
  GROUP BY backend""")[1]
if _pf:
    _tot = int(sql(f"SELECT count(*) FROM {FQ}.llm_col_raw")[1][0][0])
    for be, n in _pf:
        print(f"  WARNING: {be} 컬럼 응답 파싱 실패 {n}건 (coalesce로 NON_PII 처리됨)")
        if int(n) > _tot * 0.05:
            raise RuntimeError(f"{be} 컬럼 응답 파싱 실패 {n}/{_tot} — 5% 초과, 중단")
else:
    print("  parse_fail=0 (컬럼 트랙)")

# S2 hybrid: 0.4*rule + 0.6*llm
# 가중 결합 이유: rule_s(정규식)는 정밀(precision)하나 비정형 PII를 놓치고, llm_s(LLM)는 의미를
# 이해해 재현(recall)이 높음 → 둘을 합쳐 상호 보완. 비정형(이름·주소)이 많은 컬럼 판정에서 LLM이
# 더 결정적이라 0.6으로 더 높게 가중(rule 0.4). 임계값 >= 0.5는 P/R 균형점(과탐·누락을 함께 억제).
sql(f"DELETE FROM {FQ}.col_predictions WHERE stage='S2'")
sql(f"""INSERT INTO {FQ}.col_predictions
  WITH j AS (
    SELECT l.backend, l.table_name, l.column_name,
      CASE WHEN r.rule_is_pii THEN r.rule_conf ELSE 0.0 END AS rule_s,
      CASE WHEN l.llm_is_pii THEN l.llm_conf ELSE (1.0 - l.llm_conf) END AS llm_s,
      r.rule_category, l.llm_category, l.llm_is_pii
    FROM {FQ}.llm_col l JOIN {FQ}.rule_results r USING (table_name, column_name))
  SELECT 'S2' AS stage, backend, table_name, column_name,
    (round(0.4*rule_s + 0.6*llm_s, 3) >= 0.5) AS pred_is_pii,
    CASE WHEN round(0.4*rule_s + 0.6*llm_s,3) < 0.5 THEN 'NON_PII'
         WHEN llm_is_pii THEN llm_category ELSE rule_category END AS pred_category
  FROM j""")
print("  col_predictions(S2) written")
display(sql(f"SELECT backend, count(*) n FROM {FQ}.col_predictions WHERE stage='S2' GROUP BY backend")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## ============ span 트랙 ============
# MAGIC ## 이 단계가 하는 일 (span 1/3): gpt-oss span 추출 — ai_query 전체 코퍼스
# MAGIC `span_llm.SPAN_PROMPT` 를 `rt.sqllit()` 로 이스케이프해 `ai_query`로 text_corpus 전체의 PII span
# MAGIC 부분문자열을 추출, `span_llm_raw` 에 적재합니다. (S5 span도 이 raw를 재사용합니다.)

# COMMAND ----------

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_llm_raw
  (backend STRING, doc_id STRING, llm_raw STRING)""")
sql(f"DELETE FROM {FQ}.span_llm_raw WHERE backend='gpt-oss-120b'")

_expr = f"ai_query('{GPTOSS_EP}', concat('{rt.sqllit(SPAN_PROMPT)}', text))"
sql(f"""INSERT INTO {FQ}.span_llm_raw
  SELECT 'gpt-oss-120b', doc_id, {_expr} FROM {FQ}.text_corpus""")
print("  gpt-oss span raw done:",
      sql(f"SELECT count(*) FROM {FQ}.span_llm_raw WHERE backend='gpt-oss-120b'")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (span 2/3): qwen span 추출 — Python 표본 (선택)
# MAGIC qwen은 throughput 제약으로 `S5_QWEN_SAMPLE_DOCS`(=300) 표본만 처리합니다(`xxhash64(doc_id)` 로
# MAGIC 결정적 표집). 실패는 1회 재시도 후 raw·coverage에서 제외(분모 정직화), 5% 초과 시 hard fail.
# MAGIC qwen 미설정 시 자동 스킵.

# COMMAND ----------

def _infer_qwen_spans(backend, docs):
    """qwen 1개 백엔드의 span 추론 + 1회 재시도를 수행하고 적재용 rows 리스트를 반환.

    동일 백엔드 루프 본문을 가독성 위해 추출한 헬퍼(동작 불변). 4가지 제어 흐름:
      1) 정상: 각 doc을 llm_client(SPAN_PROMPT+text, max_tokens=400)로 호출해 rows에 누적.
      2) 진행 출력: 50건마다 처리량·누적 실패 수를 출력(throughput 가시화).
      3) 1회 재시도: 1차 실패분만 모아 한 번 더 호출(일시적 오류 흡수).
      4) fail-closed(>5%): 재시도 후에도 실패가 5% 초과면 RuntimeError로 중단(오염 적재 금지);
         5% 이하 잔여 실패는 WARNING 출력 후 raw·coverage에서 제외(분모 정직화).
    """
    rows, failed, t0 = [], [], time.time()
    for i, (doc, text) in enumerate(docs):
        try:
            rows.append({"backend": backend, "doc_id": doc,
                         "llm_raw": llm_client(SPAN_PROMPT + text, backend=backend, max_tokens=400)})
        except Exception as e:
            failed.append((doc, text, str(e)[:120]))
        if (i + 1) % 50 == 0:
            print(f"    {backend} span {i+1}/{len(docs)} ({time.time()-t0:.0f}s, fail={len(failed)})")
    if failed:
        print(f"    {backend} span 1차 실패 {len(failed)}건 → 일괄 재시도")
        still = []
        for doc, text, _ in failed:
            try:
                rows.append({"backend": backend, "doc_id": doc,
                             "llm_raw": llm_client(SPAN_PROMPT + text, backend=backend, max_tokens=400)})
            except Exception as e:
                still.append((doc, str(e)[:120]))
        if len(still) > len(docs) * 0.05:
            raise RuntimeError(f"{backend} span 최종 실패 {len(still)}/{len(docs)} — 5% 초과, 중단. 예: {still[0]}")
        if still:
            print(f"    WARNING: {backend} span 최종 실패 {len(still)}건 — raw·coverage에서 제외: "
                  f"{[d for d, _ in still]}")
    print(f"  {backend} span raw done ({len(rows)}/{len(docs)} docs, {time.time()-t0:.0f}s)")
    return rows

# COMMAND ----------

# backend -> 처리된 doc_id 목록(coverage 분모). gpt-oss는 전체.
covered = {"gpt-oss-120b": [d for (d,) in sql(f"SELECT doc_id FROM {FQ}.text_corpus")[1]]}

for backend in [b for b in LLM_BACKENDS if b != "gpt-oss-120b"]:
    # qwen은 throughput 제약 → S5_QWEN_SAMPLE_DOCS 만큼만 결정적 표집(xxhash64 정렬)해 처리량을 한정.
    docs = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT {S5_QWEN_SAMPLE_DOCS}")[1]
    rows = _infer_qwen_spans(backend, docs)
    load_rows(f"{FQ}.span_llm_raw", rows, where=f"backend='{backend}'")
    covered[backend] = [r["doc_id"] for r in rows]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (span 3/3): raw → S2 span 예측 (정규식 ∪ LLM)
# MAGIC `recover_spans()` 로 LLM 부분문자열의 char offset을 복원(환각=드롭)하고, `merge_regex_llm()` 로
# MAGIC 정규식 span ∪ LLM span을 합쳐 `span_predictions`(stage='S2')에 적재합니다. coverage도 backend별
# MAGIC 처리 doc만 기록(공정 recall). (S5 LLM-only span은 동일 raw에서 50_S5가 적재.)
# MAGIC
# MAGIC > 왜 이렇게? **LLM은 글자 위치(offset)를 정확히 세지 못하므로** 값(부분문자열)만 신뢰해 `text.find`로
# MAGIC > 위치를 되찾고(못 찾으면 환각으로 보고 드롭), 합칠 때는 **정규식을 우선**(정형 PII에 더 정밀)하며 LLM은
# MAGIC > 겹치지 않는 부분만 채웁니다.
# MAGIC >
# MAGIC > 또한 `recover_spans` 는 값을 **긴 것부터(longest-value-first)** 위치 점유해, 짧은 항목(예: `"1234"`)이
# MAGIC > 더 긴 정답 값(전화번호 등)의 자리를 먼저 차지해 드롭시키는 **shadowing(잠식)을 방지**합니다.

# COMMAND ----------

text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]}


def _row(s):
    return {"start_char": s["start"], "end_char": s["end"], "entity_type": s["entity_type"],
            "pii_value": s["pii_value"], "score": s["score"]}


for backend in LLM_BACKENDS:
    raw = sql(f"SELECT doc_id, llm_raw FROM {FQ}.span_llm_raw WHERE backend='{backend}'")[1]
    s2, n_parse_fail = [], 0
    for doc, llm_raw in raw:
        text = text_map.get(doc, "")
        parsed = extract_json(llm_raw)
        items = parsed if isinstance(parsed, list) else (parsed.get("spans") if isinstance(parsed, dict) else None)
        if items is None:
            n_parse_fail += 1
            items = []
        llm_spans = recover_spans(text, items)
        for s in merge_regex_llm(regex_spans(text), llm_spans):
            s2.append({"stage": "S2", "backend": backend, "doc_id": doc, **_row(s)})
    sql(f"DELETE FROM {FQ}.span_predictions WHERE stage='S2' AND backend='{backend}'")
    if s2:
        load_rows(f"{FQ}.span_predictions", s2)
    sql(f"DELETE FROM {FQ}.span_coverage WHERE stage='S2' AND backend='{backend}'")
    load_rows(f"{FQ}.span_coverage",
              [{"stage": "S2", "backend": backend, "doc_id": d} for d in covered[backend]])
    print(f"  {backend}: S2={len(s2)} spans (parse_fail={n_parse_fail}/{len(raw)})")
    if raw and n_parse_fail > len(raw) * 0.05:
        raise RuntimeError(f"{backend} span 응답 파싱 실패 {n_parse_fail}/{len(raw)} — 5% 초과, 중단")

# COMMAND ----------

# MAGIC %md ## 결과확인: S2 span_predictions 셀별 분포 + 유형별

# COMMAND ----------

display(sql(f"""SELECT stage, backend, count(*) n
FROM {FQ}.span_predictions WHERE stage='S2' GROUP BY stage, backend ORDER BY backend""")[1])

# COMMAND ----------

# MAGIC %md ### PERSON/ADDRESS가 S2에서 회수됨 (S1=0건 → S2>0) — LLM 효과 확인

# COMMAND ----------

display(sql(f"""SELECT backend, entity_type, count(*) n
FROM {FQ}.span_predictions WHERE stage='S2' AND entity_type IN ('PERSON','ADDRESS')
GROUP BY backend, entity_type ORDER BY backend, entity_type""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 비교 (러닝 스코어보드)
# MAGIC | 단계 | span exact F1 | 라우팅 | 비용/1k | 한 줄 |
# MAGIC |---|---|---|---|---|
# MAGIC | S1 | 0.73 | — | $0 | 정형만 — 이름·주소 0% |
# MAGIC | **S2 ← 지금 여기** | **0.9448** | **100%** | **$0.132** | LLM이 비정형 회수(정확 급상승) — 단 전 문서 LLM이라 비용↑ |
# MAGIC | S3 (예정) | — | — | — | NER cascade로 비용↓ |
# MAGIC | S4 (예정) | — | — | — | 파인튜닝 NER = 최적 |
# MAGIC | S5 (예정) | — | — | — | LLM 단독(대조군) |
# MAGIC
# MAGIC > 정확도는 크게 올랐지만 **모든 문서를 LLM에 보내 비용이 최대**입니다 — 다음 단계(S3)가 NER로 LLM 호출을 줄입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`30_S3_pattern_ner_llm/S3_ner_base_cascade`](../30_S3_pattern_ner_llm/S3_ner_base_cascade)** [ML 클러스터] — NER + 저신뢰 cascade로 비용을 낮춥니다. 점수는 90_eval에서 산출.
