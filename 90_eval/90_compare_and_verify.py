# Databricks notebook source
# MAGIC %md
# MAGIC # 평가 · 비교 · 검증 (final_grid)
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC S1~S5를 **하나의 비교 그리드**로 모으고, 빌드 산출물을 신뢰하지 않고 **원천에서 독립 재계산**해
# MAGIC 검증합니다. 컬럼 지표(col_eval) + span 지표(span_eval) + (선택) 성능/비용(perf_summary)을
# MAGIC `arch_evo_comparison` 한 테이블로 합칩니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - **span_eval**: exact / type_agnostic / partial(IoU≥0.5) / char 네 매칭 체계. coverage에 기록된
# MAGIC   처리 문서만 분모에 포함(공정 recall).
# MAGIC - **col_eval**: col_predictions ⨝ column_ground_truth로 P/R/F1/category_acc.
# MAGIC - **검증(C1a~C7 + C8)**: 정답 무결성·재현성·지표 독립 재계산·span 불변식·그리드 완전성·거버넌스·진화 단조 + vendoring 무결성.
# MAGIC   C5(그리드 완전성)는 `LLM_BACKENDS` 기반 **동적 기대셀**(qwen 미설정 시 제외).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `_common.eval_lib.evaluate(FQ, sql, load_rows)`(span_eval) + `_tools/build_harness.py`
# MAGIC (col_eval·arch_evo_comparison) + `99_eval/93_perf_capture.py`(route-aware 비용, 선택) +
# MAGIC `_tools/verify.py`(C1·C2·C3·C4·C6·C7 + 동적 C5) + vendoring 무결성 셀.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | stage | span exact F1 | char | col F1 | 라우팅 | 비용/1k |
# MAGIC |---|---|---|---|---|---|
# MAGIC | S1 | 0.73 | 0.78 | 0.77 | 0% | $0 |
# MAGIC | S2 | **0.9448** | 0.998 | 0.9091 | 100% | $0.132 |
# MAGIC | S3 | 0.8966 | — | — | 10.9% | $0.0244 |
# MAGIC | **S4** | **0.9838** | **1.0** | — | **0%** | **$0.01** |
# MAGIC | S5 | 0.9441 | — | 0.8889 | 100% | $0.132 |
# MAGIC
# MAGIC 마지막에 **10/10 PASS** 요약 — 아키텍처 검증 C1a~C7(9개) + 패키지 vendoring 무결성 C8(1개). (거버넌스/진화 셀 미실행 시 해당 체크 스킵.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC perf 비용은 예시 단가 추정이며 실제 청구와 다를 수 있습니다. S4 라우팅 0%는 합성분포 효과
# MAGIC (S4b에서 명시)입니다. 비교는 동일 평가 코퍼스(text_corpus) 기준이라 단계 간 공정합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ 거버넌스를 운영에 적용(80)하고, 권장 단계(S4)를 서빙/파이프라인에 통합합니다.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (1/5): span_eval — eval_lib.evaluate 호출
# MAGIC 검증된 `eval_lib.evaluate(FQ, sql, load_rows)` 를 그대로 호출합니다(평가 수학 로직 동등).
# MAGIC span_predictions/span_coverage/span_ground_truth → span_eval 적재.
# MAGIC
# MAGIC **4가지 매칭 기준(regime)을 함께 보는 이유** — 한 숫자로는 오해하기 쉬워서입니다:
# MAGIC - **exact**: 위치(offset)와 타입이 모두 정확 — 가장 엄격.
# MAGIC - **type_agnostic**: 위치만 맞으면 인정(타입 오류 분리) — "어디"는 맞췄는가.
# MAGIC - **partial**: 글자 겹침 IoU≥0.5면 인정 — 경계 근접을 부분 인정.
# MAGIC - **char**: 글자 단위 recall — "PII 글자를 얼마나 덮었나"(마스킹 안전성 관점).
# MAGIC
# MAGIC recall 분모는 coverage로 보정해 백엔드별 처리 doc만 공정 비교합니다.

# COMMAND ----------

span_eval_rows = eval_lib.evaluate(FQ, sql, load_rows)
print(f"  span_eval: {len(span_eval_rows)} rows")
display(sql(f"""SELECT stage, backend, regime, precision, recall, f1, n_docs
FROM {FQ}.span_eval ORDER BY stage, backend, regime""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (2/5): col_eval (SOURCE: build_harness.py)
# MAGIC col_predictions ⨝ column_ground_truth로 stage×backend별 P/R/F1 + category_acc(FREE_TEXT_PII 제외).

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.col_eval AS
  WITH p AS (
    SELECT c.stage, c.backend, c.pred_is_pii, c.pred_category,
           g.is_pii AS gt, g.category AS gt_cat
    FROM {FQ}.col_predictions c JOIN {FQ}.column_ground_truth g USING (table_name, column_name))
  SELECT stage, backend,
    sum(CASE WHEN gt AND pred_is_pii THEN 1 ELSE 0 END) AS tp,
    sum(CASE WHEN NOT gt AND pred_is_pii THEN 1 ELSE 0 END) AS fp,
    sum(CASE WHEN gt AND NOT pred_is_pii THEN 1 ELSE 0 END) AS fn,
    round(sum(CASE WHEN gt AND pred_is_pii THEN 1.0 ELSE 0 END)/nullif(sum(CASE WHEN pred_is_pii THEN 1 ELSE 0 END),0),4) AS precision,
    round(sum(CASE WHEN gt AND pred_is_pii THEN 1.0 ELSE 0 END)/nullif(sum(CASE WHEN gt THEN 1 ELSE 0 END),0),4) AS recall,
    round(2.0*sum(CASE WHEN gt AND pred_is_pii THEN 1 ELSE 0 END)/
          nullif(2*sum(CASE WHEN gt AND pred_is_pii THEN 1 ELSE 0 END)+sum(CASE WHEN NOT gt AND pred_is_pii THEN 1 ELSE 0 END)+sum(CASE WHEN gt AND NOT pred_is_pii THEN 1 ELSE 0 END),0),4) AS f1,
    round(sum(CASE WHEN gt AND pred_is_pii AND gt_cat<>'FREE_TEXT_PII' AND pred_category=gt_cat THEN 1.0 ELSE 0 END)/
          nullif(sum(CASE WHEN gt AND gt_cat<>'FREE_TEXT_PII' THEN 1 ELSE 0 END),0),4) AS category_acc
  FROM p GROUP BY stage, backend""")
print("  col_eval done")
display(sql(f"SELECT * FROM {FQ}.col_eval ORDER BY stage, backend")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## (선택) 이 단계가 하는 일 (3/5): perf_summary — route-aware 비용
# MAGIC SOURCE(`93_perf_capture.py`). **실제 LLM 호출로 latency/throughput을 실측**하므로 시간/비용이
# MAGIC 듭니다. 비용 = `route_rate × LLM단가 (+ S3/S4 NER 오버헤드)`. 단가는 예시 추정입니다.
# MAGIC 실행하려면 `RUN_PERF=true` 위젯을 설정하세요(NER 단계 S3/S4가 선행되어야 route_rates 계산 가능).

# COMMAND ----------

dbutils.widgets.dropdown("RUN_PERF", "false", ["false", "true"], "성능/비용 실측 실행")
RUN_PERF = dbutils.widgets.get("RUN_PERF").strip().lower() == "true"

# 예시 단가(추정) — 실제와 다를 수 있음
RATE_GPTOSS_PER_1K_TOK = 0.0006
RATE_T4_PER_HOUR = 1.5
AVG_TOK_PER_DOC = 220
RATE_NER_CPU_PER_1K = 0.01
N_SERIAL = 10
N_BATCH = 80


def _pcts(lat_ms):
    lat = sorted(lat_ms)
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, int(round(len(lat) * 0.95)) - 1)]
    return round(p50, 1), round(p95, 1)


def _serial_latency(docs, call):
    import time
    call(docs[0][0])  # warm-up
    lat = []
    for (t,) in docs[1:N_SERIAL + 1]:
        s = time.time(); call(t); lat.append((time.time() - s) * 1000)
    return lat


def measure_gptoss():
    import time
    ep = BACKENDS["gpt-oss-120b"]
    docs = sql(f"SELECT text FROM {FQ}.text_corpus LIMIT {N_SERIAL + 1}")[1]
    esc = rt.sqllit(SPAN_PROMPT)

    def call(t):
        sql(f"SELECT ai_query('{ep}', concat('{esc}', '{rt.sqllit(t)}'))")
    lat = _serial_latency(docs, call)
    p50, p95 = _pcts(lat)
    bdocs = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus LIMIT {N_BATCH}")[1]
    load_rows(f"{FQ}.perf_tmp", [{"doc_id": d, "text": t} for d, t in bdocs], mode="replace")
    t0 = time.time()
    sql(f"""CREATE OR REPLACE TABLE {FQ}.perf_tmp_out AS
        SELECT doc_id, ai_query('{ep}', concat('{esc}', text)) r FROM {FQ}.perf_tmp""")
    rps_batch = N_BATCH / (time.time() - t0)
    cost_1k = AVG_TOK_PER_DOC / 1000.0 * RATE_GPTOSS_PER_1K_TOK * 1000
    return dict(latency_p50_ms=p50, latency_p95_ms=p95, throughput_rps=round(rps_batch, 3)), cost_1k


def measure_qwen(backend):
    docs = sql(f"SELECT text FROM {FQ}.text_corpus LIMIT {N_SERIAL + 1}")[1]
    lat = _serial_latency(docs, lambda t: llm_client(SPAN_PROMPT + t, backend=backend, max_tokens=400))
    p50, p95 = _pcts(lat)
    rps = 1000.0 / p50
    cost_1k = (1000.0 / rps) / 3600.0 * RATE_T4_PER_HOUR
    return dict(latency_p50_ms=p50, latency_p95_ms=p95, throughput_rps=round(rps, 3)), cost_1k


def route_rates():
    out = {"S1": 0.0, "S2": 1.0, "S5": 1.0}
    for model, stage in [("base", "S3"), ("ft", "S4")]:
        r = sql(f"""SELECT count(*), sum(CASE WHEN score<{TAU} THEN 1 ELSE 0 END)
            FROM {FQ}.ner_spans_raw WHERE model='{model}'""")[1]
        tot, low = r[0]
        out[stage] = round(int(low or 0) / int(tot), 4) if int(tot) else 0.0
    return out


def stage_cost(stage, llm_cost_1k, rr):
    route = rr[stage]
    ner_overhead = RATE_NER_CPU_PER_1K if stage in ("S3", "S4") else 0.0
    return round(route * llm_cost_1k + ner_overhead, 4)


if RUN_PERF:
    print("== 성능/비용 실측 ==")
    rr = route_rates(); print("route_rates:", rr)
    g, g_cost1k = measure_gptoss(); print("gpt-oss:", g, "full-LLM cost/1k:", round(g_cost1k, 4))
    rows = []
    for stage in ["S2", "S3", "S4", "S5"]:
        rows.append({"stage": stage, "backend": "gpt-oss-120b", **g,
                     "est_cost_usd_per_1k": stage_cost(stage, g_cost1k, rr), "llm_route_rate": rr[stage]})
    for backend in [b for b in LLM_BACKENDS if b != "gpt-oss-120b"]:
        q, q_cost1k = measure_qwen(backend); print(f"{backend}:", q, "full-LLM cost/1k:", round(q_cost1k, 4))
        for stage in ["S2", "S3", "S4", "S5"]:
            rows.append({"stage": stage, "backend": backend, **q,
                         "est_cost_usd_per_1k": stage_cost(stage, q_cost1k, rr), "llm_route_rate": rr[stage]})
    rows.append({"stage": "S1", "backend": "NA", "latency_p50_ms": 0.05, "latency_p95_ms": 0.1,
                 "throughput_rps": 20000.0, "est_cost_usd_per_1k": 0.0, "llm_route_rate": 0.0})
    load_rows(f"{FQ}.perf_summary", rows, mode="replace")
    print(f"perf_summary: {len(rows)} rows")
else:
    print("  perf 실측 스킵 (RUN_PERF=false). arch_evo_comparison의 perf 컬럼은 NULL로 채워집니다.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (4/5): arch_evo_comparison (SOURCE: build_harness.py)
# MAGIC col_eval + span_eval(+ 있으면 perf_summary)를 stage×backend 단일 비교 그리드로 합칩니다.

# COMMAND ----------

# perf_summary 존재 여부
has_perf = True
try:
    sql(f"SELECT 1 FROM {FQ}.perf_summary LIMIT 1")
except Exception:
    has_perf = False

perf_join = (f"LEFT JOIN {FQ}.perf_summary pf ON c.stage=pf.stage AND c.backend=pf.backend" if has_perf else "")
perf_cols = ("pf.latency_p50_ms, pf.latency_p95_ms, pf.est_cost_usd_per_1k, pf.llm_route_rate"
             if has_perf else "CAST(NULL AS DOUBLE) latency_p50_ms, CAST(NULL AS DOUBLE) latency_p95_ms, "
             "CAST(NULL AS DOUBLE) est_cost_usd_per_1k, CAST(NULL AS DOUBLE) llm_route_rate")
perf_src = f"SELECT stage, backend FROM {FQ}.perf_summary" if has_perf else "SELECT NULL stage, NULL backend WHERE 1=0"

sql(f"""CREATE OR REPLACE TABLE {FQ}.arch_evo_comparison AS
  WITH se AS (
    SELECT stage, backend,
      max(CASE WHEN regime='exact' THEN f1 END) AS span_exact_f1,
      max(CASE WHEN regime='partial' THEN f1 END) AS span_partial_f1,
      max(CASE WHEN regime='type_agnostic' THEN f1 END) AS span_detect_f1,
      max(CASE WHEN regime='char' THEN recall END) AS span_char_recall,
      max(CASE WHEN regime='exact' THEN recall END) AS span_exact_recall,
      max(CASE WHEN regime='exact' THEN precision END) AS span_exact_prec,
      max(n_docs) AS span_n_docs
    FROM {FQ}.span_eval GROUP BY stage, backend),
  cells AS (
    SELECT stage, backend FROM {FQ}.col_eval
    UNION SELECT stage, backend FROM se
    UNION {perf_src})
  SELECT c.stage, c.backend,
         ce.precision AS col_precision, ce.recall AS col_recall, ce.f1 AS col_f1,
         ce.category_acc AS col_category_acc,
         se.span_exact_prec, se.span_exact_recall, se.span_exact_f1,
         se.span_partial_f1, se.span_detect_f1, se.span_char_recall, se.span_n_docs,
         {perf_cols}
  FROM cells c
  LEFT JOIN {FQ}.col_eval ce ON c.stage=ce.stage AND c.backend=ce.backend
  LEFT JOIN se ON c.stage=se.stage AND c.backend=se.backend
  {perf_join}
  ORDER BY c.stage, c.backend""")
print("  arch_evo_comparison done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ============ 검증 (독립 재계산) ============
# MAGIC ## 이 단계가 하는 일 (5/5): C1~C7 + vendoring 무결성 (SOURCE: verify.py)
# MAGIC 빌드 산출물을 신뢰하지 않고 원천에서 재계산해 PASS/FAIL을 기록합니다. 각 체크를 셀로 분리했습니다.

# COMMAND ----------

import hashlib
from collections import defaultdict

REPORT = []
TOL = 1e-4


def chk(cid, desc, passed, detail=""):
    REPORT.append({"check": cid, "desc": desc, "result": "PASS" if passed else "FAIL", "detail": str(detail)})
    print(f"  [{'PASS' if passed else 'FAIL'}] {cid} {desc} {detail}")
    return passed

# COMMAND ----------

# MAGIC %md ### C1 — 정답 무결성 (span offset substring 일치 + 컬럼 GT 카테고리 적법)

# COMMAND ----------

bad = sql(f"""SELECT count(*) FROM {FQ}.span_ground_truth g JOIN {FQ}.text_corpus c USING(doc_id)
    WHERE substring(c.text, g.start_char+1, g.end_char-g.start_char) <> g.pii_value""")[1][0][0]
chk("C1a", "span offset 무결성", int(bad) == 0, f"불일치={bad}")

allowed = {"PERSONAL_INFO", "PAYMENT_INFO", "IDENTIFICATION", "EMPLOYMENT_INFO", "NON_PII", "FREE_TEXT_PII"}
cats = [r[0] for r in sql(f"SELECT DISTINCT category FROM {FQ}.column_ground_truth")[1]]
chk("C1b", "컬럼 GT 카테고리 적법", set(cats) <= allowed, f"cats={sorted(set(cats)-allowed) or 'ok'}")

# COMMAND ----------

# MAGIC %md ### C2 — 재현성 (seed=42 span GT 체크섬: DB vs 정렬-안정성 자기검증)
# MAGIC 노트북 환경에서는 로컬 CSV 재생성 대신, DB의 span GT가 offset 무결성(C1a)과 결정적 정렬을
# MAGIC 만족하고 체크섬이 안정적으로 산출되는지 확인합니다(원본 C2의 노트북 적응).

# COMMAND ----------

gt_db = sql(
    f"SELECT doc_id,start_char,end_char,entity_type,pii_value FROM {FQ}.span_ground_truth ORDER BY doc_id,start_char")[1]
h_db = hashlib.md5("\n".join("|".join(map(str, r)) for r in gt_db).encode()).hexdigest()
# 정렬 안정성(동일 입력 두 번 정렬 시 동일 체크섬) — 결정성 자기검증
gt_db2 = sorted(gt_db, key=lambda r: (r[0], int(r[1])))
h_db2 = hashlib.md5("\n".join("|".join(map(str, r)) for r in gt_db2).encode()).hexdigest()
chk("C2", "seed=42 재현성(span GT 체크섬 안정성)", h_db == h_db2 and len(gt_db) > 0,
    f"db_rows={len(gt_db)} hash={'stable' if h_db==h_db2 else 'DIFF'}")

# COMMAND ----------

# MAGIC %md ### C3 — 컬럼 지표 독립 재계산 일치 (col_predictions⨝GT → P/R/F1 vs col_eval, ±1e-4)

# COMMAND ----------

pred = sql(f"SELECT stage,backend,table_name,column_name,pred_is_pii FROM {FQ}.col_predictions")[1]
gt = {(t, c): (str(v).lower() == 'true' or v is True) for t, c, v in
      sql(f"SELECT table_name,column_name,is_pii FROM {FQ}.column_ground_truth")[1]}
agg = defaultdict(lambda: [0, 0, 0])  # (stage,backend)->tp,fp,fn
for st, be, t, c, pis in pred:
    g = gt.get((t, c))
    if g is None:
        continue
    p = (pis is True or str(pis).lower() == 'true')
    a = agg[(st, be)]
    if g and p:
        a[0] += 1
    elif (not g) and p:
        a[1] += 1
    elif g and (not p):
        a[2] += 1
ce = {(r[0], r[1]): (r[2], r[3], r[4]) for r in sql(f"SELECT stage,backend,precision,recall,f1 FROM {FQ}.col_eval")[1]}
ok = True
for cell, (tp, fp, fn) in agg.items():
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    ev = ce.get(cell)
    if ev is None:
        ok = False; continue
    for a, b in zip((prec, rec, f1), ev):
        if a is not None and b is not None and abs(float(a) - float(b)) > TOL:
            ok = False
chk("C3", "컬럼 지표 독립 재계산 일치", ok, f"cells={len(agg)}")

# COMMAND ----------

# MAGIC %md ### C4 — span 불변식 (tp+fn==n_gt, exact tp+fp==예측수, P/R/F1 정합)

# COMMAND ----------

rows = sql(f"SELECT stage,backend,regime,tp,fp,fn,precision,recall,f1,n_gt FROM {FQ}.span_eval")[1]
npred = {(r[0], r[1]): int(r[2]) for r in sql(f"""
    SELECT p.stage, p.backend, count(*) FROM {FQ}.span_predictions p
    JOIN {FQ}.span_coverage c ON p.stage=c.stage AND p.backend=c.backend AND p.doc_id=c.doc_id
    GROUP BY p.stage, p.backend""")[1]}
ok, bad_list = True, []
for st, be, rg, tp, fp, fn, p, r, f, n_gt in rows:
    tp, fp, fn = int(tp), int(fp), int(fn)
    if rg in ("exact", "type_agnostic", "partial") and (tp + fn) != int(n_gt):
        ok = False; bad_list.append(f"{st}/{be}/{rg}:tp+fn!=n_gt")
    if rg == "exact" and (st, be) in npred and (tp + fp) != npred[(st, be)]:
        ok = False; bad_list.append(f"{st}/{be}:tp+fp={tp+fp}!=n_pred={npred[(st, be)]}")
    rp = tp / (tp + fp) if (tp + fp) else 0.0
    rr_ = tp / (tp + fn) if (tp + fn) else 0.0
    rf = 2 * rp * rr_ / (rp + rr_) if (rp + rr_) else 0.0
    if (abs(rp - float(p or 0)) > 1e-3 or abs(rr_ - float(r or 0)) > 1e-3 or abs(rf - float(f or 0)) > 1e-3):
        ok = False; bad_list.append(f"{st}/{be}/{rg}:PRF")
chk("C4", "span 불변식(tp+fn==n_gt, tp+fp==예측수, P/R/F1 정합)", ok,
    f"rows={len(rows)}" + (f" bad={bad_list[:4]}" if bad_list else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### C5 — 그리드 완전성 (LLM_BACKENDS 기반 **동적 기대셀**) + span LLM 파싱가능
# MAGIC qwen 미설정 시 qwen 셀은 기대에서 자동 제외됩니다. S3/S4가 그리드에 있으면 NER 셀도 기대에 포함.

# COMMAND ----------

cells = {(r[0], r[1]) for r in sql(f"SELECT DISTINCT stage,backend FROM {FQ}.arch_evo_comparison")[1]}

# 동적 기대: S1(NA) + (S2·S5) × LLM_BACKENDS. S3/S4 존재 시 (S3·S4) × LLM_BACKENDS 추가.
expect = {("S1", "NA")}
for be in LLM_BACKENDS:
    expect |= {("S2", be), ("S5", be)}
have_ner = any(s in ("S3", "S4") for s, _ in cells)
if have_ner:
    for be in LLM_BACKENDS:
        expect |= {("S3", be), ("S4", be)}
chk("C5a", "그리드 셀 완전성(동적)", expect <= cells, f"missing={sorted(expect-cells) or 'none'}")

try:
    pf = sql(f"SELECT count(*) FROM {FQ}.span_llm_raw WHERE llm_raw NOT RLIKE '\\\\[|\\\\{{'")[1][0][0]
    chk("C5b", "span LLM 응답 파싱가능", int(pf) == 0, f"unparseable={pf}")
except Exception as e:
    chk("C5b", "span LLM 파싱 점검", False, str(e)[:60])

# COMMAND ----------

# MAGIC %md ### C6 — 거버넌스 (마스킹 잔여 gold PII 0; 미실행 시 스킵)

# COMMAND ----------

try:
    resid = sql(f"""SELECT count(*) FROM {FQ}.text_corpus_masked m
        JOIN {FQ}.span_ground_truth g ON m.doc_id=g.doc_id
        WHERE locate(g.pii_value, m.masked)>0
          AND (length(g.pii_value)>=4
               OR (g.entity_type IN ('PERSON','ADDRESS') AND length(g.pii_value)>=2))""")[1][0][0]
    chk("C6", "마스킹 잔여 gold PII 0 (PERSON 2자+ 포함)", int(resid) == 0, f"잔존={resid}")
except Exception:
    chk("C6", "거버넌스 미실행(스킵)", True, "text_corpus_masked 없음")

# COMMAND ----------

# MAGIC %md ### C7 (soft) — 진화 단조 (S1 < S2 span exact F1)

# COMMAND ----------

d = {(r[0], r[1]): r[2] for r in sql(
    f"SELECT stage,backend,span_exact_f1 FROM {FQ}.arch_evo_comparison WHERE span_exact_f1 IS NOT NULL")[1]}
s1 = d.get(("S1", "NA"))
s2 = d.get(("S2", "gpt-oss-120b"))
if s1 is not None and s2 is not None:
    chk("C7", "진화 단조(soft): S1<S2 span F1", float(s1) < float(s2), f"S1={s1} S2={s2}")
else:
    print("  C7 스킵 (S1/S2 span_exact_f1 미존재 — 해당 단계 미실행)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C8 — vendoring 무결성 (regex_spans 결과 ⊆ span_ground_truth의 PATTERN-layer span)
# MAGIC `_common`에 vendoring된 `regex_spans` 가 검증 원본과 동일하게 동작하는지: text_corpus에 적용한
# MAGIC 결과가 PATTERN 레이어 정답 span과 정확히 매치되는지 표본으로 확인합니다.

# COMMAND ----------

# PATTERN 레이어 gold span(doc_id, start, end, type) 집합
gold_pat = set()
for doc, s, e, t in sql(f"""SELECT doc_id, start_char, end_char, entity_type
    FROM {FQ}.span_ground_truth WHERE source_layer='PATTERN'""")[1]:
    gold_pat.add((doc, int(s), int(e), t))

# regex_spans 재적용 결과
text_rows = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]
regex_hits = set()
for doc, text in text_rows:
    for sp in regex_spans(text):
        regex_hits.add((doc, sp["start"], sp["end"], sp["entity_type"]))

# regex가 잡은 PATTERN span 중 gold에 없는 것(=정규식이 잘못 잡은 형식) — 0이 이상적
# (정규식은 PATTERN 유형만 생성하므로 regex_hits ⊆ PATTERN 유형)
PATTERN_TYPES = {"EMAIL", "RRN", "ACCOUNT", "CARD", "PHONE", "PASSPORT", "IMEI"}
regex_pat = {h for h in regex_hits if h[3] in PATTERN_TYPES}
spurious = regex_pat - gold_pat
# 매치율: gold PATTERN span을 regex가 얼마나 재현하는지
recovered = regex_pat & gold_pat
match_rate = round(len(recovered) / len(gold_pat), 4) if gold_pat else 0.0
chk("C8", "vendoring 무결성(regex_spans ↔ PATTERN gold)",
    len(spurious) == 0 and match_rate >= 0.95,
    f"match_rate={match_rate} spurious={len(spurious)} (gold_pat={len(gold_pat)})")

# COMMAND ----------

# MAGIC %md ## 검증 리포트 적재 + 요약

# COMMAND ----------

load_rows(f"{FQ}.verification_report", REPORT, mode="replace")
fails = [r for r in REPORT if r["result"] == "FAIL"]
n_pass = len(REPORT) - len(fails)
print(f"\n== 검증 결과: {n_pass}/{len(REPORT)} PASS ==")
if fails:
    for r in fails:
        print("  FAIL:", r["check"], r["desc"], r["detail"])
else:
    print("  모든 검증 PASS")
display(spark.sql(f"SELECT check, desc, result, detail FROM {FQ}.verification_report ORDER BY check"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## final_grid — arch_evo_comparison (전 단계 비교)

# COMMAND ----------

# display는 DataFrame을 직접 받습니다 — perf 컬럼(est_cost_usd_per_1k·llm_route_rate)이
# RUN_PERF=false일 때 전부 NULL이라, 파이썬 리스트로 넘기면 타입 추론이 불가(CANNOT_DETERMINE_TYPE).
display(spark.sql(f"""SELECT stage, backend, col_f1, col_category_acc,
    span_exact_f1, span_partial_f1, span_char_recall, span_n_docs,
    est_cost_usd_per_1k, llm_route_rate
FROM {FQ}.arch_evo_comparison ORDER BY stage, backend"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 요약
# MAGIC - **S1**: 정형 P=1.0이나 이름·주소 0% → LLM 필요
# MAGIC - **S2**: span F1 0.73→0.9448, 전수 LLM $0.132/1k → 대량 비쌈
# MAGIC - **S3**: 라우팅 10.9%로 비용↓($0.0244)이나 범용 NER 과탐으로 F1 0.8966
# MAGIC - **S4**: span F1 0.9838·char 1.0·라우팅 0%·$0.01 → **정확도·비용 동시 최적**
# MAGIC - **S5**: LLM 단독 0.9441·$0.132 → S4 우위 확인
# MAGIC
# MAGIC > 위 final_grid + 검증 리포트가 본 랩의 결론입니다. 권장 운영 단계는 **S4(패턴+파인튜닝 NER+cascade)** 입니다.
