# Databricks notebook source
# MAGIC %md
# MAGIC # S3 — 패턴 + 사전학습 NER + LLM cascade  [ML 클러스터 노트북]
# MAGIC
# MAGIC > **클러스터 사양**: Databricks Runtime **17.3 ML (CPU)** 단일노드, **16-core**(예: Standard_D16ds_v5 / m5d.4xlarge). **GPU 불필요** — KoELECTRA-small CPU 추론. 이 노트북은 잡 제출 없이 ML 클러스터에 **attach해 셀을 직접 실행**합니다. (LLM cascade는 ai_query/serving 호출이라 클러스터 종류 무관.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [푸는 문제]
# MAGIC S2는 모든 문서를 LLM에 보내 비쌌습니다. S3는 **사전학습 한국어 NER**(KoELECTRA-small-v3-modu-ner)을
# MAGIC 1차 필터로 두어 이름·주소를 싸게 잡고, **NER 신뢰도가 낮은 span만 LLM으로 확인(cascade)** 해
# MAGIC 비용을 크게 낮춥니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - 정규식(PATTERN) ∪ NER(PERSON/ADDRESS) span 결합.
# MAGIC - cascade: NER score `< τ(0.70)` 인 저신뢰 span만 LLM에 "이게 진짜 PII냐" 묻고 keep/drop.
# MAGIC   고신뢰 span은 LLM을 거치지 않으므로 **LLM 호출량(=비용)이 라우팅율만큼만** 발생합니다.
# MAGIC - **fail-closed**: cascade 응답 파싱 불능/호출 실패는 drop 처리(PII로 위장 금지).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `_tools/ner_job_notebook.py`(base 추론부) + `_tools/build_ner.py`(S3 cascade·finalize·
# MAGIC column_verdict, `_parse_keep` 검증본). `ner_spans_raw(model='base')` 생성 → cascade →
# MAGIC `span_predictions`(S3) + `col_predictions`(S3).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | 지표 | 값 |
# MAGIC |---|---|
# MAGIC | span exact F1 | **0.8966** (S2 0.9448보다 낮음 = 범용 NER 과탐) |
# MAGIC | LLM 라우팅율 | **10.9%** (S2의 100% 대비 급감) |
# MAGIC | 비용 | **$0.0244 / 1k docs** |

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC 범용 사전학습 NER은 도메인 밖 토큰을 **과탐**합니다(상점명·일반명사를 이름으로). 비용은 내렸지만
# MAGIC 정확도(F1)는 S2보다 떨어집니다 — 정확도를 비용과 맞바꾼 셈입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ S4에서 **도메인 데이터로 NER을 파인튜닝**해 과탐을 없애고, 정확도·비용을 동시에 잡습니다.

# COMMAND ----------

# MAGIC %md ## 의존성 설치 (seqeval) + Python 재시작

# COMMAND ----------

# MAGIC %pip install -q seqeval

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 모델/상수 (SOURCE: ner_job_notebook.py)
# MAGIC base NER은 `Leo97/KoELECTRA-small-v3-modu-ner`. NER 레이어는 PERSON/ADDRESS만 담당(정형 PII는
# MAGIC 정규식 레이어). HF 파이프라인의 offset_mapping을 쓰므로 MeCab 불필요.

# COMMAND ----------

import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, pipeline)

BASE_NER = "Leo97/KoELECTRA-small-v3-modu-ner"
DEVICE = 0 if torch.cuda.is_available() else -1
print("device:", "cuda" if DEVICE == 0 else "cpu")

# 평가 코퍼스 로드
corpus = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]]
print(f"corpus={len(corpus)} docs")


def map_base_label(grp):
    g = (grp or "").upper()
    if g.startswith("PS") or g.startswith("PER"):
        return "PERSON"
    if g.startswith("LC") or g.startswith("LOC"):
        return "ADDRESS"
    return None  # OG/DT/기타는 NER GT 범위 밖 → 제외


def infer(nlp, label_map, score_floor=0.0):
    rows = []
    BATCH = 64
    texts = [t for _, t in corpus]
    ids = [d for d, _ in corpus]
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        try:
            results = nlp(chunk)
        except Exception as e:
            print("infer batch err", str(e)[:100]); results = [[] for _ in chunk]
        if isinstance(results, dict):
            results = [results]
        for did, ents in zip(ids[i:i + BATCH], results):
            for e in (ents or []):
                et = label_map(e.get("entity_group"))
                if et is None:
                    continue
                sc = float(e.get("score", 0.0))
                if sc < score_floor:
                    continue
                rows.append({"doc_id": did, "start_char": int(e["start"]),
                             "end_char": int(e["end"]), "entity_type": et, "score": sc})
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (1/4): base NER 추론 → ner_spans_raw(model='base')
# MAGIC KoELECTRA base로 전체 코퍼스를 추론해 PERSON/ADDRESS span을 추출합니다.

# COMMAND ----------

print("== base NER inference ==")
btok = AutoTokenizer.from_pretrained(BASE_NER)
bmdl = AutoModelForTokenClassification.from_pretrained(BASE_NER)
print("base id2label sample:", dict(list(bmdl.config.id2label.items())[:8]))
base_nlp = pipeline("token-classification", model=bmdl, tokenizer=btok,
                    aggregation_strategy="simple", device=DEVICE)
base_rows = infer(base_nlp, map_base_label)
print("base spans:", len(base_rows))

# ner_spans_raw(model, doc_id, start_char, end_char, entity_type, score)
# S3 노트북은 base만 적재(model='base' 행 교체). ft 행은 S4b가 추가.
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.ner_spans_raw
  (model STRING, doc_id STRING, start_char INT, end_char INT, entity_type STRING, score DOUBLE)""")
load_rows(f"{FQ}.ner_spans_raw",
          [{"model": "base", **r} for r in base_rows], where="model='base'")
print("  ner_spans_raw(base) written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## cascade 헬퍼 (SOURCE: build_ner.py — _parse_keep 검증본)
# MAGIC 저신뢰 span을 LLM으로 확인합니다. `_parse_keep` 은 JSON 파싱 우선, 폴백은 정규화 문자열 정확
# MAGIC 패턴만(공백/개행 취약한 문자열 포함매칭 제거). gpt-oss는 ai_query 배치, qwen은 Python.

# COMMAND ----------

from collections import defaultdict

MODEL2STAGE = {"base": "S3", "ft": "S4"}


def _ctx(text, s, e):
    return text[max(0, s - CASCADE_CTX):e + CASCADE_CTX]


def _confirm_prompt(span, etype, context):
    return (f"다음 문맥에서 \"{span}\"(추정 유형: {etype})이(가) 실제 개인정보인지 판단하세요. "
            f"개인 이름은 PERSON, 주소는 ADDRESS입니다. 문맥: …{context}… "
            f'JSON만: {{"keep": true 또는 false}}')


def _parse_keep(resp):
    j = extract_json(resp)
    if isinstance(j, dict) and isinstance(j.get("keep"), bool):
        return j["keep"]
    norm = "".join((resp or "").lower().split())
    if '"keep":true' in norm:
        return True
    if '"keep":false' in norm:
        return False
    return None


GPTOSS_EP = BACKENDS["gpt-oss-120b"]


def gptoss_cascade(lowconf, text_map):
    """lowconf: list[(key, doc, s, e, type)] → dict key->keep(bool). ai_query 배치."""
    if not lowconf:
        return {}
    rows = [{"k": k, "prompt": _confirm_prompt(text_map[d][s:e], t, _ctx(text_map[d], s, e))}
            for (k, d, s, e, t) in lowconf]
    load_rows(f"{FQ}.cascade_tmp", rows, mode="replace")
    sql(f"""CREATE OR REPLACE TABLE {FQ}.cascade_out AS
        SELECT k, ai_query('{GPTOSS_EP}', prompt) AS resp FROM {FQ}.cascade_tmp""")
    keep, n_unparsed = {}, 0
    for k, resp in sql(f"SELECT k, resp FROM {FQ}.cascade_out")[1]:
        v = _parse_keep(resp)
        if v is None:
            n_unparsed += 1
        keep[k] = bool(v)  # 파싱 불능 → drop (fail-closed)
    if n_unparsed:
        print(f"    WARNING: gpt-oss cascade 응답 파싱 불능 {n_unparsed}/{len(lowconf)} → drop 처리")
    return keep


def qwen_cascade(backend, lowconf, text_map):
    """예외/파싱불능 → keep=False(fail-closed, gpt-oss와 대칭). 실패율 5% 초과 시 hard fail."""
    keep, n_err, n_unparsed = {}, 0, 0
    for (k, d, s, e, t) in lowconf:
        try:
            r = llm_client(_confirm_prompt(text_map[d][s:e], t, _ctx(text_map[d], s, e)),
                           backend=backend, max_tokens=200)
            v = _parse_keep(r)
            if v is None:
                n_unparsed += 1
            keep[k] = bool(v)
        except Exception:
            n_err += 1
            keep[k] = False
    if n_err or n_unparsed:
        print(f"    WARNING: {backend} cascade 호출실패={n_err} 파싱불능={n_unparsed} /{len(lowconf)} → drop 처리")
    if lowconf and n_err > len(lowconf) * 0.05:
        raise RuntimeError(f"{backend} cascade 호출 실패 {n_err}/{len(lowconf)} — 5% 초과, 중단")
    return keep

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (2/4): finalize — 정규식 ∪ (고신뢰 NER ∪ 확인된 저신뢰 NER)
# MAGIC `finalize()` (build_ner.py 이식): 저신뢰 span을 cascade로 확인하고, 최종 span = 정규식 ∪
# MAGIC (고신뢰 NER ∪ keep된 저신뢰 NER). gpt-oss는 전체 docs, qwen은 표본(throughput).

# COMMAND ----------

text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]}

ner = defaultdict(lambda: defaultdict(list))  # model -> doc -> [(s,e,type,score)]
for model, doc, s, e, t, sc in sql(
        f"SELECT model, doc_id, start_char, end_char, entity_type, score FROM {FQ}.ner_spans_raw WHERE model='base'")[1]:
    ner[model][doc].append((int(s), int(e), t, float(sc)))


def finalize(model, backend, sample_docs=None):
    stage = MODEL2STAGE[model]
    docs = sample_docs if sample_docs is not None else list(text_map.keys())
    lowconf = []
    for d in docs:
        for i, (s, e, t, sc) in enumerate(ner[model].get(d, [])):
            if sc < TAU:
                lowconf.append((f"{d}#{i}", d, s, e, t))
    print(f"  {stage}/{backend}: docs={len(docs)} lowconf_routed={len(lowconf)}")
    keep = (gptoss_cascade(lowconf, text_map) if backend == "gpt-oss-120b"
            else qwen_cascade(backend, lowconf, text_map))
    out = []
    for d in docs:
        text = text_map[d]
        spans = list(regex_spans(text))
        occ = [(x["start"], x["end"]) for x in spans]
        for i, (s, e, t, sc) in enumerate(ner[model].get(d, [])):
            if sc < TAU and not keep.get(f"{d}#{i}", False):
                continue  # cascade drop
            if all(e <= cs or s >= ce for cs, ce in occ):
                spans.append({"start": s, "end": e, "entity_type": t, "pii_value": text[s:e], "score": sc})
                occ.append((s, e))
        for x in spans:
            out.append({"stage": stage, "backend": backend, "doc_id": d,
                        "start_char": x["start"], "end_char": x["end"], "entity_type": x["entity_type"],
                        "pii_value": x["pii_value"], "score": x["score"]})
    sql(f"DELETE FROM {FQ}.span_predictions WHERE stage='{stage}' AND backend='{backend}'")
    if out:
        load_rows(f"{FQ}.span_predictions", out)
    sql(f"DELETE FROM {FQ}.span_coverage WHERE stage='{stage}' AND backend='{backend}'")
    load_rows(f"{FQ}.span_coverage",
              [{"stage": stage, "backend": backend, "doc_id": d} for d in docs])
    print(f"    {stage}/{backend}: {len(out)} spans written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (3/4): S3 span 예측 적재 (gpt-oss 전체 + qwen 표본)
# MAGIC gpt-oss는 전체 코퍼스(전체 cascade), qwen은 `S5_QWEN_SAMPLE_DOCS`(=300) 표본. qwen 미설정 시 스킵.

# COMMAND ----------

# span_predictions/coverage 테이블 보장(S1/S2 미실행 단독 attach 대비)
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_predictions
  (stage STRING, backend STRING, doc_id STRING, start_char INT, end_char INT,
   entity_type STRING, pii_value STRING, score DOUBLE)""")
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_coverage
  (stage STRING, backend STRING, doc_id STRING)""")

sample = [r[0] for r in sql(
    f"SELECT doc_id FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT {S5_QWEN_SAMPLE_DOCS}")[1]]

for backend in LLM_BACKENDS:
    if backend == "gpt-oss-120b":
        finalize("base", backend)                       # 전체
    else:
        finalize("base", backend, sample_docs=sample)   # 표본

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (4/4): 컬럼 verdict (S3) — col_predictions(S3, backend='NA')
# MAGIC `column_verdict()` 이식: rule_is_pii OR (해당 컬럼 텍스트에 base NER span 존재) → PII.

# COMMAND ----------

sql(f"DELETE FROM {FQ}.col_predictions WHERE stage='S3'")
sql(f"""INSERT INTO {FQ}.col_predictions
  WITH ner_cols AS (
    SELECT DISTINCT c.table_name, c.column_name
    FROM {FQ}.ner_spans_raw n JOIN {FQ}.text_corpus c USING (doc_id)
    WHERE n.model='base')
  SELECT 'S3' AS stage, 'NA' AS backend, r.table_name, r.column_name,
    (r.rule_is_pii OR nc.table_name IS NOT NULL) AS pred_is_pii,
    CASE WHEN r.rule_is_pii THEN r.rule_category
         WHEN nc.table_name IS NOT NULL THEN 'PERSONAL_INFO' ELSE 'NON_PII' END AS pred_category
  FROM {FQ}.rule_results r
  LEFT JOIN ner_cols nc ON r.table_name=nc.table_name AND r.column_name=nc.column_name""")
print("  col_predictions(S3) written")

# COMMAND ----------

# MAGIC %md ## 결과확인: S3 span_predictions 셀별 분포 + 라우팅 규모

# COMMAND ----------

display(sql(f"""SELECT stage, backend, count(*) n
FROM {FQ}.span_predictions WHERE stage='S3' GROUP BY stage, backend ORDER BY backend""")[1])

# COMMAND ----------

# MAGIC %md ### base NER 라우팅 규모 (score<τ 비율) — 라우팅율 미리보기

# COMMAND ----------

display(sql(f"""SELECT count(*) AS total_ner_spans,
  sum(CASE WHEN score < {TAU} THEN 1 ELSE 0 END) AS low_conf_routed,
  round(sum(CASE WHEN score < {TAU} THEN 1 ELSE 0 END)/count(*), 4) AS route_rate
FROM {FQ}.ner_spans_raw WHERE model='base'""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 비교 (러닝 스코어보드)
# MAGIC | 단계 | span exact F1 | 라우팅 | 비용/1k | 한 줄 |
# MAGIC |---|---|---|---|---|
# MAGIC | S1 | 0.73 | — | $0 | 정형만 |
# MAGIC | S2 | 0.9448 | 100% | $0.132 | 전수 LLM(정확하나 비쌈) |
# MAGIC | **S3 ← 지금 여기** | **0.8966** | **10.9%** | **$0.0244** | NER가 저신뢰 span만 LLM로 → 비용 급감, 단 범용 NER 과탐으로 F1 소폭↓ |
# MAGIC | S4 (예정) | — | — | — | 파인튜닝 NER = 최적 |
# MAGIC | S5 (예정) | — | — | — | LLM 단독(대조군) |
# MAGIC
# MAGIC > 비용은 S2의 약 1/5로 떨어졌지만 범용 NER의 과탐 때문에 F1이 S2보다 낮습니다 — 다음 단계(S4)가 파인튜닝으로 이를 해결합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`40_S4_pattern_nerft_llm/S4b_ner_finetune_cascade`](../40_S4_pattern_nerft_llm/S4b_ner_finetune_cascade)** [ML 클러스터] — NER을 파인튜닝해 과탐을 제거합니다(S4a 학습 코퍼스 선행 필요).
