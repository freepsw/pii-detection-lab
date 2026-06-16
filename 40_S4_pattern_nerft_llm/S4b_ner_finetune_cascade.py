# Databricks notebook source
# MAGIC %md
# MAGIC # S4 — 패턴 + 파인튜닝 NER + LLM cascade  [ML 클러스터 노트북]
# MAGIC
# MAGIC > **클러스터 사양**: Databricks Runtime **17.3 ML (CPU)** 단일노드, **16-core**(예: Standard_D16ds_v5 / m5d.4xlarge). **GPU 불필요** — KoELECTRA-small CPU 파인튜닝. ML 클러스터에 attach해 셀을 직접 실행합니다.
# MAGIC >
# MAGIC > **예상 소요시간**: KoELECTRA-small 4 epoch 파인튜닝(약 4천 docs) + 전체 추론 + cascade는 16-core CPU에서 대략 **15~30분**(클러스터/큐 상태에 따라 가변). 모델 다운로드 첫 1회는 추가 수 분.
# MAGIC >
# MAGIC > **⚠️ 선행 실행 필수**: 같은 폴더의 [`S4a_generate_train_corpus`](./S4a_generate_train_corpus) 를 **먼저 실행**해야 합니다 — 학습 코퍼스 테이블(`text_corpus_train` 등, seed=777 분리)을 만들어 두어야 이 노트북의 파인튜닝이 동작합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [푸는 문제]
# MAGIC S3의 범용 NER은 도메인 밖 토큰을 과탐했습니다. S4는 **도메인 라벨로 KoELECTRA를 파인튜닝**해
# MAGIC PERSON/ADDRESS를 정밀하게 잡습니다. 과탐이 줄어 cascade로 라우팅되는 저신뢰 span도 거의 없어져
# MAGIC **정확도와 비용을 동시에** 잡습니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC - **train-test 분리**: 학습은 S4a가 만든 분리 코퍼스(`text_corpus_train`, seed=777)로만 수행하고,
# MAGIC   평가는 기존 `text_corpus`(seed=42)에서 — 암기 누수 없는 정직한 일반화 측정.
# MAGIC - 파인튜닝된 NER은 신뢰도가 높아 `score<τ` 인 저신뢰 span이 거의 없음 → **cascade 라우팅 ≈ 0%**.
# MAGIC - 나머지는 S3와 동일(정규식 ∪ NER, fail-closed cascade).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `_tools/ner_job_notebook.py`(파인튜닝 + ft 추론, `text_corpus_train` 으로만 학습) +
# MAGIC `_tools/build_ner.py`(S4 cascade). 인코더 `monologg/koelectra-small-v3-discriminator`, **4 epoch**.
# MAGIC 모델은 `{MODELS_VOL}/koelectra_ner_ft` 에 저장.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: final_grid 기대수치]
# MAGIC | 지표 | gpt-oss | qwen |
# MAGIC |---|---|---|
# MAGIC | span exact F1 | **0.9838** | 0.9806 |
# MAGIC | span char recall | **1.0** | — |
# MAGIC | LLM 라우팅율 | **0%** | 0% |
# MAGIC | 비용 | **$0.01 / 1k docs** | — |
# MAGIC
# MAGIC **정확도·비용 동시 최적** — 모든 단계 중 최고 F1 + 최저 비용.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계 — 정직한 명시]
# MAGIC 라우팅 **0%** 는 학습·평가가 동일 합성 분포라 파인튜닝 NER이 거의 모든 span을 고신뢰로 잡은
# MAGIC **합성분포 효과**입니다. 실데이터는 도메인 시프트로 저신뢰 span이 생겨 cascade가 일부 작동할
# MAGIC 것입니다(그래도 S3보다 낮을 것으로 기대). 수치는 본 합성 셋 기준임을 분명히 합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** — S1~S5를 한 그리드로 비교하고 10/10 검증을 수행합니다.

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
# MAGIC ## 모델/상수 + 데이터 로드 (SOURCE: ner_job_notebook.py)
# MAGIC 평가 코퍼스(text_corpus, seed=42)와 **분리 학습 코퍼스**(text_corpus_train, seed=777)를 로드합니다.
# MAGIC 학습 코퍼스가 없으면 먼저 S4a를 실행하라는 명확한 오류를 냅니다.

# COMMAND ----------

import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, pipeline,
                          DataCollatorForTokenClassification)

FT_ENCODER = "monologg/koelectra-small-v3-discriminator"  # S4 파인튜닝 인코더
LABELS = ["O", "B-PERSON", "I-PERSON", "B-ADDRESS", "I-ADDRESS"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEVICE = 0 if torch.cuda.is_available() else -1
print("device:", "cuda" if DEVICE == 0 else "cpu")

# 평가 코퍼스 (추론 대상)
corpus = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]]
print(f"corpus={len(corpus)} docs (eval, seed=42)")

# 분리 학습 코퍼스 (파인튜닝 대상)
try:
    train_corpus = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus_train")[1]]
    train_gt = {}
    for d, s, e, t in sql(
            f"SELECT doc_id, start_char, end_char, entity_type FROM {FQ}.span_gt_train")[1]:
        if t in ("PERSON", "ADDRESS"):
            train_gt.setdefault(d, []).append((int(s), int(e), t))
except Exception as e:
    raise RuntimeError(
        "text_corpus_train/span_gt_train 없음 — 먼저 S4a(S4a_generate_train_corpus)를 실행하세요.") from e
print(f"train corpus={len(train_corpus)} docs (disjoint, seed=777), gt PERSON/ADDRESS docs={len(train_gt)}")


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
# MAGIC ## 이 단계가 하는 일 (1/5): 파인튜닝 (text_corpus_train으로만 학습)
# MAGIC subword 토큰을 BIO 라벨로 인코딩하고 KoELECTRA를 4 epoch 학습합니다. train/eval split은 학습
# MAGIC 코퍼스 **내부**에서만 나눕니다(평가 코퍼스는 학습에 전혀 노출되지 않음).

# COMMAND ----------

print("== fine-tune KoELECTRA (PERSON/ADDRESS) ==")
ft_tok = AutoTokenizer.from_pretrained(FT_ENCODER)


def encode(doc_text_spans):
    text, spans = doc_text_spans
    enc = ft_tok(text, truncation=True, max_length=128, return_offsets_mapping=True)
    labels = []
    for (s, e) in enc["offset_mapping"]:
        if s == e:
            labels.append(-100); continue
        lab = "O"
        for (gs, ge, gt_) in spans:
            if s >= gs and e <= ge:
                lab = ("B-" if s == gs else "I-") + gt_
                break
            if s < ge and e > gs:  # 부분 겹침
                lab = ("B-" if s <= gs else "I-") + gt_
                break
        labels.append(L2I[lab])
    enc["labels"] = labels
    enc.pop("offset_mapping")
    return enc


# 학습/epoch-eval 모두 분리 학습 코퍼스 내부에서만
import random
docs = [(d, t) for d, t in train_corpus]
random.seed(42); random.shuffle(docs)
split = int(len(docs) * 0.85)
train_docs, eval_docs = docs[:split], docs[split:]


class DS(torch.utils.data.Dataset):
    def __init__(self, docs):
        self.items = [encode((t, train_gt.get(d, []))) for d, t in docs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


collator = DataCollatorForTokenClassification(ft_tok)
ft_mdl = AutoModelForTokenClassification.from_pretrained(
    FT_ENCODER, num_labels=len(LABELS),
    id2label={i: l for l, i in L2I.items()}, label2id=L2I)

import seqeval.metrics as sm


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_l, pred_l = [], []
    for pr, la in zip(preds, p.label_ids):
        tl, pl = [], []
        for pi, li in zip(pr, la):
            if li == -100:
                continue
            tl.append(LABELS[li]); pl.append(LABELS[pi])
        true_l.append(tl); pred_l.append(pl)
    return {"f1": sm.f1_score(true_l, pred_l),
            "precision": sm.precision_score(true_l, pred_l),
            "recall": sm.recall_score(true_l, pred_l)}


args = TrainingArguments(
    output_dir="/tmp/koelectra_ft", num_train_epochs=4, per_device_train_batch_size=16,
    per_device_eval_batch_size=32, learning_rate=5e-5, logging_steps=50,
    eval_strategy="epoch", save_strategy="no", report_to=[], seed=42)
trainer = Trainer(model=ft_mdl, args=args, train_dataset=DS(train_docs),
                  eval_dataset=DS(eval_docs), data_collator=collator,
                  compute_metrics=compute_metrics)
trainer.train()
ft_eval = trainer.evaluate()
print("FT eval:", ft_eval)

# COMMAND ----------

# MAGIC %md ## 모델 저장 → {MODELS_VOL}/koelectra_ner_ft

# COMMAND ----------

ft_mdl.save_pretrained(f"{MODELS_VOL}/koelectra_ner_ft")
ft_tok.save_pretrained(f"{MODELS_VOL}/koelectra_ner_ft")
print(f"  saved → {MODELS_VOL}/koelectra_ner_ft")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (2/5): ft 추론 → ner_spans_raw(model='ft')
# MAGIC 파인튜닝 모델로 평가 코퍼스를 추론합니다. ft 라벨은 이미 PERSON/ADDRESS이므로 label_map은 그대로 통과.

# COMMAND ----------

print("== ft NER inference ==")
ft_nlp = pipeline("token-classification", model=ft_mdl, tokenizer=ft_tok,
                  aggregation_strategy="simple", device=DEVICE)
ft_rows = infer(ft_nlp, lambda g: g if g in ("PERSON", "ADDRESS") else None)
print("ft spans:", len(ft_rows))

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.ner_spans_raw
  (model STRING, doc_id STRING, start_char INT, end_char INT, entity_type STRING, score DOUBLE)""")
load_rows(f"{FQ}.ner_spans_raw", [{"model": "ft", **r} for r in ft_rows], where="model='ft'")

# ft 파인튜닝 메트릭 기록
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.ner_ft_metrics
  (model STRING, entity_type STRING, precision DOUBLE, recall DOUBLE, f1 DOUBLE, support INT)""")
load_rows(f"{FQ}.ner_ft_metrics", [{
    "model": "ft_eval", "entity_type": "ALL",
    "precision": float(ft_eval.get("eval_precision", 0.0)),
    "recall": float(ft_eval.get("eval_recall", 0.0)),
    "f1": float(ft_eval.get("eval_f1", 0.0)), "support": len(eval_docs)}],
    where="model='ft_eval'")
print("  ner_spans_raw(ft) + ner_ft_metrics written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## cascade 헬퍼 (SOURCE: build_ner.py — _parse_keep 검증본)
# MAGIC S3와 동일한 fail-closed cascade. gpt-oss는 ai_query 배치, qwen은 Python.

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
        keep[k] = bool(v)
    if n_unparsed:
        print(f"    WARNING: gpt-oss cascade 응답 파싱 불능 {n_unparsed}/{len(lowconf)} → drop 처리")
    return keep


def qwen_cascade(backend, lowconf, text_map):
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
# MAGIC ## 이 단계가 하는 일 (3/5): finalize — 정규식 ∪ (고신뢰 ft-NER ∪ 확인된 저신뢰)
# MAGIC `finalize()` (build_ner.py 이식, model='ft' → stage='S4').

# COMMAND ----------

text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus")[1]}

ner = defaultdict(lambda: defaultdict(list))
for model, doc, s, e, t, sc in sql(
        f"SELECT model, doc_id, start_char, end_char, entity_type, score FROM {FQ}.ner_spans_raw WHERE model='ft'")[1]:
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
                continue
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
# MAGIC ## 이 단계가 하는 일 (4/5): S4 span 예측 적재 (gpt-oss 전체 + qwen 표본)

# COMMAND ----------

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_predictions
  (stage STRING, backend STRING, doc_id STRING, start_char INT, end_char INT,
   entity_type STRING, pii_value STRING, score DOUBLE)""")
sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.span_coverage
  (stage STRING, backend STRING, doc_id STRING)""")

sample = [r[0] for r in sql(
    f"SELECT doc_id FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT {S5_QWEN_SAMPLE_DOCS}")[1]]

for backend in LLM_BACKENDS:
    if backend == "gpt-oss-120b":
        finalize("ft", backend)
    else:
        finalize("ft", backend, sample_docs=sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일 (5/5): 컬럼 verdict (S4) — col_predictions(S4, backend='NA')
# MAGIC rule_is_pii OR (해당 컬럼 텍스트에 ft NER span 존재) → PII.

# COMMAND ----------

sql(f"DELETE FROM {FQ}.col_predictions WHERE stage='S4'")
sql(f"""INSERT INTO {FQ}.col_predictions
  WITH ner_cols AS (
    SELECT DISTINCT c.table_name, c.column_name
    FROM {FQ}.ner_spans_raw n JOIN {FQ}.text_corpus c USING (doc_id)
    WHERE n.model='ft')
  SELECT 'S4' AS stage, 'NA' AS backend, r.table_name, r.column_name,
    (r.rule_is_pii OR nc.table_name IS NOT NULL) AS pred_is_pii,
    CASE WHEN r.rule_is_pii THEN r.rule_category
         WHEN nc.table_name IS NOT NULL THEN 'PERSONAL_INFO' ELSE 'NON_PII' END AS pred_category
  FROM {FQ}.rule_results r
  LEFT JOIN ner_cols nc ON r.table_name=nc.table_name AND r.column_name=nc.column_name""")
print("  col_predictions(S4) written")

# COMMAND ----------

# MAGIC %md ## 결과확인: S4 span_predictions 셀별 분포 + 라우팅 규모 (≈0%)

# COMMAND ----------

display(sql(f"""SELECT stage, backend, count(*) n
FROM {FQ}.span_predictions WHERE stage='S4' GROUP BY stage, backend ORDER BY backend""")[1])

# COMMAND ----------

# MAGIC %md ### ft NER 라우팅 규모 (score<τ 비율) — 합성분포에서 0%에 근접

# COMMAND ----------

display(sql(f"""SELECT count(*) AS total_ner_spans,
  sum(CASE WHEN score < {TAU} THEN 1 ELSE 0 END) AS low_conf_routed,
  round(sum(CASE WHEN score < {TAU} THEN 1 ELSE 0 END)/count(*), 4) AS route_rate
FROM {FQ}.ner_spans_raw WHERE model='ft'""")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 비교 (러닝 스코어보드)
# MAGIC | 단계 | span exact F1 | 라우팅 | 비용/1k | 한 줄 |
# MAGIC |---|---|---|---|---|
# MAGIC | S1 | 0.73 | — | $0 | 정형만 |
# MAGIC | S2 | 0.9448 | 100% | $0.132 | 전수 LLM(비쌈) |
# MAGIC | S3 | 0.8966 | 10.9% | $0.0244 | 범용 NER cascade(과탐으로 F1↓) |
# MAGIC | **S4 ← 지금 여기** | **0.9838** | **0%** | **$0.01** | 파인튜닝 NER = **최고 F1 + 최저 비용**(라우팅 0%) |
# MAGIC | S5 (예정) | — | — | — | LLM 단독(대조군) |
# MAGIC
# MAGIC > **S4가 정확도·비용을 동시에 최적화**합니다(파인튜닝으로 NER 신뢰도↑ → LLM 호출 0%). 다음 S5는 LLM 단독 대조군이며, 90_eval에서 전 단계를 독립 재계산으로 검증합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`90_eval/90_compare_and_verify`](../90_eval/90_compare_and_verify)** — 전 단계 비교 그리드 + 10/10 검증.
