# Databricks notebook source
# MAGIC %md
# MAGIC # 60b. Day-2 재학습 — NER 챔피언/챌린저 + 사람 승인 승격 게이트 (Rung C)  [ML 클러스터]
# MAGIC
# MAGIC > **클러스터 사양**: S4b와 동일(Databricks Runtime **17.3 ML CPU**, 16-core). **예상 ~20–30분**(챌린저 1회 파인튜닝).
# MAGIC > **선행**: `S4b`가 챔피언 모델 `{MODELS_VOL}/koelectra_ner_ft`를 이미 저장했어야 한다.
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC 규칙(60a)으로 못 고치는 드리프트도 있다 — **새 유형의 이름**(예: 로마자·외국식 표기)이 유입되면 한국어
# MAGIC 이름으로 파인튜닝된 챔피언 NER이 놓친다. 가장 비싼 레버 = **NER 재학습**. 단 자가라벨 재학습은
# MAGIC model-collapse 위험이 커서 **반드시 사람 승인 승격 게이트** 뒤에 둔다.
# MAGIC
# MAGIC ## [직관·원리]
# MAGIC champion(현행 S4 모델) vs **challenger**(신규 패턴을 추가 학습한 후보)를 만들고, **3중 게이트**로만 승격한다:
# MAGIC ① held-out 신규패턴(stress) recall Δ ≥ 0.01(실제 개선) ∧ ② main 비퇴행(기존 능력 유지) ∧ ③ canary 비퇴행(망각 방지).
# MAGIC 통과해도 **자동 교체 없음** — `ner_champion_log`에 기록하고 사람(APPROVE 위젯)이 승인해야 active 모델이 바뀐다.
# MAGIC 문제 시 로그 한 줄로 **즉시 롤백**.
# MAGIC
# MAGIC ## [코드로 보기]
# MAGIC 파인튜닝은 `40_S4_pattern_nerft_llm/S4b`의 encode/Trainer 패턴을 그대로 재사용. SOURCE 설계:
# MAGIC `02_/09_monitor/{build_ner_retrain_corpus,build_ner_champion}.py`. **MLflow 레지스트리·Jobs 미사용**(랩 제약) —
# MAGIC 버전은 모델 폴더(`koelectra_ner_ft_v2`), alias는 `ner_champion_log` 테이블로 대체. 실제 학습이라 게이트 수치는 **실측**.
# MAGIC 신규 패턴은 합성(영문 이름) — 실데이터에선 Auditor/사람 라벨로 약지도 코퍼스를 만든다(개념: `07_운영_모니터링_Day2.md`).
# MAGIC
# MAGIC ## [결과: 기대수치]
# MAGIC | 지표 | 값(실측·합성) |
# MAGIC |---|---|
# MAGIC | held-out 신규패턴(stress) PERSON recall | champion **낮음** → challenger **높음**(Δ) |
# MAGIC | main(기존) 비퇴행 | challenger ≈ champion |
# MAGIC | canary(seed777) 비퇴행 | challenger ≈ champion |
# MAGIC | 게이트 | Δ≥0.01 ∧ main 비퇴행 ∧ canary 비퇴행 → **PROMOTE/KEEP** |
# MAGIC | 승격 | 기본 기록만; `APPROVE=true`에서만 active 교체 + 롤백 가능 |
# MAGIC
# MAGIC ## [한계]
# MAGIC 합성 신규패턴(영문 이름)으로 OOD를 모사 — 실데이터 시프트와 다를 수 있다(상대 추세로). model-collapse 완화는
# MAGIC 본 랩에서 게이트(Δ·canary)만 시연; 전체 M-a..M-g(혼합비 캡·합의 게이트·LLM 중재 등)는 `02_/09_monitor` 참조.
# MAGIC MLflow/Jobs 기반 운영 등록·스케줄도 `02_` 참조. 메인 산출물은 불변(전부 `*_v2`/`*_stress`/`ner_champion_log` 격리).
# MAGIC
# MAGIC ## [다음 단계로]
# MAGIC ➡️ 개념·성숙도 사다리·내 운영 이식: **[`docs/learn/07_운영_모니터링_Day2.md`](../docs/learn/07_운영_모니터링_Day2.md)**.

# COMMAND ----------

# MAGIC %pip install -q seqeval

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md ## 챔피언 로드 + 신규패턴 코호트 생성 (train seed=2028 / held-out stress seed=2029)

# COMMAND ----------

import os
import random as _rnd
import re as _re
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
                          Trainer, pipeline, DataCollatorForTokenClassification)

try:
    from faker import Faker
except ImportError:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "faker"]); from faker import Faker

FT_ENCODER = "monologg/koelectra-small-v3-discriminator"
LABELS = ["O", "B-PERSON", "I-PERSON", "B-ADDRESS", "I-ADDRESS"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEVICE = 0 if torch.cuda.is_available() else -1
CHAMP_DIR = f"{MODELS_VOL}/koelectra_ner_ft"
CHAL_DIR = f"{MODELS_VOL}/koelectra_ner_ft_v2"

try:
    _champ_tok = AutoTokenizer.from_pretrained(CHAMP_DIR)
    _champ_mdl = AutoModelForTokenClassification.from_pretrained(CHAMP_DIR)
except Exception as e:
    raise RuntimeError(f"챔피언 모델 없음({CHAMP_DIR}) — 먼저 40_S4_pattern_nerft_llm/S4b를 실행하세요.") from e
print("champion 로드 OK:", CHAMP_DIR)

# --- 신규패턴(영문 이름) 생성기 — 한국어 NER 챔피언이 OOD로 놓치도록 ---
_ffake = Faker("en_US")
_kfake = Faker("ko_KR")
_SLOT_RE = _re.compile(r"\{([A-Z_]+)\}")


def _phone():
    return f"010-{_rnd.randint(1000,9999)}-{_rnd.randint(1000,9999)}"


def fill_gm(template, gen_map):
    parts, spans, cursor, pos = [], [], 0, 0
    for m in _SLOT_RE.finditer(template):
        lit = template[pos:m.start()]; parts.append(lit); cursor += len(lit)
        et = m.group(1); val = gen_map[et]()
        parts.append(val)
        spans.append({"start": cursor, "end": cursor + len(val), "entity_type": et, "pii_value": val})
        cursor += len(val); pos = m.end()
    parts.append(template[pos:]); text = "".join(parts)
    for s in spans:
        assert text[s["start"]:s["end"]] == s["pii_value"]
    return text, spans


NEWNAME_TPL = [
    "외국인 고객 {PERSON}님 본인확인 완료, 회신 연락처 {PHONE}.",
    "명의자 {PERSON} 가입신청 접수 — 담당 배정, 연락처 {PHONE}.",
    "{PERSON} 고객 상담 요청 접수. 본인확인 진행.",
    "해외 거주 {PERSON}님 문의, 회신 {PHONE} 남김.",
]


def _gen_newname(n, seed):
    Faker.seed(seed); _rnd.seed(seed)
    corpus, gt = [], []
    for i in range(n):
        text, spans = fill_gm(NEWNAME_TPL[i % len(NEWNAME_TPL)], {"PERSON": _ffake.name, "PHONE": _phone})
        doc = f"newname.{seed}.{i:04d}"
        corpus.append((doc, text))
        for s in spans:
            gt.append((doc, s["start"], s["end"], s["entity_type"], s["pii_value"]))
    return corpus, gt


train_corpus_nn, train_gt_nn = _gen_newname(180, 2028)     # 챌린저 학습용 신규패턴
stress_corpus, stress_gt = _gen_newname(80, 2029)          # held-out 신규패턴(게이트 평가)
print(f"신규패턴: train={len(train_corpus_nn)} docs / stress(held-out)={len(stress_corpus)} docs")

# stress 코호트 적재(gold) — 격리 테이블
sql(f"CREATE OR REPLACE TABLE {FQ}.text_corpus_stress (doc_id STRING, text STRING)")
load_rows(f"{FQ}.text_corpus_stress", [{"doc_id": d, "text": t} for d, t in stress_corpus])
sql(f"""CREATE OR REPLACE TABLE {FQ}.span_ground_truth_stress
  (doc_id STRING, start_char INT, end_char INT, entity_type STRING, pii_value STRING)""")
load_rows(f"{FQ}.span_ground_truth_stress",
          [{"doc_id": d, "start_char": s, "end_char": e, "entity_type": t, "pii_value": v}
           for d, s, e, t, v in stress_gt])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 약지도 챌린저 코퍼스 조립 → `text_corpus_train_v2` / `span_gt_train_v2`
# MAGIC 기존 학습셋(seed=777, 표준 한국어 이름) + 신규패턴(영문 이름). 학습-평가 분리 유지(평가 stress는 별도 seed).
# MAGIC 실데이터에선 신규패턴 라벨을 Auditor/사람이 만든다(여기선 합성 gold).

# COMMAND ----------

base_train = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus_train")[1]]
base_gt = {}
for d, s, e, t in sql(f"SELECT doc_id, start_char, end_char, entity_type FROM {FQ}.span_gt_train")[1]:
    if t in ("PERSON", "ADDRESS"):
        base_gt.setdefault(d, []).append((int(s), int(e), t))

train_gt_map = dict(base_gt)
v2_corpus = list(base_train)
for d, t in train_corpus_nn:
    v2_corpus.append((d, t))
for d, s, e, et, v in train_gt_nn:
    if et in ("PERSON", "ADDRESS"):
        train_gt_map.setdefault(d, []).append((int(s), int(e), et))
print(f"challenger 학습셋 v2: {len(v2_corpus)} docs (기존 {len(base_train)} + 신규 {len(train_corpus_nn)})")

# 격리 적재(참고/추적용)
sql(f"CREATE OR REPLACE TABLE {FQ}.text_corpus_train_v2 (doc_id STRING, text STRING)")
load_rows(f"{FQ}.text_corpus_train_v2", [{"doc_id": d, "text": t} for d, t in v2_corpus])

# COMMAND ----------

# MAGIC %md ## 챌린저 파인튜닝 (S4b 패턴 재사용) → `koelectra_ner_ft_v2`

# COMMAND ----------

_tok = AutoTokenizer.from_pretrained(FT_ENCODER)


def encode(text, spans):
    enc = _tok(text, truncation=True, max_length=128, return_offsets_mapping=True)
    labels = []
    for (s, e) in enc["offset_mapping"]:
        if s == e:
            labels.append(-100); continue
        lab = "O"
        for (gs, ge, gt_) in spans:
            if s >= gs and e <= ge:
                lab = ("B-" if s == gs else "I-") + gt_; break
            if s < ge and e > gs:
                lab = ("B-" if s <= gs else "I-") + gt_; break
        labels.append(L2I[lab])
    enc["labels"] = labels; enc.pop("offset_mapping"); return enc


class DS(torch.utils.data.Dataset):
    def __init__(self, docs):
        self.items = [encode(t, train_gt_map.get(d, [])) for d, t in docs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


_docs = list(v2_corpus); _rnd.seed(42); _rnd.shuffle(_docs)
_split = int(len(_docs) * 0.85)
collator = DataCollatorForTokenClassification(_tok)
chal_mdl = AutoModelForTokenClassification.from_pretrained(
    FT_ENCODER, num_labels=len(LABELS), id2label={i: l for l, i in L2I.items()}, label2id=L2I)
args = TrainingArguments(output_dir="/tmp/koelectra_v2", num_train_epochs=4, per_device_train_batch_size=16,
                        per_device_eval_batch_size=32, learning_rate=5e-5, logging_steps=50,
                        eval_strategy="no", save_strategy="no", report_to=[], seed=42)
Trainer(model=chal_mdl, args=args, train_dataset=DS(_docs[:_split]), data_collator=collator).train()
chal_mdl.save_pretrained(CHAL_DIR); _tok.save_pretrained(CHAL_DIR)
print(f"  challenger 저장 → {CHAL_DIR}")

# COMMAND ----------

# MAGIC %md ## 승격 게이트 실측 — champion vs challenger (stress Δ ∧ main 비퇴행 ∧ canary)

# COMMAND ----------

def _nlp(model, tok):
    return pipeline("token-classification", model=model, tokenizer=tok,
                    aggregation_strategy="simple", device=DEVICE)


def infer_person(nlp, corpus):
    """corpus=[(doc,text)] → {doc: set((start,end)) for PERSON}."""
    out = {}
    texts = [t for _, t in corpus]; ids = [d for d, _ in corpus]
    for i in range(0, len(texts), 64):
        res = nlp(texts[i:i + 64])
        if isinstance(res, dict):
            res = [res]
        for did, ents in zip(ids[i:i + 64], res):
            ps = {(int(e["start"]), int(e["end"])) for e in (ents or []) if e.get("entity_group") == "PERSON"}
            if ps:
                out[did] = ps
    return out


def recall_person(pred, gold):
    """type-agnostic offset 겹침(IoU>0) recall."""
    tot = sum(len(v) for v in gold.values())
    if not tot:
        return 0.0
    hit = 0
    for d, gset in gold.items():
        pset = pred.get(d, set())
        for (gs, ge) in gset:
            if any(not (pe <= gs or ps >= ge) for (ps, pe) in pset):
                hit += 1
    return round(hit / tot, 4)


def gold_person(table, ids=None):
    where = "entity_type='PERSON'"
    if ids is not None:
        where += f" AND doc_id IN ({','.join(repr(d) for d in ids)})"
    g = {}
    for d, s, e in sql(f"SELECT doc_id, start_char, end_char FROM {table} WHERE {where}")[1]:
        g.setdefault(d, set()).add((int(s), int(e)))
    return g


# 평가 코호트: stress(신규·held-out) / main 표본 / canary(seed777 표본)
main_samp = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT 200")[1]]
canary_samp = [(d, t) for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus_train ORDER BY xxhash64(doc_id) LIMIT 200")[1]]
g_stress = gold_person(f"{FQ}.span_ground_truth_stress")
g_main = gold_person(f"{FQ}.span_ground_truth", ids=[d for d, _ in main_samp])
g_canary = gold_person(f"{FQ}.span_gt_train", ids=[d for d, _ in canary_samp])

champ_nlp = _nlp(_champ_mdl, _champ_tok)
chal_nlp = _nlp(chal_mdl, _tok)
champ_stress = recall_person(infer_person(champ_nlp, stress_corpus), g_stress)
chal_stress = recall_person(infer_person(chal_nlp, stress_corpus), g_stress)
champ_main = recall_person(infer_person(champ_nlp, main_samp), g_main)
chal_main = recall_person(infer_person(chal_nlp, main_samp), g_main)
champ_canary = recall_person(infer_person(champ_nlp, canary_samp), g_canary)
chal_canary = recall_person(infer_person(chal_nlp, canary_samp), g_canary)

PROMOTE_DELTA, MAIN_FLOOR, CANARY_FLOOR = 0.01, -0.005, -0.01
d_stress = round(chal_stress - champ_stress, 4)
main_ok = (chal_main - champ_main) >= MAIN_FLOOR
canary_ok = (chal_canary - champ_canary) >= CANARY_FLOOR
gate = "PROMOTE" if (d_stress >= PROMOTE_DELTA and main_ok and canary_ok) else "KEEP"
print(f"  held-out 신규패턴 recall: champion {champ_stress} → challenger {chal_stress} (Δ {d_stress})")
print(f"  main(기존) recall:        champion {champ_main} → challenger {chal_main} (비퇴행={main_ok})")
print(f"  canary(seed777) recall:   champion {champ_canary} → challenger {chal_canary} (비퇴행={canary_ok})")
print(f"  → 게이트: **{gate}** (Δstress≥{PROMOTE_DELTA} ∧ main≥{MAIN_FLOOR} ∧ canary≥{CANARY_FLOOR})")

# COMMAND ----------

# MAGIC %md ## 사람 승인 + champion_log (alias 대체) + 롤백

# COMMAND ----------

import datetime as _dt
dbutils.widgets.dropdown("APPROVE", "false", ["false", "true"], "challenger 승격 승인(사람)")
APPROVE = dbutils.widgets.get("APPROVE").strip().lower() == "true"

sql(f"""CREATE TABLE IF NOT EXISTS {FQ}.ner_champion_log
  (ts STRING, decision STRING, champ_stress DOUBLE, chal_stress DOUBLE, delta_stress DOUBLE,
   main_ok BOOLEAN, canary_ok BOOLEAN, approved BOOLEAN, active_model_dir STRING)""")
active = CHAL_DIR if (gate == "PROMOTE" and APPROVE) else CHAMP_DIR
row = {"ts": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
       "decision": gate, "champ_stress": champ_stress, "chal_stress": chal_stress, "delta_stress": d_stress,
       "main_ok": bool(main_ok), "canary_ok": bool(canary_ok), "approved": bool(gate == "PROMOTE" and APPROVE),
       "active_model_dir": active}
load_rows(f"{FQ}.ner_champion_log", [row])
if gate == "PROMOTE" and APPROVE:
    print(f"  ✓ (APPROVE=true) 승격 — active NER = {CHAL_DIR}. 롤백: ner_champion_log에 active=champion 행 추가 또는 widget=false 재실행.")
elif gate == "PROMOTE":
    print("  ⏸ 게이트 PROMOTE지만 미승인 — active=champion 유지. 승격하려면 APPROVE 위젯=true 후 재실행(사람 승인).")
else:
    print("  게이트 KEEP — challenger 미승격, champion 유지(model-collapse/망각 방어).")
display(spark.sql(f"SELECT decision, delta_stress, main_ok, canary_ok, approved, active_model_dir FROM {FQ}.ner_champion_log ORDER BY ts DESC"))

# COMMAND ----------

# MAGIC %md ## 모니터 self-check (인쇄 — 아키텍처 10/10과 별개)

# COMMAND ----------

bres = []
def bchk(name, ok, detail):
    bres.append(bool(ok)); print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name} — {detail}")


# B1 챌린저가 신규패턴을 학습(stress recall이 champion 이상)
bchk("B1 챌린저 신규패턴 학습", chal_stress >= champ_stress, f"champ {champ_stress} → chal {chal_stress}")
# B2 게이트 일관성
bchk("B2 게이트 일관성", (gate == "PROMOTE") == (d_stress >= PROMOTE_DELTA and main_ok and canary_ok),
     f"gate={gate} Δ={d_stress}")
# B3 사람 승인 게이트(미승인 시 active=champion)
b3 = sql(f"SELECT active_model_dir FROM {FQ}.ner_champion_log ORDER BY ts DESC LIMIT 1")[1][0][0]
bchk("B3 사람승인 게이트", (b3 == CHAL_DIR) == (gate == "PROMOTE" and APPROVE), f"active={b3.split('/')[-1]}, APPROVE={APPROVE}")
# B4 메인 격리(메인 span_predictions·ner_spans_raw 불변 — 60b는 *_v2/*_stress/champion_log만 기록)
b4 = int(sql(f"SELECT count(*) FROM {FQ}.span_predictions WHERE doc_id LIKE 'newname.%' OR doc_id LIKE 'drift.%'")[1][0][0])
bchk("B4 메인 격리(불변)", b4 == 0, f"메인 누수={b4}")
print(f"\n== 재학습 self-check {sum(bres)}/{len(bres)} PASS == (아키텍처 90_eval 10/10과 독립)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ Day-2 전체 개념·성숙도 사다리(A 규칙·B LLM튜닝·C NER재학습)·내 운영 이식·MLflow/Jobs 운영 등록은
# MAGIC **[`docs/learn/07_운영_모니터링_Day2.md`](../docs/learn/07_운영_모니터링_Day2.md)**. 아키텍처 재검증은 `90_eval`(10/10 불변).
