# Databricks notebook source
# MAGIC %md
# MAGIC # 60a. Day-2 모니터링 — 라벨 없는 드리프트 탐지 + 규칙 자동교정 폐곡선 (Rung A)
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC S1~S5로 아키텍처를 **구축(Day-1)** 했다면, 운영(Day-2)에서는 **새 PII 형식이 라벨 없이 유입**돼
# MAGIC 탐지기가 조용히 늙는다. 정답이 없어 "정확도가 떨어졌다"는 것조차 측정이 안 된다. 이 노트북은
# MAGIC **라벨 없이** 저하를 탐지하고(신호+LLM 심판), 가장 싼 레버(정규식)부터 **제안→섀도우검증→사람 승인**으로
# MAGIC 고치는 Day-2 폐곡선을 보인다.
# MAGIC
# MAGIC ## [직관·원리]
# MAGIC 4단계 멘탈모델: ① **싼 신호**(룰이 조용한데 LLM은 활성 = 새 형식 의심) → ② **비싼 심판**(고-recall LLM
# MAGIC Auditor로 준-정답=silver) → ③ **심판을 frozen gold로 역채점**(auditor가 못 믿을 만하면 `degraded`로 교정 보류)
# MAGIC → ④ 저하 구간에 **규칙 교정안 제안**(shape→정규식→섀도우평가). 성숙도 사다리에서 **Rung A = 규칙**(가장
# MAGIC 싸고 결정론적이라 *지금* 닫을 수 있는 루프). 드리프트는 seed=2027(메인 42·학습 777과 분리).
# MAGIC
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `02_/09_monitor/{build_detectors_drift,build_health,build_audit,build_remediation}.py` 이식.
# MAGIC 드리프트 생성기는 랩의 `01_generate_data` 패턴을 인라인(02_의 생성기 모듈은 랩에 없어 재작성). 평가 수학은
# MAGIC `_common/eval_lib`와 동일 regime을 노트북 인라인으로 미러(eval_lib는 메인 테이블 하드코딩이라 재지정 불가).
# MAGIC **메인 8개 테이블은 절대 건드리지 않는다** — 전부 `*_drift`/`monitor_*`/`audit_*`/`remediation_*` 격리.
# MAGIC
# MAGIC ## [결과: 기대수치]
# MAGIC | 지표 | 값 |
# MAGIC |---|---|
# MAGIC | 드리프트 코호트 | 5종 정형변형 ~62 docs (regex 100% 미탐 self-assert) |
# MAGIC | 라벨없는 신호 | 룰침묵·LLM활성 비율 **↑(WARN/CRIT)** vs 메인 baseline |
# MAGIC | Auditor 자기보정 | frozen gold recall **≈0.9+** → `degraded=False` |
# MAGIC | **규칙 회복(gold 기준·결정론)** | 구조화 정형 recall **0.0 → ~1.0**(제안 정규식 섀도우 적용) |
# MAGIC | 거버넌스 | 기본 **제안만**(검토 대기); `APPROVE=true` 위젯 시에만 `S1_FIXED`(드리프트 한정) |
# MAGIC
# MAGIC ## [한계]
# MAGIC silver(LLM Auditor) 기준 수치는 **비결정**(gpt-oss가 gpt-oss를 감사 = 순환성) → 헤드라인·검증은 **gold 기준**만 단언,
# MAGIC silver는 예시. 합성 상한치(상대 추세로 읽기). NER 재학습(비싼 레버)은 `60b`/문서 참조. 일일 스케줄은 Jobs로
# MAGIC 감싸면 됨(이 랩은 Jobs 미사용 — `07_운영_모니터링_Day2.md`).
# MAGIC
# MAGIC ## [다음 단계로]
# MAGIC ➡️ **[`60b_ner_challenger_retrain`](./60b_ner_challenger_retrain)** — 규칙으로 못 고치는 이름·주소 드리프트는
# MAGIC NER 챔피언/챌린저 **재학습 + 사람 승인 승격 게이트**로. 개념·운영 적용은 `../docs/learn/07_운영_모니터링_Day2.md`.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 드리프트 코호트 생성 — 신규 정형 PII 변형 (seed=2027)
# MAGIC 운영에 새로 들어온 5종 변형(외국인/무하이픈 주민번호·신형 여권·유선전화·변형 계좌)을 주입한다. 모두
# MAGIC **현재 9종 정규식이 정직하게 놓치도록**(R4 self-assert) 설계 — "거짓 저하 스토리"를 방지한다.

# COMMAND ----------

import random as _rnd
import re as _re

try:
    from faker import Faker
except ImportError:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "faker"])
    from faker import Faker

DRIFT_SEED = 2027
_fake = Faker("ko_KR")

# 정규식이 커버하려는 정형 타입(이 타입의 누락 = 진짜 recall leakage)
REGEX_COVERABLE = {"RRN", "PHONE", "CARD", "ACCOUNT", "EMAIL", "IMEI", "PASSPORT"}
STRUCTURED_TAGS = {"foreign_rrn", "nohyphen_rrn", "new_passport", "landline", "variant_account"}
TYPE_LAYER = {"RRN": "PATTERN", "PHONE": "PATTERN", "PASSPORT": "PATTERN", "ACCOUNT": "PATTERN",
              "PERSON": "NER"}


# --- 신규 변형 생성기 (현재 정규식이 미탐하도록) ---
def _foreign_rrn():   # 외국인 성별코드 5-8 → [1-4] 거부
    return f"{_rnd.randint(0,99):02d}{_rnd.randint(1,12):02d}{_rnd.randint(1,28):02d}-{_rnd.choice([5,6,7,8])}{_rnd.randint(100000,999999)}"


def _nohyphen_rrn():  # 하이픈 없는 13자리 → 하이픈 필수 정규식 미탐
    return f"{_rnd.randint(0,99):02d}{_rnd.randint(1,12):02d}{_rnd.randint(1,28):02d}{_rnd.choice([1,2,3,4])}{_rnd.randint(100000,999999)}"


def _new_passport():  # 영문1+숫자3+영문1+숫자4 (M123A4567) → [MSROD]\d{8} 미탐
    return _rnd.choice("MSROD") + f"{_rnd.randint(0,999):03d}" + _rnd.choice("ABCDEFGHJKLMN") + f"{_rnd.randint(0,9999):04d}"


def _landline():      # 02/0XX 유선 → 01X 시작 필수 PHONE 정규식 미탐
    area = _rnd.choice(["02", "031", "032", "051", "053", "042", "062", "070"])
    mid = _rnd.randint(1000, 9999) if (area == "02" and _rnd.random() < 0.5) else _rnd.randint(100, 999)
    return f"{area}-{mid}-{_rnd.randint(1000,9999)}"


def _variant_account():  # 6-2-6 / 3-3-6 / 4-3-6 → \d{3}-\d{2}-\d{6} 미탐
    shape = _rnd.choice(["6-2-6", "3-3-6", "4-3-6"])
    if shape == "6-2-6":
        return f"{_rnd.randint(100000,999999)}-{_rnd.randint(10,99)}-{_rnd.randint(100000,999999)}"
    if shape == "3-3-6":
        return f"{_rnd.randint(100,999)}-{_rnd.randint(100,999)}-{_rnd.randint(100000,999999)}"
    return f"{_rnd.randint(1000,9999)}-{_rnd.randint(100,999)}-{_rnd.randint(100000,999999)}"


_SLOT_RE = _re.compile(r"\{([A-Z_]+)\}")


def fill_gm(template, gen_map):
    """{TYPE} 슬롯을 gen_map으로 채우고 (text, spans) 반환. text[s:e]==value self-assert (랩 fill 패턴)."""
    parts, spans, cursor, pos = [], [], 0, 0
    for m in _SLOT_RE.finditer(template):
        lit = template[pos:m.start()]; parts.append(lit); cursor += len(lit)
        et = m.group(1); val = gen_map[et]()
        parts.append(val)
        spans.append({"start": cursor, "end": cursor + len(val), "entity_type": et, "pii_value": val})
        cursor += len(val); pos = m.end()
    parts.append(template[pos:]); text = "".join(parts)
    for s in spans:
        assert text[s["start"]:s["end"]] == s["pii_value"], (text, s)
    return text, spans


def _gm(**ov):
    base = {"PERSON": _fake.name, "RRN": _nohyphen_rrn, "PASSPORT": _new_passport,
            "PHONE": _landline, "ACCOUNT": _variant_account}
    base.update(ov)
    return base


# (tag, n, [templates], gen_map) — 위험단서(고객·명의·본인확인) 포함, 정형 슬롯만 변형값 주입
STRUCTURED_PATTERNS = [
    ("foreign_rrn", 14, ["외국인 고객 본인확인, 주민번호 {RRN} 대조 완료. 담당 {PERSON}.",
                         "명의자 {PERSON} 가입신청, 주민등록번호 {RRN} 확인.",
                         "{PERSON} 고객 신원확인 — 주민번호 {RRN} 조회."], _gm(RRN=_foreign_rrn)),
    ("nohyphen_rrn", 12, ["본인확인 처리, 주민번호 {RRN} 입력 확인. 고객 {PERSON}.",
                          "가입 상담 — 명의자 주민등록번호 {RRN} 대조.",
                          "{PERSON} 고객 주민번호 {RRN} 기준 가입이력 조회."], _gm(RRN=_nohyphen_rrn)),
    ("new_passport", 12, ["외국인 가입 상담, 여권 {PASSPORT} 기준 신원확인. 담당 {PERSON}.",
                          "명의자 {PERSON} 여권번호 {PASSPORT} 확인 후 개통.",
                          "고객 {PERSON} 여권 {PASSPORT} 대조 완료."], _gm(PASSPORT=_new_passport)),
    ("landline", 12, ["유선 회선 문의, 회신 연락처 {PHONE} 확인. 고객 {PERSON}.",
                      "명의자 {PERSON} 유선전화 {PHONE}로 상담 요청.",
                      "고객 사무실 대표번호 {PHONE} 등록, 담당 {PERSON}."], _gm(PHONE=_landline)),
    ("variant_account", 12, ["자동이체 계좌 {ACCOUNT} 변경 요청. 명의자 {PERSON}.",
                             "환불계좌 {ACCOUNT} 등록 요망. 고객 {PERSON}.",
                             "{PERSON} 고객 출금계좌 {ACCOUNT} 본인확인 완료."], _gm(ACCOUNT=_variant_account)),
]


def generate_drift():
    Faker.seed(DRIFT_SEED); _rnd.seed(DRIFT_SEED)
    corpus, gt = [], []
    for tag, n, tpls, gm in STRUCTURED_PATTERNS:
        for i in range(n):
            text, spans = fill_gm(tpls[i % len(tpls)], gm)
            doc = f"drift.{tag}.{i:04d}"
            corpus.append((doc, text))
            for s in spans:
                gt.append((doc, s["start"], s["end"], s["entity_type"], s["pii_value"],
                           TYPE_LAYER.get(s["entity_type"], "PATTERN")))
    return corpus, gt


corpus_rows, gt_rows = generate_drift()
# R4 정직성: 구조화 변형 정형 PII gold가 실제로 현재 정규식에 '미탐'인지 self-assert
_tbd = dict(corpus_rows)
_checked = _missed = 0
_caught = []
for d, s, e, t, v, _l in gt_rows:
    if d.split(".")[1] not in STRUCTURED_TAGS or t not in REGEX_COVERABLE:
        continue
    _checked += 1
    if any(r["entity_type"] == t and not (r["end"] <= s or r["start"] >= e) for r in regex_spans(_tbd[d])):
        _caught.append((d, t, v))
    else:
        _missed += 1
assert not _caught, f"R4 위반: 구조화 변형 {len(_caught)}건이 이미 정규식에 탐지됨(거짓 스토리). 예: {_caught[:3]}"
print(f"  ✓ R4 정직성: 구조화 변형 정형 PII {_missed}/{_checked} 전부 정규식 미탐(100%)")
print(f"  드리프트: {len(corpus_rows)} docs / {len(gt_rows)} gold spans")

# COMMAND ----------

# MAGIC %md ### 드리프트 테이블 적재 (타입 명시 → load_rows 캐스팅; 메인 불변)

# COMMAND ----------

sql(f"""CREATE OR REPLACE TABLE {FQ}.text_corpus_drift (doc_id STRING, text STRING)""")
load_rows(f"{FQ}.text_corpus_drift", [{"doc_id": d, "text": t} for d, t in corpus_rows], mode="append")
sql(f"""CREATE OR REPLACE TABLE {FQ}.span_ground_truth_drift
  (doc_id STRING, start_char INT, end_char INT, entity_type STRING, pii_value STRING, source_layer STRING)""")
load_rows(f"{FQ}.span_ground_truth_drift",
          [{"doc_id": d, "start_char": s, "end_char": e, "entity_type": t, "pii_value": v, "source_layer": l}
           for d, s, e, t, v, l in gt_rows], mode="append")
# 적재 무결성 + 메인 누수 0
_chk = sql(f"""SELECT
  (SELECT count(*) FROM {FQ}.span_ground_truth_drift g JOIN {FQ}.text_corpus_drift c USING(doc_id)
     WHERE substring(c.text, g.start_char+1, g.end_char-g.start_char) <> g.pii_value),
  (SELECT count(*) FROM {FQ}.text_corpus_drift s JOIN {FQ}.text_corpus m ON s.text=m.text)""")[1][0]
assert int(_chk[0]) == 0 and int(_chk[1]) == 0, f"드리프트 무결성/누수 실패: offset={_chk[0]} main_leak={_chk[1]}"
display(spark.sql(f"SELECT count(*) docs FROM {FQ}.text_corpus_drift"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 현행 탐지기를 드리프트에 적용 (격리: `*_drift`, 메인 불변)
# MAGIC S1(정규식)과 S2/S5(gpt-oss `ai_query` 1배치)를 드리프트 코호트에 돌린다. `span_predictions`/`span_coverage`의
# MAGIC **스키마만 복제**(clone-schema)해 타입을 상속하고 메인은 건드리지 않는다.

# COMMAND ----------

def _sqllit(s):
    return s.replace("'", "''")


# clone-schema (타입 상속, 행 0) — 메인 불변
sql(f"CREATE OR REPLACE TABLE {FQ}.span_predictions_drift AS SELECT * FROM {FQ}.span_predictions WHERE 1=0")
sql(f"CREATE OR REPLACE TABLE {FQ}.span_coverage_drift AS SELECT * FROM {FQ}.span_coverage WHERE 1=0")
sql(f"CREATE OR REPLACE TABLE {FQ}.span_llm_raw_drift (backend STRING, doc_id STRING, llm_raw STRING)")

text_map = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus_drift")[1]}
docs = list(text_map)


def _row(stage, backend, doc, s):
    return {"stage": stage, "backend": backend, "doc_id": doc, "start_char": s["start"],
            "end_char": s["end"], "entity_type": s["entity_type"], "pii_value": s["pii_value"], "score": s["score"]}


# S1 (정규식) — 드리프트 변형은 설계상 정규식이 못 잡으므로 0건일 수 있음(=룰 침묵 신호의 근거)
s1_rows = [_row("S1", "NA", d, s) for d, t in text_map.items() for s in regex_spans(t)]
if s1_rows:
    load_rows(f"{FQ}.span_predictions_drift", s1_rows)
load_rows(f"{FQ}.span_coverage_drift", [{"stage": "S1", "backend": "NA", "doc_id": d} for d in docs])

# S2/S5 (gpt-oss ai_query 1배치) → span_llm_raw_drift → recover → merge
bk = "gpt-oss-120b"
ep = BACKENDS[bk]
sql(f"""INSERT INTO {FQ}.span_llm_raw_drift
  SELECT '{bk}', doc_id, ai_query('{ep}', concat('{_sqllit(SPAN_PROMPT)}', text)) FROM {FQ}.text_corpus_drift""")
raw = sql(f"SELECT doc_id, llm_raw FROM {FQ}.span_llm_raw_drift WHERE backend='{bk}'")[1]
s2_rows, s5_rows, n_parse_fail = [], [], 0
for doc, llm_raw in raw:
    parsed = extract_json(llm_raw)
    items = parsed if isinstance(parsed, list) else (parsed.get("spans") if isinstance(parsed, dict) else None)
    if items is None:
        n_parse_fail += 1; items = []
    llm_spans = recover_spans(text_map.get(doc, ""), items)
    s5_rows += [_row("S5", bk, doc, s) for s in llm_spans]
    s2_rows += [_row("S2", bk, doc, s) for s in merge_regex_llm(regex_spans(text_map.get(doc, "")), llm_spans)]
for stage, rows in [("S2", s2_rows), ("S5", s5_rows)]:
    if rows:
        load_rows(f"{FQ}.span_predictions_drift", rows)
    load_rows(f"{FQ}.span_coverage_drift", [{"stage": stage, "backend": bk, "doc_id": d} for d in docs])
print(f"  S1 spans={len(s1_rows)} · S2 spans={len(s2_rows)} · S5 spans={len(s5_rows)} · LLM parse_fail={n_parse_fail}/{len(raw)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 라벨 없는 건강 신호 — "룰이 조용한데 LLM은 활성" (vs 메인 baseline)
# MAGIC 정답 없이 탐지기 I/O만으로 드리프트를 의심하는 GT-free 신호. 핵심 = **rule-silent·LLM-active 비율**:
# MAGIC S1=0건인데 S2(LLM)>0건인 문서 비율이 메인 baseline 대비 급등하면 새 형식 유입 신호다.

# COMMAND ----------

def _silent_rate(pred, corpus):
    n = int(sql(f"SELECT count(*) FROM {corpus}")[1][0][0]) or 1
    silent = int(sql(f"""
      WITH d AS (SELECT doc_id FROM {corpus}),
        s1c AS (SELECT doc_id, count(*) n FROM {pred} WHERE stage='S1' GROUP BY doc_id),
        llc AS (SELECT doc_id, count(*) n FROM {pred} WHERE stage='S2' AND backend='gpt-oss-120b' GROUP BY doc_id)
      SELECT count(*) FROM d LEFT JOIN s1c USING(doc_id) LEFT JOIN llc USING(doc_id)
      WHERE coalesce(llc.n,0)>0 AND coalesce(s1c.n,0)=0""")[1][0][0] or 0)
    return round(silent / n, 4)


def _parse_fail_rate(raw_tbl):
    try:
        tot = int(sql(f"SELECT count(*) FROM {raw_tbl} WHERE backend='gpt-oss-120b'")[1][0][0]) or 0
    except Exception:
        return None
    if not tot:
        return None
    pf = int(sql(f"SELECT count(*) FROM {raw_tbl} WHERE backend='gpt-oss-120b' AND llm_raw NOT RLIKE '\\\\[|\\\\{{'")[1][0][0] or 0)
    return round(pf / tot, 4)


drift_silent = _silent_rate(f"{FQ}.span_predictions_drift", f"{FQ}.text_corpus_drift")
base_silent = _silent_rate(f"{FQ}.span_predictions", f"{FQ}.text_corpus")
drift_pf = _parse_fail_rate(f"{FQ}.span_llm_raw_drift")


def _breach(drift, base):
    if base is not None and drift > max(base * 3, 0.2):
        return "CRIT"
    if base is not None and drift > max(base * 2, 0.1):
        return "WARN"
    return "OK"


health = [{"detector": "rule", "metric": "s1_silent_llm_active_rate", "slice": "drift",
           "value": drift_silent, "baseline": base_silent, "breach": _breach(drift_silent, base_silent)},
          {"detector": "llm", "metric": "llm_parse_fail_rate", "slice": "drift",
           "value": drift_pf, "baseline": None, "breach": "OK"}]
load_rows(f"{FQ}.monitor_health", health, mode="replace")
print(f"  rule-silent·LLM-active: drift={drift_silent} vs main baseline={base_silent} → {health[0]['breach']}")
display(spark.sql(f"SELECT detector, metric, value, baseline, breach FROM {FQ}.monitor_health ORDER BY detector"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. LLM-as-judge Auditor — 실버(준-정답) + frozen-gold 자기보정
# MAGIC 라벨이 없으니 **고-recall LLM**이 준-정답(silver)을 만든다. 그리고 "누가 심판을 감사하나" — 매 사이클
# MAGIC frozen gold(메인 `span_ground_truth`) 표본에 auditor 자신의 recall을 측정해, 낮으면 `degraded=True`로
# MAGIC **교정을 보류**한다(fail-safe). ⚠️ gpt-oss가 gpt-oss를 감사 = 순환성 → 헤드라인은 §5의 **gold 기준**으로만 단언.

# COMMAND ----------

AUDIT_SPAN_PROMPT = (
    "당신은 개인정보 감사관입니다. 다음 한국어 텍스트에서 개인정보(PII)에 해당하는 부분 문자열을 "
    "최대한 빠짐없이(고-recall) 찾아 JSON 배열로만 출력하세요. 의심되면 포함하세요(놓치는 것보다 과탐이 낫다). "
    "각 원소 형식: {\"value\": \"원문에 그대로 나타난 문자열\", \"type\": \"유형\"}. "
    "유형: PERSON, PHONE(유선 02·031 포함), RRN(무하이픈·외국인 성별코드 5-8 포함), CARD, "
    "ACCOUNT(다양한 포맷), EMAIL, ADDRESS, IMEI, PASSPORT(신형 영문+숫자 포함). "
    "value는 텍스트에 등장한 그대로 복사. 없으면 []. 설명·코드펜스 없이 JSON 배열만.\n\n텍스트:\n")
CALIB_RECALL_FLOOR, CALIB_PRECISION_FLOOR, CALIB_DOCS = 0.90, 0.70, 120


def audit_scan(doc_text):
    """{doc:text} → {doc:[span]} (gpt-oss ai_query 1배치, 고-recall 프롬프트)."""
    if not doc_text:
        return {}
    sql(f"CREATE OR REPLACE TABLE {FQ}.audit_tmp (k STRING, prompt STRING)")
    load_rows(f"{FQ}.audit_tmp", [{"k": d, "prompt": AUDIT_SPAN_PROMPT + t} for d, t in doc_text.items()])
    sql(f"CREATE OR REPLACE TABLE {FQ}.audit_out AS SELECT k, ai_query('{ep}', prompt) resp FROM {FQ}.audit_tmp")
    out = {}
    for k, resp in sql(f"SELECT k, resp FROM {FQ}.audit_out")[1]:
        items = extract_json(resp)
        if isinstance(items, dict):
            items = items.get("spans") if isinstance(items.get("spans"), list) else [items]
        out[k] = recover_spans(doc_text[k], items if isinstance(items, list) else [])
    return out


# 4-1) 드리프트 실버
scanned = audit_scan(text_map)
audit_rows = []
for d, spans in scanned.items():
    rs = regex_spans(text_map[d])
    for sp in spans:
        cosign = any(r["entity_type"] == sp["entity_type"] and not (r["end"] <= sp["start"] or r["start"] >= sp["end"]) for r in rs)
        audit_rows.append({"doc_id": d, "start_char": sp["start"], "end_char": sp["end"],
                           "entity_type": sp["entity_type"], "pii_value": sp["pii_value"], "regex_cosign": bool(cosign)})
load_rows(f"{FQ}.audit_spans", audit_rows, mode="replace")
sql(f"CREATE OR REPLACE VIEW {FQ}.audit_spans_silver AS SELECT * FROM {FQ}.audit_spans")

# 4-2) frozen-gold 자기보정 (type_agnostic offset)
samp = [r[0] for r in sql(f"SELECT doc_id FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT {CALIB_DOCS}")[1]]
ids = ",".join(repr(d) for d in samp)
ctext = {d: t for d, t in sql(f"SELECT doc_id, text FROM {FQ}.text_corpus WHERE doc_id IN ({ids})")[1]}
gold = {}
for d, s, e in sql(f"SELECT doc_id, start_char, end_char FROM {FQ}.span_ground_truth WHERE doc_id IN ({ids})")[1]:
    gold.setdefault(d, set()).add((int(s), int(e)))
cscan = audit_scan(ctext)
g_tot = g_hit = a_tot = a_hit = 0
for d in ctext:
    goff = gold.get(d, set())
    aoff = {(sp["start"], sp["end"]) for sp in cscan.get(d, [])}
    g_tot += len(goff); g_hit += len(goff & aoff)
    a_tot += len(aoff); a_hit += len(aoff & goff)
cal_recall = round(g_hit / g_tot, 4) if g_tot else 0.0
cal_prec = round(a_hit / a_tot, 4) if a_tot else 0.0
degraded = (cal_recall < CALIB_RECALL_FLOOR) or (cal_prec < CALIB_PRECISION_FLOOR)
load_rows(f"{FQ}.auditor_calibration", [{"entity_type": "ALL", "precision": cal_prec, "recall": cal_recall,
          "n_gt": g_tot, "degraded": bool(degraded)}], mode="replace")
print(f"  audit_spans(drift)={len(audit_rows)} · auditor 자기보정 P={cal_prec} R={cal_recall} → degraded={degraded} (floor R={CALIB_RECALL_FLOOR})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 규칙 자동교정 폐곡선 — 제안 → **gold 섀도우평가** → 사람 승인
# MAGIC 실버에서 S1이 놓친 정형 span을 모아 값→**shape**→후보 정규식으로 일반화하고, **드리프트 gold로 회복(섀도우)**·
# MAGIC **클린 메인으로 FP 가드**를 통과한 것만 제안한다. **헤드라인(결정론) = gold 기준 구조화 recall 0→~1.0**.
# MAGIC 기본은 **제안만**(검토 대기); `APPROVE=true` 위젯에서만 `S1_FIXED`를 드리프트에 영속화.

# COMMAND ----------

dbutils.widgets.dropdown("APPROVE", "false", ["false", "true"], "교정안 적용(사람 승인)")
APPROVE = dbutils.widgets.get("APPROVE").strip().lower() == "true"


def _shape(val):
    def cls(c):
        return "D" if c.isdigit() else ("L" if "A" <= c <= "Z" else ("l" if "a" <= c <= "z" else None))
    out, i, n = [], 0, len(val)
    while i < n:
        c = cls(val[i]); j = i
        while j < n and cls(val[j]) == c:
            j += 1
        out.append(f"{c}{{{j-i}}}" if c else val[i:j]); i = j
    return "".join(out)


def _shape_to_regex(shape):
    pat = _re.sub(r"D\{(\d+)\}", r"\\d{\1}", shape)
    pat = _re.sub(r"L\{(\d+)\}", r"[A-Z]{\1}", pat)
    pat = _re.sub(r"l\{(\d+)\}", r"[a-z]{\1}", pat)
    return r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])"


def augmented_spans(text, candidates):
    """base _PATTERNS + 후보로 regex_spans 동일 overlap 해소(복제)."""
    if not text:
        return []
    cand = []
    for et, rx, prio in list(span_patterns._PATTERNS) + candidates:
        for m in rx.finditer(text):
            cand.append({"start": m.start(), "end": m.end(), "entity_type": et,
                         "pii_value": m.group(0), "score": 1.0, "_p": prio})
    cand.sort(key=lambda s: (-s["_p"], -(s["end"] - s["start"]), s["start"]))
    chosen = []
    for s in cand:
        if any(not (s["end"] <= c["start"] or s["start"] >= c["end"]) for c in chosen):
            continue
        chosen.append(s)
    for c in chosen:
        c.pop("_p", None)
    return sorted(chosen, key=lambda s: s["start"])


# 진단: 실버 정형 타입인데 S1(drift)이 놓친 span
s1_off = {}
for d, s, e in sql(f"SELECT doc_id, start_char, end_char FROM {FQ}.span_predictions_drift WHERE stage='S1'")[1]:
    s1_off.setdefault(d, set()).add((int(s), int(e)))
leaked = [(d, int(s), int(e), t, v) for d, s, e, t, v in
          sql(f"SELECT doc_id, start_char, end_char, entity_type, pii_value FROM {FQ}.audit_spans_silver")[1]
          if t in REGEX_COVERABLE and (int(s), int(e)) not in s1_off.get(d, set())]

# (타입,shape) 클러스터 → 후보 정규식 + 섀도우평가(드리프트 회복 gain, 클린 메인 FP)
clusters = {}
for (d, s, e, t, v) in leaked:
    clusters.setdefault((t, _shape(v)), []).append((d, s, e, v))
clean_docs = sql(f"SELECT doc_id, text FROM {FQ}.text_corpus ORDER BY xxhash64(doc_id) LIMIT 800")[1]
clean_gold = {}
_cids = ",".join(repr(d) for d, _ in clean_docs)
for d, s, e in sql(f"SELECT doc_id, start_char, end_char FROM {FQ}.span_ground_truth WHERE doc_id IN ({_cids})")[1]:
    clean_gold.setdefault(d, set()).add((int(s), int(e)))
PRIO = {"RRN": 8, "ACCOUNT": 7, "CARD": 7, "PHONE": 6, "PASSPORT": 5, "IMEI": 4, "EMAIL": 9}
proposals, passing_comp = [], []
for (t, shape), members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    if len(members) < 2:  # 최소 지지(과적합 방지)
        continue
    body = _shape_to_regex(shape)
    try:
        rx = _re.compile(body)
    except _re.error:
        continue
    prio = PRIO.get(t, 5)
    gain = sum(1 for (d, s, e, v) in members
               if any(sp["entity_type"] == t and sp["start"] == s and sp["end"] == e
                      for sp in augmented_spans(_tbd[d], [(t, rx, prio)])))
    fp = sum(1 for d, tx in clean_docs for m in rx.finditer(tx or "") if (m.start(), m.end()) not in clean_gold.get(d, set()))
    passed = gain > 0 and fp == 0
    proposals.append({"detector": "rule", "entity_type": t, "regex": body, "support": len(members),
                      "recall_gain": gain, "clean_fp": fp, "shadow_pass": bool(passed),
                      "promoted": bool(passed and APPROVE and not degraded)})
    if passed and not degraded:
        passing_comp.append((t, rx, prio))
# NER/LLM stub 행 (60b·문서 참조)
proposals.append({"detector": "ner", "entity_type": "PERSON/ADDRESS", "regex": None, "support": None,
                  "recall_gain": None, "clean_fp": None, "shadow_pass": None, "promoted": False})
load_rows(f"{FQ}.remediation_proposals", proposals, mode="replace")

# 헤드라인(결정론): gold 기준 구조화(REGEX_COVERABLE) recall — before(S1) vs after(증강)
gold_struct = {}
for d, s, e, t in sql(f"SELECT doc_id, start_char, end_char, entity_type FROM {FQ}.span_ground_truth_drift WHERE entity_type IN {tuple(REGEX_COVERABLE)}")[1]:
    gold_struct.setdefault(d, set()).add((int(s), int(e)))
g_total = sum(len(v) for v in gold_struct.values())
before_hit = after_hit = 0
s1_fixed = []
for d, sset in gold_struct.items():
    boff = {(sp["start"], sp["end"]) for sp in regex_spans(_tbd[d])}
    aug = augmented_spans(_tbd[d], passing_comp)
    aoff = {(sp["start"], sp["end"]) for sp in aug}
    before_hit += len(sset & boff); after_hit += len(sset & aoff)
    s1_fixed += [{"stage": "S1_FIXED", "backend": "NA", "doc_id": d, "start_char": sp["start"],
                  "end_char": sp["end"], "entity_type": sp["entity_type"], "pii_value": sp["pii_value"],
                  "score": sp["score"]} for sp in aug]
before = round(before_hit / g_total, 4) if g_total else 0.0
after = round(after_hit / g_total, 4) if g_total else 0.0
n_pass = sum(1 for p in proposals if p.get("shadow_pass"))
print(f"  진단: S1 미탐 정형 실버 {len(leaked)}건 · 후보 통과 {n_pass}개")
print(f"  ★ 구조화 정형 recall(gold 기준·결정론): before={before} → after={after} (n_gold={g_total})")

# 사람 승인 게이트: 기본 제안만. APPROVE 시에만 S1_FIXED(드리프트 한정) 영속화
sql(f"DELETE FROM {FQ}.span_predictions_drift WHERE stage='S1_FIXED'")
if APPROVE and not degraded and passing_comp and s1_fixed:
    load_rows(f"{FQ}.span_predictions_drift", s1_fixed)
    print("  ✓ (APPROVE=true) S1_FIXED 적용 — 드리프트 한정. 메인 불변.")
else:
    print("  ⏸ 적용 보류(기본=사람 검토 승인). 회복은 gold 섀도우로 이미 검증됨. 적용하려면 APPROVE 위젯=true 후 재실행.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 여기까지 — Day-2 Rung A 요약
# MAGIC | 항목 | 값(예시·합성) | 의미 |
# MAGIC |---|---|---|
# MAGIC | 라벨없는 신호 | 룰침묵·LLM활성 ↑ vs 메인 | 라벨 0으로 저하 탐지 |
# MAGIC | Auditor 보정 | frozen-gold recall ≥0.9 → degraded=False | 심판을 다시 검증 |
# MAGIC | **규칙 회복(gold)** | **0.0 → ~1.0** | 가장 싼 레버로 폐곡선 |
# MAGIC | 거버넌스 | 제안→섀도우→**사람 승인** | 자동 적용 아님(기본) |
# MAGIC
# MAGIC > S4는 *만들기*, Day-2는 *운영하기* — 같은 5원칙(인간검토·섀도우·심판보정·격리·단일평가)이 더 비싼 레버에 다시 적용된다.

# COMMAND ----------

# MAGIC %md ## 모니터 self-check (인쇄 — 아키텍처 10/10과 별개)

# COMMAND ----------

mres = []
def mchk(name, ok, detail):
    mres.append(bool(ok)); print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name} — {detail}")


# M1 드리프트 offset 무결성
m1 = int(sql(f"""SELECT count(*) FROM {FQ}.span_ground_truth_drift g JOIN {FQ}.text_corpus_drift c USING(doc_id)
  WHERE substring(c.text, g.start_char+1, g.end_char-g.start_char) <> g.pii_value""")[1][0][0])
mchk("M1 드리프트 offset 무결성", m1 == 0, f"불일치={m1}")
# M2 R4 정직성(현재 정규식이 구조화 변형 미탐)
mchk("M2 R4 정직성", _missed == _checked and _checked > 0, f"정규식 미탐 {_missed}/{_checked}")
# M3 라벨없는 저하 관측(드리프트 silent > 메인)
mchk("M3 라벨없는 저하 신호", drift_silent > base_silent, f"drift={drift_silent} > main={base_silent}")
# M4 결정론적 회복(gold 기준 after > before)
mchk("M4 규칙 회복(gold)", after > before, f"{before} → {after}")
# M5 메인 격리(메인 span_predictions에 S1_FIXED/드리프트 0건)
m5 = int(sql(f"SELECT count(*) FROM {FQ}.span_predictions WHERE stage='S1_FIXED' OR doc_id LIKE 'drift.%'")[1][0][0])
mchk("M5 메인 격리(불변)", m5 == 0, f"메인 누수={m5}")
# M6 거버넌스(미승인 시 S1_FIXED 드리프트에도 0)
m6 = int(sql(f"SELECT count(*) FROM {FQ}.span_predictions_drift WHERE stage='S1_FIXED'")[1][0][0])
mchk("M6 사람승인 게이트", (m6 > 0) == (APPROVE and not degraded and bool(passing_comp)), f"S1_FIXED(drift)={m6}, APPROVE={APPROVE}")
load_rows(f"{FQ}.monitor_verification_report",
          [{"check": f"M{i+1}", "result": "PASS" if v else "FAIL"} for i, v in enumerate(mres)], mode="replace")
print(f"\n== 모니터 self-check {sum(mres)}/{len(mres)} PASS == (아키텍처 90_eval 10/10과 독립)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`60b_ner_challenger_retrain`](./60b_ner_challenger_retrain)** — 이름·주소처럼 정규식으로 못 잡는 드리프트는
# MAGIC NER **재학습(챔피언/챌린저) + 사람 승인 승격 게이트**로 닫는다(가장 비싼 레버). 개념·성숙도 사다리·내 운영 이식은
# MAGIC **[`docs/learn/07_운영_모니터링_Day2.md`](../docs/learn/07_운영_모니터링_Day2.md)**.
