# Databricks notebook source
# MAGIC %md
# MAGIC # S4a — 분리 학습 코퍼스 생성 (seed=777)
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC S4는 KoELECTRA를 **파인튜닝**합니다. 만약 평가 코퍼스(seed=42)로 학습하고 같은 코퍼스에
# MAGIC 추론·평가하면 F1이 학습 데이터 암기(train-test 누수)로 부풀려집니다(실측 0.9998). 이 노트북은
# MAGIC **다른 seed(777)로 완전 분리된 학습 코퍼스**를 만들어 누수를 차단합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC 동일 템플릿·동일 분포에서 seed만 다르게 생성합니다 → 현실 시나리오("과거 라벨링된 상담
# MAGIC 데이터로 파인튜닝 → 신규 데이터에 적용")와 동일. 짧은 템플릿+유한한 이름 풀 때문에 seed가
# MAGIC 달라도 드물게(~0.8%) 평가 코퍼스와 **verbatim 일치**하는 문장이 나오므로, 그런 doc은 학습에서
# MAGIC 제외해 **암기 누수 0**을 보장합니다. 평가는 기존 `text_corpus`(약 4,860 docs)에서 그대로 산출돼
# MAGIC 타 단계와 비교 가능합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC SOURCE: `00_data/03_generate_train_corpus.py`. 생성기(`gen_text_column`·`TYPE_MAP`)는 seed=42
# MAGIC 데이터 생성과 동일 로직을 인라인하되 `TRAIN_SEED=777` 로 시딩합니다. verbatim 중복 판정의
# MAGIC '평가 텍스트'는 노트북 네이티브로 `{FQ}.text_corpus` 에서 읽습니다(원본의 CSV 스캔과 동치).
# MAGIC
# MAGIC 🔧 **내 데이터로 파인튜닝하려면**: 이 합성 생성기 대신 내 라벨 표본을 같은 스키마의
# MAGIC `text_corpus_train`(doc_id·text) / `span_gt_train`(doc_id·offset·entity_type·pii_value)에 적재하고,
# MAGIC 학습-평가 분리(평가셋과 verbatim 중복 제거)를 유지하세요. → `../docs/learn/06_내_데이터에_적용.md`

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: 기대 산출물]
# MAGIC `{FQ}.text_corpus_train`(doc_id·text, disjoint seed=777) + `{FQ}.span_gt_train`
# MAGIC (doc_id·start/end·entity_type·pii_value). 적재 무결성: 텍스트 NULL 0건 + GT offset substring 일치.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC 합성 데이터라 학습·평가 분포가 동일합니다(실데이터의 도메인 시프트 없음). 그래도 verbatim 누수
# MAGIC 제거로 '암기'는 배제되어, 모델의 일반화(같은 분포 내 신규 문장 탐지)는 정직하게 측정됩니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC ➡️ **[`S4b_ner_finetune_cascade`](./S4b_ner_finetune_cascade)** [ML 클러스터] 가 이 학습 코퍼스로만 파인튜닝합니다(train-test 분리 보존).

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 생성기 인라인 (seed=42 데이터 생성과 동일 로직, 이 노트북은 seed=777로 시딩)
# MAGIC `01_generate_data` 의 `gen_text_column`/`TYPE_MAP`/템플릿을 그대로 옮깁니다.

# COMMAND ----------

import random as _random
import re as _re

try:
    from faker import Faker
except ImportError:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "faker"])
    from faker import Faker

TRAIN_SEED = 777  # 평가 corpus(seed=42)와 분리
fake = Faker("ko_KR")

_SLOT_RE = _re.compile(r"\{([A-Z_]+)\}")


def gen_phone():
    return f"010-{_random.randint(1000,9999)}-{_random.randint(1000,9999)}"


def gen_rrn():
    yy, mm, dd = _random.randint(0, 99), _random.randint(1, 12), _random.randint(1, 28)
    g = _random.choice([1, 2, 3, 4])
    return f"{yy:02d}{mm:02d}{dd:02d}-{g}{_random.randint(100000, 999999)}"


def gen_card():
    digits = [_random.randint(0, 9) for _ in range(15)]
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    full = digits + [(10 - (s % 10)) % 10]
    n = "".join(map(str, full))
    return f"{n[0:4]}-{n[4:8]}-{n[8:12]}-{n[12:16]}"


def gen_account():
    return f"{_random.randint(100,999)}-{_random.randint(10,99)}-{_random.randint(100000,999999)}"


def gen_imei():
    return "".join(str(_random.randint(0, 9)) for _ in range(15))


def gen_email():
    return fake.email()


def gen_address():
    return fake.address().replace("\n", " ")


def _person():
    return fake.name()


def _phone():
    p = gen_phone()
    return p.replace("-", "") if _random.random() < 0.30 else p


def _card():
    c = gen_card()
    return c.replace("-", "") if _random.random() < 0.25 else c


def _passport():
    return _random.choice("MSROD") + "".join(str(_random.randint(0, 9)) for _ in range(8))


SLOT_GEN = {
    "PERSON": _person, "PHONE": _phone, "RRN": gen_rrn, "CARD": _card,
    "ACCOUNT": gen_account, "EMAIL": gen_email, "ADDRESS": gen_address,
    "IMEI": gen_imei, "PASSPORT": _passport,
}

TEMPLATES = {
    "consult_note": [
        "고객 {PERSON}님이 {PHONE}로 요금 문의함. 본인확인 위해 주민번호 {RRN} 대조 완료.",
        "{PERSON} 고객 단말 분실 신고 접수. IMEI {IMEI}, 회신 연락처 {PHONE}.",
        "명의자 {PERSON}, 자동이체 계좌 {ACCOUNT} 변경 요청. 확인메일 {EMAIL}로 발송함.",
        "{ADDRESS} 거주 {PERSON} 고객 이전설치 요청, 연락처 {PHONE} 남김.",
        "결제카드 {CARD} 승인오류 문의. 고객 {PERSON}, 회신 번호 {PHONE}.",
        "요금제 변경 가능 여부 문의. 추가 비용 없음 안내하고 종료.",
        "{PERSON}님 명의 회선 해지 접수, 위약금 안내 완료.",
        "외국인 가입 상담, 여권 {PASSPORT} 기준 신원확인. 담당 {PERSON}.",
        "데이터 속도 저하 클레임. 기지국 점검 요청 등록함.",
        "{PERSON} 고객 명의도용 의심 신고, 주민번호 {RRN}로 가입이력 조회.",
    ],
    "complaint_body": [
        "{PERSON}입니다. {PHONE}으로 상담원이 전화했는데 응대가 불친절했습니다. 이메일 {EMAIL}로 답변 바랍니다.",
        "기지국 문제로 통화가 자꾸 끊깁니다. 제 번호는 {PHONE}이고 주소는 {ADDRESS}입니다.",
        "앱에서 요금 조회가 안 됩니다. 오류 화면 첨부하니 확인 부탁드립니다.",
        "{PERSON} 본인입니다. 카드 {CARD} 이중청구 건 환불 요청합니다. 환불계좌 {ACCOUNT}.",
        "약정 안내를 제대로 못 받았습니다. 시정 요구합니다.",
        "명의자 {PERSON}, 가입 시 받은 사은품 미수령. 연락처 {PHONE}로 회신 주세요.",
    ],
    "memo": [
        "{PERSON} VIP 고객, 재연락 {PHONE}",
        "환불계좌 {ACCOUNT} 등록 요망",
        "",
        "",
        "{PERSON} 미납 건, 독촉 연락 {PHONE}",
        "본인확인 완료 {RRN}",
        "특이사항 없음",
    ],
    "review_text": [
        "매장 직원이 친절했어요. 개통도 빨랐습니다.",
        "요금제 설명이 좀 부족했네요. 그래도 만족합니다.",
        "대기 시간이 길었지만 처리는 깔끔했어요.",
        "상담사 {PERSON}님 덕분에 빠르게 해결했어요. 직통 {PHONE} 공유받음.",
        "신호가 약한 지역인데 안내가 좋았습니다.",
        "{PERSON} 점장님이 추천해준 요금제 아주 만족스러워요.",
        "가격이 합리적이고 사은품도 좋았습니다.",
        "재방문 의사 있어요. 추천합니다.",
    ],
}


def fill(template):
    parts, spans, cursor, pos = [], [], 0, 0
    for m in _SLOT_RE.finditer(template):
        lit = template[pos:m.start()]
        parts.append(lit)
        cursor += len(lit)
        etype = m.group(1)
        value = SLOT_GEN[etype]()
        parts.append(value)
        spans.append({"start": cursor, "end": cursor + len(value),
                      "entity_type": etype, "pii_value": value})
        cursor += len(value)
        pos = m.end()
    parts.append(template[pos:])
    text = "".join(parts)
    for s in spans:
        assert text[s["start"]:s["end"]] == s["pii_value"], (text, s)
    return text, spans


def gen_text_column(column):
    tpl = _random.choice(TEMPLATES[column])
    return fill(tpl)


print("생성기 인라인 완료 (TRAIN_SEED=777)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일: seed=777 학습 코퍼스 생성 (평가 코퍼스 verbatim 중복 제외)
# MAGIC 평가 corpus의 컬럼 분포에 근사(memo 2000 / consult_note 1200 / complaint_body 800 /
# MAGIC review_text 1000). 빈 메모(의도된 결측)와 평가 텍스트와 verbatim 일치하는 doc은 제외합니다.

# COMMAND ----------

N_PER_COLUMN = {"memo": 2000, "consult_note": 1200, "complaint_body": 800, "review_text": 1000}

# 평가 텍스트 = 노트북 네이티브로 text_corpus에서(원본 _eval_texts()의 CSV 스캔과 동치)
eval_texts = {t for (t,) in sql(f"SELECT text FROM {FQ}.text_corpus")[1]}

Faker.seed(TRAIN_SEED)
_random.seed(TRAIN_SEED)

n_dropped = 0
corpus_rows, gt_rows = [], []
for column, n in N_PER_COLUMN.items():
    for i in range(n):
        text, spans = gen_text_column(column)
        if not text:           # 빈 메모(의도된 결측) — 평가 corpus 빌드와 동일하게 제외
            continue
        if text in eval_texts:  # 평가 corpus와 verbatim 일치 → 암기 누수 방지로 제외
            n_dropped += 1
            continue
        doc_id = f"train.{column}.{i:06d}"
        corpus_rows.append({"doc_id": doc_id, "text": text})
        for s in spans:
            gt_rows.append({"doc_id": doc_id, "start_char": s["start"], "end_char": s["end"],
                            "entity_type": s["entity_type"], "pii_value": s["pii_value"]})

print(f"  (평가 corpus와 verbatim 중복 {n_dropped}건 학습에서 제외)")

# self-assert: offset 무결성 (로컬)
_text_by_doc = {r["doc_id"]: r["text"] for r in corpus_rows}
for g in gt_rows:
    assert _text_by_doc[g["doc_id"]][g["start_char"]:g["end_char"]] == g["pii_value"], \
        f"offset mismatch {g['doc_id']}"

n_pa = sum(1 for g in gt_rows if g["entity_type"] in ("PERSON", "ADDRESS"))
print(f"  ✓ train_corpus: {len(corpus_rows):,} docs / train_span_gt: {len(gt_rows):,} spans "
      f"(PERSON/ADDRESS {n_pa:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 테이블 적재: text_corpus_train / span_gt_train
# MAGIC `load_rows(mode='replace')` 로 Delta에 직접 적재합니다(DataFrame 경로라 임베디드 따옴표 안전).

# COMMAND ----------

load_rows(f"{FQ}.text_corpus_train", corpus_rows, mode="replace")
# start/end는 STRING으로 적재되므로 INT 캐스팅 뷰로 재생성(원본 read_files CAST와 동치)
load_rows(f"{FQ}.span_gt_train", gt_rows, mode="replace")
sql(f"""CREATE OR REPLACE TABLE {FQ}.span_gt_train AS
  SELECT doc_id, CAST(start_char AS INT) AS start_char, CAST(end_char AS INT) AS end_char,
         entity_type, pii_value FROM {FQ}.span_gt_train""")
print("  loaded text_corpus_train / span_gt_train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 적재 무결성 검증 셀 (NULL 0, offset substring 일치)
# MAGIC 원본 `load()` 의 무결성 검증 SQL을 그대로 이식합니다. 둘 중 하나라도 0이 아니면 실패시킵니다.

# COMMAND ----------

_chk = sql(f"""SELECT
  (SELECT count(*) FROM {FQ}.text_corpus_train WHERE text IS NULL) AS null_text,
  (SELECT count(*) FROM {FQ}.span_gt_train g JOIN {FQ}.text_corpus_train c USING(doc_id)
    WHERE substring(c.text, g.start_char+1, g.end_char-g.start_char) <> g.pii_value) AS offset_mismatch""")[1]
null_text, offset_mismatch = int(_chk[0][0]), int(_chk[0][1])
if null_text or offset_mismatch:
    raise RuntimeError(f"학습 corpus 적재 무결성 실패: null_text={null_text} offset_mismatch={offset_mismatch}")

_cnt = sql(f"""SELECT (SELECT count(*) FROM {FQ}.text_corpus_train) AS n_train,
  (SELECT count(*) FROM {FQ}.span_gt_train) AS n_span_gt""")[1]
print(f"  무결성 OK — text_corpus_train={_cnt[0][0]} span_gt_train={_cnt[0][1]} "
      f"(null_text={null_text}, offset_mismatch={offset_mismatch})")
display(sql(f"SELECT * FROM {FQ}.text_corpus_train LIMIT 5")[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`S4b_ner_finetune_cascade`](./S4b_ner_finetune_cascade)** [ML 클러스터] — 이 분리 코퍼스로 KoELECTRA를 파인튜닝하고 S4 cascade를 수행합니다.
# MAGIC
# MAGIC > 참고: S3(`30_S3_pattern_ner_llm/S3_ner_base_cascade`)도 ML 클러스터 노트북입니다. S3는 사전학습 NER만 쓰므로 이 학습 코퍼스가 필요 없지만, S4b는 위에서 만든 `text_corpus_train` 이 선행되어야 합니다.
