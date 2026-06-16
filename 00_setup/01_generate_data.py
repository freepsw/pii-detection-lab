# Databricks notebook source
# MAGIC %md
# MAGIC # 0-1. 합성 데이터 생성 + raw 볼륨 업로드
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC PII 탐지 아키텍처를 비교하려면 **정답(ground truth)을 아는** 데이터가 필요합니다. 실제 고객
# MAGIC 데이터는 정답 라벨이 없고 외부 반출도 불가합니다. 그래서 한국 통신사 CS 도메인을 본떠
# MAGIC **합성 데이터 + 이중 정답(컬럼단위·span단위)** 을 `seed=42` 로 결정적으로 생성합니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [직관/원리]
# MAGIC 핵심 설계는 **template-insertion**: 텍스트 템플릿의 `{TYPE}` 슬롯을 값으로 치환하면서
# MAGIC 각 값의 `(start, end)` char offset을 동시에 기록합니다. 따라서 생성 시점에
# MAGIC `text[start:end] == pii_value` 가 100% 보장됩니다(사후 정렬/탐지 불필요 → offset 드리프트 차단).
# MAGIC 또한 정형 PII(전화·카드·주민번호 등)는 `source_layer=PATTERN`, 이름·주소는 `NER` 로 귀속해
# MAGIC **단계별 recall 차이가 데이터에 내장**됩니다(S1 패턴만으론 PERSON/ADDRESS 미탐 → S3/S4 NER이 회수).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [코드로 보기]
# MAGIC 생성 로직(`kr_pii_data_generators` · `text_templates_ko` · 9개 테이블 빌더)을 이 노트북에
# MAGIC 인라인했습니다 — 고객이 합성·라벨링 로직을 직접 검토할 수 있도록. 모두 순수 Python(faker)이며
# MAGIC `seed=42` 로 고정되어 검증 에이전트가 동일하게 재생성할 수 있습니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## [결과: 기대 산출물]
# MAGIC | 산출물 | 행수(approx) | 비고 |
# MAGIC |---|---|---|
# MAGIC | subscribers / billing | 2,000 / 2,000 | 구조화 + memo 자유텍스트 |
# MAGIC | call_records / employees | 5,000 / 300 | 구조화 |
# MAGIC | cs_consultations / reviews | 1,200 / 1,500 | 텍스트 네이티브 |
# MAGIC | column_ground_truth | 51 컬럼 | is_pii·category(자유텍스트=FREE_TEXT_PII) |
# MAGIC | span_ground_truth | 수천 span | doc_id·offset·entity_type·source_layer |
# MAGIC | type_map | 9 유형 | entity_type→category→mask_token→layer |
# MAGIC
# MAGIC text_corpus 평가 코퍼스는 약 **4,860 docs** 규모입니다(02_load_tables에서 long 포맷 변환).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [한계]
# MAGIC 합성 값은 '형식만 맞는' 가짜 PII입니다(카드번호만 Luhn 유효). 실제 분포와 다를 수 있으며,
# MAGIC S4의 라우팅 0% 같은 일부 수치는 합성 분포의 영향을 받습니다(90_eval에서 정직하게 명시).

# COMMAND ----------

# MAGIC %md
# MAGIC ## [다음 단계로]
# MAGIC 생성한 CSV를 `RAW_VOL` 에 업로드한 뒤 → `02_load_tables` 로 Delta 테이블 9종 + text_corpus를 만듭니다.

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 생성 경로 선택 (위젯)
# MAGIC 패키지에는 이미 검증된 CSV가 `../data/` 에 커밋돼 있습니다. 기본값은 **기존 CSV 업로드**입니다.
# MAGIC 합성 로직을 처음부터 재실행하려면 `regenerate=true` 로 바꾸세요(동일 seed=42 → 바이트 동일 산출).

# COMMAND ----------

dbutils.widgets.dropdown("regenerate", "false", ["false", "true"], "데이터 재생성 여부")
REGENERATE = dbutils.widgets.get("regenerate").strip().lower() == "true"

import os

# 패키지 data/ 디렉터리(노트북 워크스페이스 경로 기준) — 기존 CSV 업로드 경로
_nb_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
_pkg_root = os.path.dirname(os.path.dirname(_nb_path))      # .../03_pii_detection_lab
DATA_WS = f"/Workspace{_pkg_root}/data"
# 로컬(드라이버) 작업 디렉터리 — 재생성 시 CSV를 쓸 곳
LOCAL_DATA = "/tmp/pii_lab_data"
os.makedirs(LOCAL_DATA, exist_ok=True)

CSV_FILES = [
    "subscribers.csv", "billing.csv", "call_records.csv", "employees.csv",
    "cs_consultations.csv", "reviews.csv",
    "column_ground_truth.csv", "span_ground_truth.csv", "type_map.csv",
]
print(f"REGENERATE={REGENERATE} | DATA_WS={DATA_WS} | RAW_VOL={RAW_VOL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 생성기 (1) — `kr_pii_data_generators`: 한국형 PII 합성 값 (스키마 독립)
# MAGIC 모두 형식만 맞는 합성 값(카드만 Luhn 유효). `Faker("ko_KR")` + `seed=42`.

# COMMAND ----------

import csv as _csv
import random as _random

try:
    from faker import Faker
except ImportError:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "faker"])
    from faker import Faker

SEED = 42
fake = Faker("ko_KR")
Faker.seed(SEED)
_random.seed(SEED)


def gen_phone():
    return f"010-{_random.randint(1000,9999)}-{_random.randint(1000,9999)}"


def gen_rrn():
    """주민등록번호 형식 (합성, 유효 주민번호 아님). YYMMDD-Gxxxxxx"""
    yy, mm, dd = _random.randint(0, 99), _random.randint(1, 12), _random.randint(1, 28)
    g = _random.choice([1, 2, 3, 4])
    return f"{yy:02d}{mm:02d}{dd:02d}-{g}{_random.randint(100000, 999999)}"


def gen_card():
    """16자리 카드번호 (Luhn 유효), 4-4-4-4 형식"""
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


def gen_card_expiry():
    return f"{_random.randint(1,12):02d}/{_random.randint(26,32)}"


def gen_account():
    """은행 계좌번호 형식"""
    return f"{_random.randint(100,999)}-{_random.randint(10,99)}-{_random.randint(100000,999999)}"


def gen_imei():
    return "".join(str(_random.randint(0, 9)) for _ in range(15))


def gen_email():
    return fake.email()


def gen_address():
    return fake.address().replace("\n", " ")


def gen_birth():
    return fake.date_of_birth(minimum_age=19, maximum_age=80).isoformat()


print("PII 값 생성기 로드 완료 (gen_phone/rrn/card/account/imei/email/address/birth)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 생성기 (2) — `text_templates_ko`: 비정형 한국어 텍스트 + 정확한 char offset
# MAGIC `fill()` 이 `{TYPE}` 슬롯을 채우면서 생성 시점에 `text[start:end]==pii_value` 를 self-assert 합니다.

# COMMAND ----------

import re as _re

_SLOT_RE = _re.compile(r"\{([A-Z_]+)\}")


def _person():
    return fake.name()


def _phone():
    p = gen_phone()
    return p.replace("-", "") if _random.random() < 0.30 else p


def _rrn():
    return gen_rrn()


def _card():
    c = gen_card()
    return c.replace("-", "") if _random.random() < 0.25 else c


def _account():
    return gen_account()


def _email():
    return gen_email()


def _address():
    return gen_address()


def _imei():
    return gen_imei()


def _passport():
    return _random.choice("MSROD") + "".join(str(_random.randint(0, 9)) for _ in range(8))


SLOT_GEN = {
    "PERSON": _person, "PHONE": _phone, "RRN": _rrn, "CARD": _card,
    "ACCOUNT": _account, "EMAIL": _email, "ADDRESS": _address,
    "IMEI": _imei, "PASSPORT": _passport,
}

# entity_type → (category, presidio_entity, mask_token, source_layer)
TYPE_MAP = {
    "RRN":      ("IDENTIFICATION", "KR_RRN",          "[주민번호]",     "PATTERN"),
    "PHONE":    ("PERSONAL_INFO",  "PHONE_NUMBER",    "[전화번호]",     "PATTERN"),
    "EMAIL":    ("PERSONAL_INFO",  "EMAIL_ADDRESS",   "[이메일]",       "PATTERN"),
    "CARD":     ("PAYMENT_INFO",   "CREDIT_CARD",     "[카드번호]",     "PATTERN"),
    "ACCOUNT":  ("PAYMENT_INFO",   "KR_BANK_ACCOUNT", "[계좌번호]",     "PATTERN"),
    "IMEI":     ("IDENTIFICATION", "KR_IMEI",         "[단말식별번호]", "PATTERN"),
    "PASSPORT": ("IDENTIFICATION", "KR_PASSPORT",     "[여권번호]",     "PATTERN"),
    "PERSON":   ("PERSONAL_INFO",  "PERSON",          "[이름]",         "NER"),
    "ADDRESS":  ("PERSONAL_INFO",  "LOCATION",        "[주소]",         "NER"),
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
    """템플릿의 {TYPE} 슬롯을 채우고 (text, spans) 반환. text[start:end]==pii_value 보장."""
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


print("텍스트 템플릿 로드 완료 (TYPE_MAP 9종, TEMPLATES 4컬럼)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 생성기 (3) — 9개 테이블 빌더 + 이중 정답 누적
# MAGIC `01_generate_data.py` 의 테이블 생성 함수를 인라인. 날짜는 고정 앵커(2026-06-01) 기준으로
# MAGIC 생성해 seed=42 완전 재현을 보장합니다(상대 '오늘' 기준 비결정성 제거).

# COMMAND ----------

import datetime as _dt

ANCHOR = _dt.date(2026, 6, 1)
ANCHOR_DT = _dt.datetime(2026, 6, 1)

N_SUB, N_BILL, N_CALL, N_EMP = 2000, 2000, 5000, 300
N_CONSULT, N_REVIEW = 1200, 1500

SPANS = []  # (doc_id, table, column, row_key, start, end, entity_type, pii_value, source_layer)


def _emit_text(table, column, row_key):
    text, spans = gen_text_column(column)
    doc_id = f"{table}.{column}.{row_key}"
    for s in spans:
        SPANS.append([doc_id, table, column, row_key, s["start"], s["end"],
                      s["entity_type"], s["pii_value"], TYPE_MAP[s["entity_type"]][3]])
    return text


def _write(name, header, rows):
    path = os.path.join(LOCAL_DATA, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)
    print(f"  ✓ {name}: {len(rows):,} rows ({len(header)} cols)")
    return path


def gen_subscribers():
    header = ["subscriber_id", "name_ko", "rrn", "phone", "email", "address",
              "birth_date", "gender", "imei", "x_contact", "card_type",
              "plan_code", "status", "region_code", "join_date", "memo"]
    plans = ["5G_PREMIUM", "5G_STANDARD", "LTE_BASIC", "LTE_Safe", "DATA_ONLY"]
    statuses = ["ACTIVE", "SUSPENDED", "DORMANT", "TERMINATED"]
    regions = ["02", "031", "032", "051", "053", "042", "062", "064"]
    card_types = ["CREDIT", "DEBIT", "PREPAID"]
    rows = []
    for i in range(N_SUB):
        sid = f"SUB{i+100000:08d}"
        memo = _emit_text("subscribers", "memo", sid)
        rows.append([
            sid, fake.name(), gen_rrn(), gen_phone(), gen_email(), gen_address(),
            gen_birth(), _random.choice(["M", "F"]), gen_imei(), gen_phone(),
            _random.choice(card_types), _random.choice(plans), _random.choice(statuses),
            _random.choice(regions),
            fake.date_between(start_date="-5y", end_date=ANCHOR).isoformat(),
            memo,
        ])
    return _write("subscribers.csv", header, rows)


def gen_billing():
    header = ["bill_id", "subscriber_id", "card_number", "card_expiry",
              "bank_account", "amount", "billing_month", "paid_yn", "due_date"]
    rows = []
    for i in range(N_BILL):
        rows.append([
            f"BILL{i+1:09d}", f"SUB{_random.randint(100000, 100000+N_SUB-1):08d}",
            gen_card(), gen_card_expiry(), gen_account(),
            _random.randint(15000, 250000), f"2025-{_random.randint(1,12):02d}",
            _random.choice(["Y", "N"]),
            fake.date_between(start_date="-1y", end_date=ANCHOR).isoformat(),
        ])
    return _write("billing.csv", header, rows)


def gen_calls():
    header = ["call_id", "caller_no", "callee_no", "duration_sec",
              "cell_id", "call_ts", "call_type"]
    rows = []
    for i in range(N_CALL):
        rows.append([
            f"CALL{i+1:010d}", gen_phone(), gen_phone(), _random.randint(0, 3600),
            f"CELL{_random.randint(10000,99999)}",
            fake.date_time_between(start_date=ANCHOR_DT - _dt.timedelta(days=90), end_date=ANCHOR_DT).isoformat(sep=" "),
            _random.choice(["VOICE", "SMS", "VIDEO", "DATA"]),
        ])
    return _write("call_records.csv", header, rows)


def gen_employees():
    header = ["emp_id", "name_ko", "email", "phone", "salary",
              "dept", "position", "hire_date"]
    depts = ["네트워크운영", "고객서비스", "마케팅", "재무", "인사", "기술개발"]
    positions = ["사원", "대리", "과장", "차장", "부장", "팀장"]
    rows = []
    for i in range(N_EMP):
        rows.append([
            f"EMP{i+1:06d}", fake.name(), gen_email(), gen_phone(),
            _random.randint(35000000, 120000000),
            _random.choice(depts), _random.choice(positions),
            fake.date_between(start_date="-15y", end_date=ANCHOR).isoformat(),
        ])
    return _write("employees.csv", header, rows)


def gen_consultations():
    header = ["consult_id", "subscriber_id", "channel", "consult_ts",
              "consult_note", "complaint_body", "category_code"]
    channels = ["콜센터", "챗봇", "이메일", "매장"]
    cats = ["BILLING", "NETWORK", "DEVICE", "PLAN_CHANGE", "COMPLAINT", "ETC"]
    rows = []
    for i in range(N_CONSULT):
        cid = f"CONS{i+1:08d}"
        note = _emit_text("cs_consultations", "consult_note", cid)
        body = _emit_text("cs_consultations", "complaint_body", cid) if _random.random() < 0.6 else ""
        rows.append([
            cid, f"SUB{_random.randint(100000, 100000+N_SUB-1):08d}",
            _random.choice(channels),
            fake.date_time_between(start_date=ANCHOR_DT - _dt.timedelta(days=180), end_date=ANCHOR_DT).isoformat(sep=" "),
            note, body, _random.choice(cats),
        ])
    return _write("cs_consultations.csv", header, rows)


def gen_reviews():
    header = ["review_id", "store_name", "review_text", "rating"]
    stores = ["텔코강남점", "텔코스토어 종로", "텔코플라자 부산서면", "텔코 대전둔산점",
              "telco 광주충장로", "텔코 수원역점"]
    rows = []
    for i in range(N_REVIEW):
        rid = f"REV{i+1:08d}"
        text = _emit_text("reviews", "review_text", rid)
        rows.append([rid, _random.choice(stores), text, _random.randint(1, 5)])
    return _write("reviews.csv", header, rows)


def gen_column_gt():
    header = ["table_name", "column_name", "is_pii", "category"]
    P, N, F = "true", "false", "FREE_TEXT_PII"
    gt = [
        ("subscribers", "subscriber_id", N, "NON_PII"),
        ("subscribers", "name_ko", P, "PERSONAL_INFO"),
        ("subscribers", "rrn", P, "IDENTIFICATION"),
        ("subscribers", "phone", P, "PERSONAL_INFO"),
        ("subscribers", "email", P, "PERSONAL_INFO"),
        ("subscribers", "address", P, "PERSONAL_INFO"),
        ("subscribers", "birth_date", P, "PERSONAL_INFO"),
        ("subscribers", "gender", N, "NON_PII"),
        ("subscribers", "imei", P, "IDENTIFICATION"),
        ("subscribers", "x_contact", P, "PERSONAL_INFO"),
        ("subscribers", "card_type", N, "NON_PII"),
        ("subscribers", "plan_code", N, "NON_PII"),
        ("subscribers", "status", N, "NON_PII"),
        ("subscribers", "region_code", N, "NON_PII"),
        ("subscribers", "join_date", N, "NON_PII"),
        ("subscribers", "memo", P, F),
        ("billing", "bill_id", N, "NON_PII"),
        ("billing", "subscriber_id", N, "NON_PII"),
        ("billing", "card_number", P, "PAYMENT_INFO"),
        ("billing", "card_expiry", P, "PAYMENT_INFO"),
        ("billing", "bank_account", P, "PAYMENT_INFO"),
        ("billing", "amount", N, "NON_PII"),
        ("billing", "billing_month", N, "NON_PII"),
        ("billing", "paid_yn", N, "NON_PII"),
        ("billing", "due_date", N, "NON_PII"),
        ("call_records", "call_id", N, "NON_PII"),
        ("call_records", "caller_no", P, "PERSONAL_INFO"),
        ("call_records", "callee_no", P, "PERSONAL_INFO"),
        ("call_records", "duration_sec", N, "NON_PII"),
        ("call_records", "cell_id", N, "NON_PII"),
        ("call_records", "call_ts", N, "NON_PII"),
        ("call_records", "call_type", N, "NON_PII"),
        ("employees", "emp_id", N, "NON_PII"),
        ("employees", "name_ko", P, "PERSONAL_INFO"),
        ("employees", "email", P, "PERSONAL_INFO"),
        ("employees", "phone", P, "PERSONAL_INFO"),
        ("employees", "salary", P, "EMPLOYMENT_INFO"),
        ("employees", "dept", N, "NON_PII"),
        ("employees", "position", N, "NON_PII"),
        ("employees", "hire_date", N, "NON_PII"),
        ("cs_consultations", "consult_id", N, "NON_PII"),
        ("cs_consultations", "subscriber_id", N, "NON_PII"),
        ("cs_consultations", "channel", N, "NON_PII"),
        ("cs_consultations", "consult_ts", N, "NON_PII"),
        ("cs_consultations", "consult_note", P, F),
        ("cs_consultations", "complaint_body", P, F),
        ("cs_consultations", "category_code", N, "NON_PII"),
        ("reviews", "review_id", N, "NON_PII"),
        ("reviews", "store_name", N, "NON_PII"),
        ("reviews", "review_text", P, F),
        ("reviews", "rating", N, "NON_PII"),
    ]
    return _write("column_ground_truth.csv", header, gt)


def gen_span_gt():
    header = ["doc_id", "table_name", "column_name", "row_key",
              "start_char", "end_char", "entity_type", "pii_value", "source_layer"]
    return _write("span_ground_truth.csv", header, SPANS)


def gen_type_map():
    header = ["entity_type", "category", "presidio_entity", "mask_token", "source_layer"]
    rows = [[k, v[0], v[1], v[2], v[3]] for k, v in TYPE_MAP.items()]
    return _write("type_map.csv", header, rows)


print("테이블 빌더 로드 완료 (subscribers/billing/calls/employees/consultations/reviews + GT 3종)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이 단계가 하는 일: 데이터 준비 (재생성 또는 기존 CSV 복사)
# MAGIC `REGENERATE=true` 면 생성기를 실행해 `/tmp/pii_lab_data` 에 CSV를 쓰고, 아니면 패키지 `../data/`
# MAGIC 에 커밋된 검증 CSV를 그대로 사용합니다. 어느 경로든 다음 셀에서 `RAW_VOL` 로 업로드됩니다.

# COMMAND ----------

if REGENERATE:
    print(f"== 혼합 데이터 + 이중 정답 생성 (seed={SEED}) ==")
    gen_subscribers()
    gen_billing()
    gen_calls()
    gen_employees()
    gen_consultations()
    gen_reviews()
    gen_column_gt()
    gen_span_gt()
    gen_type_map()
    print(f"\n  span 정답 총 {len(SPANS):,}건")
    SRC_DIR = LOCAL_DATA
    print(f"생성 완료 → {SRC_DIR}")
else:
    SRC_DIR = DATA_WS
    print(f"재생성 스킵 — 패키지 커밋 CSV 사용: {SRC_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## raw 볼륨 업로드
# MAGIC `RAW_VOL` (`/Volumes/<catalog>/<schema>/raw`)에 9개 CSV를 업로드합니다. dbutils.fs.cp로
# MAGIC 워크스페이스/드라이버 로컬 경로를 모두 처리합니다.

# COMMAND ----------

import shutil

for name in CSV_FILES:
    src = os.path.join(SRC_DIR, name)
    dst_vol = f"{RAW_VOL}/{name}"
    if SRC_DIR == DATA_WS:
        # 워크스페이스 파일 → 볼륨: dbutils.fs.cp (file: 스킴)
        dbutils.fs.cp(f"file:{src}", dst_vol)
    else:
        # 드라이버 로컬 → 볼륨: 직접 복사(볼륨은 FUSE 마운트)
        shutil.copyfile(src, dst_vol)
    print(f"  ✓ uploaded {name} → {dst_vol}")

# COMMAND ----------

# MAGIC %md ## 업로드 확인

# COMMAND ----------

display(dbutils.fs.ls(RAW_VOL))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC ➡️ **[`02_load_tables`](./02_load_tables)** — raw CSV를 Delta 테이블 9종 + text_corpus로 적재.
