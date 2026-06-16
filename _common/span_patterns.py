#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
span_patterns.py — 정규식 기반 span 탐지 (PATTERN 레이어: 정형 PII).
S1(패턴) 및 S2/S3/S4의 패턴 레이어가 공유. NER 레이어(PERSON/ADDRESS)는 탐지 못함(의도).

regex_spans(text) -> [ {start,end,entity_type,pii_value,score} ]  (char offset, 겹침 해소)
"""
import re

# (entity_type, 정규식, priority)  priority 큰 것이 겹칠 때 우선.
# 숫자 런은 \b 로 독립 런만 매칭(카드16 vs IMEI15 vs 전화11 구분).
_PATTERNS = [
    ("EMAIL",    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), 9),
    ("RRN",      re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"), 8),
    ("ACCOUNT",  re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{6}(?![\d-])"), 7),
    ("CARD",     re.compile(r"(?<![\d-])\d{4}-\d{4}-\d{4}-\d{4}(?![\d-])"), 7),
    ("CARD",     re.compile(r"(?<!\d)\d{16}(?!\d)"), 6),
    ("PHONE",    re.compile(r"(?<![\d-])01[016789]-\d{3,4}-\d{4}(?![\d-])"), 6),
    ("PHONE",    re.compile(r"(?<!\d)01[016789]\d{7,8}(?!\d)"), 5),
    ("PASSPORT", re.compile(r"(?<![A-Za-z0-9])[MSROD]\d{8}(?![A-Za-z0-9])"), 5),
    ("IMEI",     re.compile(r"(?<!\d)\d{15}(?!\d)"), 4),
]


def regex_spans(text):
    if not text:
        return []
    cand = []
    for etype, rx, prio in _PATTERNS:
        for m in rx.finditer(text):
            cand.append({"start": m.start(), "end": m.end(), "entity_type": etype,
                         "pii_value": m.group(0), "score": 1.0, "_prio": prio})
    # 겹침 해소: priority 높은 것, 같으면 긴 것 우선
    cand.sort(key=lambda s: (-s["_prio"], -(s["end"] - s["start"]), s["start"]))
    chosen = []
    for s in cand:
        if any(not (s["end"] <= c["start"] or s["start"] >= c["end"]) for c in chosen):
            continue
        chosen.append(s)
    for c in chosen:
        c.pop("_prio", None)
    chosen.sort(key=lambda s: s["start"])
    return chosen


if __name__ == "__main__":
    samples = [
        "고객 홍길동님이 010-1234-5678로 문의. 주민번호 901010-1234567 확인.",
        "카드 1234-5678-9012-3456 승인오류, 계좌 123-45-678901, imei 356938035643809",
        "연락처 01098765432, 여권 M12345678, 메일 a.b@kt.com",
    ]
    for t in samples:
        print(t)
        for s in regex_spans(t):
            assert t[s["start"]:s["end"]] == s["pii_value"]
            print("   ", s)
