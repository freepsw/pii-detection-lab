#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
span_llm.py — LLM 기반 span 추출 + offset 복원 (S2의 LLM 파트, S5 LLM-only 공유).

LLM은 char 위치 계산이 부정확하므로 '원문에 그대로 나타난 부분문자열 + 유형'만 받고,
Python이 text.find로 offset을 복원한다(환각=find 실패→드롭, 중복=비겹침 위치 배정).
"""
import re

SPAN_PROMPT = (
    "다음 한국어 텍스트에서 개인정보(PII)에 해당하는 부분 문자열을 모두 찾아 JSON 배열로만 출력하세요.\n"
    "각 원소 형식: {\"value\": \"원문에 그대로 나타난 문자열\", \"type\": \"유형\"}\n"
    "유형 목록: PERSON(사람 이름), PHONE(전화번호), RRN(주민등록번호), CARD(카드번호), "
    "ACCOUNT(계좌번호), EMAIL(이메일), ADDRESS(주소), IMEI, PASSPORT(여권번호)\n"
    "규칙: value는 반드시 텍스트에 등장한 그대로(공백·하이픈·형식 포함) 복사한다. "
    "개인정보가 없으면 빈 배열 []. 설명·코드펜스 없이 JSON 배열만 출력.\n\n텍스트:\n")

_TYPE_NORM = {
    "PERSON": "PERSON", "이름": "PERSON", "NAME": "PERSON",
    "PHONE": "PHONE", "전화": "PHONE", "전화번호": "PHONE", "휴대폰": "PHONE", "MOBILE": "PHONE",
    "RRN": "RRN", "주민등록번호": "RRN", "주민번호": "RRN", "SSN": "RRN",
    "CARD": "CARD", "카드": "CARD", "카드번호": "CARD", "CREDIT_CARD": "CARD",
    "ACCOUNT": "ACCOUNT", "계좌": "ACCOUNT", "계좌번호": "ACCOUNT", "BANK": "ACCOUNT",
    "EMAIL": "EMAIL", "이메일": "EMAIL", "메일": "EMAIL",
    "ADDRESS": "ADDRESS", "주소": "ADDRESS", "LOCATION": "ADDRESS",
    "IMEI": "IMEI", "단말식별번호": "IMEI",
    "PASSPORT": "PASSPORT", "여권": "PASSPORT", "여권번호": "PASSPORT",
}


def norm_type(t):
    if not t:
        return "UNKNOWN"
    t = str(t).strip().upper()
    return _TYPE_NORM.get(t, _TYPE_NORM.get(str(t).strip(), t))


def recover_spans(text, items, default_score=0.8):
    """items: list[{value,type}] → [{start,end,entity_type,pii_value,score}] (비겹침)."""
    if not text or not items:
        return []
    claimed = []  # (start,end)
    out = []
    # [fix R2-008] 긴 값부터 위치 점유 — LLM 출력 순서에 따라 짧은 항목(예: "1234")이
    # 긴 정답 span(전화번호 등)의 자리를 선점해 드롭시키는 shadowing 방지.
    items = sorted(items, key=lambda it: -len(it.get("value") or "")
                   if isinstance(it, dict) and isinstance(it.get("value"), str) else 0)
    for it in items:
        if not isinstance(it, dict):
            continue
        v = it.get("value")
        if not v or not isinstance(v, str):
            continue
        etype = norm_type(it.get("type"))
        # 비겹침 첫 출현 위치 탐색
        start = 0
        while True:
            idx = text.find(v, start)
            if idx == -1:
                break
            end = idx + len(v)
            if all(end <= cs or idx >= ce for cs, ce in claimed):
                claimed.append((idx, end))
                out.append({"start": idx, "end": end, "entity_type": etype,
                            "pii_value": v, "score": default_score})
                break
            start = idx + 1
    out.sort(key=lambda s: s["start"])
    return out


def merge_regex_llm(regex_spans, llm_spans):
    """S2용: 정규식 span 우선, 겹치지 않는 LLM span 추가."""
    merged = list(regex_spans)
    occ = [(s["start"], s["end"]) for s in merged]
    for s in llm_spans:
        if all(s["end"] <= cs or s["start"] >= ce for cs, ce in occ):
            merged.append(s); occ.append((s["start"], s["end"]))
    merged.sort(key=lambda x: x["start"])
    return merged
