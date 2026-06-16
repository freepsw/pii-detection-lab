# -*- coding: utf-8 -*-
"""
column_prompt.py — 컬럼 PII 분류 프롬프트 (검증 데모 _tools/build_columns.py에서 추출, 단일 출처).
이 프롬프트는 S2(패턴+LLM)·S5(LLM-only) 컬럼 트랙이 gpt-oss/qwen 양 백엔드에 동일하게 사용한다.
원본과 바이트 동등(추출만, 로직 변경 없음).
"""

PROMPT_HEAD = (
    "당신은 데이터 거버넌스 전문가입니다. 아래 데이터베이스 컬럼이 개인정보(PII)인지 "
    "한국 개인정보보호법 기준으로 판정하세요.\n")
PROMPT_TAIL = (
    "\n\n카테고리 정의:\n- PERSONAL_INFO: 이름/휴대폰/이메일/주소/생년월일\n"
    "- PAYMENT_INFO: 신용카드번호/계좌번호\n"
    "- IDENTIFICATION: 주민등록번호/여권번호/IMEI 등 개인 식별 가능한 고유번호\n"
    "- EMPLOYMENT_INFO: 급여/인사평가 등 고용 민감정보\n"
    "- NON_PII: 코드/상태/집계수치/업무용 날짜(가입일/청구월/납기일/입사일) 등\n\n"
    "판정 규칙:\n1) 컬럼명보다 샘플값의 실제 패턴을 우선 고려한다.\n"
    "2) 생년월일은 PERSONAL_INFO지만 가입일/청구월/납기일/입사일 같은 업무 날짜는 NON_PII다.\n"
    "3) 카드 종류 코드(CREDIT/DEBIT/PREPAID)는 결제수단 유형일 뿐 PII가 아니다.\n"
    "4) 15자리 단말 식별번호(IMEI)는 IDENTIFICATION이다.\n"
    "5) 자유텍스트(상담메모/민원/리뷰)에 개인정보가 섞여 있으면 PII로 본다.\n\n"
    'JSON만 출력(설명 금지): {"is_pii": true 또는 false, "category": "위 5개 중 하나", '
    '"confidence": 0.0~1.0, "reason": "한 문장 근거"}')


def prompt_text(table, column, samples):
    return f"{PROMPT_HEAD}컬럼명: {column}\n테이블: {table}\n샘플값(최대 10개): {samples}{PROMPT_TAIL}"
