#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90_llm_client.py — 백엔드 추상화 (Python 경로; 주로 S3/S4 NER cascade에서 사용).

- llm_client(prompt, backend=...) 가 gpt-oss-120b / qwen3-4b 둘 다 동일 인터페이스로 호출.
- 응답은 OpenAI choices[0].message.content 와 pyfunc predictions[0] 양형식 처리(in-region 04 이식).
- extract_json: 방어적 JSON 파싱(전체→```json 펜스→첫 {..}→첫 [..]).
- 온클러스터(노트북)는 WorkspaceClient() 기본 인증, 로컬은 w= 주입.

SQL 단계(S2/S5 컬럼·span 배치)는 ai_query('<endpoint>', ...)로 처리하므로 이 모듈을 쓰지 않는다.
"""
import json
import re
import time

try:
    from backend_config import BACKENDS
except Exception:  # 파일명 prefix(90_) 회피용 폴백
    BACKENDS = {"gpt-oss-120b": "databricks-gpt-oss-120b", "qwen3-4b": "qwen3-4b-t4"}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_ARR = re.compile(r"\[.*\]", re.DOTALL)


def extract_json(text):
    """문자열에서 첫 유효 JSON(객체/배열)을 최대한 복구. 실패 시 None."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    candidates = [text]
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    mo = _OBJ.search(text)
    if mo:
        candidates.append(mo.group(0))
    ma = _ARR.search(text)
    if ma:
        candidates.append(ma.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _extract_chat(resp):
    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
    if choices:
        c0 = choices[0]
        msg = getattr(c0, "message", None) if not isinstance(c0, dict) else c0.get("message")
        if msg is not None:
            content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
            if content is not None:
                return content
    preds = resp.predictions if hasattr(resp, "predictions") else resp
    try:
        first = preds[0]
        return first.get("response", first) if isinstance(first, dict) else first
    except Exception:
        return preds


def llm_client(prompt, backend="gpt-oss-120b", system=None, temperature=0.0,
               max_tokens=512, w=None, retries=2, backoff_s=2.0):
    """단일 프롬프트를 백엔드에 질의하고 텍스트 응답 반환."""
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    endpoint = BACKENDS[backend]
    if w is None:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
    msgs = []
    if system:
        msgs.append(ChatMessage(role=ChatMessageRole.SYSTEM, content=system))
    msgs.append(ChatMessage(role=ChatMessageRole.USER, content=prompt))
    last = None
    for attempt in range(retries + 1):
        try:
            resp = w.serving_endpoints.query(
                name=endpoint, messages=msgs,
                temperature=temperature, max_tokens=max_tokens)
            return _extract_chat(resp)
        except Exception as e:
            last = e
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"llm_client failed ({backend}/{endpoint}): {last}")


def llm_json(prompt, backend="gpt-oss-120b", **kw):
    """llm_client 후 extract_json. (raw, parsed) 반환."""
    raw = llm_client(prompt, backend=backend, **kw)
    return raw, extract_json(raw)
