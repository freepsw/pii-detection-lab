# -*- coding: utf-8 -*-
"""
lab_runtime.py — 실행 substrate (검증 데모의 _tools/_dbx.py 노트북 대체).
※ 이 파일은 sys.path로 import되는 순수 모듈입니다(노트북 아님 — Databricks notebook source 헤더 없음).

설계: 기존 _dbx.py는 로컬에서 `databricks auth token --profile`(subprocess CLI)로
토큰을 주입하고 statement_execution API로 SQL을 돌렸다 — 노트북에서는 동작하지 않는다.
이 모듈은 노트북 네이티브로 전환한다:
  - SQL 실행: spark.sql (노트북에 attach된 컴퓨트 사용, warehouse_id 불필요)
  - 인증: WorkspaceClient() — 실행 사용자 컨텍스트 자동 인증(토큰 주입 없음)
  - 반환형: _dbx.run_sql과 동일한 (columns, rows) 튜플 — 검증 로직 재사용 호환

검증된 헬퍼(sqllit, 멱등 DELETE→append)는 동작을 보존한다.
로컬 실행(개발자 검증용)은 databricks-connect로 spark를 확보하는 fallback을 둔다.
"""


def get_spark():
    """노트북 전역 spark. 로컬이면 databricks-connect fallback."""
    try:
        return spark  # noqa: F821  (노트북 전역)
    except NameError:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()


def get_client():
    """노트북: WorkspaceClient() 네이티브 인증. 로컬: 프로파일 fallback."""
    from databricks.sdk import WorkspaceClient
    try:
        return WorkspaceClient()
    except Exception:
        import os
        return WorkspaceClient(profile=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))


def sql(query, spark=None):
    """SQL 실행 → (columns, rows) 튜플. _dbx.run_sql 호환(검증 로직이 그대로 소비).

    rows는 list[list] (각 셀은 파이썬 기본형). DDL/DML도 동일 인터페이스.
    """
    sp = spark or get_spark()
    df = sp.sql(query)
    cols = list(df.columns)
    rows = [list(r) for r in df.collect()] if cols else []
    return cols, rows


def sqllit(s):
    """SQL 문자열 리터럴 이스케이프(작은따옴표) — 검증된 동작 보존.
    ai_query 프롬프트를 SQL에 인라인할 때 필수.
    """
    return (s or "").replace("'", "''")


def load_rows(table, rows, mode="append", where=None, spark=None):
    """dict 리스트 → Delta 테이블 적재. _dbx.load_rows의 노트북 네이티브 대체.

    - 모든 값 STRING 적재(검증 로직과 동일). rows[0] 키 순서 = 컬럼 순서.
    - where 지정 시 INSERT 전 해당 조건 DELETE(재실행 멱등 — 검증된 패턴).
    - mode='replace'면 테이블 덮어쓰기(overwriteSchema).
    JSONL roundtrip 대신 spark.createDataFrame를 쓰되, 임베디드 따옴표/개행은
    DataFrame 경로라 이스케이프 문제 없음(SQL 인라인이 아님).
    """
    from pyspark.sql.types import StructType, StructField, StringType
    from pyspark.sql.functions import col
    sp = spark or get_spark()
    assert rows, "load_rows: empty rows"
    keys = list(rows[0].keys())
    data = [tuple(str(r.get(k)) if r.get(k) is not None else None for k in keys) for r in rows]
    # 모든 값을 STRING으로 적재(검증 로직과 동일). 명시 StringType 스키마로 타입 추론 흔들림 차단
    # (전부 None인 컬럼이 void로 추론돼 재적재 시 schema merge가 깨지는 문제 방지).
    schema = StructType([StructField(k, StringType(), True) for k in keys])
    sdf = sp.createDataFrame(data, schema=schema)
    if mode == "replace":
        (sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table))
        return
    if where:
        try:
            sp.sql(f"DELETE FROM {table} WHERE {where}")
        except Exception:
            pass  # 테이블 미존재 시 첫 적재
    # 기존 테이블이 있으면 그 선언 타입(예: start_char INT·score DOUBLE)에 맞춰 캐스팅 후 append.
    # (STRING 그대로 append하면 INT 컬럼과 schema merge 충돌 → DELTA_FAILED_TO_MERGE_FIELDS.
    #  원본 _dbx의 INSERT-cast 동작과 동치.) 테이블이 없으면 STRING 스키마로 생성.
    if sp.catalog.tableExists(table):
        tgt = sp.table(table).schema
        sdf = sdf.select([col(k).cast(tgt[k].dataType).alias(k) for k in keys])
        (sdf.write.mode("append").saveAsTable(table))
    else:
        (sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(table))


def upload_file(local_path, volume_path, client=None):
    """로컬 파일 → UC Volume 업로드(노트북에서 dbutils.fs 또는 SDK files)."""
    w = client or get_client()
    with open(local_path, "rb") as f:
        w.files.upload(volume_path, f, overwrite=True)
    return volume_path
