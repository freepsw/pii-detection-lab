# Databricks notebook source
# MAGIC %md
# MAGIC # 0. 설정 (Single Source of Truth)
# MAGIC
# MAGIC > ⚠️ **`03_pii_detection_lab` 폴더 전체를 import했는지 먼저 확인하세요** (Workspace > Import, 디렉터리 단위). 단일 노트북만 가져오면 `_common`·`data` 상대 경로가 깨져 아래 셀이 실패합니다.
# MAGIC
# MAGIC **이 노트북의 위젯값만 바꾸면 전체 랩에 전파됩니다.** 다른 모든 노트북은 첫 셀에서
# MAGIC `%run ../00_setup/00_config` 로 이 설정을 상속합니다 (catalog·schema·백엔드·임계값 + `_common` 모듈 import).
# MAGIC
# MAGIC | 위젯 | 설명 | 예시 |
# MAGIC |---|---|---|
# MAGIC | `catalog` | 작업할 Unity Catalog 카탈로그 | `main` |
# MAGIC | `schema` | 작업 스키마(자동 생성 시도) | `pii_lab` |
# MAGIC | `gptoss_endpoint` | Foundation Model 엔드포인트명 | `databricks-gpt-oss-120b` |
# MAGIC | `qwen_endpoint` | (선택) in-region 커스텀 서빙명. **비우면 gpt-oss 단독 실행** | `` |

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "1. Catalog")
dbutils.widgets.text("schema", "pii_lab", "2. Schema")
dbutils.widgets.text("gptoss_endpoint", "databricks-gpt-oss-120b", "3. gpt-oss endpoint")
dbutils.widgets.text("qwen_endpoint", "", "4. qwen endpoint (선택)")

# COMMAND ----------

# MAGIC %md ## 설정 변수 도출 (이 셀 실행 후 FQ·BACKENDS·LLM_BACKENDS 등이 전역에 노출됨)

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
FQ = f"{CATALOG}.{SCHEMA}"
RAW_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
MODELS_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/models"

GPTOSS = dbutils.widgets.get("gptoss_endpoint").strip()
QWEN = dbutils.widgets.get("qwen_endpoint").strip() or None

# 백엔드 단일 출처 — qwen 위젯이 비면 gpt-oss 단독(전 단계가 LLM_BACKENDS를 순회하므로 자동 스킵)
BACKENDS = {"gpt-oss-120b": GPTOSS}
if QWEN:
    BACKENDS["qwen3-4b"] = QWEN
LLM_BACKENDS = list(BACKENDS.keys())

# cascade·표본 상수 (검증 데모 backend_config.py에서 통합 이식)
TAU = 0.70                  # NER 신뢰도 임계값 — 미만만 LLM 라우팅(비용 레버)
CASCADE_CTX = 40            # cascade 시 span 좌우 컨텍스트 글자수
PERF_SAMPLE_DOCS = 200      # 성능 벤치 고정 표본
S5_QWEN_SAMPLE_DOCS = 300   # qwen 처리량 제약 표본
CATEGORIES = ["PERSONAL_INFO", "PAYMENT_INFO", "IDENTIFICATION", "EMPLOYMENT_INFO", "NON_PII"]

print(f"FQ={FQ} | RAW_VOL={RAW_VOL}")
print(f"LLM_BACKENDS={LLM_BACKENDS}  (qwen 미설정 시 gpt-oss 단독)")

# COMMAND ----------

# MAGIC %md ## `_common` 모듈 import 경로 설정
# MAGIC 현재 노트북 경로에서 `_common`을 sys.path에 추가해 검증된 모듈(바이트 동일)을 import합니다.

# COMMAND ----------

import os
import sys

# 실행 중 노트북의 워크스페이스 경로 → 패키지 루트(03_pii_detection_lab)/_common 도출
_nb_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
# .../03_pii_detection_lab/00_setup/00_config → 루트는 두 단계 위
_pkg_root = os.path.dirname(os.path.dirname(_nb_path))
_common_ws = f"/Workspace{_pkg_root}/_common"
if _common_ws not in sys.path:
    sys.path.insert(0, _common_ws)
print("_common 경로:", _common_ws)

# COMMAND ----------

import lab_runtime as rt          # sql / load_rows / sqllit / get_client (노트북 네이티브)
import span_patterns              # 정규식 9종 (검증 바이트 동일)
import span_llm                   # SPAN_PROMPT + recover_spans / merge_regex_llm
import column_prompt              # 컬럼 분류 프롬프트
import llm_client as _llm_mod     # 백엔드 추상화
import eval_lib                   # span 평가 지표

# 백엔드 단일 출처를 llm_client에 주입(바이트 동일 모듈을 수정하지 않고 런타임 오버라이드)
_llm_mod.BACKENDS = BACKENDS
from llm_client import llm_client, extract_json
from span_patterns import regex_spans
from span_llm import SPAN_PROMPT, recover_spans, merge_regex_llm

# 짧은 sql 헬퍼(노트북 셀에서 바로 사용)
def sql(q):
    return rt.sql(q, spark=spark)

def load_rows(table, rows, mode="append", where=None):
    return rt.load_rows(table, rows, mode=mode, where=where, spark=spark)

print("공통 모듈 로드 완료: rt, span_patterns, span_llm, column_prompt, llm_client, eval_lib")

# COMMAND ----------

# MAGIC %md ## 카탈로그·스키마·볼륨 생성 (권한 있으면)
# MAGIC 권한이 없으면 관리자에게 사전 생성을 요청하고 위젯에 기존 카탈로그·스키마를 지정하세요.

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    print(f"(카탈로그 생성 스킵 — 권한 없거나 기존 사용: {str(e)[:80]})")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.models")
print(f"준비 완료: {FQ} (raw·models 볼륨 포함)")
