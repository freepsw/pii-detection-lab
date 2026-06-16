# Databricks notebook source
# MAGIC %md
# MAGIC # 0-0. 사전점검 (Preflight) — 읽기 전에 1분
# MAGIC
# MAGIC ## [푸는 문제]
# MAGIC 긴 학습 문서를 읽고 노트북을 돌리기 시작한 **뒤에야** 환경 문제(클러스터·권한·엔드포인트·import 사고)를
# MAGIC 발견하면 늦습니다. 이 노트북은 **Run All 한 번으로 1분 안에** 그 블로커들을 먼저 잡아냅니다.
# MAGIC
# MAGIC ## [직관/원리]
# MAGIC 모든 점검은 **읽기 전용**(존재 확인·DESCRIBE·디렉터리 나열)이며 **비차단**입니다 — WARN이 떠도
# MAGIC 멈추지 않고, 각 항목에 '조치' 포인터를 함께 출력합니다. (단, 첫 셀의 `%run ../00_setup/00_config` 는
# MAGIC 표준 설정을 상속하며 카탈로그·스키마·볼륨이 없으면 생성 시도합니다 — 어차피 가장 먼저 실행할 셀입니다.)
# MAGIC
# MAGIC ## [결과: 기대 산출물]
# MAGIC `✅ PASS` / `⚠️ WARN` 목록 + 마지막 `N/M PASS` 요약. 전부 PASS면 `README` §4.2 순서로 진행하세요.
# MAGIC
# MAGIC ## [한계]
# MAGIC 엔드포인트는 **존재 여부**만 확인합니다(쿼리하지 않음 — qwen 콜드스타트 15~32분 회피). 실제 응답 품질은
# MAGIC S2부터 확인됩니다. HuggingFace(S3/S4) 도달성은 점검하지 않고 안내만 합니다.
# MAGIC
# MAGIC ## [다음 단계로]
# MAGIC 전부 PASS → `00_setup/00_config`(이미 상속됨) 확인 후 `01_generate_data` → `02_load_tables` → S1…

# COMMAND ----------

# MAGIC %run ../00_setup/00_config

# COMMAND ----------

# MAGIC %md ## 점검 실행 (읽기 전용 · 비차단)

# COMMAND ----------

import os

results = []
def chk(name, ok, detail, fix=""):
    results.append(bool(ok))
    print(f"{'✅ PASS' if ok else '⚠️  WARN'}  {name} — {detail}")
    if (not ok) and fix:
        print(f"          ↳ 조치: {fix}")

# 1) 클러스터 런타임 (S3/S4 NER에 ML 17.3 필요)
try:
    sv = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "")
except Exception:
    sv = ""
chk("클러스터 런타임", ("17.3" in sv and "ml" in sv.lower()),
    f"sparkVersion='{sv or '미상'}'",
    "ML 17.3 LTS 클러스터에 연결하세요(S3/S4 NER 학습·추론에 필요). 01_환경_설정 §2.")

# 2) 카탈로그·스키마·볼륨 (00_config가 생성 시도)
try:
    spark.sql(f"DESCRIBE SCHEMA {FQ}")
    vols = set()
    for r in spark.sql(f"SHOW VOLUMES IN {FQ}").collect():
        d = r.asDict()
        vols.add(d.get("volume_name") or list(d.values())[-1])
    chk("카탈로그·스키마·볼륨", {"raw", "models"} <= vols,
        f"{FQ} · 볼륨={sorted(vols)}",
        "raw·models 볼륨이 없으면 00_config Run All(생성 권한 필요). 권한 없으면 관리자에 사전생성 요청. 01_환경_설정 §1.")
except Exception as e:
    chk("카탈로그·스키마·볼륨", False, f"{FQ} 접근 실패: {str(e)[:80]}",
        "00_config의 카탈로그/스키마 생성 권한 확인 또는 위젯에 기존 카탈로그·스키마 지정. 01_환경_설정 §1.")

# 3) data/ 디렉터리 = '폴더 전체 import' 여부 (단일 노트북 import 사고 포착)
try:
    data_ws = f"/Workspace{_pkg_root}/data"
    ncsv = len([f for f in os.listdir(data_ws) if f.endswith(".csv")])
    chk("data/ 디렉터리(폴더 전체 import)", ncsv >= 9, f"{data_ws} — CSV {ncsv}개",
        "단일 노트북만 import했을 수 있습니다. `03_pii_detection_lab` 디렉터리 전체(zip/.dbc)를 다시 import하세요.")
except Exception as e:
    chk("data/ 디렉터리(폴더 전체 import)", False, f"{str(e)[:90]}",
        "폴더 전체 import 여부 확인(단일 노트북 import 시 data/ 가 없습니다).")

# 4·5) LLM 엔드포인트 — 존재만 확인(쿼리 X, 콜드스타트 회피)
w = rt.get_client()
try:
    w.serving_endpoints.get(BACKENDS["gpt-oss-120b"])
    chk("gpt-oss 엔드포인트", True, f"{BACKENDS['gpt-oss-120b']} 존재 확인")
except Exception as e:
    chk("gpt-oss 엔드포인트", False, f"{BACKENDS['gpt-oss-120b']} 없음: {str(e)[:70]}",
        "Serving 메뉴에서 엔드포인트·region 가용 확인. 없으면 다른 Foundation Model로 교체하거나 qwen 배포. 01_환경_설정 §3.")

if "qwen3-4b" in BACKENDS:
    try:
        w.serving_endpoints.get(BACKENDS["qwen3-4b"])
        chk("qwen 엔드포인트(선택)", True, f"{BACKENDS['qwen3-4b']} 존재 확인(첫 호출 시 콜드스타트 가능)")
    except Exception as e:
        chk("qwen 엔드포인트(선택)", False, f"{BACKENDS['qwen3-4b']} 없음: {str(e)[:70]}",
            "in-region 커스텀 서빙 배포 필요(선택 항목). 01_환경_설정 §4. 비우면 gpt-oss 단독으로 동작.")
else:
    print("ℹ️  qwen 미설정 — gpt-oss 단독으로 전체 랩 동작(정상).")

print("ℹ️  S3/S4(NER)는 HuggingFace에서 KoELECTRA 다운로드가 필요합니다(폐쇄망은 01_환경_설정 §5 오프라인 절차).")

# COMMAND ----------

# MAGIC %md ## 요약

# COMMAND ----------

n_ok = sum(results)
print(f"=== 사전점검 {n_ok}/{len(results)} PASS ===")
if n_ok == len(results):
    print("환경 준비 완료 → README §4.2 순서대로 00_setup/00_config(상속됨) → 01_generate_data → 02_load_tables → S1… 진행.")
else:
    print("⚠️ WARN 항목의 '조치'를 먼저 해결하세요(일부는 선택/특정 단계 한정). 이 점검은 아무 데이터도 변경하지 않습니다(읽기 전용).")
