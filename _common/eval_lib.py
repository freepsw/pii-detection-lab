# -*- coding: utf-8 -*-
"""
eval_lib.py — span 정확도 평가 (검증 데모 99_eval/92_eval_spans.py 이식).

이식 규칙: 평가 수학(_f1·_iou·매칭 로직)은 원본과 **로직 동등**. 변경점은 단 하나 —
_dbx.run_sql/_dbx.load_rows 의존을 제거하고 sql_fn·load_fn 헬퍼를 주입받는다
(노트북에서는 lab_runtime.sql / lab_runtime.load_rows 를 넘긴다).
세 매칭 체계 + char: exact / type_agnostic / partial(IoU>=0.5) / char.
coverage(span_coverage)에 기록된 처리 문서만 분모에 포함(공정 recall).
"""
from collections import defaultdict

THETA = 0.5


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _iou(a, b):
    ov = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if ov == 0:
        return 0.0
    un = (a[1] - a[0]) + (b[1] - b[0]) - ov
    return ov / un if un else 0.0


def evaluate(FQ, sql_fn, load_fn):
    """span_predictions/span_coverage/span_ground_truth → span_eval 적재 + 반환.
    sql_fn(query)->(cols,rows), load_fn(table, rows, mode='replace').
    """
    gt = defaultdict(list)
    for doc, s, e, t in sql_fn(
            f"SELECT doc_id, start_char, end_char, entity_type FROM {FQ}.span_ground_truth")[1]:
        gt[doc].append((int(s), int(e), t))
    cov = defaultdict(set)
    for st, be, doc in sql_fn(f"SELECT stage, backend, doc_id FROM {FQ}.span_coverage")[1]:
        cov[(st, be)].add(doc)
    preds = defaultdict(lambda: defaultdict(list))
    cells = set()
    for st, be, doc, s, e, t in sql_fn(
            f"SELECT stage, backend, doc_id, start_char, end_char, entity_type FROM {FQ}.span_predictions")[1]:
        preds[(st, be)][doc].append((int(s), int(e), t))
        cells.add((st, be))
    cells |= set(cov.keys())

    out = []
    for cell in sorted(cells):
        covered = cov.get(cell) or set(preds[cell].keys())
        n_gt = sum(len(gt[d]) for d in covered)
        ex = dict(tp=0, fp=0, fn=0)
        ta = dict(tp=0, fp=0, fn=0)
        pa = dict(tp=0, fp=0, fn=0)
        ch = dict(tp=0, fp=0, fn=0)
        for doc in covered:
            G = gt.get(doc, [])
            P = preds[cell].get(doc, [])
            gs, ps = set(G), set(P)
            ex["tp"] += len(gs & ps); ex["fp"] += len(ps - gs); ex["fn"] += len(gs - ps)
            gse = set((s, e) for s, e, _ in G); pse = set((s, e) for s, e, _ in P)
            ta["tp"] += len(gse & pse); ta["fp"] += len(pse - gse); ta["fn"] += len(gse - pse)
            used = [False] * len(P)
            tp_p = 0
            for (gsx, gex, gt_) in G:
                best, bi = 0.0, -1
                for i, (psx, pex, pt_) in enumerate(P):
                    if used[i] or pt_ != gt_:
                        continue
                    v = _iou((gsx, gex), (psx, pex))
                    if v >= THETA and v > best:
                        best, bi = v, i
                if bi >= 0:
                    used[bi] = True; tp_p += 1
            pa["tp"] += tp_p; pa["fn"] += len(G) - tp_p; pa["fp"] += len(P) - tp_p
            gchar = set()
            for s, e, _ in G:
                gchar |= set(range(s, e))
            pchar = set()
            for s, e, _ in P:
                pchar |= set(range(s, e))
            ch["tp"] += len(gchar & pchar); ch["fp"] += len(pchar - gchar); ch["fn"] += len(gchar - pchar)
        for regime, d in [("exact", ex), ("type_agnostic", ta), ("partial", pa), ("char", ch)]:
            p, r, f = _f1(d["tp"], d["fp"], d["fn"])
            out.append({"stage": cell[0], "backend": cell[1], "regime": regime,
                        "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                        "precision": p, "recall": r, "f1": f,
                        "n_docs": len(covered), "n_gt": n_gt})
    load_fn(f"{FQ}.span_eval", out, mode="replace")
    return out
