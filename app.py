# -*- coding: utf-8 -*-
"""
도시가스 공급량 중장기 추정 (2036 ~ 2045)

- 저장소의 '2045중장기사업계획.xlsx'(2035년까지 작성)를 읽어 2036~2045를 추세 모델로 채움
- 모든 모델은 '데이터에서 계수를 추정'하는 방식 (임의의 감쇠/고정값 없음)
- 모델 6종 중 선택, 용도별로 다른 모델 지정 가능, 모델별 결과/검증오차 비교
- 원본 서식·수식을 유지한 엑셀로 다운로드
"""
import io
import os
import re
from copy import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openpyxl import load_workbook
from openpyxl.formula.translate import Translator
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="공급량 중장기 추정", layout="wide")

TARGET_END = 2045
PLAN_END = 2035  # 원래 계획이 작성된 마지막 해
# 확정값 (2026~2029년, 사용자가 확인한 엑셀 값). 저장소 엑셀이 예전 값이면 이 값으로 교체해 사용합니다.
CONFIRMED_VALUES = {
    "주택용_취사/개별난방": {2026: 24689242, 2027: 24762440, 2028: 25114005, 2029: 24944797},
    "주택용_중집난방": {2026: 836497, 2027: 835248, 2028: 843196, 2029: 834350},
    "업무난방": {2026: 693593, 2027: 684270, 2028: 677113, 2029: 670119},
    "일반용": {2026: 3381184, 2027: 3351860, 2028: 3342768, 2029: 3303216},
    "냉난방용": {2026: 1403814, 2027: 1399607, 2028: 1442951, 2029: 1424315},
    "산업용": {2026: 8678288, 2027: 8727394, 2028: 8718795, 2029: 8592138},
    "열병합용": {2026: 248368, 2027: 248970, 2028: 278738, 2029: 249837},
    "연료전지용": {2026: 203147, 2027: 591953, 2028: 862788, 2029: 867490},
    "수송용(BIO포함)": {2026: 3174626, 2027: 2973354, 2028: 2511844, 2029: 2155094},
    "주한미군용": {2026: 57073, 2027: 56144, 2028: 54682, 2029: 53338},
}

HH_TITLE_H = 84  # 오른쪽 표 머리글을 왼쪽 표 머리글 높이에 맞추는 제목 영역 높이(px)
GSHEET_ID = "1gIhArPlLBJ9fwlaqXtZWxiKlSK9hbRuz6HcDw_Yf7Is"  # 월별 상품별 공급량 구글 시트 (취사용·개별난방용 행 사용)
PLOT_CFG = {"scrollZoom": True, "displaylogo": False}  # 마우스 휠 확대/축소


def _merge_over(over, conf_over):
    """가정용 시뮬레이션 대체값 + 확정값 교체분을 행별 Series로 합침"""
    out = {r: s.copy() for r, s in over.items()}
    for r, d in conf_over.items():
        cs = pd.Series(d, dtype=float)
        out[r] = pd.concat([cs, out[r]]) if r in out else cs
    return out


def show_chart(fig):
    """휠 확대 + 드래그 = 화면 이동(pan)"""
    fig.update_layout(dragmode="pan")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

REPO_PLAN_FILE = "2045중장기사업계획.xlsx"
YEAR_RE = re.compile(r"^\s*(\d{4})\s*년?\s*$")
TOTAL_LABELS = {"합계", "총합계", "총계"}
FOLLOW_GLOBAL = "(전체 모델 따름)"

# ─────────────────────────────────────────────────────────
# 1. 추세 모델 (모두 데이터로 계수를 추정)
# ─────────────────────────────────────────────────────────
MODELS = {
    "1. 선형 회귀": "linear",
    "2. 2차 곡선": "quad",
    "3. 로그 추세": "log",
    "4. 지수 추세": "exp",
    "5. 연평균 성장률(CAGR)": "cagr",
    "6. Holt 지수평활": "holt",
}
MODEL_NAME = {v: k for k, v in MODELS.items()}
MODEL_NAME.update({"hh": "가정용 기온 시뮬레이션", "hold": "임시 유지 (담당부서 자료 대기)"})
MODEL_DESC = {
    "linear": "**y = a + b·t** 직선을 최소제곱법으로 적합해 연장합니다. 매년 **일정한 양**씩 증가/감소하는 흐름에 맞고, 설명이 가장 쉽습니다.",
    "quad": "**y = a + b·t + c·t²** 곡선을 적합합니다. 증감폭이 점점 **커지거나 작아지는(가속·둔화)** 흐름을 반영합니다. 다만 10년 외삽 시 방향이 꺾일 수 있어 주의가 필요합니다.",
    "log": "**y = a + b·ln(t)** 를 적합합니다. 초기에 크게 변하고 이후 변화폭이 줄어드는 **성숙·포화형** 패턴에 맞습니다. 증가/감소 방향은 바뀌지 않습니다.",
    "exp": "**ln(y) = a + b·t** 를 회귀합니다. 매년 **일정한 비율(복리)** 로 변하는 흐름에 맞고, 값이 항상 양수입니다.",
    "cagr": "학습 기간 **첫해와 마지막해**로 구한 연평균 증감률을 매년 적용합니다. 가장 단순하지만 양 끝 값에 민감합니다.",
    "holt": "수준(level)과 추세(trend)를 **최근 값에 더 가중**해 갱신하는 지수평활입니다. 평활계수는 오차가 최소가 되도록 **데이터에서 자동 추정**되며, 최근 흐름에 민감합니다.",
}


def _holt(y):
    """Holt 선형 지수평활: alpha/beta를 격자탐색으로 추정(1단계 예측오차 SSE 최소)"""
    best = None
    grid = np.arange(0.05, 1.0, 0.05)
    for a in grid:
        for b in grid:
            lv, tr, sse = y[0], y[1] - y[0], 0.0
            for v in y[1:]:
                pred = lv + tr
                sse += (v - pred) ** 2
                new_lv = a * v + (1 - a) * pred
                tr = b * (new_lv - lv) + (1 - b) * tr
                lv = new_lv
            if best is None or sse < best[0]:
                best = (sse, lv, tr, a, b)
    return best


def adjust_path(last, out, mult=1.0, phi=1.0):
    """[추가 옵션] 모델이 추정한 경로의 '연간 증감폭'을 사용자가 조정
       mult : 기울기 배율 (1.0 = 모델 그대로, 0.5 = 증감폭 절반)
       phi  : 감쇠계수 (k년째 증감폭에 phi**k 적용, 1.0 = 적용 안 함)"""
    if mult == 1.0 and phi == 1.0:
        return out
    path = np.concatenate([[last], out])
    d = np.diff(path) * mult * phi ** np.arange(1, len(out) + 1)
    return last + np.cumsum(d)


def project(years, vals, n, model, train_start=None, anchor=True, mult=1.0, phi=1.0, rate_pct=None):
    """
    years/vals : 실적·계획 연도와 값 (NaN 허용)
    n          : 이후 몇 년을 예측할지
    anchor     : True면 마지막 해 값에서 끊김 없이 이어지도록 모델선을 평행이동
    mult/phi   : [추가 옵션] 기울기 배율 / 감쇠계수 (기본값이면 모델 결과 그대로)
    rate_pct   : [추가 옵션] 값이 있으면 모델 대신 '마지막 값 × (1+증감률)^t' 사용
    반환 : (예측값 ndarray 또는 None, 비고 문자열 또는 None)
    """
    years = np.asarray(years, float)
    vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    if train_start is not None:
        ok &= years >= train_start
    if ok.sum() == 0:
        return None, None
    yr, y = years[ok], vals[ok]
    last = y[-1]
    k = np.arange(1, n + 1)
    note = None

    # 학습 점이 부족하면 선형으로 대체
    need = {"linear": 2, "quad": 3, "log": 2, "exp": 2, "cagr": 2, "holt": 4}[model]
    use = model
    if len(y) < need:
        if len(y) >= 2:
            use, note = "linear", f"학습 점 {len(y)}개로 부족해 선형으로 대체"
        else:
            return np.full(n, last), "학습 점 1개: 마지막 값 유지"
    if use == "exp" and (y <= 0).any():
        use, note = "linear", "0 이하 값이 있어 지수 추세 불가 → 선형으로 대체"

    t = yr - yr[0]
    tn = (yr[-1] + k) - yr[0]

    if use == "cagr":
        span = yr[-1] - yr[0]
        if y[0] > 0 and last > 0 and span > 0:
            r = (last / y[0]) ** (1 / span) - 1
        else:
            r, note = 0.0, "시작/끝 값이 0 이하라 증감률 0 적용"
        out = last * (1 + r) ** k
    elif use == "holt":
        _, lv, tr, _, _ = _holt(y)
        out = lv + tr * k
        if anchor:
            out = out + (last - (lv + tr * 0))  # 마지막 실적과 평활 수준의 차이만큼 이동
    else:
        if use == "linear":
            c = np.polyfit(t, y, 1); f = lambda x: np.polyval(c, x)
        elif use == "quad":
            c = np.polyfit(t, y, 2); f = lambda x: np.polyval(c, x)
        elif use == "log":
            c = np.polyfit(np.log(t + 1), y, 1); f = lambda x: np.polyval(c, np.log(x + 1))
        else:  # exp
            c = np.polyfit(t, np.log(y), 1); f = lambda x: np.exp(np.polyval(c, x))
        out = f(tn)
        if anchor:
            out = out + (last - f(t[-1]))
    if rate_pct is not None and not (isinstance(rate_pct, float) and np.isnan(rate_pct)):
        out = last * (1 + float(rate_pct) / 100.0) ** k
        note = f"직접 증감률 {float(rate_pct):+.2f}%/년 적용"
    else:
        out = adjust_path(last, out, mult, phi)
    return np.maximum(0, out), note


def backtest_mape(years, vals, model, train_start, anchor, holdout=3):
    """마지막 3개 연도를 가리고 예측했을 때의 평균 오차율(검증용). 점이 부족하면 NaN"""
    years = np.asarray(years, float)
    vals = np.asarray(vals, float)
    ok = ~np.isnan(vals) & (years >= train_start)
    yr, y = years[ok], vals[ok]
    if len(y) < holdout + 4:
        return np.nan
    pred, _ = project(yr[:-holdout], y[:-holdout], holdout, model, None, anchor)
    if pred is None:
        return np.nan
    test = y[-holdout:]
    return float(np.mean(np.abs(pred - test) / np.maximum(np.abs(test), 1e-9)))


def build_projection(layout, global_model, train_start, anchor, row_model,
                     mult=1.0, phi=1.0, row_mult=None, row_rate=None):
    """반환: (예측 DataFrame[엑셀행 x 연도], 적용모델 dict, 비고 dict)
       mult/phi/row_mult/row_rate 는 [추가 옵션]. 기본값이면 모델 추정 결과 그대로."""
    row_mult, row_rate = row_mult or {}, row_rate or {}
    hist, base_end = layout["hist"], layout["base_end"]
    hy = [y for y in hist.columns if y <= base_end]
    fut = list(range(base_end + 1, TARGET_END + 1))
    rows, used, notes = {}, {}, {}
    for r in layout["data_rows"]:
        m = row_model.get(r) or global_model
        rm = row_mult.get(r)
        p, nt = project(hy, hist.loc[r, hy].values.astype(float), len(fut), m, train_start, anchor,
                        mult=(rm / 100.0 if rm is not None else mult), phi=phi, rate_pct=row_rate.get(r))
        if p is not None:
            rows[r], used[r] = p, m
            if nt:
                notes[r] = nt
    return pd.DataFrame(rows, index=fut).T, used, notes


# ─────────────────────────────────────────────────────────
# 2. 엑셀 구조 인식 / 쓰기
# ─────────────────────────────────────────────────────────
def _norm(s):
    return re.sub(r"\s+", "", str(s)) if s is not None else ""


def _find_header(ws):
    best = None
    for r in range(1, min(ws.max_row, 40) + 1):
        cols = {}
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if v is None:
                continue
            m = YEAR_RE.match(str(v))
            if m:
                cols[int(m.group(1))] = c
        if len(cols) >= 5 and (best is None or len(cols) > len(best[1])):
            best = (r, cols)
    return best


def parse_layout(ws, ws_v):
    h = _find_header(ws)
    if not h:
        return None
    header_row, ycols = h
    first_col = min(ycols.values())
    label_col = None
    for c in range(first_col - 1, 0, -1):
        n = sum(isinstance(ws.cell(r, c).value, str) and ws.cell(r, c).value.strip() != ""
                for r in range(header_row + 1, header_row + 16))
        if n >= 3:
            label_col = c
            break
    if label_col is None:
        return None
    total_row, data_rows = None, []
    for r in range(header_row + 1, ws.max_row + 1):
        lab = ws.cell(r, label_col).value
        if lab is None or str(lab).strip() == "":
            continue
        if _norm(lab) in TOTAL_LABELS:
            total_row = r
            break
        data_rows.append(r)
    if total_row is None or not data_rows:
        return None
    years = sorted(ycols)

    def num(r, c):
        v = ws_v.cell(r, c).value
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan

    labels = {r: str(ws.cell(r, label_col).value).strip() for r in data_rows}
    hist = pd.DataFrame({r: [num(r, ycols[y]) for y in years] for r in data_rows}, index=years).T
    filled = [y for y in years if hist[y].notna().any()]
    if not filled:
        return None
    return dict(header_row=header_row, ycols=ycols, label_col=label_col, labels=labels,
                data_rows=data_rows, total_row=total_row, hist=hist, base_end=max(filled))


def write_workbook(file_bytes, sheet_name, layout, proj, used, global_model, train_start, anchor, extra=None):
    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb[sheet_name]
    ycols = dict(layout["ycols"])
    base_end = layout["base_end"]
    src_col = ycols[base_end]
    hr, tr = layout["header_row"], layout["total_row"]
    fut = list(proj.columns)

    src_hdr = ws.cell(hr, src_col)
    for y in fut:
        if y in ycols:
            continue
        new_c = max(ycols.values()) + 1
        ycols[y] = new_c
        h = ws.cell(hr, new_c)
        h._style = copy(src_hdr._style)
        h.value = f"{y}년" if isinstance(src_hdr.value, str) and "년" in src_hdr.value else y
    for y in fut:
        w = ws.column_dimensions[get_column_letter(src_col)].width
        if w:
            ws.column_dimensions[get_column_letter(ycols[y])].width = w

    def style_from_src(cell, src):
        if not cell.has_style:
            cell._style = copy(src._style)
        cell.number_format = src.number_format

    for r in layout["data_rows"]:
        if r not in proj.index:
            continue
        src = ws.cell(r, src_col)
        for y in fut:
            c = ws.cell(r, ycols[y])
            c.value = int(round(float(proj.loc[r, y])))
            style_from_src(c, src)

    # 옵션2: 확정 이후 기존 계획 구간을 시뮬레이션 값으로 대체 (기존 계획값은 메모 시트에 기록)
    ov_log = []
    for r, ser in ((extra or {}).get("overrides") or {}).items():
        for y, v in ser.items():
            if y in ycols:
                c = ws.cell(r, ycols[y])
                ov_log.append([layout["labels"][r], y, c.value if not (isinstance(c.value, str) and c.value.startswith("=")) else "(수식)", int(round(float(v)))])
                c.value = int(round(float(v)))

    first_r, last_r = min(layout["data_rows"]), max(layout["data_rows"])
    src_t = ws.cell(tr, src_col)
    for y in fut:
        c = ws.cell(tr, ycols[y])
        if isinstance(src_t.value, str) and src_t.value.startswith("="):
            c.value = Translator(src_t.value, origin=src_t.coordinate).translate_formula(c.coordinate)
        else:
            L = get_column_letter(ycols[y])
            c.value = f"=SUM({L}{first_r}:{L}{last_r})"
        style_from_src(c, src_t)

    for r in range(tr + 1, ws.max_row + 1):
        src = ws.cell(r, src_col)
        if isinstance(src.value, str) and src.value.startswith("="):
            for y in fut:
                c = ws.cell(r, ycols[y])
                c.value = Translator(src.value, origin=src.coordinate).translate_formula(c.coordinate)
                style_from_src(c, src)

    # 배경색: 확정 구간(2026~확정 연도) = 노랑, 추정 구간(이후~2045) = 연한 파랑
    conf = (extra or {}).get("conf_year")
    if conf:
        from openpyxl.styles import PatternFill
        f_conf = PatternFill("solid", start_color="FFE699", end_color="FFE699")
        f_est = PatternFill("solid", start_color="DDEBF7", end_color="DDEBF7")
        for y, col in ycols.items():
            if y < 2026:
                continue
            fill = f_conf if y <= conf else f_est
            for r in list(layout["data_rows"]) + [tr]:
                ws.cell(r, col).fill = fill
            if y > conf:
                ws.cell(hr, col).fill = f_est
        ws.cell(hr + 0, 1)  # no-op

    memo_name = "2045확장_적용내역"
    if memo_name in wb.sheetnames:
        del wb[memo_name]
    memo = wb.create_sheet(memo_name)
    memo.append(["항목", "내용"])
    if conf:
        memo.append(["배경색", f"2026~{conf}년 노랑(확정), {conf + 1}~{TARGET_END}년 연한 파랑(추정)"])
    memo.append(["기준(마지막 계획) 연도", base_end])
    memo.append(["학습 시작 연도", train_start])
    memo.append(["전체 적용 모델", MODEL_NAME.get(global_model, global_model)])
    memo.append(["마지막 해 값에서 이어붙이기", "예" if anchor else "아니오"])
    extra = extra or {}
    memo.append(["추가 옵션(사용자 조정)", "적용" if extra.get("on") else "미적용 (모델 추정값 그대로)"])
    if extra.get("on"):
        memo.append(["기울기 배율(%)", extra.get("mult_pct", 100)])
        memo.append(["감쇠계수", extra.get("phi", 1.0)])
    for nt in extra.get("notes", []):
        memo.append(list(nt))
    memo.append([])
    if ov_log:
        memo.append(["[옵션2] 대체된 기존 계획값", "연도", "기존 계획", "시뮬레이션"])
        for row in ov_log:
            memo.append([row[0], row[1], row[2], row[3]])
        memo.append([])
    memo.append(["용도", "적용 모델", "기울기 배율(%) 개별", "직접 증감률(%/년)"])
    for r in layout["data_rows"]:
        if r in used:
            rm = (extra.get("row_mult") or {}).get(r)
            rr = (extra.get("row_rate") or {}).get(r)
            memo.append([layout["labels"][r], MODEL_NAME.get(used[r], used[r]),
                         "" if rm is None else rm, "" if rr is None else rr])
    for col, w in (("A", 28), ("B", 32), ("C", 22), ("D", 22)):
        memo.column_dimensions[col].width = w
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


# ─────────────────────────────────────────────────────────
# 3. 파일 로딩
# ─────────────────────────────────────────────────────────
def _repo_plan_path():
    """저장소의 계획 엑셀: '2045_수정파일*.xlsx' 중 가장 최근(이름순) 파일 우선, 없으면 기존 파일명"""
    import glob
    try:
        bases = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    except NameError:
        bases = [os.getcwd()]
    for d in bases:
        c = sorted(glob.glob(os.path.join(d, "2045_수정파일*.xlsx")))
        if c:
            return c[-1]
        p = os.path.join(d, REPO_PLAN_FILE)
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_layouts(data: bytes):
    wb_f = load_workbook(io.BytesIO(data))
    wb_v = load_workbook(io.BytesIO(data), data_only=True)
    out = {}
    for name in wb_f.sheetnames:
        lay = parse_layout(wb_f[name], wb_v[name])
        if lay:
            out[name] = lay
    return out


# ─────────────────────────────────────────────────────────
# 3-1. 가정용 기온 시뮬레이션
#   ① 일 기온 → 난방도일(HDD) → 월(또는 일) 공급량 관계를 실적으로 적합  (데이터 추정)
#   ② 연평균 기온이 매년 +rate℃ 오른다고 가정하고, 일 기온 전체를 평행이동해 연간 물량 변화율을 계산
#   ③ 공동주택 신규유입(활황/비활황 주기) 증가율을 곱해 2036~2045 가정용 물량 산정
#   V(Y) = V(2035) × Π(1+신규유입 증가율) × 기온효과(Y)    ※ 2035년 계획값을 출발점으로 함
# ─────────────────────────────────────────────────────────
HH_TEMP_FILE = "기온_198001_202512.xlsx"
HH_SUPPLY_FILE = "공급량실적_계획_실적_MJ.xlsx"


def _repo_file(name):
    try:
        bases = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    except NameError:
        bases = [os.getcwd()]
    for d in bases:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _read_repo_bytes(name):
    p = _repo_file(name)
    if not p:
        return None
    with open(p, "rb") as f:
        return f.read()


def _household_rows(layout):
    """계획표에서 가정용 두 행(취사/개별난방, 중집난방)의 엑셀 행번호를 찾는다"""
    r5 = r6 = None
    for r, lab in layout["labels"].items():
        n = _norm(lab)
        if "주택용" not in n and "가정용" not in n:
            continue
        if "난방" in n and ("중집" in n or "중앙" in n):
            r6 = r
        elif "취사" in n or "개별" in n:
            r5 = r
    return r5, r6


@st.cache_data(show_spinner=False)
def load_daily_temp(data: bytes):
    df = pd.read_excel(io.BytesIO(data))
    dcol = next(c for c in df.columns if any(k in str(c) for k in ("날짜", "일자")))
    tcol = next(c for c in df.columns if "기온" in str(c) and "최" not in str(c))
    out = pd.DataFrame({"날짜": pd.to_datetime(df[dcol], errors="coerce"),
                        "T": pd.to_numeric(df[tcol], errors="coerce")}).dropna(subset=["날짜"]).sort_values("날짜")
    out["T"] = out["T"].interpolate(limit=3)          # 1~3일 결측은 보간
    out = out.dropna(subset=["T"])
    out["연"], out["월"] = out["날짜"].dt.year, out["날짜"].dt.month
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_household_monthly(data: bytes):
    """공급량 실적 시트에서 취사/개별난방(취사용+개별난방용) 월 공급량 [GJ]"""
    x = pd.ExcelFile(io.BytesIO(data))
    sh = next((s for s in x.sheet_names if "실적" in s), x.sheet_names[0])
    df = x.parse(sh)
    need = ["취사용", "개별난방용"]
    miss = [c for c in need + ["연", "월"] if c not in df.columns]
    if miss:
        raise ValueError(f"공급량 실적 시트에 {miss} 컬럼이 없습니다")
    df = df.dropna(subset=need + ["연", "월"])
    out = pd.DataFrame({"연": df["연"].astype(int).values, "월": df["월"].astype(int).values,
                        "S": (df[need].sum(axis=1) / 1000.0).values})      # MJ → GJ
    return out[out["S"] > 0].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_daily_sheet(data: bytes, name: str):
    """구글 시트를 내보낸 일별 파일(일자, 공급량(MJ), 평균기온) 읽기"""
    if name.lower().endswith(".csv"):
        df = None
        for enc in ("utf-8-sig", "cp949"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            raise ValueError("CSV를 읽을 수 없습니다")
    else:
        df = pd.read_excel(io.BytesIO(data))
    cols = [str(c) for c in df.columns]
    dcol = next(c for c in cols if "일자" in c or "날짜" in c)
    scol = next((c for c in cols if "공급량" in c and "MJ" in c.upper()), None)
    div = 1000.0
    if scol is None:
        scol = next(c for c in cols if "공급량" in c)
        div = 1.0
    tcol = next(c for c in cols if "기온" in c and "최" not in c)
    df.columns = cols
    out = pd.DataFrame({"날짜": pd.to_datetime(df[dcol], errors="coerce"),
                        "S": pd.to_numeric(df[scol], errors="coerce") / div,
                        "T": pd.to_numeric(df[tcol], errors="coerce")}).dropna()
    out = out[out["S"] > 0]
    out["연"], out["월"] = out["날짜"].dt.year, out["날짜"].dt.month
    return out.reset_index(drop=True)


def _als(y, H, yi, nY, iters=150):
    """S = k_연도 × (a + b·HDD) 적합 (k: 연도별 규모효과, 교대최소제곱)"""
    k = np.ones(nY)
    for _ in range(iters):
        kk = k[yi]
        a0, b0 = np.linalg.lstsq(np.c_[kk, kk * H], y, rcond=None)[0]
        mu = a0 + b0 * H
        k = np.bincount(yi, y * mu, nY) / np.maximum(np.bincount(yi, mu * mu, nY), 1e-12)
        k = k / k.mean()
    fit = k[yi] * (a0 + b0 * H)
    return a0, b0, k, fit


@st.cache_data(show_spinner=False)
def fit_hdd_model(temp_df, obs_df, level):
    """기준온도(Tb)를 14~24℃에서 탐색해 설명력(R²)이 가장 높은 난방도일 모델을 선택"""
    best = None
    for Tb in np.arange(14.0, 24.01, 0.5):
        if level == "월":
            h = (np.maximum(0, Tb - temp_df["T"]).groupby([temp_df["연"], temp_df["월"]]).sum()
                 .rename("H").reset_index())
            m = obs_df.merge(h, on=["연", "월"])
        else:
            m = obs_df.copy()
            m["H"] = np.maximum(0, Tb - m["T"])
        if len(m) < 24:
            continue
        y, H = m["S"].values.astype(float), m["H"].values.astype(float)
        yrs, yi = np.unique(m["연"].values, return_inverse=True)
        a0, b0, k, fit = _als(y, H, yi, len(yrs))
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best["r2"]:
            best = dict(Tb=float(Tb), a=float(a0), b=float(b0), r2=float(r2), n=int(len(m)),
                        k=pd.Series(k, index=yrs.astype(int)), level=level)
    return best


def parse_gsheet_household(csv_text: str):
    """구글 시트 CSV에서 '2018-01' 형식 월 헤더 + '취사용'/'개별난방용' 행을 찾아 월별 합계(GJ) 반환 (시트 단위 MJ)"""
    raw = pd.read_csv(io.StringIO(csv_text), header=None, dtype=str, keep_default_na=False)
    pat = re.compile(r"^\s*(\d{4})[-./](\d{1,2})\s*$")
    hdr = None
    for i in range(len(raw)):
        if sum(bool(pat.match(str(v))) for v in raw.iloc[i].values) >= 12:
            hdr = i
            break
    if hdr is None:
        raise ValueError("월 헤더(예: 2018-01)를 찾지 못했습니다")
    cols = {j: pat.match(str(v)).groups() for j, v in enumerate(raw.iloc[hdr].values) if pat.match(str(v))}

    def find_row(label):
        for i in range(hdr + 1, min(len(raw), hdr + 60)):
            if any(str(v).strip() == label for v in raw.iloc[i, :4].values):
                return i
        raise ValueError(f"'{label}' 행을 찾지 못했습니다")

    def num(v):
        v = str(v).replace(",", "").strip()
        try:
            return float(v)
        except ValueError:
            return np.nan
    r_a, r_b = find_row("취사용"), find_row("개별난방용")
    rows = []
    for j, (y, m) in cols.items():
        a, b = num(raw.iat[r_a, j]), num(raw.iat[r_b, j])
        if np.isnan(a) or np.isnan(b):
            continue
        rows.append((int(y), int(m), (a + b) / 1000.0))      # MJ → GJ
    out = pd.DataFrame(rows, columns=["연", "월", "S"])
    return out[out["S"] > 0].sort_values(["연", "월"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_gsheet_household(sheet_id: str):
    import requests
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return parse_gsheet_household(r.content.decode("utf-8", errors="replace"))


def get_household_monthly(sup_b):
    """월별 취사용+개별난방용 실적: 구글 시트 → (실패 시) 저장소 공급량 실적 엑셀 순서"""
    try:
        df = fetch_gsheet_household(GSHEET_ID)
        if len(df) >= 60:
            return df, "구글 시트 (취사용+개별난방용)"
    except Exception:
        pass
    if sup_b:
        return load_household_monthly(sup_b), "저장소 공급량 실적 엑셀 (구글 시트에 연결하지 못함)"
    return None, ""


def render_history_section(y0):
    """화면 상단: 2018~2025 가정용(취사용+개별난방용) 과거 실적과 추세"""
    if not st.toggle("📊 과거 실적 분석 (가정용 취사용+개별난방용)", value=False, key="hist_on",
                     help="켜면 가정용 과거 실적(월별 → 연간)과 기준연도 대비 증감·추세를 보여줍니다."):
        return
    st.subheader("📊 가정용 과거 실적 분석 (취사용+개별난방용)")
    sup_b = _read_repo_bytes(HH_SUPPLY_FILE)
    try:
        mdf, src = get_household_monthly(sup_b)
    except Exception as e:
        st.warning(f"과거 실적을 읽지 못했습니다: {e}")
        return
    if mdf is None:
        st.warning("월별 실적 데이터가 없습니다.")
        return
    Y1 = 2025
    mdf = mdf[mdf["연"] <= Y1]
    yrs_av = sorted(int(y) for y in mdf["연"].unique())
    opts = [y for y in yrs_av if y <= Y1 - 3]
    if not opts:
        st.warning("분석할 연도 데이터가 부족합니다.")
        return
    sel_col, _sp0 = st.columns([1, 2])
    with sel_col:
        Y0 = st.selectbox("기준연도 (이 해 대비 증감 비교)", opts, index=opts.index(2018) if 2018 in opts else 0,
                          key="hist_base_year", help=f"선택한 연도부터 {Y1}년까지의 실적과 추세를 보여줍니다.")
    temp_b = _read_repo_bytes(HH_TEMP_FILE)
    temp_df = load_daily_temp(temp_b) if temp_b else None
    calib, ref_years, normal = None, None, None
    if temp_df is not None:
        try:
            calib = fit_hdd_model(temp_df, mdf, "월")
        except Exception:
            calib = None
    # 월이 빠진 해는 모델로 보완
    filled = {}
    ann = mdf.groupby("연")["S"].sum()
    cnt = mdf.groupby("연")["S"].count()
    for y in range(Y0, Y1 + 1):
        if y in ann.index and cnt[y] < 12:
            miss = sorted(set(range(1, 13)) - set(mdf[mdf["연"] == y]["월"]))
            if calib is not None and y in calib["k"].index and temp_df is not None:
                add = 0.0
                for m in miss:
                    d = temp_df[(temp_df["연"] == y) & (temp_df["월"] == m)]
                    if len(d) == 0:
                        add = np.nan
                        break
                    add += calib["k"][y] * (calib["a"] + calib["b"] * float(np.maximum(0, calib["Tb"] - d["T"]).sum()))
                if not np.isnan(add):
                    ann[y] += add
                    filled[y] = miss
    ann = ann.reindex(range(Y0, Y1 + 1)).dropna()
    if len(ann) < 3:
        st.warning("분석할 연도 데이터가 부족합니다.")
        return
    rows = {"연": ann.index.tolist(), "실제": (ann.values / 1e6).tolist()}
    norm = None
    if calib is not None:
        full_y = _full_years(temp_df)
        ref_years = full_y[-10:]
        base = season_total(calib, temp_df, ref_years, 0.0)
        norm = pd.Series({y: calib["k"][y] * base for y in ann.index if y in calib["k"].index})
        rows["기온보정"] = [norm.get(y, np.nan) / 1e6 for y in ann.index]
    tbl = pd.DataFrame(rows).set_index("연")
    a0, a1 = tbl["실제"].iloc[0], tbl["실제"].iloc[-1]
    n = len(tbl) - 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{tbl.index[0]}년 공급량", f"{a0:,.2f} 백만GJ")
    c2.metric(f"{tbl.index[-1]}년 공급량", f"{a1:,.2f} 백만GJ", f"{a1 / a0 - 1:+.1%} ({tbl.index[0]}년 대비)")
    c3.metric(f"{tbl.index[0]}년 대비 증감률 (연평균)", f"{(a1 / a0) ** (1 / n) - 1:+.2%}")
    _xx = tbl.index.values.astype(float)
    _sl = np.polyfit(_xx, tbl["실제"].values, 1)[0]
    c4.metric("추세선 기울기", f"{_sl:+.2f} 백만GJ/년", f"{_sl / a0:+.2%} ({tbl.index[0]}년 대비 연평균)")
    x = tbl.index.values.astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tbl.index, y=tbl["실제"], mode="lines+markers+text", name="실제 공급량",
                             line=dict(color="black", width=4), text=[f"{v:.1f}" for v in tbl["실제"]], textposition="top center"))
    allv = list(tbl["실제"].values)
    b1, b0 = np.polyfit(x, tbl["실제"].values, 1)
    fig.add_trace(go.Scatter(x=tbl.index, y=b0 + b1 * x, mode="lines", name="공급량 추세선",
                             line=dict(color="black", width=2, dash="dash")))
    allv += list(b0 + b1 * x)
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.35 + 0.1
    fig.update_xaxes(dtick=1, tickformat="d")
    fig.update_yaxes(title="백만 GJ", tickformat=",.1f", range=[lo - pad, hi + pad])
    fig.update_layout(height=400, margin=dict(t=60, b=10), legend=dict(orientation="h", y=-0.2),
                      title=dict(text=f"취사용+개별난방용 연간 공급량 ({tbl.index[0]}∼{tbl.index[-1]}년)", x=0.01, xanchor="left", y=0.95, font=dict(size=15)))
    show_chart(fig)
    note = f"자료: {src}. "
    if filled:
        note += " / ".join(f"{y}년 {', '.join(str(m) + '월' for m in ms)}은 실적이 없어 기온-공급량 모델로 보완" for y, ms in filled.items()) + ". "
    note += "점선은 선택한 기준연도부터 2025년까지 실제 공급량의 직선 추세입니다. 연도별 공급량은 기온(추운 해·따뜻한 해)의 영향을 그대로 포함합니다."
    st.caption(note)
    with st.expander("📋 연도별 표", expanded=False):
        t = tbl.copy()
        t["실제 증감"] = t["실제"].pct_change()
        t = t[["실제", "실제 증감"]].rename(columns={"실제": "실제(백만GJ)"})
        t.index = [f"{i}년" for i in t.index]
        fm = {c: "{:,.2f}" for c in t.columns if "백만" in c}
        fm.update({c: "{:+.1%}" for c in t.columns if "증감" in c})
        st.dataframe(t.style.format(fm, na_rep=""), use_container_width=True)
    st.markdown("---")


def _full_years(temp_df):
    cnt = temp_df.groupby("연").size()
    return [int(y) for y in cnt.index if cnt[y] >= 360]


def season_total(calib, temp_df, ref_years, shift):
    """평년 기온(최근 N년 일 기온)을 shift℃ 평행이동했을 때의 연간 모델 물량(상대값)"""
    d = temp_df[temp_df["연"].isin(ref_years)]
    hdd = float(np.maximum(0, calib["Tb"] - (d["T"] + shift)).groupby(d["연"]).sum().mean())
    n = 12.0 if calib["level"] == "월" else 365.25
    return calib["a"] * n + calib["b"] * hdd


def _bell(pos, n):
    """활황기 안에서의 위치(0~n-1) → 0(시작·끝)~1(중간 최대) 종 모양"""
    sh = [np.sin(np.pi * k / (n - 1)) for k in range(n)]
    return sh[pos] / max(sh)


def simulate_household(v5, v6, years, calib, temp_df, ref_years, p):
    """기준연도(2035) 값에서 출발해 연도별 물량 산정
       취사/개별난방 = V × Π(1+신규유입 증가율) × 기온 효과
       중집난방      = 취사/개별난방과 같은 증감 비율(기준연도 대비)을 그대로 적용"""
    base = season_total(calib, temp_df, ref_years, p["dT0"])
    G = 1.0
    rows = []
    for Y in years:
        d = p["rate"] * (Y - p["base_year"])
        R = season_total(calib, temp_df, ref_years, p["dT0"] + d) / base
        pos = (Y - p["boom_start"]) % p["cycle"]
        boom = (Y >= p["boom_start"]) and pos < p["boom_len"]
        if boom:
            g = p["g_end"] if p["boom_len"] <= 1 else p["g_start"] + (p["g_end"] - p["g_start"]) * _bell(pos, p["boom_len"])
        else:
            g = p["g_non"]
        G *= 1 + g / 100.0
        rows.append(dict(연=Y, 기온상승=d, 활황=boom, 신규유입=g, G=G, R=R,
                         v5=v5 * G * R, v6=v6 * G * R,
                         v5_nw=v5 * G,        # 기온 상승 없음 (신규유입만)
                         v5_ng=v5 * R))       # 신규유입 없음 (기온만)
    return pd.DataFrame(rows).set_index("연")


def obs_temp_trends(temp_df):
    """관측 연평균 기온의 선형 추세(℃/년): 기간별 참고값"""
    ann = temp_df.groupby("연")["T"].mean()
    full = set(_full_years(temp_df))
    ann = ann[ann.index.isin(full)]
    last = int(ann.index.max())
    out = []
    for a in (1980, 2000, 2010, last - 9):
        x = ann.loc[a:last]
        if len(x) >= 5:
            out.append((a, last, float(np.polyfit(x.index.values, x.values, 1)[0])))
    return out


def render_household_section(layout, hist, hy, labels, base_end, y0, anchor_year, scope2):
    """화면 최상단: 가정용(취사/개별난방) 기온 시뮬레이션. 중집난방은 같은 증감 비율을 따라감.
       anchor_year : 시뮬레이션 출발(기준) 연도. 옵션1 = 계획 마지막 해(2035), 옵션2 = 확정 마지막 해(2029)
       반환: {apply, rows{엑셀행: Series(연도)}, notes[[항목,내용]]}"""
    out = {"apply": False, "rows": {}, "notes": []}
    st.subheader("🌡️ 가정용 기온 시뮬레이션 적용")
    r5, r6 = _household_rows(layout)
    if r5 is None or r6 is None:
        st.info("계획표에서 '주택용_취사/개별난방', '주택용_중집난방' 행을 찾지 못해 가정용 시뮬레이션을 건너뜁니다.")
        return out
    if anchor_year not in hist.columns or np.isnan(hist.loc[r5, anchor_year]) or np.isnan(hist.loc[r6, anchor_year]):
        st.warning(f"{anchor_year}년 가정용 값이 엑셀에 없어 시뮬레이션을 건너뜁니다. (옵션1로 바꾸거나 엑셀을 확인하세요)")
        return out

    apply = True  # 가정용 시뮬레이션은 항상 전체 추정에 반영
    half, _sp = st.columns([1, 1])
    with half:
        rate = st.number_input("연평균 기온 상승 (℃/년)", 0.0, 0.5, 0.117, 0.005, format="%.3f", key="hh_rate",
                               help="기본 0.117 = 2016∼2025년 관측 연평균 기온의 선형 추세. 아래 표에서 기간별 추세를 확인하세요.")
    start_txt = f"{anchor_year}년 {'확정' if scope2 else '계획'}값"
    trend_slot = half.container()
    show_tr = True  # 기간별 추세 표는 항상 표시
    # ── 공동주택 신규유입(분양/입주) 요약표 : 기온 표 오른쪽 ──
    _bs = int(st.session_state.get("hh_bs", 2030)); _bl = int(st.session_state.get("hh_bl", 4))
    _cy = int(st.session_state.get("hh_cy", 10)); _lo = st.session_state.get("hh_g0", 0.3)
    _hi = st.session_state.get("hh_g1", 0.7); _gn = st.session_state.get("hh_gn", 0.0)
    _rows, _cur = [], None
    for _y in range(anchor_year + 1, TARGET_END + 1):
        _b = _y >= _bs and (_y - _bs) % _cy < _bl
        if _cur is None or _cur[2] != _b:
            _cur = [_y, _y, _b]; _rows.append(_cur)
        else:
            _cur[1] = _y
    _tb = pd.DataFrame([(f"{a}∼{b}", "분양/입주 증가 예상" if on else ("평시 (증가 없음)" if _gn == 0 else "평시 (소폭 증가)"),
                         f"+{_lo:g}∼{_hi:g}%/년" if on else f"+{_gn:g}%/년") for a, b, on in _rows],
                       columns=["기간", "공동주택 분양/입주", "신규유입 증가율"])
    _on = [r[2] for r in _rows]

    def _hl2(row):
        return ["background-color: #FFE699; font-weight: 700; color: #000" if _on[row.name] else "" for _ in row]
    with _sp:
        st.markdown("<div style='height:%dpx;font-weight:600;padding-top:6px'>공동주택 신규유입 물량</div>" % HH_TITLE_H, unsafe_allow_html=True)
        st.table(_tb.style.hide(axis="index").apply(_hl2, axis=1)
                 .set_table_styles([{"selector": "th", "props": [("background-color", "#D9D9D9"), ("color", "#000"), ("font-weight", "700")]}])
                 .set_properties(**{"text-align": "center"}))
        st.caption("▶ 강조된 기간: 공동주택 분양/입주 증가로 신규유입 물량이 늘어나는 기간 (시작·끝 해는 낮고 중간 해가 최대)")
    _g0, _g1 = st.session_state.get("hh_g0", 0.3), st.session_state.get("hh_g1", 0.7)
    st.info(f"- 연평균기온 {rate:g}도 상승 가정\n- 공동주택 신규유입 {_g0:g}%∼{_g1:g}% 반영")

    # ── 데이터 ──
    temp_b, sup_b = _read_repo_bytes(HH_TEMP_FILE), _read_repo_bytes(HH_SUPPLY_FILE)
    u_t = u_s = u_d = None
    if temp_b is None or sup_b is None:
        with st.expander("📂 기온·공급량 데이터 (저장소에 파일이 없어 업로드가 필요합니다)", expanded=True):
            u_t = st.file_uploader("일 평균기온 엑셀 (기온_198001_202512.xlsx)", type=["xlsx"], key="hh_up_t")
            u_s = st.file_uploader("월별 공급량 실적 엑셀 (공급량실적_계획_실적_MJ.xlsx)", type=["xlsx"], key="hh_up_s")
    temp_b = u_t.getvalue() if u_t is not None else temp_b
    sup_b = u_s.getvalue() if u_s is not None else sup_b
    try:
        temp_df = load_daily_temp(temp_b) if temp_b else None
    except Exception as e:
        st.error(f"기온 파일을 읽을 수 없습니다: {e}")
        return out
    daily_df = None
    if u_d is not None:
        try:
            daily_df = load_daily_sheet(u_d.getvalue(), u_d.name)
            if len(daily_df) < 300:
                st.warning(f"일별 시트에서 공급량이 있는 날이 {len(daily_df)}일뿐이라 월 실적으로 보정합니다.")
                daily_df = None
        except Exception as e:
            st.warning(f"일별 시트를 읽지 못해 월 실적으로 보정합니다: {e}")
    if temp_df is None and daily_df is not None:
        temp_df = daily_df[["날짜", "T", "연", "월"]]
    if temp_df is None or (sup_b is None and daily_df is None):
        st.warning("기온·공급량 파일이 필요합니다. 위 '기온·공급량 데이터'에서 업로드하세요.")
        return out
    try:
        if daily_df is not None:
            src = st.radio("보정 기준", ["일별 시트 (업로드)", "월별 실적 (저장소)"], horizontal=True, key="hh_src") if sup_b else "일별 시트 (업로드)"
        else:
            src = "월별 실적 (저장소)"
        if src.startswith("일별"):
            calib = fit_hdd_model(temp_df, daily_df, "일")
        else:
            obs, _src = get_household_monthly(sup_b)
            calib = fit_hdd_model(temp_df, obs, "월")
    except Exception as e:
        st.error(f"기온-공급량 관계를 적합하지 못했습니다: {e}")
        return out
    if not calib or calib["b"] <= 0 or calib["a"] <= 0:
        st.error("기온-공급량 관계가 유효하지 않습니다(난방 민감도 ≤ 0). 데이터를 확인해주세요.")
        return out

    if show_tr:
        with trend_slot:
            trs = obs_temp_trends(temp_df)
            tr_df = pd.DataFrame([(f"{a}∼{b}", f"{t:+.3f}") for a, b, t in trs], columns=["기간", "추세 (℃/년)"])
            hit = [abs(t - rate) < 0.0005 for _, _, t in trs]  # 적용 중인 값과 같은 기간을 강조

            def _hl(row):
                on = hit[row.name]
                return ["background-color: #FFE699; font-weight: 700; color: #000" if on else "" for _ in row]
            sty_tr = (tr_df.style.hide(axis="index").apply(_hl, axis=1)
                      .set_table_styles([{"selector": "th", "props": [("background-color", "#D9D9D9"), ("color", "#000"), ("font-weight", "700")]}])
                      .set_properties(**{"text-align": "center"}))
            st.table(sty_tr)
            if any(hit):
                st.caption("▶ 강조된 기간의 추세를 적용했습니다.")

    # ── 시나리오 가정 ──
    v5, v6 = float(hist.loc[r5, anchor_year]), float(hist.loc[r6, anchor_year])
    full_y = _full_years(temp_df)
    with st.expander("⚙️ 시나리오 가정 (공동주택 신규유입 · 기준 기온)", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            boom_start = st.number_input("활황 시작 연도", 2026, 2045, 2030, 1, key="hh_bs")
            boom_len = st.number_input("활황 기간 (년)", 1, 10, 4, 1, key="hh_bl", help="2030∼2033 = 4년")
            cycle = st.number_input("반복 주기 (년)", 5, 20, 10, 1, key="hh_cy", help="10년 주기 → 2040∼2045 재활황")
        with a2:
            g_start = st.number_input("활황기 증가율 최저 — 시작·끝 해 (%/년)", 0.0, 3.0, 0.3, 0.05, format="%.2f", key="hh_g0",
                                      help="활황기 첫 해와 마지막 해의 증가율. 중간으로 갈수록 커집니다.")
            g_end = st.number_input("활황기 증가율 최대 — 중간 해 (%/년)", 0.0, 3.0, 0.7, 0.05, format="%.2f", key="hh_g1",
                                    help="활황기 가운데 해의 증가율(종 모양으로 변함).")
            g_non = st.number_input("비활황기 증가율 (%/년)", -3.0, 3.0, 0.0, 0.05, format="%.2f", key="hh_gn",
                                    help="기온이 같을 때 신규 공동주택 유입으로 생기는 소폭 증가분(가정값).")
        with a3:
            n_ref = st.slider("평년 산정 기간 (최근 N년)", 5, min(30, len(full_y)), min(10, len(full_y)), key="hh_ref")
    dT0 = 0.0  # 출발 연도(3개년 롤링 평균 기온)를 평년 수준으로 보고 상승분만 더함
    ref_years = full_y[-n_ref:]
    p = dict(rate=rate, dT0=dT0, base_year=anchor_year, boom_start=int(boom_start), boom_len=int(boom_len),
             cycle=int(cycle), g_start=g_start, g_end=g_end, g_non=g_non)
    years = list(range(anchor_year + 1, TARGET_END + 1))
    sim = simulate_household(v5, v6, years, calib, temp_df, ref_years, p)

    # ── 근거 요약 ──
    s01 = season_total(calib, temp_df, ref_years, dT0 + 0.1) / season_total(calib, temp_df, ref_years, dT0) - 1
    s10 = season_total(calib, temp_df, ref_years, dT0 + 1.0) / season_total(calib, temp_df, ref_years, dT0) - 1
    kk5 = calib["k"].iloc[-5:]
    k_slope = np.polyfit(kk5.index.values, np.log(kk5.values), 1)[0] if len(kk5) >= 3 else np.nan
    per = "월" if calib["level"] == "월" else "일"
    # ── 결과 요약 (취사/개별난방 기준, 중집난방 제외) ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{anchor_year}년 취사/개별난방" + (" (확정)" if scope2 else ""), f"{v5:,.0f}")
    k2.metric("2040년", f"{sim.loc[2040, 'v5']:,.0f}", f"{sim.loc[2040, 'v5'] / v5 - 1:+.1%}")
    k3.metric(f"{TARGET_END}년", f"{sim.loc[TARGET_END, 'v5']:,.0f}", f"{sim.loc[TARGET_END, 'v5'] / v5 - 1:+.1%}")
    k4.metric(f"{TARGET_END}년 기온 효과 / 신규유입 효과",
              f"{sim.loc[TARGET_END, 'R'] - 1:+.1%} / {sim.loc[TARGET_END, 'G'] - 1:+.1%}")
    hy_a = [y for y in hy if 2026 <= y <= anchor_year]
    hist5 = hist.loc[r5, hy_a].astype(float)
    M = 1e6
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=hist5.index, y=hist5.values / M, mode="lines+markers",
                              name="확정" if scope2 else "계획/실적", line=dict(color="gray", width=3)))
    for col, nm, colr, dash, w in (("v5_nw", "기온 상승 없음 (신규유입만)", "#2ca02c", "dot", 2),
                                   ("v5_ng", "신규유입 없음 (기온만)", "#ff7f0e", "dot", 2),
                                   ("v5", "시뮬레이션 (기온+신규유입)", "black", "solid", 4)):
        figA.add_trace(go.Scatter(x=[anchor_year] + years, y=[v5 / M] + list(sim[col].values / M),
                                  mode="lines+markers" if col == "v5" else "lines",
                                  name=nm, line=dict(color=colr, width=w, dash=dash),
                                  hovertemplate="%{x}년: %{y:,.2f} 백만GJ<extra></extra>"))
    figA.add_vrect(x0=anchor_year + 0.5, x1=TARGET_END + 0.5, fillcolor="royalblue", opacity=0.07,
                   annotation_text="시뮬레이션", annotation_position="inside top")
    figA.update_xaxes(dtick=2, tickformat="d")
    figA.update_yaxes(title="백만 GJ", tickformat=",.1f")
    # Y축: 기존(0부터) 대비 변화가 약 2.5배 크게 보이도록 범위를 좁힘
    allv = np.concatenate([hist5.values, sim[["v5", "v5_nw", "v5_ng"]].values.ravel()]) / M
    span_old = allv.max() * 1.15
    half = max(span_old / 2.5 / 2, (allv.max() - allv.min()) * 0.6)
    mid = (allv.max() + allv.min()) / 2
    figA.update_yaxes(range=[mid - half, mid + half])
    figA.update_layout(height=460, margin=dict(t=70, b=10), legend=dict(orientation="h", y=-0.22),
                       title=dict(text="취사/개별난방 물량 (중집난방 제외)", x=0.01, xanchor="left", y=0.96, font=dict(size=15)))
    figB = go.Figure()
    for nm, colr, w, yv in (("합계 변화율", "black", 4, (sim["v5"] / v5 - 1) * 100),
                            ("기온 효과", "#ff7f0e", 2, (sim["R"] - 1) * 100),
                            ("신규유입 효과", "#2ca02c", 2, (sim["G"] - 1) * 100)):
        figB.add_trace(go.Scatter(x=[anchor_year] + years, y=[0] + list(yv.values), mode="lines+markers", name=nm,
                                  line=dict(color=colr, width=w), hovertemplate="%{x}년: %{y:+.2f}%<extra></extra>"))
    figB.add_hline(y=0, line_color="gray", line_width=1)
    figB.update_xaxes(dtick=2, tickformat="d")
    ymax = max(10.0, 1.6 * max(abs((sim["R"] - 1) * 100).max(), abs((sim["G"] - 1) * 100).max(), abs((sim["v5"] / v5 - 1) * 100).max()))
    figB.update_yaxes(title=f"{anchor_year}년 대비 (%)", ticksuffix="%", range=[-ymax, ymax])
    figB.update_layout(height=420, margin=dict(t=70, b=10), legend=dict(orientation="h", y=-0.22),
                       title=dict(text=f"{anchor_year}년 대비 변화율과 요인 분해", x=0.01, xanchor="left", y=0.96, font=dict(size=15)))
    show_chart(figA)
    show_chart(figB)
    st.caption(f"중집난방은 취사/개별난방과 같은 비율로 증감합니다: {anchor_year}년 {v6:,.0f} → "
               f"2040년 {sim.loc[2040, 'v6']:,.0f} → {TARGET_END}년 {sim.loc[TARGET_END, 'v6']:,.0f} ({sim.loc[TARGET_END, 'v6'] / v6 - 1:+.1%}).")

    with st.expander("📋 연도별 산정표 (취사/개별난방)", expanded=False):
        t = pd.DataFrame({
            "구분": np.where(sim["활황"], "활황기", "비활황기"),
            f"기온 상승(℃, {anchor_year}년 대비)": sim["기온상승"],
            "신규유입 증가율(%)": sim["신규유입"],
            "기온 효과(누적)": sim["R"] - 1,
            "신규유입 효과(누적)": sim["G"] - 1,
            labels[r5]: sim["v5"],
        })
        t["전년 대비"] = t[labels[r5]].pct_change()
        t.loc[years[0], "전년 대비"] = sim.loc[years[0], "v5"] / v5 - 1
        t.index = [f"{y}년" for y in t.index]
        st.dataframe(t.style.format({f"기온 상승(℃, {anchor_year}년 대비)": "{:.3f}", "신규유입 증가율(%)": "{:.2f}",
                                      "기온 효과(누적)": "{:+.2%}", "신규유입 효과(누적)": "{:+.2%}",
                                      labels[r5]: "{:,.0f}", "전년 대비": "{:+.2%}"}), use_container_width=True)
        st.caption("산식: V(Y) = V(출발연도) × Π(1+신규유입 증가율) × 기온 효과(Y). 기온 효과는 평년 일 기온을 평행이동해 난방도일을 다시 계산한 값입니다. "
                   "일 기온을 균등하게 올리므로 겨울철이 더 빨리 오르는 지구온난화 패턴은 반영하지 않습니다.")
    st.caption("※ 가정용을 확정(위 토글)하면 취사/개별난방과 중집난방 값이 아래 전체 추정·그래프·엑셀에 들어가며, 두 행에는 추세 모델·기울기 옵션이 적용되지 않습니다.")
    st.markdown("---")

    out["apply"] = bool(apply)
    out["rows"] = {r5: sim["v5"], r6: sim["v6"]}
    out["notes"] = [
        ["가정용 기온 시뮬레이션", "적용(확정)" if apply else "미적용"],
        ["추정 범위", f"옵션2: {anchor_year + 1}~{TARGET_END}년 ({anchor_year}년까지 확정)" if scope2 else f"옵션1: {anchor_year + 1}~{TARGET_END}년"],
        ["연평균 기온 상승(℃/년)", rate],
        ["난방도일 기준온도(℃) / R²", f"{calib['Tb']:.1f} / {calib['r2']:.3f}"],
        ["기온 +0.1℃ 영향(연간 물량)", f"{s01:+.2%}"],
        ["평년 산정 기간", f"{ref_years[0]}~{ref_years[-1]}"],
        ["공동주택 활황기", f"{int(boom_start)}년부터 {int(boom_len)}년간, {int(cycle)}년 주기, 증가율 {g_start:.2f}~{g_end:.2f}%/년(시작·끝 최저, 중간 최대)"],
        ["비활황기 증가율(%/년)", g_non],
        ["중집난방", "취사/개별난방과 같은 증감 비율(출발연도 대비) 적용"],
    ]
    return out


def main():
    st.title("🔥 도시가스 공급량 중장기 추정 (~2045)")
    st.caption("2035년까지 작성된 계획을 2045년까지 연장합니다. 가정용은 기온·공동주택 시나리오로, 나머지 용도는 추세 모델(데이터에서 계수 추정)로 계산합니다. (단위: GJ)")

    # ── 데이터 파일 (저장소 파일 우선, 필요 시 교체) ──
    repo_path = _repo_plan_path()
    with st.sidebar:
        st.header("📂 데이터")
        up = st.file_uploader("다른 엑셀로 계산하려면 업로드 (선택)", type=["xlsx"], key="p45_up")
    if up is not None:
        data, src_name = up.getvalue(), up.name
    elif repo_path:
        with open(repo_path, "rb") as f:
            data = f.read()
        src_name = os.path.basename(repo_path)
    else:
        st.info(f"👈 저장소에 '{REPO_PLAN_FILE}'이 없습니다. 왼쪽에서 엑셀을 업로드하세요.")
        return

    try:
        cands = load_layouts(data)
    except Exception as e:
        st.error(f"엑셀을 열 수 없습니다: {e}")
        return
    if not cands:
        st.error("연도 헤더(예: 2025년)와 '합계' 행이 있는 표를 찾지 못했습니다.")
        return
    sheet = list(cands)[0]
    if len(cands) > 1:
        sheet = st.sidebar.selectbox("시트", list(cands))
    layout = cands[sheet]
    base_end = layout["base_end"]
    if base_end >= TARGET_END:
        # 엑셀에 이미 2045년까지 값이 들어 있어도(이전 추정 결과), 원래 계획 마지막 해(2035)까지만 계획으로 보고 이후는 다시 추정해 덮어씁니다.
        base_end = PLAN_END
        layout = dict(layout, base_end=PLAN_END)
    hist = layout["hist"]
    hy = [y for y in hist.columns if y <= base_end]
    labels = layout["labels"]
    # 저장소 엑셀(업로드 아님)의 2026~2029년 값이 확정값과 다르면 확정값으로 교체 (엑셀 다운로드에도 반영)
    conf_over = {}
    if up is None:
        hist = hist.copy()
        for r in layout["data_rows"]:
            for y, v in CONFIRMED_VALUES.get(labels[r], {}).items():
                if y in hist.columns and (np.isnan(hist.loc[r, y]) or abs(hist.loc[r, y] - v) > 0.5):
                    hist.loc[r, y] = v
                    conf_over.setdefault(r, {})[y] = v
        layout = dict(layout, hist=hist)
        if conf_over:
            st.info("저장소 엑셀의 2026∼2029년 값이 확정값과 달라, 확정값(2026∼2029년)으로 교체해서 계산·다운로드합니다. "
                    "저장소 엑셀을 확정값 파일로 교체하면 이 안내는 사라집니다.")
    st.caption(f"📄 {src_name} · {base_end}년까지의 계획을 {TARGET_END}년까지 연장")

    # ── 추정 범위 (옵션1/2) ──
    scope2 = True  # 2030~2045년 추정이 기본 (옵션 선택 삭제)
    anchor_year = base_end
    if scope2:
        cands_y = [y for y in hy if 2025 <= y < base_end]
        anchor_year = st.number_input("확정 마지막 연도 (이 해까지는 확정, 다음 해부터 추정)", min_value=min(cands_y), max_value=max(cands_y),
                                      value=2029 if 2029 in cands_y else max(cands_y), step=1, key="p45_conf")
        anchor_year = int(anchor_year)
        st.caption(f"✔ {anchor_year}년까지 확정. 가정용(취사/개별·중집)은 {anchor_year + 1}∼{base_end}년 기존 계획 대신 기온 시뮬레이션 값을 사용하고, "
                   f"그 외 용도는 {anchor_year + 1}∼{base_end}년 기존 계획을 유지한 채 {base_end + 1}년부터 추세 모델로 추정합니다.")

    # ── 사이드바: 추세 옵션 ──
    with st.sidebar:
        st.markdown("---")
        st.header("📈 추세 옵션")
        mlabel = st.radio("추세 모델", list(MODELS), index=0, key="p45_model")
        gm = MODELS[mlabel]
        start_opts = [y for y in hy if y <= base_end - 3]
        default_start = 2029 if 2029 in start_opts else start_opts[0]
        train_start = st.select_slider(
            "학습 시작 연도", options=start_opts, value=default_start, key="p45_start",
            help="이 연도부터 {}년까지의 값으로 모델을 학습합니다. 기본 2029년 = 추세로 작성된 구간(2026∼2028 확정계획의 설비 증설 등 단절 구간 제외)".format(base_end))
        anchor = st.checkbox(f"{base_end}년 값에서 이어붙이기", value=True, key="p45_anchor",
                             help="모델선과 마지막 해 값의 차이만큼 평행이동해서 2036년에 값이 갑자기 튀지 않게 합니다.")
        st.caption(f"학습 구간: {train_start}∼{base_end}년 ({base_end - train_start + 1}개 점)")
        st.markdown("---")
        st.header("🖼️ 표시 옵션")
        y0 = st.checkbox("Y축을 0부터 표시", value=True, key="p45_y0",
                         help="체크하면 변동 비율이 실제 크기 그대로 보입니다. 해제하면 축을 확대해 작은 변화도 크게 보입니다.")
        st.markdown("---")
        st.header("🚚 수송용")
        hold_tr = st.checkbox("2035년 수준으로 임시 유지", value=True, key="p45_trhold",
                              help="수송용은 담당부서 자료를 받기로 해서, 자료 수령 전까지 2035년 값을 그대로 둡니다. 예측값이 아니라 자리표시입니다. 해제하면 선택한 추세 모델로 계산합니다.")

    render_history_section(y0)
    hh = render_household_section(layout, hist, hy, labels, base_end, y0, anchor_year, scope2)
    # 옵션2: 가정용 기존 계획(anchor+1~base_end)을 시뮬레이션 값으로 대체
    over = {}
    if scope2 and hh["apply"]:
        hist = hist.copy()
        for r, ser in hh["rows"].items():
            over[r] = ser[[y for y in ser.index if y <= base_end]]
            for y, v in over[r].items():
                hist.loc[r, y] = v
        layout = dict(layout, hist=hist)

    st.markdown("---")
    other = st.toggle("🔎 타용도 분석 (가정용 외 용도의 추세 모델·결과 그래프 보기)", value=False, key="p45_other",
                      help="켜면 가정용 외 용도의 추세 모델 선택, 기울기 조정, 합계·용도별 그래프와 상세표가 나타납니다. 꺼도 엑셀 다운로드에는 현재 설정이 그대로 반영됩니다.")
    adv, mult_pct, phi = False, 100, 1.0
    row_model, row_mult, row_rate = {}, {}, {}
    if other:
        st.markdown(f"#### 나머지 용도 추세 모델: {mlabel}")
        st.info(MODEL_DESC[gm])

        # ── [추가 옵션] 기울기 직접 조정 ──
        adv = st.toggle("⚙️ 추가 옵션 (기울기 직접 조정)", value=False, key="p45_adv",
                        help="모델이 추정한 증감 추세의 기울기를 시나리오 가정으로 직접 조정합니다. 꺼두면 모델 추정값 그대로입니다.")
        mult_pct, phi = 100, 1.0
        if adv:
            a1, a2 = st.columns(2)
            with a1:
                mult_pct = st.slider("기울기 배율 (%)", 0, 200, 100, 5, key="p45_mult",
                                     help="100% = 모델 추정 그대로, 50% = 연간 증감폭을 절반으로, 0% = 증감 없음(보합), 150% = 1.5배 가파르게")
            with a2:
                phi = st.slider("감쇠계수", 0.50, 1.00, 1.00, 0.01, key="p45_phi",
                                help="1.00 = 적용 안 함. 0.90이면 매년 증감폭이 전년의 90%로 줄어듭니다(매년 10%씩 감소).")
            st.caption("⚠️ 이 값은 모델이 데이터에서 추정한 것이 아니라 **사용자가 정하는 시나리오 가정**입니다. "
                       "엑셀의 '2045확장_적용내역' 시트에 함께 기록됩니다.")

        # ── 용도별 모델/기울기 개별 지정 ──
        seed = []
        for r in layout["data_rows"]:
            if np.isnan(hist.loc[r, hy].values.astype(float)).all():
                continue
            row = {"행": r, "용도": labels[r], "모델": FOLLOW_GLOBAL}
            if adv:
                row["기울기 배율(%)"] = np.nan
                row["직접 증감률(%/년)"] = np.nan
            seed.append(row)
        col_cfg = {"모델": st.column_config.SelectboxColumn("모델", options=[FOLLOW_GLOBAL] + list(MODELS)),
                   "행": st.column_config.NumberColumn("엑셀 행", width="small")}
        if adv:
            col_cfg["기울기 배율(%)"] = st.column_config.NumberColumn(
                "기울기 배율(%)", min_value=0, max_value=300, step=5, format="%.0f",
                help="비워두면 위의 전체 배율을 따릅니다")
            col_cfg["직접 증감률(%/년)"] = st.column_config.NumberColumn(
                "직접 증감률(%/년)", step=0.1, format="%.2f",
                help="입력하면 모델 대신 '마지막 값 × (1+증감률)^t'로 계산합니다 (예: -0.5)")
        with st.expander("🔧 용도별로 다른 모델 / 기울기 적용 (선택)", expanded=False):
            st.caption("용도마다 흐름이 다르므로 필요하면 모델을 따로 지정하세요. 아래 '모델 비교'에서 용도별 결과를 먼저 볼 수 있습니다."
                       + (" 기울기 배율·직접 증감률은 비워두면 적용되지 않습니다." if adv else " 기울기는 위의 '추가 옵션'을 켜면 용도별로도 조정할 수 있습니다."))
            edited = st.data_editor(
                pd.DataFrame(seed), hide_index=True, use_container_width=True, disabled=["행", "용도"],
                column_config=col_cfg, key=f"p45_ed_{sheet}_{int(adv)}")
        row_model, row_mult, row_rate = {}, {}, {}
        for _, rw in edited.iterrows():
            r = int(rw["행"])
            row_model[r] = None if rw["모델"] == FOLLOW_GLOBAL else MODELS[rw["모델"]]
            if adv:
                if pd.notna(rw.get("기울기 배율(%)")):
                    row_mult[r] = float(rw["기울기 배율(%)"])
                if pd.notna(rw.get("직접 증감률(%/년)")):
                    row_rate[r] = float(rw["직접 증감률(%/년)"])


    tr_rows = [r for r in layout["data_rows"] if "수송" in labels[r]]

    def finalize(p, u):
        """가정용(확정 시)은 시뮬레이션 값, 수송용은 임시 유지 값으로 교체"""
        p, u = p.copy(), dict(u)
        if hold_tr:
            for r in tr_rows:
                if r in p.index:
                    p.loc[r] = float(hist.loc[r, base_end])
                    u[r] = "hold"
        if hh["apply"]:
            for r, ser in hh["rows"].items():
                if r in p.index:
                    p.loc[r] = ser.reindex(p.columns).values
                    u[r] = "hh"
        return p, u

    proj, used, notes = build_projection(layout, gm, train_start, anchor, row_model,
                                         mult_pct / 100.0, phi, row_mult, row_rate)
    proj, used = finalize(proj, used)
    adjusted = bool(adv and (mult_pct != 100 or phi != 1.0 or row_mult or row_rate))
    proj0 = finalize(*build_projection(layout, gm, train_start, anchor, row_model)[:2])[0] if adjusted else None
    if proj.empty:
        st.error("예측할 수 있는 용도 행이 없습니다.")
        return
    rows = [r for r in layout["data_rows"] if r in proj.index]
    full = pd.concat([hist.loc[rows, hy], proj.loc[rows]], axis=1)
    full.index = [labels[r] for r in rows]
    tot = full.sum()
    if other:
        warn = {r: n for r, n in notes.items() if not n.startswith("직접")}
        if warn:
            st.warning("⚠️ " + " / ".join(f"{labels[r]}: {n}" for r, n in warn.items()))

        # ── 결과 요약 ──
        st.markdown("### 결과")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{base_end}년 합계", f"{tot[base_end]:,.0f}")
        k2.metric("2040년 합계", f"{tot[2040]:,.0f}", f"{tot[2040] / tot[base_end] - 1:+.1%}")
        k3.metric(f"{TARGET_END}년 합계", f"{tot[TARGET_END]:,.0f}", f"{tot[TARGET_END] / tot[base_end] - 1:+.1%}")
        k4.metric(f"연평균 증감률 ({base_end}→{TARGET_END})",
                  f"{(tot[TARGET_END] / tot[base_end]) ** (1 / (TARGET_END - base_end)) - 1:+.2%}")

        t1, t2, t3, t4 = st.tabs(["📈 합계 추이", "🧪 모델 비교 (용도별)", "🧱 용도별 추이", "📋 연도별 상세표"])

        with t1:
            M = 1e6
            hist_tot = tot[[y for y in tot.index if 2026 <= y <= base_end]]
            sel = tot[[y for y in tot.index if y >= base_end]]
            label_years = {hist_tot.index[0], base_end, 2040, TARGET_END}
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_tot.index, y=hist_tot.values / M, mode="lines+markers+text", name="계획/실적",
                line=dict(color="gray", width=3), textposition="top center",
                text=[f"{v / M:.1f}" if y in label_years else "" for y, v in hist_tot.items()],
                hovertemplate="%{x}년: %{y:,.2f} 백만GJ<extra></extra>"))
            ymax = max(hist_tot.max(), tot.max())
            if adjusted:
                t0 = proj0.loc[rows].sum(axis=0)
                ymax = max(ymax, t0.max())
                fig.add_trace(go.Scatter(
                    x=[base_end] + list(t0.index), y=[hist_tot[base_end] / M] + list(t0.values / M), mode="lines",
                    name=f"{mlabel} (조정 전)", line=dict(color="gray", width=2, dash="dot"),
                    hovertemplate="%{x}년: %{y:,.2f} 백만GJ<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=sel.index, y=sel.values / M, mode="lines+markers+text",
                name=f"{mlabel}" + (" (기울기 조정)" if adjusted else ""),
                line=dict(color="royalblue", width=4), textposition="top center",
                text=[f"{v / M:.1f}" if y in label_years else "" for y, v in sel.items()],
                hovertemplate="%{x}년: %{y:,.2f} 백만GJ<extra></extra>"))
            fig.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="royalblue", opacity=0.07,
                          annotation_text="추세 예측", annotation_position="inside top")
            fig.update_xaxes(dtick=2, tickformat="d")
            fig.update_yaxes(title="백만 GJ", tickformat=",.0f" if y0 else ",.1f")
            if y0:
                fig.update_yaxes(range=[0, ymax / M * 1.15])
            fig.update_layout(height=440, margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
            show_chart(fig)
            st.caption("Y축을 0부터 표시합니다. 사이드바 '표시 옵션'에서 확대 보기로 바꿀 수 있습니다." if y0
                       else "⚠️ Y축을 확대해서 표시 중입니다. 작은 변화도 크게 보일 수 있습니다.")

        with t2:
            st.caption("용도를 고르면 6개 모델의 예측을 한 화면에서 비교합니다. 검증오차는 학습 구간의 마지막 3개 연도를 가리고 예측했을 때의 평균 오차율로, "
                       "값이 작을수록 최근 흐름을 잘 맞춘 모델입니다. (점이 적어 참고용이며, 계획값 자체가 추세로 작성된 구간이라 '계획과의 일치도'로 이해하세요.) 이 탭은 추가 옵션 적용 전 모델 원본 값입니다.")
            pick = st.selectbox("용도", ["합계"] + [labels[r] for r in rows], key="p45_pick")
            if pick == "합계":
                series = hist.loc[rows, hy].sum(axis=0).values.astype(float)
            else:
                rr = [r for r in rows if labels[r] == pick][0]
                series = hist.loc[rr, hy].values.astype(float)
            figm = go.Figure()
            figm.add_trace(go.Scatter(x=hy, y=series, mode="lines+markers", name="계획/실적",
                                      line=dict(color="gray", width=3)))
            recs, ymax_c = [], float(np.nanmax(series))
            for lab, key in MODELS.items():
                if pick == "합계":
                    pm, _, _ = build_projection(layout, key, train_start, anchor, {})
                    pm, _ = finalize(pm, {})
                    p = pm.loc[rows].sum(axis=0).values
                else:
                    p, _ = project(hy, series, TARGET_END - base_end, key, train_start, anchor)
                ymax_c = max(ymax_c, float(np.nanmax(p)))
                fy = list(range(base_end + 1, TARGET_END + 1))
                figm.add_trace(go.Scatter(x=[base_end] + fy, y=[series[-1]] + list(p), mode="lines",
                                          name=lab, line=dict(width=4 if key == gm else 1.8)))
                mape = backtest_mape(hy, series, key, train_start, anchor)
                recs.append({"모델": lab, "2040년": p[4], f"{TARGET_END}년": p[-1],
                             f"{TARGET_END}년 vs {base_end}년": p[-1] / series[-1] - 1 if series[-1] else np.nan,
                             "검증오차(MAPE)": mape})
            if pick != "합계" and rr in hh["rows"] and hh["apply"]:
                hv = hh["rows"][rr].reindex(range(base_end + 1, TARGET_END + 1)).values
                ymax_c = max(ymax_c, float(np.nanmax(hv)))
                figm.add_trace(go.Scatter(x=[base_end] + list(range(base_end + 1, TARGET_END + 1)), y=[series[-1]] + list(hv),
                                          mode="lines+markers", name="★ 가정용 기온 시뮬레이션 (확정)",
                                          line=dict(color="black", width=4)))
            figm.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="royalblue", opacity=0.07)
            figm.update_xaxes(dtick=2, tickformat="d")
            figm.update_layout(height=440, legend=dict(orientation="h", y=-0.25), margin=dict(t=20), yaxis_title="GJ")
            if y0:
                figm.update_yaxes(range=[0, ymax_c * 1.1])
            show_chart(figm)
            if pick == "합계":
                st.caption("합계는 가정용 확정값과 수송용 임시 유지를 반영한 값이며, 각 모델은 나머지 용도에 적용됩니다.")
            elif pick != "합계" and rr in tr_rows and hold_tr:
                st.caption("수송용은 담당부서 자료 수령 전까지 2035년 수준을 임시로 유지 중이라 최종 결과에는 아래 모델선이 쓰이지 않습니다.")
            st.dataframe(pd.DataFrame(recs).set_index("모델").style.format(
                {"2040년": "{:,.0f}", f"{TARGET_END}년": "{:,.0f}", f"{TARGET_END}년 vs {base_end}년": "{:+.1%}",
                 "검증오차(MAPE)": "{:.1%}"}, na_rep="-"), use_container_width=True)

        with t3:
            long = full.reset_index().melt(id_vars="index", var_name="연", value_name="값").rename(columns={"index": "용도"})
            long = long[long["연"] >= 2026]
            long["구분"] = np.where(long["연"] <= base_end, "계획/실적", "추세예측")
            bridge = long[long["연"] == base_end].copy()
            bridge["구분"] = "추세예측"
            long_l = pd.concat([long, bridge], ignore_index=True)
            order = {"용도": list(full.index)}
            as_idx = st.checkbox(f"{base_end}년 = 100 지수로 보기 (용도별 규모 차이가 클 때 추천)", value=True, key="p45_idx")
            if as_idx:
                basev = long[long["연"] == base_end].set_index("용도")["값"]
                long_l["표시값"] = long_l.apply(lambda x: x["값"] / basev[x["용도"]] * 100 if basev[x["용도"]] else np.nan, axis=1)
                ylab = f"{base_end}년=100"
            else:
                long_l["표시값"], ylab = long_l["값"], "GJ"
            figl = px.line(long_l, x="연", y="표시값", color="용도", line_dash="구분", markers=True,
                           category_orders=order, labels={"표시값": ylab})
            figl.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="royalblue", opacity=0.07)
            figl.update_xaxes(dtick=2, tickformat="d")
            figl.update_layout(height=480)
            show_chart(figl)
            figb = px.bar(long, x="연", y="값", color="용도", category_orders=order, title="연도별 용도 구성 (누적)")
            figb.add_vline(x=base_end + 0.5, line_dash="dash", line_color="royalblue")
            figb.update_xaxes(dtick=2, tickformat="d")
            figb.update_layout(height=460)
            show_chart(figb)

        with t4:
            table = full.copy()
            table.loc["합계"] = tot
            table.loc["전년 대비 증감"] = tot.pct_change()
            table.columns = [f"{y}년" + ("" if y <= base_end else " ▶") for y in table.columns]
            num_rows = [i for i in table.index if i != "전년 대비 증감"]
            sty = table.style.format("{:,.0f}", subset=pd.IndexSlice[num_rows, :], na_rep="")
            sty = sty.format("{:+.2%}", subset=pd.IndexSlice[["전년 대비 증감"], :], na_rep="")
            st.dataframe(sty, use_container_width=True)
            st.caption(f"▶ 표시는 {base_end}년 이후 예측 값입니다. 가정용은 기온 시뮬레이션(확정 시), 수송용은 임시 유지, 나머지는 선택한 추세 모델입니다.")


    # ── 엑셀 다운로드 ──
    st.markdown("### 엑셀 다운로드")
    notes_x = list(hh["notes"]) if hh["rows"] else []
    if hold_tr and tr_rows:
        notes_x.append(["수송용", f"{base_end}년 수준 임시 유지 (담당부서 자료 수령 후 교체 필요)"])
    extra = {"on": adjusted, "mult_pct": mult_pct, "phi": phi, "row_mult": row_mult, "row_rate": row_rate, "notes": notes_x, "overrides": _merge_over(over, conf_over),
             "conf_year": anchor_year if scope2 else 2029}
    out = write_workbook(data, sheet, layout, proj, used, gm, train_start, anchor, extra)
    st.download_button(
        "📥 2045 확장 엑셀 다운로드", data=out,
        file_name=f"중장기공급량추정_{TARGET_END}_{mlabel.split('. ')[1].split('(')[0].strip().replace(' ', '')}{'_기울기조정' if adjusted else ''}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("원본 서식·수식은 유지되고, 합계/전년대비/증감/비율 행은 수식으로 2045년까지 확장됩니다. 적용한 모델은 엑셀 끝의 '2045확장_적용내역' 시트에 기록됩니다.")


if __name__ == "__main__":
    main()
