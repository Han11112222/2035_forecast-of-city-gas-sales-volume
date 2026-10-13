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

    memo_name = "2045확장_적용내역"
    if memo_name in wb.sheetnames:
        del wb[memo_name]
    memo = wb.create_sheet(memo_name)
    memo.append(["항목", "내용"])
    memo.append(["기준(마지막 계획) 연도", base_end])
    memo.append(["학습 시작 연도", train_start])
    memo.append(["전체 적용 모델", MODEL_NAME.get(global_model, global_model)])
    memo.append(["마지막 해 값에서 이어붙이기", "예" if anchor else "아니오"])
    extra = extra or {}
    memo.append(["추가 옵션(사용자 조정)", "적용" if extra.get("on") else "미적용 (모델 추정값 그대로)"])
    if extra.get("on"):
        memo.append(["기울기 배율(%)", extra.get("mult_pct", 100)])
        memo.append(["감쇠계수", extra.get("phi", 1.0)])
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
    try:
        bases = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    except NameError:
        bases = [os.getcwd()]
    for d in bases:
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
# 4. 화면
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 공급량 중장기 추정 (2036~2045)")
    st.caption("2035년까지 작성된 계획을 추세 모델로 2045년까지 연장합니다. 모든 모델은 데이터에서 계수를 추정하며, 임의로 정한 감쇠·고정값은 쓰지 않습니다. (단위: GJ)")

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
        src_name = REPO_PLAN_FILE
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
        st.warning(f"이미 {base_end}년까지 값이 있습니다.")
        return
    hist = layout["hist"]
    hy = [y for y in hist.columns if y <= base_end]
    labels = layout["labels"]
    st.caption(f"📄 {src_name} · {base_end}년까지의 계획을 {TARGET_END}년까지 연장")

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
            help="이 연도부터 {}년까지의 값으로 모델을 학습합니다. 기본 2029년 = 추세로 작성된 구간(2026~2028 확정계획의 설비 증설 등 단절 구간 제외)".format(base_end))
        anchor = st.checkbox(f"{base_end}년 값에서 이어붙이기", value=True, key="p45_anchor",
                             help="모델선과 마지막 해 값의 차이만큼 평행이동해서 2036년에 값이 갑자기 튀지 않게 합니다.")
        st.caption(f"학습 구간: {train_start}~{base_end}년 ({base_end - train_start + 1}개 점)")
        st.markdown("---")
        st.header("🖼️ 표시 옵션")
        y0 = st.checkbox("Y축을 0부터 표시", value=True, key="p45_y0",
                         help="체크하면 변동 비율이 실제 크기 그대로 보입니다. 해제하면 축을 확대해 작은 변화도 크게 보입니다.")

    st.markdown(f"#### 선택한 모델: {mlabel}")
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

    proj, used, notes = build_projection(layout, gm, train_start, anchor, row_model,
                                         mult_pct / 100.0, phi, row_mult, row_rate)
    adjusted = bool(adv and (mult_pct != 100 or phi != 1.0 or row_mult or row_rate))
    proj0, _, _ = build_projection(layout, gm, train_start, anchor, row_model) if adjusted else (None, None, None)
    if proj.empty:
        st.error("예측할 수 있는 용도 행이 없습니다.")
        return
    rows = [r for r in layout["data_rows"] if r in proj.index]
    full = pd.concat([hist.loc[rows, hy], proj.loc[rows]], axis=1)
    full.index = [labels[r] for r in rows]
    tot = full.sum()
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
        hist_tot = tot[[y for y in tot.index if y <= base_end]]
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
        fig.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="green", opacity=0.05,
                      annotation_text="추세 예측", annotation_position="inside top")
        fig.update_xaxes(dtick=2, tickformat="d")
        fig.update_yaxes(title="백만 GJ", tickformat=",.0f" if y0 else ",.1f")
        if y0:
            fig.update_yaxes(range=[0, ymax / M * 1.15])
        fig.update_layout(height=440, margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
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
        figm.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="green", opacity=0.05)
        figm.update_xaxes(dtick=2, tickformat="d")
        figm.update_layout(height=440, legend=dict(orientation="h", y=-0.25), margin=dict(t=20), yaxis_title="GJ")
        if y0:
            figm.update_yaxes(range=[0, ymax_c * 1.1])
        st.plotly_chart(figm, use_container_width=True)
        st.dataframe(pd.DataFrame(recs).set_index("모델").style.format(
            {"2040년": "{:,.0f}", f"{TARGET_END}년": "{:,.0f}", f"{TARGET_END}년 vs {base_end}년": "{:+.1%}",
             "검증오차(MAPE)": "{:.1%}"}, na_rep="-"), use_container_width=True)

    with t3:
        long = full.reset_index().melt(id_vars="index", var_name="연", value_name="값").rename(columns={"index": "용도"})
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
        figl.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="green", opacity=0.05)
        figl.update_xaxes(dtick=2, tickformat="d")
        figl.update_layout(height=480)
        st.plotly_chart(figl, use_container_width=True)
        figb = px.bar(long, x="연", y="값", color="용도", category_orders=order, title="연도별 용도 구성 (누적)")
        figb.add_vline(x=base_end + 0.5, line_dash="dash", line_color="green")
        figb.update_xaxes(dtick=2, tickformat="d")
        figb.update_layout(height=460)
        st.plotly_chart(figb, use_container_width=True)

    with t4:
        table = full.copy()
        table.loc["합계"] = tot
        table.loc["전년 대비 증감"] = tot.pct_change()
        table.columns = [f"{y}년" + ("" if y <= base_end else " ▶") for y in table.columns]
        num_rows = [i for i in table.index if i != "전년 대비 증감"]
        sty = table.style.format("{:,.0f}", subset=pd.IndexSlice[num_rows, :], na_rep="")
        sty = sty.format("{:+.2%}", subset=pd.IndexSlice[["전년 대비 증감"], :], na_rep="")
        st.dataframe(sty, use_container_width=True)
        st.caption(f"▶ 표시는 {base_end}년 이후 추세 예측 값입니다.")

    # ── 엑셀 다운로드 ──
    st.markdown("### 엑셀 다운로드")
    extra = {"on": adjusted, "mult_pct": mult_pct, "phi": phi, "row_mult": row_mult, "row_rate": row_rate}
    out = write_workbook(data, sheet, layout, proj, used, gm, train_start, anchor, extra)
    st.download_button(
        "📥 2045 확장 엑셀 다운로드", data=out,
        file_name=f"중장기공급량추정_{TARGET_END}_{mlabel.split('. ')[1].split('(')[0].strip().replace(' ', '')}{'_기울기조정' if adjusted else ''}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("원본 서식·수식은 유지되고, 합계/전년대비/증감/비율 행은 수식으로 2045년까지 확장됩니다. 적용한 모델은 엑셀 끝의 '2045확장_적용내역' 시트에 기록됩니다.")


if __name__ == "__main__":
    main()
