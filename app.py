# -*- coding: utf-8 -*-
"""
중장기 공급량 추정 : 2036 ~ 2045 빈칸 자동 채움 (Streamlit)

- 업로드한 '중장기 공급량 추정' 엑셀의 표 구조(연도 헤더 행, 용도 행, 합계 행)를 자동 인식
- 5가지 방법 중 선택 (+ 용도별로 방법 개별 지정 가능)
- 원본 서식/수식 유지한 채 2036~2045 열을 채워 엑셀로 다운로드

사용법
  1) 단독 실행 :  streamlit run plan_extension_2045.py
  2) 기존 앱에 추가 :  from plan_extension_2045 import render_plan_extension_2045
                       (모드 선택에서 호출)  render_plan_extension_2045()
"""
import io
import re
from copy import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl import load_workbook
from openpyxl.formula.translate import Translator
from openpyxl.utils import get_column_letter

TARGET_END = 2045
YEAR_RE = re.compile(r"^\s*(\d{4})\s*년?\s*$")
TOTAL_LABELS = {"합계", "총합계", "총계"}

METHODS = {
    "1. 선형 추세": "linear",
    "2. 감쇠 추세 (추천)": "damped",
    "3. 연평균 성장률(CAGR)": "cagr",
    "4. 보합 (마지막 해 유지)": "hold",
    "5. 용도별 직접 입력 증감률": "custom",
}
METHOD_DESC = {
    "linear": "최근 N년의 연간 증감량(기울기)이 2045년까지 그대로 이어진다고 가정합니다. 가장 단순하지만 장기로 갈수록 값이 크게 벌어집니다.",
    "damped": "최근 N년의 기울기로 시작하되 매년 감쇠계수만큼 기울기가 줄어듭니다. 급격한 증가/감소가 점차 안정화되는 시나리오입니다.",
    "cagr": "최근 N년의 연평균 성장률(%)이 매년 유지된다고 가정합니다. 감소 용도는 감소폭이 점점 작아지고 0에 수렴합니다.",
    "hold": "마지막 해(2035년) 값을 2045년까지 유지합니다. 연료전지처럼 설비용량이 고정된 용도에 적합합니다.",
    "custom": "용도별로 직접 입력한 연 증감률(%)을 적용합니다. 정책·시장 전망 등 외부 가정을 반영할 때 사용합니다.",
}
FOLLOW_GLOBAL = "(전체 방법 따름)"


# ─────────────────────────────────────────────────────────
# 1. 엑셀 구조 인식
# ─────────────────────────────────────────────────────────
def _norm(s):
    return re.sub(r"\s+", "", str(s)) if s is not None else ""


def _find_header(ws):
    """연도(2025년 등)가 5개 이상 가로로 나열된 행을 헤더로 인식"""
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
    """시트에서 헤더행/연도열/용도행/합계행을 찾는다. 실패하면 None"""
    h = _find_header(ws)
    if not h:
        return None
    header_row, ycols = h
    first_col = min(ycols.values())

    label_col = None
    for c in range(first_col - 1, 0, -1):
        n = sum(
            isinstance(ws.cell(r, c).value, str) and ws.cell(r, c).value.strip() != ""
            for r in range(header_row + 1, header_row + 16)
        )
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
    hist = pd.DataFrame(
        {r: [num(r, ycols[y]) for y in years] for r in data_rows}, index=years
    ).T  # 행: 엑셀 행번호, 열: 연도

    filled_years = [y for y in years if hist[y].notna().any()]
    if not filled_years:
        return None
    base_end = max(filled_years)

    return dict(
        header_row=header_row, ycols=ycols, label_col=label_col, labels=labels,
        data_rows=data_rows, total_row=total_row, hist=hist, base_end=base_end,
    )


# ─────────────────────────────────────────────────────────
# 2. 예측 로직 (5가지)
# ─────────────────────────────────────────────────────────
def project(years, vals, n, method, window=7, phi=0.9, rate_pct=0.0):
    """마지막 실적 값을 기준(anchor)으로 n년치를 예측. 음수는 0으로 처리"""
    years = np.asarray(years, float)
    vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    if ok.sum() == 0:
        return None
    y_ok, v_ok = years[ok], vals[ok]
    last = v_ok[-1]
    k = np.arange(1, n + 1)

    if method == "custom":
        out = last * (1 + rate_pct / 100.0) ** k
        return np.maximum(0, out)
    if method == "hold" or len(v_ok) < 2:
        return np.full(n, last)

    yw, vw = y_ok[-window:], v_ok[-window:]
    if len(vw) < 2:
        return np.full(n, last)
    slope = np.polyfit(yw, vw, 1)[0]

    if method == "linear":
        out = last + slope * k
    elif method == "damped":
        out, v, st_ = [], last, slope
        for _ in range(n):
            st_ *= phi
            v += st_
            out.append(v)
        out = np.array(out)
    elif method == "cagr":
        span = yw[-1] - yw[0]
        r = (vw[-1] / vw[0]) ** (1 / span) - 1 if vw[0] > 0 and vw[-1] > 0 and span > 0 else 0.0
        out = last * (1 + r) ** k
    else:
        out = np.full(n, last)
    return np.maximum(0, out)


def window_cagr_pct(years, vals, window=7):
    """직접입력 기본값으로 쓸 최근 N년 CAGR(%)"""
    ok = ~np.isnan(vals)
    y, v = np.asarray(years, float)[ok][-window:], np.asarray(vals, float)[ok][-window:]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0 or y[-1] == y[0]:
        return 0.0
    return round(((v[-1] / v[0]) ** (1 / (y[-1] - y[0])) - 1) * 100, 2)


def build_projection(layout, global_method, window, phi, row_cfg):
    """
    row_cfg: {엑셀행번호: (override_method_key or None, rate_pct)}
    반환: (예측 DataFrame[행번호 x 연도], 적용방법 dict)
    """
    hist = layout["hist"]
    base_end = layout["base_end"]
    hy = [y for y in hist.columns if y <= base_end]
    fut = list(range(base_end + 1, TARGET_END + 1))
    n = len(fut)
    rows, used = {}, {}
    for r in layout["data_rows"]:
        vals = hist.loc[r, hy].values.astype(float)
        ov, rate = row_cfg.get(r, (None, 0.0))
        m = ov or global_method
        p = project(hy, vals, n, m, window, phi, rate)
        if p is not None:
            rows[r] = p
            used[r] = m
    return pd.DataFrame(rows, index=fut).T, used


# ─────────────────────────────────────────────────────────
# 3. 엑셀 쓰기 (원본 서식/수식 유지)
# ─────────────────────────────────────────────────────────
def write_workbook(file_bytes, sheet_name, layout, proj, used, global_method, window, phi, row_cfg):
    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb[sheet_name]
    ycols = dict(layout["ycols"])
    base_end = layout["base_end"]
    src_col = ycols[base_end]
    hr, tr = layout["header_row"], layout["total_row"]
    fut = list(proj.columns)

    # (1) 필요한 연도 열 확보 (없으면 마지막 열 오른쪽에 추가)
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
        L, SL = get_column_letter(ycols[y]), get_column_letter(src_col)
        w = ws.column_dimensions[SL].width
        if w:
            ws.column_dimensions[L].width = w

    def style_from_src(cell, src):
        if not cell.has_style:
            cell._style = copy(src._style)
        cell.number_format = src.number_format

    # (2) 용도별 예측값
    for r in layout["data_rows"]:
        if r not in proj.index:
            continue
        src = ws.cell(r, src_col)
        for y in fut:
            c = ws.cell(r, ycols[y])
            c.value = int(round(float(proj.loc[r, y])))
            style_from_src(c, src)

    # (3) 합계 행
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

    # (4) 합계 아래의 수식 행(전년대비 증감, 열량/부피 증감 등) 확장
    for r in range(tr + 1, ws.max_row + 1):
        src = ws.cell(r, src_col)
        if isinstance(src.value, str) and src.value.startswith("="):
            for y in fut:
                c = ws.cell(r, ycols[y])
                c.value = Translator(src.value, origin=src.coordinate).translate_formula(c.coordinate)
                style_from_src(c, src)

    # (5) 적용 내역 시트
    memo_name = "2045확장_적용내역"
    if memo_name in wb.sheetnames:
        del wb[memo_name]
    memo = wb.create_sheet(memo_name)
    inv = {v: k for k, v in METHODS.items()}
    memo.append(["항목", "내용"])
    memo.append(["기준(마지막 계획) 연도", base_end])
    memo.append(["전체 적용 방법", inv.get(global_method, global_method)])
    memo.append(["추세 계산 기간(최근 N년)", window])
    memo.append(["감쇠계수", phi])
    memo.append([])
    memo.append(["용도", "적용 방법", "직접입력 증감률(%/년)"])
    for r in layout["data_rows"]:
        if r in used:
            rate = row_cfg.get(r, (None, 0.0))[1] if used[r] == "custom" else ""
            memo.append([layout["labels"][r], inv.get(used[r], used[r]), rate])
    memo.column_dimensions["A"].width = 28
    memo.column_dimensions["B"].width = 32
    memo.column_dimensions["C"].width = 22

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


# ─────────────────────────────────────────────────────────
# 4. Streamlit 화면
# ─────────────────────────────────────────────────────────
def render_plan_extension_2045():
    st.subheader("🗓️ 중장기 공급량 2036~2045 빈칸 채우기")
    st.caption("2035년까지 작성된 계획표를 올리면, 선택한 방법으로 2045년까지 채운 엑셀을 내려받을 수 있습니다.")

    up = st.file_uploader("중장기 공급량 추정 엑셀 (.xlsx)", type=["xlsx"], key="plan2045_up")
    if up is None:
        st.info("👆 엑셀 파일을 업로드하세요.")
        return
    data = up.getvalue()

    try:
        wb_f = load_workbook(io.BytesIO(data))
        wb_v = load_workbook(io.BytesIO(data), data_only=True)
    except Exception as e:
        st.error(f"엑셀을 열 수 없습니다: {e}")
        return

    cands = {}
    for name in wb_f.sheetnames:
        lay = parse_layout(wb_f[name], wb_v[name])
        if lay:
            cands[name] = lay
    if not cands:
        st.error("연도 헤더(예: 2025년)와 '합계' 행이 있는 표를 찾지 못했습니다. 시트 구조를 확인해주세요.")
        return

    sheet = st.selectbox("작업할 시트", list(cands), index=0)
    layout = cands[sheet]
    base_end = layout["base_end"]
    if base_end >= TARGET_END:
        st.warning(f"이미 {base_end}년까지 값이 있습니다.")
        return
    st.success(
        f"인식 결과: 용도 {len(layout['data_rows'])}개 행 · 실적/계획 {base_end}년까지 → "
        f"{base_end + 1}~{TARGET_END}년({TARGET_END - base_end}년) 채움"
    )

    # ── 방법 선택 ──
    st.markdown("##### 1) 채움 방법")
    mlabel = st.radio("방법", list(METHODS), index=1, horizontal=True, key="plan2045_m")
    gm = METHODS[mlabel]
    st.info("ℹ️ " + METHOD_DESC[gm])

    c1, c2 = st.columns(2)
    with c1:
        n_hist = len([y for y in layout["hist"].columns if y <= base_end])
        window = st.slider("추세 계산 기간 (최근 N년)", 3, min(15, n_hist), min(7, n_hist),
                           help="선형·감쇠·CAGR에서 최근 몇 년의 흐름을 볼지 정합니다. 기본 7년 = 2029~2035")
    with c2:
        phi = st.slider("감쇠계수 (감쇠 추세용)", 0.5, 1.0, 0.9, 0.05,
                        help="1.0이면 선형과 동일, 낮을수록 빨리 안정화")

    # ── 용도별 개별 설정 ──
    hy = [y for y in layout["hist"].columns if y <= base_end]
    seed = []
    for r in layout["data_rows"]:
        lab = layout["labels"][r]
        vals = layout["hist"].loc[r, hy].values.astype(float)
        seed.append({
            "행": r, "용도": lab,
            "방법": "4. 보합 (마지막 해 유지)" if "연료전지" in lab else FOLLOW_GLOBAL,
            "증감률(%/년)": window_cagr_pct(hy, vals, 7),
        })
    with st.expander("🔧 용도별로 방법/증감률 개별 지정 (선택)", expanded=False):
        st.caption("기본값: 연료전지는 설비용량 고정으로 보고 '보합'. 증감률은 5번(직접입력)일 때만 사용되며 초기값은 최근 7년 CAGR입니다.")
        edited = st.data_editor(
            pd.DataFrame(seed),
            hide_index=True, use_container_width=True, disabled=["행", "용도"],
            column_config={
                "방법": st.column_config.SelectboxColumn("방법", options=[FOLLOW_GLOBAL] + list(METHODS)),
                "증감률(%/년)": st.column_config.NumberColumn("증감률(%/년)", step=0.1, format="%.2f"),
                "행": st.column_config.NumberColumn("엑셀 행", width="small"),
            },
            key=f"plan2045_ed_{sheet}",
        )
    row_cfg = {}
    for _, rw in edited.iterrows():
        ov = None if rw["방법"] == FOLLOW_GLOBAL else METHODS[rw["방법"]]
        row_cfg[int(rw["행"])] = (ov, float(rw["증감률(%/년)"] or 0.0))

    proj, used = build_projection(layout, gm, window, phi, row_cfg)
    if proj.empty:
        st.error("예측할 수 있는 용도 행이 없습니다.")
        return

    # ── 방법별 합계 비교 ──
    st.markdown("##### 2) 방법별 합계 비교")
    hist_tot = layout["hist"][hy].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hy, y=hist_tot.values, mode="lines+markers", name="계획/실적",
                             line=dict(color="gray", width=3)))
    comp = {}
    for lab, key in METHODS.items():
        cfg_c = {r: (None, v[1]) for r, v in row_cfg.items()}
        p_c, _ = build_projection(layout, key, window, phi, cfg_c)
        tot = p_c.sum()
        comp[lab] = tot
        sel = key == gm
        fig.add_trace(go.Scatter(
            x=[base_end] + list(tot.index), y=[hist_tot[base_end]] + list(tot.values),
            mode="lines", name=lab, line=dict(width=4 if sel else 1.5, dash="solid" if sel else "dot"),
        ))
    sel_tot = proj.sum()
    fig.add_trace(go.Scatter(x=[base_end] + list(sel_tot.index), y=[hist_tot[base_end]] + list(sel_tot.values),
                             mode="lines+markers", name="★ 현재 적용(용도별 설정 포함)",
                             line=dict(color="black", width=3)))
    fig.update_xaxes(dtick=2, tickformat="d")
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("점선/실선은 '전체 방법'을 모든 용도에 적용했을 때의 합계이고, ★ 검은 선이 용도별 개별 설정까지 반영한 실제 다운로드 값입니다.")

    cmp_df = pd.DataFrame({k: [v.get(2040, np.nan), v.get(TARGET_END, np.nan)] for k, v in comp.items()},
                          index=["2040년", f"{TARGET_END}년"]).T
    cmp_df[f"{TARGET_END}년 vs {base_end}년"] = cmp_df[f"{TARGET_END}년"] / hist_tot[base_end] - 1
    st.dataframe(cmp_df.style.format({"2040년": "{:,.0f}", f"{TARGET_END}년": "{:,.0f}",
                                       f"{TARGET_END}년 vs {base_end}년": "{:+.1%}"}), use_container_width=True)

    # ── 미리보기 ──
    st.markdown("##### 3) 채워질 값 미리보기")
    prev_years = [y for y in hy if y > base_end - 3]
    table = pd.concat([layout["hist"].loc[:, prev_years], proj], axis=1)
    table.index = [layout["labels"][r] for r in table.index]
    table.loc["합계"] = table.sum()
    table.columns = [f"{y}년" for y in table.columns]
    st.dataframe(table.style.format("{:,.0f}"), use_container_width=True)
    yoy = (sel_tot / pd.concat([pd.Series({base_end: hist_tot[base_end]}), sel_tot]).shift(1).iloc[1:] - 1)
    st.caption("합계 전년 대비: " + " · ".join(f"{y} {v:+.2%}" for y, v in yoy.items()))

    # ── 다운로드 ──
    st.markdown("##### 4) 엑셀 다운로드")
    out = write_workbook(data, sheet, layout, proj, used, gm, window, phi, row_cfg)
    st.download_button(
        "📥 2045 확장 엑셀 다운로드", data=out,
        file_name=f"중장기공급량추정_{TARGET_END}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("원본 서식·수식은 유지되고, 합계/전년대비/증감 행은 수식으로 확장됩니다. "
               "다른 시트를 참조하는 수식 행(예: 부피 증감)은 엑셀에서 열어 참조 범위를 한 번 확인해주세요.")


if __name__ == "__main__":
    try:
        st.set_page_config(page_title="2045 계획 확장", layout="wide")
    except Exception:
        pass
    st.title("🔥 중장기 공급량 2045 확장")
    render_plan_extension_2045()
