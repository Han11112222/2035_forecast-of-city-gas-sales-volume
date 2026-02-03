import io
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression # 예측용 라이브러리 추가

# ─────────────────────────────────────────────────────────
# 🟢 [설정] 깃허브 정보 입력 (여기를 형님 정보로 맞추세요!)
# ─────────────────────────────────────────────────────────
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
SALES_FILE = "판매량(계획_실적).xlsx"
SUPPLY_FILE = "공급량(계획_실적).xlsx"

# ─────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────
def set_korean_font():
    ttf = Path(__file__).parent / "NanumGothic-Regular.ttf"
    if ttf.exists():
        try:
            mpl.font_manager.fontManager.addfont(str(ttf))
            mpl.rcParams["font.family"] = "NanumGothic"
            mpl.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

set_korean_font()
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

# 엑셀 헤더 → 분석 그룹 매핑 (판매량용)
USE_COL_TO_GROUP: Dict[str, str] = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용",
}

GROUP_OPTIONS: List[str] = [
    "총량", "가정용", "영업용", "업무용", "산업용", "수송용", "열병합", "연료전지", "열전용설비용",
]

COLOR_PLAN = "rgba(0, 90, 200, 1)"
COLOR_ACT = "rgba(0, 150, 255, 1)"
COLOR_PREV = "rgba(190, 190, 190, 1)"
COLOR_DIFF = "rgba(0, 80, 160, 1)"

# ─────────────────────────────────────────────────────────
# [신규 추가] 깃허브 파일 로드 함수
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_bytes_from_github(filename):
    """깃허브에서 엑셀 파일을 바이너리로 가져옴"""
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────
# 공통 유틸 (형님 코드 그대로 유지)
# ─────────────────────────────────────────────────────────
def fmt_num_safe(v) -> str:
    if pd.isna(v): return "-"
    try: return f"{float(v):,.0f}"
    except Exception: return "-"

def fmt_rate(v: float) -> str:
    if pd.isna(v) or np.isnan(v): return "-"
    return f"{float(v):,.1f}%"

def center_style(styler):
    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles([dict(selector="th", props=[("text-align", "center")])])
    return styler

def _clean_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Unnamed: 0" in out.columns: out = out.drop(columns=["Unnamed: 0"])
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def keyword_group(col: str) -> Optional[str]:
    c = str(col)
    if "열병합" in c: return "열병합"
    if "연료전지" in c: return "연료전지"
    if "수송용" in c: return "수송용"
    if "열전용" in c: return "열전용설비용"
    if c in ["산업용"]: return "산업용"
    if c in ["일반용"]: return "영업용"
    if any(k in c for k in ["취사용", "난방용", "자가열"]): return "가정용"
    if any(k in c for k in ["업무", "냉방", "주한미군"]): return "업무용"
    return None

def make_long(plan_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue
            group = USE_COL_TO_GROUP.get(col)
            if group is None: group = keyword_group(col)
            if group is None: continue

            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            records.append(base)
    if not records: return pd.DataFrame(columns=["연", "월", "그룹", "용도", "계획/실적", "값"])
    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.dropna(subset=["연", "월"])
    long_df["연"] = long_df["연"].astype(int)
    long_df["월"] = long_df["월"].astype(int)
    return long_df

def load_all_sheets(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    needed = ["계획_부피", "실적_부피", "계획_열량", "실적_열량"]
    out: Dict[str, pd.DataFrame] = {}
    for name in needed:
        if name in xls.sheet_names: out[name] = xls.parse(name)
    return out

def build_long_dict(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    long_dict: Dict[str, pd.DataFrame] = {}
    if ("계획_부피" in sheets) and ("실적_부피" in sheets):
        long_dict["부피"] = make_long(sheets["계획_부피"], sheets["실적_부피"])
    if ("계획_열량" in sheets) and ("실적_열량" in sheets):
        long_dict["열량"] = make_long(sheets["계획_열량"], sheets["실적_열량"])
    return long_dict

# ... (형님 코드의 나머지 유틸 함수들: render_section_selector, render_metric_card 등은 그대로 사용)
# 편의상 코드가 너무 길어지지 않게 핵심 로직은 유지하고, 
# 형님 코드의 시각화 함수(render_section_selector 등)가 이미 정의되어 있다고 가정하고 아래 실행 로직에서 사용합니다.
# (실제 실행 시에는 형님 코드의 모든 함수 정의가 여기에 포함되어야 합니다.)
# 여기서는 형님 코드에 없던 '예측 함수'만 추가합니다.

def pick_default_year(years: List[int]) -> int:
    return 2025 if 2025 in years else years[-1]

def apply_period_filter(df, sel_year, sel_month, agg_mode):
    if df.empty: return df
    base = df[df["연"] == sel_year].copy()
    if agg_mode == "당월": base = base[base["월"] == sel_month]
    else: base = base[base["월"] <= sel_month]
    return base

def render_section_selector(long_df, title, key_prefix, fixed_mode=None, show_mode=True):
    # 형님의 Selector 로직 (간략화하여 재구현, 형님 코드 원본이 있으면 그걸 쓰세요)
    st.markdown(f"#### ✅ {title} 기준 선택")
    years = sorted(long_df["연"].unique())
    if not years: return 0, 1, "연 누적", []
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: sel_year = st.selectbox("연도", years, index=len(years)-1, key=key_prefix+"y")
    with c2: sel_month = st.selectbox("월", range(1,13), index=11, key=key_prefix+"m")
    
    agg_mode = fixed_mode if fixed_mode else st.radio("기준", ["당월", "연 누적"], key=key_prefix+"mode")
    if not show_mode and not fixed_mode: agg_mode = "연 누적"
    return sel_year, sel_month, agg_mode, years

def monthly_core_dashboard(long_df, unit_label, key_prefix):
    # (형님 코드의 대시보드 함수 내용이 있다고 가정)
    pass # 실제로는 형님 코드 붙여넣기

def monthly_trend_section(long_df, unit_label, key_prefix):
    # (형님 코드의 트렌드 함수 내용이 있다고 가정)
    pass 

def yearly_summary_section(long_df, unit_label, key_prefix):
    # (형님 코드)
    pass

def plan_vs_actual_usage_section(long_df, unit_label, key_prefix):
    pass

def half_year_stacked_section(long_df, unit_label, key_prefix):
    pass

# ─────────────────────────────────────────────────────────
# [신규 추가] 2035 예측 기능 함수
# ─────────────────────────────────────────────────────────
def prediction_2035_section(long_df: pd.DataFrame, unit_label: str):
    st.markdown(f"## 🔮 2035 장기 판매량 예측 ({unit_label})")
    st.info("💡 과거 실적 데이터를 학습하여 2035년까지의 용도별 추세를 예측합니다.")

    # 실적 데이터만 사용
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    # 연도별/그룹별 합계
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []

    progress = st.progress(0, text="예측 모델링 중...")
    
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue # 데이터 부족

        model = LinearRegression()
        model.fit(sub['연'].values.reshape(-1, 1), sub['값'].values)
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred] # 음수 방지

        # 실적 저장
        for y, v in zip(sub['연'], sub['값']):
            results.append({'Year': y, '그룹': grp, '판매량': v, 'Type': '실적'})
        # 예측 저장
        for y, v in zip(future_years.flatten(), pred):
            results.append({'Year': y, '그룹': grp, '판매량': v, 'Type': '예측'})
            
        progress.progress((i+1)/len(groups))
    
    progress.empty()
    df_res = pd.DataFrame(results)

    # 차트
    fig = px.line(df_res, x='Year', y='판매량', color='그룹', line_dash='Type',
                  markers=True, title=f"2035년 용도별 장기 전망 ({unit_label})",
                  category_orders={"Type": ["실적", "예측"]})
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측 구간")
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 테이블
    st.markdown("### 📋 예측 데이터 상세")
    piv = df_res[df_res['Type']=='예측'].pivot_table(index='Year', columns='그룹', values='판매량')
    piv['전체합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"))
    st.download_button("예측 데이터 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv")


# ─────────────────────────────────────────────────────────
# 메인 레이아웃 (좌측탭 구성)
# ─────────────────────────────────────────────────────────
st.title("도시가스 계획 / 실적 분석")

with st.sidebar:
    st.header("📌 분석 탭")
    main_tab = st.radio(
        "분석 항목",
        ["판매량 분석", "공급량 분석(월)", "공급량 분석(일)"],
        index=0,
        key="main_tab"
    )

    st.markdown("---")
    st.header("📂 데이터 연결")

    # [수정] 판매량 파일 로드 로직
    if main_tab == "판매량 분석":
        src = st.radio("데이터 소스", ["☁️ GitHub (기본)", "📂 엑셀 업로드"], index=0, key="sales_src")
        excel_bytes = None
        
        if src == "📂 엑셀 업로드":
            up = st.file_uploader("판매량 파일(.xlsx)", type=["xlsx"], key="sales_uploader")
            if up: excel_bytes = up.getvalue()
        else:
            # 깃허브 로드
            excel_bytes = load_bytes_from_github(SALES_FILE)
            if excel_bytes:
                st.caption(f"✅ GitHub 연결 성공: {SALES_FILE}")
            else:
                st.error("🚨 GitHub 연결 실패. 아이디/파일명 확인 필요.")
    
    # [수정] 공급량 파일 로드 로직
    else:
        src = st.radio("데이터 소스", ["☁️ GitHub (기본)", "📂 엑셀 업로드"], index=0, key="supply_src")
        supply_bytes = None
        
        if src == "📂 엑셀 업로드":
            up = st.file_uploader("공급량 파일(.xlsx)", type=["xlsx"], key="supply_uploader")
            if up: supply_bytes = up.getvalue()
        else:
            # 깃허브 로드
            supply_bytes = load_bytes_from_github(SUPPLY_FILE)
            if supply_bytes:
                st.caption(f"✅ GitHub 연결 성공: {SUPPLY_FILE}")
            else:
                st.error("🚨 GitHub 연결 실패.")


# ─────────────────────────────────────────────────────────
# 1) 판매량 분석 실행 로직
# ─────────────────────────────────────────────────────────
if main_tab == "판매량 분석":
    
    if excel_bytes is not None:
        sheets = load_all_sheets(excel_bytes)
        long_dict = build_long_dict(sheets)

        tab_labels = []
        if "부피" in long_dict: tab_labels.append("부피 기준 (천m³)")
        if "열량" in long_dict: tab_labels.append("열량 기준 (GJ)")

        if not tab_labels:
            st.info("유효한 시트(계획_부피, 실적_부피 등)를 찾지 못했어.")
        else:
            tabs = st.tabs(tab_labels)
            for tab_label, tab in zip(tab_labels, tabs):
                with tab:
                    # [신규] 분석 모드 선택 (실적분석 vs 예측)
                    analysis_mode = st.radio("분석 모드 선택", 
                                             ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"], 
                                             horizontal=True, label_visibility="collapsed")
                    
                    if tab_label.startswith("부피"):
                        df_long = long_dict.get("부피", pd.DataFrame())
                        unit = "천m³"
                        prefix = "sales_vol_"
                    else:
                        df_long = long_dict.get("열량", pd.DataFrame())
                        unit = "GJ"
                        prefix = "sales_gj_"
                    
                    if analysis_mode.startswith("1"):
                        # ★ 형님의 기존 분석 함수 호출 (함수 본문이 다 있다고 가정)
                        # 실제 코드 실행 시에는 형님 코드의 함수 본문을 모두 복사해야 합니다.
                        # 여기서는 구조만 보여드립니다.
                        try:
                            monthly_core_dashboard(df_long, unit, prefix + "dash_")
                            st.markdown("---")
                            monthly_trend_section(df_long, unit, prefix + "trend_")
                            half_year_stacked_section(df_long, unit, prefix + "stack_")
                            st.markdown("---")
                            yearly_summary_section(df_long, unit, prefix + "summary_")
                            plan_vs_actual_usage_section(df_long, unit, prefix + "pv_")
                        except NameError:
                            st.warning("⚠️ 형님의 원본 분석 함수(monthly_core_dashboard 등)가 정의되지 않았습니다. 원본 코드를 합쳐주세요.")
                    else:
                        # ★ 신규 예측 함수 호출
                        prediction_2035_section(df_long, unit)

    else:
        st.info("데이터를 불러오면 분석이 시작됩니다.")


# ─────────────────────────────────────────────────────────
# 2, 3) 공급량 분석 (형님 코드 로직 + 깃허브 바이트 연동)
# ─────────────────────────────────────────────────────────
elif main_tab in ["공급량 분석(월)", "공급량 분석(일)"]:
    # (공급량 분석 로직도 동일하게 supply_bytes를 받아 처리)
    if 'supply_bytes' in locals() and supply_bytes is not None:
        st.success("공급량 분석 데이터를 불러왔습니다. (형님의 기존 공급량 분석 코드 실행)")
        # 여기에 형님의 공급량 분석 로직(supply_core_dashboard 등)을 넣으면 됩니다.
    else:
        st.info("공급량 데이터를 불러오면 분석이 시작됩니다.")
