import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# 🟢 기본 설정 & 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

def set_korean_font():
    ttf = Path(__file__).parent / "NanumGothic-Regular.ttf"
    if ttf.exists():
        try:
            import matplotlib as mpl
            mpl.font_manager.fontManager.addfont(str(ttf))
            mpl.rcParams["font.family"] = "NanumGothic"
            mpl.rcParams["axes.unicode_minus"] = False
        except: pass

set_korean_font()

# 🟢 파일명 설정 (형님 파일명)
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"

# 🟢 용도 매핑 (형님 코드 그대로)
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 및 전처리 (복잡한 거 다 뺌)
# ─────────────────────────────────────────────────────────
def _clean_base(df):
    out = df.copy()
    if "Unnamed: 0" in out.columns: out = out.drop(columns=["Unnamed: 0"])
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long(plan_df, actual_df):
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue
            group = USE_COL_TO_GROUP.get(col)
            if not group: continue # 매핑 안된 컬럼은 패스
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    return long_df.dropna(subset=["연", "월"])

def load_data_simple(uploaded_file=None):
    """파일이 있으면 읽고, 업로드되면 업로드 파일 읽음"""
    try:
        # 1. 업로드된 파일이 있으면 우선 사용
        if uploaded_file:
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        
        # 2. 없으면 로컬(같은 폴더) 파일 찾기
        elif Path(DEFAULT_SALES_XLSX).exists():
            return pd.ExcelFile(DEFAULT_SALES_XLSX, engine='openpyxl')
        
        return None
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 기존 실적 분석 (형님 로직 단순화)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 연도 필터
    years = sorted(long_df['연'].unique())
    if not years: return
    
    c1, c2 = st.columns([1, 2])
    with c1: sel_year = st.selectbox("기준 연도", years, index=len(years)-1)
    
    # 데이터 필터
    df_sub = long_df[long_df['연'] == sel_year]
    
    # KPI
    plan_sum = df_sub[df_sub['계획/실적']=='계획']['값'].sum()
    act_sum = df_sub[df_sub['계획/실적']=='실적']['값'].sum()
    
    k1, k2 = st.columns(2)
    k1.metric("연간 계획", f"{plan_sum:,.0f} {unit_label}")
    k2.metric("연간 실적", f"{act_sum:,.0f} {unit_label}", delta=f"{act_sum-plan_sum:,.0f}")
    
    st.markdown("---")
    
    # 월별 추이 그래프
    st.markdown("#### 📅 월별 추이")
    grp = df_sub.groupby(['월', '계획/실적'])['값'].sum().reset_index()
    fig1 = px.line(grp, x='월', y='값', color='계획/실적', markers=True)
    fig1.update_xaxes(dtick=1)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 용도별 구성비
    st.markdown("#### 🧱 용도별 실적")
    grp_use = df_sub[df_sub['계획/실적']=='실적'].groupby(['월', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(grp_use, x='월', y='값', color='그룹')
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측 (신규 추가)
# ─────────────────────────────────────────────────────────
def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    st.caption("과거 실적 데이터를 기반으로 선형 회귀(Linear Regression) 예측")
    
    # 실적 데이터만 사용
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    # 연도별/그룹별 합계
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 예측 모델링
        model = LinearRegression()
        model.fit(sub['연'].values.reshape(-1, 1), sub['값'].values)
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred] # 음수 제거
        
        # 저장
        for y, v in zip(sub['연'], sub['값']):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '실적'})
        for y, v in zip(future_years.flatten(), pred):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '예측'})
            
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    # 그래프
    fig = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', markers=True,
                  title=f"2035년까지의 장기 전망 ({unit_label})")
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="Forecast")
    st.plotly_chart(fig, use_container_width=True)
    
    # 다운로드
    piv = df_res[df_res['Type']=='예측'].pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"))
    st.download_button("예측 데이터 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv")

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    with st.sidebar:
        st.header("설정")
        # 데이터 소스: 그냥 파일 있나 보고 없으면 업로드
        uploaded = None
        if not Path(DEFAULT_SALES_XLSX).exists():
            st.warning(f"⚠️ '{DEFAULT_SALES_XLSX}' 파일이 없습니다.")
            uploaded = st.file_uploader("엑셀 파일 업로드", type="xlsx")
        else:
            st.success(f"✅ '{DEFAULT_SALES_XLSX}' 파일 연결됨")
            if st.checkbox("다른 파일 업로드하기"):
                uploaded = st.file_uploader("엑셀 파일 업로드", type="xlsx")

        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 실적 분석", "2. 2035 예측"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    # 데이터 로드
    xls = load_data_simple(uploaded)
    if xls is None: return

    # 시트 읽기 및 변환
    try:
        if unit.startswith("부피"):
            df_p = xls.parse("계획_부피")
            df_a = xls.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p = xls.parse("계획_열량")
            df_a = xls.parse("실적_열량")
            unit_label = "GJ"
            
        long_df = make_long(df_p, df_a)
        
    except Exception as e:
        st.error(f"시트 로드 실패: {e}")
        return

    # 기능 실행
    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    else:
        render_prediction_2035(long_df, unit_label)

if __name__ == "__main__":
    main()
