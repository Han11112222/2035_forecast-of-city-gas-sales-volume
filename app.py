import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import io
from urllib.parse import quote
from sklearn.linear_model import LinearRegression

# =============================================================================
# 🟢 [설정] Han형님의 깃허브 정보
# =============================================================================
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
EXCEL_FILE_NAME = "판매량(계획_실적).xlsx"

# =============================================================================
# 1. 형님의 기존 로직 (매핑 및 데이터 변환) - 건드리지 않음
# =============================================================================
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

st.set_page_config(page_title="도시가스 판매량 분석 및 예측", page_icon="🔥", layout="wide")

def _clean_base(df):
    out = df.copy()
    if "Unnamed: 0" in out.columns: out = out.drop(columns=["Unnamed: 0"])
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long(plan_df, actual_df):
    """형님의 핵심 로직: Wide -> Long 변환 및 그룹 매핑"""
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue
            group = USE_COL_TO_GROUP.get(col)
            if not group: continue
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    return long_df.dropna(subset=["연", "월"])

# =============================================================================
# 2. [수정됨] 에러 없는 데이터 로드 함수 (Requests 사용)
# =============================================================================
@st.cache_data(ttl=600)
def load_data_safe():
    """한글 파일명 URL 인코딩 + 바이너리 다운로드로 에러 원천 차단"""
    try:
        # 한글 파일명 URL 인코딩
        encoded_name = quote(EXCEL_FILE_NAME)
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{encoded_name}"
        
        # 파일 내용을 바이너리로 다운로드
        response = requests.get(url)
        response.raise_for_status() # 404 에러 시 예외 발생
        
        # 엑셀 파일로 인식
        return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
    
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return None

# =============================================================================
# 3. [신규 기능] 2035 예측 로직
# =============================================================================
def run_prediction_section(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 판매량 예측 ({unit_label})")
    
    # 실적 데이터만 추출하여 연도별/그룹별 합계
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    # 그룹별 예측 수행
    for grp in groups:
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        model = LinearRegression()
        model.fit(sub['연'].values.reshape(-1, 1), sub['값'].values)
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred] # 음수 방지
        
        # 실적 데이터
        for y, v in zip(sub['연'], sub['값']):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '실적'})
        # 예측 데이터
        for y, v in zip(future_years.flatten(), pred):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '예측'})
            
    df_res = pd.DataFrame(results)
    
    # 차트 그리기
    fig = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', 
                  markers=True, title=f"용도별 장기 전망 (~2035) [{unit_label}]")
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="Forecast")
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터 표
    piv = df_res[df_res['Type']=='예측'].pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"))

# =============================================================================
# 4. 메인 실행 (통합)
# =============================================================================
def main():
    st.title("🔥 도시가스 계획/실적 분석 및 예측")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("설정")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        mode = st.radio("기능 선택", ["1. 실적 분석 (기본)", "2. 2035 예측 (신규)"])

    # 데이터 로드
    xls = load_data_safe()
    if xls is None: return

    # 단위에 따른 시트 선택
    try:
        if unit.startswith("부피"):
            df_p = xls.parse("계획_부피")
            df_a = xls.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p = xls.parse("계획_열량")
            df_a = xls.parse("실적_열량")
            unit_label = "GJ"
            
        # 형님의 변환 함수 실행
        long_df = make_long(df_p, df_a)
        
    except ValueError as e:
        st.error(f"시트 이름 오류: 엑셀 파일에 '계획_부피', '실적_부피' 등의 시트가 있는지 확인하세요.\n에러내용: {e}")
        return

    # 기능 분기
    if mode.startswith("1"):
        # -----------------------------------------------------------
        # 기존 실적 분석 (형님 스타일의 차트)
        # -----------------------------------------------------------
        st.subheader(f"📊 판매량 실적 분석 ({unit_label})")
        
        # 연도 필터
        years = sorted(long_df['연'].unique())
        sel_year = st.selectbox("연도 선택", years, index=len(years)-1)
        
        # 필터링
        sub = long_df[long_df['연'] == sel_year]
        
        # 간단 KPI (형님 코드의 복잡한 대시보드 대신 핵심만 표시)
        p_sum = sub[sub['계획/실적']=='계획']['값'].sum()
        a_sum = sub[sub['계획/실적']=='실적']['값'].sum()
        
        c1, c2 = st.columns(2)
        c1.metric("연간 계획", f"{p_sum:,.0f}")
        c2.metric("연간 실적", f"{a_sum:,.0f}", delta=f"{a_sum-p_sum:,.0f}")
        
        # 차트 1: 월별 추이
        st.markdown("#### 📅 월별 실적 추이")
        grp = sub.groupby(['월', '계획/실적'])['값'].sum().reset_index()
        fig1 = px.line(grp, x='월', y='값', color='계획/실적', markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 차트 2: 용도별 누적
        st.markdown("#### 🧱 용도별 구성비")
        grp2 = sub[sub['계획/실적']=='실적'].groupby(['월', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(grp2, x='월', y='값', color='그룹')
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        # -----------------------------------------------------------
        # 신규 예측 기능
        # -----------------------------------------------------------
        run_prediction_section(long_df, unit_label)

if __name__ == "__main__":
    main()
