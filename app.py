import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import requests
from sklearn.linear_model import LinearRegression
from urllib.parse import quote

# =============================================================================
# 🟢 [설정] Han형님의 깃허브 정보
# =============================================================================
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
EXCEL_FILE_NAME = "판매량(계획_실적).xlsx"

# 🟢 [매핑] 형님 코드의 분류 기준 그대로 적용
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

GROUP_OPTIONS = ["총량", "가정용", "영업용", "업무용", "산업용", "수송용", "열병합", "연료전지", "열전용설비용"]
COLOR_PLAN, COLOR_ACT, COLOR_PREV = "rgba(0, 90, 200, 1)", "rgba(0, 150, 255, 1)", "rgba(190, 190, 190, 1)"

st.set_page_config(page_title="도시가스 판매량 분석 및 예측", page_icon="🔥", layout="wide")

# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 (형님 로직 + GitHub)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_excel_bytes_from_github():
    """깃허브에서 엑셀 파일을 바이너리로 가져옴 (가장 안전한 방법)"""
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(EXCEL_FILE_NAME)}"
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        return None

def make_long(plan_df, actual_df):
    """형님 코드: Wide -> Long 변환 및 그룹 매핑"""
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

def _clean_base(df):
    out = df.copy()
    if "Unnamed: 0" in out.columns: out = out.drop(columns=["Unnamed: 0"])
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def load_all_data(excel_bytes):
    xls = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    data_dict = {}
    
    # 부피 데이터 처리
    if "계획_부피" in xls.sheet_names and "실적_부피" in xls.sheet_names:
        data_dict["부피"] = make_long(xls.parse("계획_부피"), xls.parse("실적_부피"))
        
    # 열량 데이터 처리
    if "계획_열량" in xls.sheet_names and "실적_열량" in xls.sheet_names:
        data_dict["열량"] = make_long(xls.parse("계획_열량"), xls.parse("실적_열량"))
        
    return data_dict

# -----------------------------------------------------------------------------
# 2. [기능 1] 형님의 실적 분석 기능 (핵심만 이식)
# -----------------------------------------------------------------------------
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 대시보드 ({unit_label})")
    
    # 1. 필터
    years = sorted(long_df['연'].unique())
    if not years: return
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: sel_year = st.selectbox("기준 연도", years, index=len(years)-1)
    with c2: sel_month = st.selectbox("기준 월", range(1, 13), index=11)
    
    # 데이터 필터링 (연누적 기준)
    df_this = long_df[(long_df['연'] == sel_year) & (long_df['월'] <= sel_month)]
    df_prev = long_df[(long_df['연'] == sel_year - 1) & (long_df['월'] <= sel_month)]
    
    # KPI 계산
    plan_sum = df_this[df_this['계획/실적']=='계획']['값'].sum()
    act_sum = df_this[df_this['계획/실적']=='실적']['값'].sum()
    prev_act_sum = df_prev[df_prev['계획/실적']=='실적']['값'].sum()
    
    # KPI 카드
    k1, k2, k3 = st.columns(3)
    k1.metric(f"계획 ({sel_month}월 누적)", f"{plan_sum:,.0f}")
    k2.metric(f"실적 ({sel_month}월 누적)", f"{act_sum:,.0f}", delta=f"{act_sum-plan_sum:,.0f} (계획대비)")
    k3.metric(f"전년 실적 ({sel_month}월 누적)", f"{prev_act_sum:,.0f}", delta=f"{act_sum-prev_act_sum:,.0f} (전년대비)")
    
    st.markdown("---")
    
    # 차트: 월별 추이 (형님 스타일)
    st.markdown("#### 📈 월별 추이 비교")
    grp_df = long_df.groupby(['연', '월', '계획/실적'])['값'].sum().reset_index()
    # 최근 3년만 표시
    recent_years = years[-3:] 
    grp_df = grp_df[grp_df['연'].isin(recent_years)]
    
    fig = px.line(grp_df, x='월', y='값', color='연', line_dash='계획/실적', markers=True)
    fig.update_layout(xaxis=dict(dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig, use_container_width=True)
    
    # 차트: 용도별 누적 (Stacked Bar)
    st.markdown("#### 🧱 용도별 구성비")
    stack_df = df_this[df_this['계획/실적']=='실적'].groupby(['월', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(stack_df, x='월', y='값', color='그룹', title=f"{sel_year}년 용도별 실적")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. [기능 2] 2035 예측 기능 (형님 데이터 구조 활용)
# -----------------------------------------------------------------------------
def render_prediction(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 판매량 예측 ({unit_label})")
    st.info("💡 과거 실적 데이터를 기반으로 용도별 선형 추세를 분석하여 예측합니다.")
    
    # 실적 데이터만 추출
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    # 연도별/그룹별 합계
    df_train_base = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train_base['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train_base[df_train_base['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 모델링
        model = LinearRegression()
        model.fit(sub['연'].values.reshape(-1, 1), sub['값'].values)
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred]
        
        # 결과 수집
        for y, v in zip(sub['연'], sub['값']):
            results.append({'Year': y, '그룹': grp, '판매량': v, 'Type': '실적'})
        for y, v in zip(future_years.flatten(), pred):
            results.append({'Year': y, '그룹': grp, '판매량': v, 'Type': '예측'})
        
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    # 예측 차트
    fig = px.line(df_res, x='Year', y='판매량', color='그룹', line_dash='Type', markers=True)
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측 구간")
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터 표
    piv = df_res[df_res['Type']=='예측'].pivot_table(index='Year', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"))
    st.download_button("예측 결과 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv")

# -----------------------------------------------------------------------------
# 4. 메인 실행
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 판매량 분석 및 예측")
    
    # 사이드바
    with st.sidebar:
        st.header("설정")
        src = st.radio("데이터 소스", ["GitHub (기본)", "파일 업로드"])
        
        excel_bytes = None
        if src == "파일 업로드":
            up = st.file_uploader("엑셀 파일", type="xlsx")
            if up: excel_bytes = up.getvalue()
        else:
            excel_bytes = load_excel_bytes_from_github()
            if excel_bytes is None:
                st.error("GitHub 데이터 로드 실패. 아이디/파일명 확인 필요.")
        
        st.markdown("---")
        mode = st.radio("메뉴", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    if excel_bytes is None:
        st.info("데이터를 불러오는 중이거나 파일이 없습니다.")
        return

    # 데이터 로딩 (형님 로직 적용)
    data_dict = load_all_data(excel_bytes)
    
    target_key = "부피" if unit.startswith("부피") else "열량"
    unit_label = "천m³" if unit.startswith("부피") else "GJ"
    
    if target_key not in data_dict:
        st.error(f"'{target_key}' 데이터를 찾을 수 없습니다. 시트명을 확인하세요.")
        return
        
    df_long = data_dict[target_key]

    # 기능 분기
    if mode.startswith("1"):
        render_analysis_dashboard(df_long, unit_label)
    else:
        render_prediction(df_long, unit_label)

if __name__ == "__main__":
    main()
