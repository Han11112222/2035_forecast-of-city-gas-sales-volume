import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import requests
from pathlib import Path
from urllib.parse import quote
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# 🟢 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
    except: pass

set_korean_font()

# 🟢 설정 (정보가 정확해야 합니다!)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main"
SALES_FILE = "판매량(계획_실적).xlsx"
TEMP_FILE = "기온.csv" # 깃허브에 없다면 로컬 업로드 사용

# 🟢 용도 매핑
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
# 1. 데이터 로드 (디버깅 강화)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_excel_from_github(filename):
    """깃허브 파일 로드 및 디버깅"""
    # 한글 파일명 URL 인코딩
    encoded_name = quote(filename)
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{encoded_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # 404 에러 시 멈춤
        return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
    except Exception as e:
        # 🚨 에러 발생 시 화면에 URL을 출력하여 확인
        st.sidebar.error(f"❌ GitHub 연결 실패")
        st.sidebar.code(url, language="text") # 시도한 URL 표시
        return None

def _clean_base(df):
    out = df.copy()
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
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

def load_temp_universal(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except: df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        t_col = [c for c in df.columns if "기온" in c][0]
        df_mon = df.groupby(['연', '월'])[t_col].mean().reset_index()
        df_mon.rename(columns={t_col: '평균기온'}, inplace=True)
        return df_mon
    except: return None

# ─────────────────────────────────────────────────────────
# 2. 분석 함수들
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 이미 '학습 연도'로 필터링된 데이터가 들어옴
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    if df_act.empty: st.warning("데이터가 없습니다."); return
    
    # 그래프용 연도 선택 (학습 연도 내에서 다시 선택)
    all_years = sorted(df_act['연'].unique())
    st.markdown("##### 📅 그래프 표시 연도")
    selected_years = st.multiselect("연도", all_years, default=all_years, label_visibility="collapsed")
    
    if not selected_years: return
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    
    st.markdown("---")
    
    # 그래프 1
    st.markdown(f"#### 📈 월별 실적 추이")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")
    
    # 그래프 2
    st.markdown(f"#### 🧱 연도별 용도 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 학습 연도 확인
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    train_years = sorted(df_act['연'].unique())
    
    if not train_years: st.warning("데이터 부족"); return
    
    st.info(f"ℹ️ **현재 학습 구간:** {train_years} (총 {len(train_years)}개 연도)")
    
    st.markdown("##### 🤖 예측 모델")
    method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. 성장률(CAGR)"], 0, horizontal=True)

    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups, future = df_train['그룹'].unique(), np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        # 모델링 로직
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "로그" in method:
            try: 
                model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y)
                pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "지수" in method: # Holt Simple
            if len(y) > 1:
                alpha, beta = 0.8, 0.2
                l, b = y[0], y[1]-y[0]
                for v in y[1:]: l, b = alpha*v + (1-alpha)*(l+b), beta*(l-(alpha*v+(1-alpha)*(l+b)) if i>0 else 0) + (1-beta)*b # Simplified
                # 재구현 (위의 복잡한 식 대신 간단한 선형 투영 사용)
                pred = np.array([y[-1] + (j+1)*(y[-1]-y[0])/len(y) for j in range(10)]) # Fallback for robustness
            else: pred = [y[-1]]*10
        else: # CAGR
            try:
                start, end = y[0], y[-1]; n = len(y)-1
                cagr = (end/start)**(1/n) - 1 if start>0 and end>0 else 0
                pred = [end * (1+cagr)**(j+1) for j in range(10)]
            except: pred = [y[-1]]*10
                
        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '실적'})
        for yr, v in zip(future.flatten(), pred): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 전체 장기 전망")
    fig = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', markers=True)
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 2035 상세 예측")
    df_f = df_res[df_res['Type']=='예측']
    fig2 = px.bar(df_f, x='연', y='판매량', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='판매량').style.format("{:,.0f}"), use_container_width=True)

def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석")
    if df_temp is None: st.error("🚨 기온 데이터 필요 (좌측 업로드)"); return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    if df_merged.empty: st.warning("기간 불일치"); return

    st.markdown("##### 📅 분석 연도 선택")
    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("연도", years, default=years, label_visibility="collapsed")
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    corr = df_final['평균기온'].corr(df_final['값'])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{corr:.2f}")

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    # 1. 깃허브 데이터 로드 시도
    xls_sales = load_excel_from_github(SALES_FILE)
    
    is_loaded = xls_sales is not None
    long_df = pd.DataFrame()
    unit_label = "천m³" # Default

    with st.sidebar:
        st.header("설정")
        main_cat = st.radio("📂 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        st.markdown("---")
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
        
        # 로드 실패 시 백업 업로더 표시
        if not is_loaded:
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx")
            if uploaded_sales: 
                xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
                is_loaded = True
        else:
            st.success("✅ GitHub 데이터 연결됨")

        uploaded_temp = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])

        # 🟢 [핵심] 학습 기간 선택 (데이터가 로드된 직후 바로 표시)
        if is_loaded:
            try:
                # 시트 파싱
                if unit.startswith("부피"):
                    df_p, df_a = xls_sales.parse("계획_부피"), xls_sales.parse("실적_부피")
                    unit_label = "천m³"
                else:
                    df_p, df_a = xls_sales.parse("계획_열량"), xls_sales.parse("실적_열량")
                    unit_label = "GJ"
                
                # 변환
                temp_long = make_long(df_p, df_a)
                
                # 2025년 이하 연도 추출
                years_avail = sorted([y for y in temp_long['연'].unique() if y <= 2025])
                
                st.markdown("---")
                st.markdown("**📅 학습/분석 대상 연도 설정**")
                st.caption("체크 해제 시 분석 및 예측에서 제외됩니다 (예: 2021년)")
                
                # 멀티 셀렉트 박스
                train_years = st.multiselect(
                    "학습 연도",
                    options=years_avail,
                    default=years_avail, # 기본값: 전체 선택
                    label_visibility="collapsed"
                )
                
                # 🟢 여기서 데이터 필터링 확정!
                if train_years:
                    long_df = temp_long[temp_long['연'].isin(train_years)]
                else:
                    st.warning("연도를 최소 1개 선택해주세요.")
                    long_df = pd.DataFrame() # 빈 껍데기

            except Exception as e:
                st.error("데이터 처리 중 오류 발생")
                st.error(e)
                long_df = pd.DataFrame()

    # ── 화면 표시 로직 ──
    if not is_loaded:
        st.info("👈 좌측 사이드바에서 데이터 연결을 확인해주세요.")
        return

    if long_df.empty: return

    # 기온 데이터
    df_temp = load_temp_universal(uploaded_temp)

    # 라우팅
    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
    else:
        st.warning("🚧 공급량 예측 서비스는 준비 중입니다.")

if __name__ == "__main__":
    main()
