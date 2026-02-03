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
# 🟢 기본 설정 & 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') # 윈도우용 기본 폰트
    except: pass

set_korean_font()

# 🟢 깃허브 설정 (틀리면 로드 안됨)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" 
SALES_FILE = "판매량(계획_실적).xlsx"
TEMP_FILE = "기온.csv"

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
# 1. 데이터 로드 및 전처리
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_github_file(filename, file_type='xlsx'):
    """깃허브 파일 로드 (실패 시 None 반환하여 업로더 유도)"""
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if file_type == 'xlsx':
            return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
        else: # csv
            try: return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            except: return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except:
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

def load_temp_universal(uploaded_file=None):
    # 1. 깃허브 우선 시도
    if uploaded_file is None:
        return load_github_file(TEMP_FILE, 'csv')
    
    # 2. 업로드 파일 처리
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except: df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except: return None

def preprocess_temp(df):
    if df is None: return None
    if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    
    # 기온 컬럼 찾기
    temp_col = [c for c in df.columns if "기온" in c]
    if not temp_col: return None
    
    monthly = df.groupby(['연', '월'])[temp_col[0]].mean().reset_index()
    monthly.rename(columns={temp_col[0]: '평균기온'}, inplace=True)
    return monthly

# ─────────────────────────────────────────────────────────
# 2. 분석 함수들
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 필터링된 데이터 사용
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    if df_act.empty: st.warning("선택된 기간에 데이터가 없습니다."); return
    
    all_years = sorted(df_act['연'].unique())
    st.markdown("##### 📅 그래프 표시 연도")
    selected_years = st.multiselect("연도 선택", all_years, default=all_years, key="viz_years", label_visibility="collapsed")
    
    if not selected_years: return
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    
    st.markdown("---")
    # 그래프 1
    st.markdown("#### 📈 월별 실적 추이")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")
    # 그래프 2
    st.markdown("#### 🧱 연도별 용도 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df_yr.pivot(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    train_years = sorted(df_act['연'].unique())
    if not train_years: st.warning("학습 데이터 부족"); return
    
    st.info(f"ℹ️ **학습 구간:** {train_years} (좌측 '학습 연도' 탭에서 제외 가능)")
    
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
        
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "로그" in method:
            try: model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y); pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "지수" in method:
             # 간단 지수 평활
             pred = np.array([y[-1] + (j+1)*(y[-1]-y[0])/len(y) for j in range(10)])
        else: # CAGR
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(10)]
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
    if df_temp is None: st.error("🚨 기온 데이터 없음. 좌측에서 업로드해주세요."); return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_temp_proc = preprocess_temp(df_temp)
    if df_temp_proc is None: st.error("기온 데이터 처리 실패"); return

    df_merged = pd.merge(df_home, df_temp_proc, on=['연', '월'], how='inner')
    if df_merged.empty: st.warning("기간 불일치"); return

    st.markdown("##### 📅 분석 연도")
    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("연도", years, default=years, key="house_years", label_visibility="collapsed")
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{df_final['평균기온'].corr(df_final['값']):.2f}")

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    # 1. 깃허브 로드 시도
    xls_sales = load_github_file(SALES_FILE, 'xlsx')
    
    is_loaded = xls_sales is not None
    long_df = pd.DataFrame()
    unit_label = "천m³"

    # 사이드바
    with st.sidebar:
        st.header("설정")
        main_cat = st.radio("📂 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        st.markdown("---")
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
        
        # 🟢 데이터 로드 (깃허브 실패 시 업로드 버튼 표시)
        if is_loaded:
            st.success("✅ GitHub 로드됨")
            uploaded_sales = None
        else:
            st.error("❌ GitHub 실패 (파일 필요)")
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx")
            if uploaded_sales: 
                xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
                is_loaded = True
        
        uploaded_temp = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])

        # 🟢 [학습 기간 선택] - 데이터가 로드되어야만 표시됨
        if is_loaded:
            try:
                if unit.startswith("부피"):
                    df_p, df_a = xls_sales.parse("계획_부피"), xls_sales.parse("실적_부피")
                    unit_label = "천m³"
                else:
                    df_p, df_a = xls_sales.parse("계획_열량"), xls_sales.parse("실적_열량")
                    unit_label = "GJ"
                
                temp_long = make_long(df_p, df_a)
                
                # 2015~2025만 학습용
                years_avail = sorted([y for y in temp_long['연'].unique() if y <= 2025])
                
                st.markdown("---")
                st.markdown("**📅 학습 대상 연도 (왜곡 제외)**")
                st.caption("체크 해제된 연도는 분석에서 빠집니다.")
                
                # 디폴트: 전체 선택
                train_years = st.multiselect(
                    "학습 연도", 
                    options=years_avail, 
                    default=years_avail,
                    label_visibility="collapsed"
                )
                
                # 데이터 필터링 적용!
                if train_years:
                    long_df = temp_long[temp_long['연'].isin(train_years)]
                else:
                    st.warning("최소 1개 연도 필요")
                    long_df = pd.DataFrame()

            except Exception as e:
                st.error("데이터 읽기 오류")
                long_df = pd.DataFrame()

    # ── 화면 표시 ──
    if not is_loaded:
        st.info("👈 좌측에서 판매량 파일을 업로드해주세요.")
        return
        
    if long_df.empty: return

    # 기온 데이터 로드 (업로드 or 깃허브)
    df_temp = load_temp_universal(uploaded_temp)

    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
    else:
        st.header("🚧 공급량 예측")
        st.warning("준비 중입니다.")

if __name__ == "__main__":
    main()
