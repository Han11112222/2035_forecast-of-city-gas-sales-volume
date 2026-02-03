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
    ttf = Path(__file__).parent / "NanumGothic-Regular.ttf"
    if ttf.exists():
        try:
            import matplotlib as mpl
            mpl.font_manager.fontManager.addfont(str(ttf))
            mpl.rcParams["font.family"] = "NanumGothic"
            mpl.rcParams["axes.unicode_minus"] = False
        except: pass

set_korean_font()

# 🟢 설정 정보 (깃허브)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
SALES_FILE_NAME = "판매량(계획_실적).xlsx"
TEMP_FILE_NAME = "기온_198001_202512.xlsx" # 또는 기온.csv

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
# 1. 데이터 로드 및 전처리 (깃허브 디폴트 적용)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_github_file(filename, file_type='xlsx'):
    """깃허브에서 파일을 강제로 로드 (Default)"""
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if file_type == 'xlsx':
            return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
        elif file_type == 'csv':
            # CSV 인코딩 처리
            try:
                return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            except:
                return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except Exception as e:
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

# 기온 데이터 전처리
def preprocess_temp(df):
    if df is None: return None
    # 날짜 컬럼 통일
    if '일자' in df.columns: df.rename(columns={'일자': '날짜'}, inplace=True)
    if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
    
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    
    # 기온 컬럼 찾기
    temp_col = [c for c in df.columns if "기온" in c]
    if not temp_col: return None
    target = temp_col[0]
    
    # 월평균
    monthly = df.groupby(['연', '월'])[target].mean().reset_index()
    monthly.rename(columns={target: '평균기온'}, inplace=True)
    return monthly

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    if df_act.empty: st.warning("데이터가 없습니다."); return
    
    all_years = sorted(df_act['연'].unique())
    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 그래프에 표시할 연도 선택")
    selected_years = st.multiselect("연도 선택", all_years, default=default_years, label_visibility="collapsed")
    if not selected_years: return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 표 1
    st.markdown("##### 📋 상세 데이터")
    piv_mon = df_mon.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(piv_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 표 2
    st.markdown("##### 📋 상세 데이터")
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측
# ─────────────────────────────────────────────────────────
def holt_linear_trend(y, n_preds):
    if len(y) < 2: return np.full(n_preds, y[0])
    alpha, beta = 0.8, 0.2
    level, trend = y[0], y[1] - y[0]
    for val in y[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + i * trend for i in range(1, n_preds + 1)])

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    st.markdown("##### 🤖 예측 모델 선택")
    method = st.radio("방법", ["1. 선형 (Linear)", "2. 2차 곡선 (Poly)", "3. 로그 (Log)", "4. 지수 평활 (Holt)", "5. 성장률 (CAGR)"], 0, horizontal=True)

    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    if df_act.empty: st.warning("학습 데이터가 없습니다."); return

    train_years = sorted(df_act['연'].unique())
    st.caption(f"ℹ️ 학습 데이터: {train_years[0]}~{train_years[-1]}년 (선택된 연도만 반영)")

    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        X = sub['연'].values
        y = sub['값'].values
        pred = []
        
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in method:
            try: model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y); pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in method: pred = holt_linear_trend(y, 10)
        else:
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(10)]
            except: pred = [y[-1]]*10
                
        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 전체 장기 전망")
    fig = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', markers=True)
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**💡 Insight:** 선택하신 학습 연도를 기반으로 미래 추세를 예측했습니다.")
    
    st.markdown("#### 🧱 2035 미래 예측 상세")
    df_f = df_res[df_res['Type']=='예측']
    fig2 = px.bar(df_f, x='연', y='판매량', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    piv = df_f.pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("다운로드", piv.to_csv().encode('utf-8-sig'), "forecast.csv")

# ─────────────────────────────────────────────────────────
# 4. [기능 3] 가정용 정밀 분석
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None:
        st.error("🚨 기온 데이터가 없습니다. (GitHub 로드 실패 또는 파일 없음)"); return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty: st.warning("데이터 기간 불일치"); return

    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도", years, default=years, label_visibility="collapsed")
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    corr = df_final['평균기온'].corr(df_final['값'])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{corr:.2f}")
        st.caption("-1에 가까울수록 반비례")

    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_final['기간'], y=df_final['값'], name="판매량", yaxis='y'))
    fig2.add_trace(go.Scatter(x=df_final['기간'], y=df_final['평균기온'], name="기온", line=dict(color='red'), yaxis='y2'))
    fig2.update_layout(yaxis=dict(title="판매량"), yaxis2=dict(title="기온", overlaying='y', side='right'))
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 메인 실행 (구조 개편)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    # 1. 깃허브 데이터 자동 로드 (Default)
    xls_sales = load_github_file(SALES_FILE_NAME, 'xlsx')
    
    # 기온 데이터 (xlsx 우선 시도, 없으면 csv 시도)
    xls_temp = load_github_file(TEMP_FILE_NAME, 'xlsx') # 엑셀 시도
    if xls_temp:
        df_temp = preprocess_temp(xls_temp.parse(0))
    else:
        # 실패시 CSV 시도 (파일명이 csv일 경우 대비)
        df_temp_csv = load_github_file("기온.csv", 'csv') 
        df_temp = preprocess_temp(df_temp_csv) if df_temp_csv is not None else None

    # 데이터 로드 상태 플래그
    is_loaded = xls_sales is not None
    long_df = pd.DataFrame()
    unit_label = "천m³"

    with st.sidebar:
        st.header("설정")
        
        # 🟢 [대분류]
        main_cat = st.radio("📂 분석 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        
        # 🟢 [소분류]
        sub_modes = ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"]
        sub_mode = st.radio("기능 선택", sub_modes)
        
        st.markdown("---")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        
        # 로드 성공 여부 표시
        if is_loaded:
            st.success(f"✅ GitHub 데이터 로드됨")
        else:
            st.warning("⚠️ GitHub 로드 실패. 파일 업로드 필요")
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx")
            if uploaded_sales: 
                xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
                is_loaded = True
            
            uploaded_temp = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])
            if uploaded_temp:
                # 업로드 로직 (간략화)
                if uploaded_temp.name.endswith('.csv'):
                    try: df_temp = preprocess_temp(pd.read_csv(uploaded_temp, encoding='cp949'))
                    except: df_temp = preprocess_temp(pd.read_csv(uploaded_temp))
                else:
                    df_temp = preprocess_temp(pd.ExcelFile(uploaded_temp, engine='openpyxl').parse(0))

        # 🟢 [학습 기간 선택] - 데이터가 있을 때만
        if is_loaded:
            try:
                if unit.startswith("부피"):
                    df_p, df_a = xls_sales.parse("계획_부피"), xls_sales.parse("실적_부피")
                    unit_label = "천m³"
                else:
                    df_p, df_a = xls_sales.parse("계획_열량"), xls_sales.parse("실적_열량")
                    unit_label = "GJ"
                long_df = make_long(df_p, df_a)
                
                # 2025년까지만 학습 데이터로
                years_avail = sorted([y for y in long_df['연'].unique() if y <= 2025])
                
                st.markdown("---")
                st.markdown("**📅 학습 대상 연도 설정**")
                train_years = st.multiselect(
                    "연도 선택", years_avail, default=years_avail, label_visibility="collapsed"
                )
                
                if train_years:
                    long_df = long_df[long_df['연'].isin(train_years)]
                else:
                    st.warning("최소 1개 연도 필요"); long_df = pd.DataFrame()

            except Exception as e:
                st.error(f"데이터 처리 오류: {e}")
                long_df = pd.DataFrame()

    # ── 메인 화면 ──
    if not is_loaded or long_df.empty:
        if not is_loaded: st.info("👈 좌측에서 데이터를 확인해주세요.")
        return

    # 라우팅
    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
    else:
        # 공급량 예측 (형님 요청: 아직 시작 전)
        st.header("🚧 공급량 예측")
        st.warning("공급량 예측 서비스는 준비 중입니다.")
        st.info("현재 '1. 판매량 예측' 메뉴만 이용 가능합니다.")

if __name__ == "__main__":
    main()
