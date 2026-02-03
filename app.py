import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import requests
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

# 🟢 설정 정보
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_TEMP_XLSX = "기온_198001_202512.xlsx" # 기온 파일 (로컬/깃허브용)

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
def _clean_base(df):
    out = df.copy()
    # Unnamed 컬럼 제거
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

def load_data_simple(uploaded_file=None):
    try:
        if uploaded_file:
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        elif Path(DEFAULT_SALES_XLSX).exists():
            return pd.ExcelFile(DEFAULT_SALES_XLSX, engine='openpyxl')
        return None
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# [기온 데이터 로드 추가] 가정용 정밀 분석을 위해 필요
def load_temp_universal(uploaded_file=None):
    try:
        # 1. 업로드된 파일 우선
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except: df = pd.read_csv(uploaded_file, encoding='cp949')
            else:
                df = pd.ExcelFile(uploaded_file, engine='openpyxl').parse(0)
        # 2. 로컬 파일 확인
        elif Path("기온.csv").exists():
            try: df = pd.read_csv("기온.csv", encoding='utf-8-sig')
            except: df = pd.read_csv("기온.csv", encoding='cp949')
        elif Path(DEFAULT_TEMP_XLSX).exists():
            df = pd.ExcelFile(DEFAULT_TEMP_XLSX, engine='openpyxl').parse(0)
        else:
            return None

        # 전처리
        if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        
        temp_col = [c for c in df.columns if "기온" in c][0]
        df_mon = df.groupby(['연', '월'])[temp_col].mean().reset_index()
        df_mon.rename(columns={temp_col: '평균기온'}, inplace=True)
        return df_mon
    except:
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_act = df_act[df_act['연'] <= 2025] 
    
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("분석할 실적 데이터가 없습니다.")
        return

    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요 (다중 선택)")
    selected_years = st.multiselect(
        "연도 선택",
        options=all_years,
        default=default_years,
        label_visibility="collapsed"
    )
    
    if not selected_years:
        st.warning("연도를 1개 이상 선택해주세요.")
        return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # [그래프 1] 월별 실적 추이
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon_compare = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    
    fig1 = px.line(
        df_mon_compare, x='월', y='값', color='연', markers=True,
        title="월별 실적 추이 비교"
    )
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### 📋 월별 상세 수치")
    pivot_mon = df_mon_compare.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(pivot_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # [그래프 2] 연도별 용도 누적
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr_usage = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    fig2 = px.bar(
        df_yr_usage, x='연', y='값', color='그룹',
        title="연도별 판매량 및 용도 구성", text_auto='.2s'
    )
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 용도별 상세 수치")
    pivot_usage = df_yr_usage.pivot(index='연', columns='그룹', values='값').fillna(0)
    pivot_usage['합계'] = pivot_usage.sum(axis=1)
    st.dataframe(pivot_usage.style.format("{:,.0f}"), use_container_width=True)

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
    pred_method = st.radio(
        "예측 방법",
        ["1. 선형 회귀 (Linear)", "2. 2차 곡선 (Polynomial)", "3. 연평균 성장률 (CAGR)", 
         "4. 지수 평활 (Holt's Trend)", "5. 로그 추세 (Logarithmic)"],
        index=0, horizontal=True
    )

    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        sub_recent = sub.tail(5)
        if len(sub_recent) < 2: sub_recent = sub
            
        X = sub_recent['연'].values.reshape(-1, 1)
        y = sub_recent['값'].values
        pred = []
        
        if "선형" in pred_method:
            model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
        elif "2차" in pred_method:
            try: coeffs = np.polyfit(X.flatten(), y, 2); pred = np.poly1d(coeffs)(future_years.flatten())
            except: model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
        elif "로그" in pred_method:
            try: 
                model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y)
                pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
        elif "지수" in pred_method: pred = holt_linear_trend(y, 10)
        else: # CAGR
            try:
                start, end = y[0], y[-1]; n = len(y)-1
                cagr = (end/start)**(1/n) - 1 if start>0 and end>0 else 0
                pred = [end * (1+cagr)**(j+1) for j in range(10)]
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
    
    st.info("**💡 Insight:** 최근 5년 추세를 반영하여 급격한 변동을 보정한 예측 결과입니다.")
    
    st.markdown("#### 🧱 2035 미래 예측 상세")
    df_f = df_res[df_res['Type']=='예측']
    fig2 = px.bar(df_f, x='연', y='판매량', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    piv = df_f.pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("💾 예측 데이터 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast.csv")

# ─────────────────────────────────────────────────────────
# 4. [기능 3] 가정용 정밀 분석
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None:
        st.error("🚨 기온 데이터가 없습니다. 좌측에서 파일을 로드해주세요.")
        return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty: st.warning("데이터 기간이 일치하지 않습니다."); return

    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도", years, default=years[-5:] if len(years)>=5 else years)
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    corr = df_final['평균기온'].corr(df_final['값'])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title=f"기온 vs 판매량")
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
# 메인 실행 (메뉴 구조 개편)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    with st.sidebar:
        st.header("설정")
        
        # 🟢 [대분류] 메뉴 선택
        main_category = st.radio("📂 분석 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        
        st.markdown("---")
        
        # 🟢 [소분류] 상세 기능 선택
        if main_category == "1. 판매량 예측":
            # 형님 요청: 모든 기능을 1번에 포함
            sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        else:
            # 2. 공급량 예측 (준비 중 표시)
            sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        
        st.markdown("---")
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        
        # 파일 업로드
        st.markdown("---")
        st.caption("데이터 파일")
        uploaded_sales = None
        if not Path(DEFAULT_SALES_XLSX).exists():
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx")
        
        uploaded_temp = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])

    # 🟢 데이터 로드 (판매량)
    xls_sales = load_data_simple(uploaded_sales)
    if xls_sales is None:
        st.info("👈 판매량 데이터가 없습니다. 파일을 폴더에 넣거나 업로드해주세요.")
        return

    # 🟢 데이터 로드 (기온)
    df_temp = load_temp_universal(uploaded_temp)

    # 🟢 데이터 변환
    try:
        if unit.startswith("부피"):
            df_p = xls_sales.parse("계획_부피")
            df_a = xls_sales.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p = xls_sales.parse("계획_열량")
            df_a = xls_sales.parse("실적_열량")
            unit_label = "GJ"
        long_df = make_long(df_p, df_a)
    except Exception as e:
        st.error(f"데이터 처리 오류: {e}")
        return

    # 🟢 [화면 라우팅] 대분류/소분류에 따른 화면 표시
    if main_category == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
            
    else: # 2. 공급량 예측
        st.warning("🚧 [공급량 예측] 서비스는 아직 준비 중입니다.")
        st.info("현재는 '1. 판매량 예측' 메뉴만 활성화되어 있습니다.")

if __name__ == "__main__":
    main()
