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
from urllib.parse import quote

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
SALES_FILE_NAME = "판매량(계획_실적).xlsx"
TEMP_FILE_NAME = "기온.csv" 

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
def load_excel_from_github_force(filename):
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
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

def load_temp_universal(file_obj):
    try:
        fname = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        if fname.endswith('.csv'):
            try: df = pd.read_csv(file_obj, encoding='utf-8-sig')
            except: df = pd.read_csv(file_obj, encoding='cp949')
        else:
            df = pd.read_excel(file_obj)
            
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
# 2. [기능 1] 실적 분석
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 판매량 실적 분석 ({unit_label})")
    
    # 🔴 필터링된 long_df가 들어오므로 여기서는 별도 필터 최소화
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    if df_act.empty:
        st.warning("선택하신 기간에 해당하는 실적 데이터가 없습니다.")
        return
        
    all_years = sorted(df_act['연'].unique())
    
    # 여기서 연도 선택은 '그래프에 표시할 연도' (학습 데이터 선택과는 별개로 시각화용)
    st.markdown("##### 📅 그래프에 표시할 연도 선택")
    selected_years = st.multiselect(
        "연도 선택", 
        all_years, 
        default=all_years, # 기본적으로 필터링된 모든 연도 표시
        label_visibility="collapsed"
    )
    if not selected_years: return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
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
    st.subheader(f"🔮 2035 장기 판매량 예측 ({unit_label})")
    
    st.markdown("##### 🤖 예측 모델 선택")
    method = st.radio(
        "예측 방법",
        ["1. 선형 회귀 (Linear)", "2. 2차 곡선 (Polynomial)", "3. 연평균 성장률 (CAGR)", 
         "4. 지수 평활 (Holt's Trend)", "5. 로그 추세 (Logarithmic)"],
        index=0, horizontal=True
    )

    # 🔴 이미 메인 함수에서 필터링된 long_df가 들어옴 (사용자가 선택한 연도만 있음)
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    if df_act.empty:
        st.warning("선택된 학습 기간에 데이터가 없습니다.")
        return

    # 학습 데이터 연도 확인
    train_years = sorted(df_act['연'].unique())
    st.caption(f"ℹ️ **학습 데이터:** {train_years[0]}~{train_years[-1]}년 중 선택된 {len(train_years)}개 연도를 사용하여 예측합니다.")

    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 전체 데이터를 다 사용 (사용자가 이미 필터링했으므로)
        X = sub['연'].values
        y = sub['값'].values
        pred = []
        
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in method:
            try: 
                # 로그모델은 연도(2020 등)를 직접 쓰면 스케일 문제 발생, 인덱스(1,2,3..) 사용 추천
                X_idx = np.arange(1, len(X)+1).reshape(-1,1)
                X_fut_idx = np.arange(len(X)+1, len(X)+11).reshape(-1,1)
                model = LinearRegression(); model.fit(np.log(X_idx), y)
                pred = model.predict(np.log(X_fut_idx))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in method: pred = holt_linear_trend(y, 10)
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
    
    st.info("**💡 Insight:** 선택하신 학습 연도 데이터를 기반으로 미래 추세를 산출했습니다. (코로나 등 특정 연도 제외 가능)")
    
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

    # 여기도 필터링된 long_df 사용
    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    
    # 데이터 병합
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty: st.warning("데이터 기간이 일치하지 않습니다."); return

    # 분석 연도 선택
    years = sorted(df_merged['연'].unique())
    st.markdown("##### 📅 분석할 연도 선택")
    sel_years = st.multiselect("분석 연도", years, default=years, label_visibility="collapsed")
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
# 메인 실행 (구조 변경)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    # ── [데이터 로드 (먼저 수행하여 사이드바에 연도 정보 제공)] ──
    xls_sales = load_excel_from_github_force(SALES_FILE_NAME)
    
    # 판매량 파일이 없으면 업로더 표시를 위해 플래그 설정
    is_sales_loaded = False
    long_df = pd.DataFrame()
    unit_label = "천m³" # 기본값

    # 1. 깃허브 로드 시도
    if xls_sales:
        is_sales_loaded = True
    
    with st.sidebar:
        st.header("설정")
        main_category = st.radio("📂 분석 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        
        st.markdown("---")
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        
        st.markdown("---")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        
        st.markdown("---")
        st.caption("데이터 파일 설정")
        
        # 파일 업로더 (깃허브 실패 시 또는 사용자 업로드 시)
        uploaded_sales = None
        sales_src = st.radio("판매량 소스", ["☁️ GitHub", "📂 파일 업로드"], key="s_src", label_visibility="collapsed")
        
        if sales_src == "📂 파일 업로드":
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx")
            if uploaded_sales:
                xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
                is_sales_loaded = True
        
        # 기온 파일
        uploaded_temp = st.file_uploader("기온 데이터(.csv, .xlsx)", type=["csv", "xlsx"])

        # 🔴 [핵심 기능] 학습 기간 선택 (데이터가 로드된 경우에만 표시)
        if is_sales_loaded:
            # 1. 일단 전체 데이터 변환해서 연도 정보 가져오기
            try:
                if unit.startswith("부피"):
                    df_p, df_a = xls_sales.parse("계획_부피"), xls_sales.parse("실적_부피")
                    unit_label = "천m³"
                else:
                    df_p, df_a = xls_sales.parse("계획_열량"), xls_sales.parse("실적_열량")
                    unit_label = "GJ"
                long_df = make_long(df_p, df_a)
                
                # 2025년까지만 학습 데이터로 허용
                available_years = sorted([y for y in long_df['연'].unique() if y <= 2025])
                
                st.markdown("---")
                st.markdown("**📅 학습/분석 대상 연도 설정**")
                st.caption("체크 해제된 연도는 분석 및 예측 학습에서 제외됩니다. (예: 2021년 제외)")
                
                selected_train_years = st.multiselect(
                    "연도 선택",
                    options=available_years,
                    default=available_years, # 디폴트: 전체 (2015~2025)
                    label_visibility="collapsed"
                )
                
                # 🔴 [데이터 필터링 적용]
                if selected_train_years:
                    long_df = long_df[long_df['연'].isin(selected_train_years)]
                else:
                    st.warning("최소 1개 이상의 연도를 선택해야 합니다.")
                    long_df = pd.DataFrame() # 빈 데이터프레임

            except Exception as e:
                st.error(f"데이터 처리 중 오류: {e}")
                is_sales_loaded = False

    # ── [메인 화면 로직] ──
    if not is_sales_loaded:
        st.info("👈 좌측에서 판매량 데이터를 연결해주세요.")
        if sales_src == "☁️ GitHub": st.error(f"GitHub 연결 실패. ({SALES_FILE_NAME})")
        return

    if long_df.empty: return

    # 기온 데이터 로드
    df_temp = load_temp_universal(uploaded_temp)

    # 화면 출력
    if main_category == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
    else:
        st.warning("🚧 [공급량 예측] 서비스는 아직 준비 중입니다.")
        st.info("'1. 판매량 예측' 메뉴를 이용해주세요.")

if __name__ == "__main__":
    main()
