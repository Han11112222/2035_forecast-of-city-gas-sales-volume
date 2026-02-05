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
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 깃허브 설정 (파일 정확히 매칭)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" 
SALES_FILE = "판매량(계획_실적).xlsx" # 판매량용
PLAN_FILE = "사업계획최종.xlsx"       # 공급량용 (신규 추가!)
TEMP_FILE = "기온.csv"

# 🟢 용도 매핑
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용", "개별난방": "가정용", "중앙난방": "가정용",
    "일반용": "영업용", "영업용_일반용1": "영업용", "영업용_일반용2": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용", 
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", "업무용_냉난방": "업무용", "업무용_주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용", "CNG": "수송용", "BIO": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_github_file(filename, file_type='xlsx'):
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if file_type == 'xlsx':
            return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
        else:
            try: return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            except: return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except: return None

def _clean_base(df):
    out = df.copy()
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
    if '연' in out.columns: out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    if '월' in out.columns: out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

# [판매량용] 계획 vs 실적 변환
def make_long_sales(plan_df, actual_df):
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            clean_col = col.strip()
            group = USE_COL_TO_GROUP.get(clean_col)
            if not group: continue
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = clean_col
            base["구분"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

# [공급량용] 단일 시트 변환 (사업계획최종)
def make_long_supply(df, label="확정계획"):
    df = _clean_base(df)
    records = []
    
    for col in df.columns:
        clean_col = col.strip()
        group = USE_COL_TO_GROUP.get(clean_col)
        if not group: continue # 매핑 안되는 컬럼(합계 등) 제외
        
        base = df[["연", "월"]].copy()
        base["그룹"] = group
        base["용도"] = clean_col
        base["구분"] = label
        base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append(base)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

def load_temp_universal(uploaded_file=None):
    if uploaded_file is None:
        return load_github_file(TEMP_FILE, 'csv')
    try:
        if uploaded_file.name.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except: df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        temp_col = [c for c in df.columns if "기온" in c][0]
        monthly = df.groupby(['연', '월'])[temp_col].mean().reset_index()
        monthly.rename(columns={temp_col: '평균기온'}, inplace=True)
        return monthly
    except: return None

# ─────────────────────────────────────────────────────────
# 2. 공통 분석 대시보드 (판매량/공급량 공유)
# ─────────────────────────────────────────────────────────
def render_common_dashboard(long_df, unit_label, title_prefix=""):
    st.subheader(f"📊 {title_prefix} 실적 분석 ({unit_label})")
    
    # 데이터 존재 확인
    if long_df.empty: st.warning("분석할 데이터가 없습니다."); return
    
    # 연도 선택
    all_years = sorted(long_df['연'].unique())
    st.markdown("##### 📅 그래프 표시 연도")
    selected_years = st.multiselect("연도 선택", all_years, default=all_years, key=f"viz_{title_prefix}", label_visibility="collapsed")
    if not selected_years: return

    df_filtered = long_df[long_df['연'].isin(selected_years)]
    st.markdown("---")

    # [그래프 1] 월별 추이
    st.markdown(f"#### 📈 월별 {title_prefix} 추이")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # [그래프 2] 연도별 용도 구성
    st.markdown(f"#### 🧱 연도별 용도 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. 예측 및 분석 함수들 (기존 유지)
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
    
    train_years = sorted(long_df['연'].unique())
    if not train_years: st.warning("학습 데이터 없음"); return
    st.info(f"ℹ️ **학습 구간:** {train_years} (좌측 사이드바에서 변경 가능)")
    
    method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)

    df_train = long_df.groupby(['연', '그룹'])['값'].sum().reset_index()
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
        elif "지수" in method: pred = holt_linear_trend(y, 10)
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
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 🧱 2035 상세 예측")
    df_f = df_res[df_res['Type']=='예측']
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='판매량').style.format("{:,.0f}"), use_container_width=True)

def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석")
    if df_temp is None: st.error("🚨 기온 데이터 없음"); return

    df_home = long_df[(long_df['그룹'] == '가정용')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    if df_merged.empty: st.warning("기간 불일치"); return

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
    
    # 1. 깃허브 데이터 로드
    xls_sales = load_github_file(SALES_FILE) # 판매량
    xls_plan = load_github_file(PLAN_FILE)   # 공급량(사업계획)
    
    # 로드 상태 플래그
    is_sales_ok = xls_sales is not None
    is_plan_ok = xls_plan is not None
    
    # 데이터프레임 초기화
    df_sales_long = pd.DataFrame()
    df_plan_long = pd.DataFrame()
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
        
        # 파일 상태 및 업로드
        if is_sales_ok: st.success("✅ 판매량 데이터 로드됨")
        else: 
            st.error("❌ 판매량 로드 실패")
            up_s = st.file_uploader("판매량(.xlsx)", type="xlsx")
            if up_s: xls_sales = pd.ExcelFile(up_s, engine='openpyxl'); is_sales_ok = True
            
        if is_plan_ok: st.success("✅ 사업계획(공급량) 로드됨")
        else:
            st.warning("⚠️ 사업계획 로드 실패 (공급량 분석용)")
            up_p = st.file_uploader("사업계획(.xlsx)", type="xlsx")
            if up_p: xls_plan = pd.ExcelFile(up_p, engine='openpyxl'); is_plan_ok = True
            
        up_t = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])

        # 🟢 [학습/분석 기간 선택]
        # 판매량, 공급량 중 현재 선택된 카테고리에 맞는 데이터를 기준으로 연도 필터 생성
        
        target_years = []
        
        # 데이터 변환 (판매량)
        if is_sales_ok:
            try:
                s_p = "계획_부피" if unit.startswith("부피") else "계획_열량"
                s_a = "실적_부피" if unit.startswith("부피") else "실적_열량"
                df_sales_long = make_long_sales(xls_sales.parse(s_p), xls_sales.parse(s_a))
                # 판매량은 2025년 이하만 실적으로 봄
                sales_years = sorted([y for y in df_sales_long['연'].unique() if y <= 2025])
            except: df_sales_long = pd.DataFrame(); sales_years = []
        else: sales_years = []

        # 데이터 변환 (공급량)
        if is_plan_ok:
            try:
                # 사업계획 파일은 시트가 하나라고 가정하거나 첫번째 시트 사용
                # 형님이 주신 csv를 보면 "데이터"라는 시트일 가능성 높음. 없으면 첫번째.
                if "데이터" in xls_plan.sheet_names:
                    raw_plan = xls_plan.parse("데이터")
                else:
                    raw_plan = xls_plan.parse(0)
                
                df_plan_long = make_long_supply(raw_plan, label="확정계획")
                # 공급량 확정계획은 2026~2028년임
                plan_years = sorted(df_plan_long['연'].unique())
            except: df_plan_long = pd.DataFrame(); plan_years = []
        else: plan_years = []

        # 현재 모드에 따른 연도 필터 표시
        st.markdown("---")
        st.markdown("**📅 학습/분석 대상 연도**")
        
        if main_cat == "1. 판매량 예측":
            if sales_years:
                train_years = st.multiselect("연도 선택", sales_years, default=sales_years, label_visibility="collapsed")
                if train_years: 
                    # 실적 데이터만 필터링 (계획 데이터는 제외하지 않음 or 필요시 조정)
                    df_sales_long = df_sales_long[df_sales_long['연'].isin(train_years) | (df_sales_long['구분']=='계획')]
            else: st.info("데이터 로드 필요")
            
        else: # 2. 공급량 예측
            if plan_years:
                # 여기서는 2026~2028이 '실적'처럼 취급됨
                target_plan_years = st.multiselect("연도 선택", plan_years, default=plan_years, label_visibility="collapsed")
                if target_plan_years:
                    df_plan_long = df_plan_long[df_plan_long['연'].isin(target_plan_years)]
            else: st.info("사업계획 데이터 필요")

    # ── 메인 로직 ──
    df_temp = load_temp_universal(up_t)

    if main_cat == "1. 판매량 예측":
        if df_sales_long.empty: st.info("👈 판매량 데이터를 확인해주세요."); return
        
        # 판매량은 '실적' 데이터만 추출해서 분석
        df_analysis = df_sales_long[df_sales_long['구분'] == '실적']
        
        if "실적분석" in sub_mode:
            render_common_dashboard(df_analysis, unit_label, title_prefix="판매량")
        elif "2035 예측" in sub_mode:
            render_prediction_2035(df_analysis, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(df_analysis, df_temp, unit_label)
            
    else: # 2. 공급량 예측
        if df_plan_long.empty: st.info("👈 사업계획(공급량) 데이터를 확인해주세요."); return
        
        # 공급량은 2026~2028 계획 데이터가 '실적(Base)' 역할을 함
        # 형님 요청: "공급량 실적(2026~2028)을 먼저 보여줘"
        
        if "실적분석" in sub_mode:
            render_common_dashboard(df_plan_long, unit_label, title_prefix="공급량(확정계획)")
        elif "2035 예측" in sub_mode:
            # 2026~2028 데이터를 학습 데이터로 사용하여 2029~2035 예측
            render_prediction_2035(df_plan_long, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(df_plan_long, df_temp, unit_label)

if __name__ == "__main__":
    main()
