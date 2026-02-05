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
    else:
        try:
            import matplotlib as mpl
            mpl.rcParams['axes.unicode_minus'] = False
            mpl.rc('font', family='Malgun Gothic')
        except: pass

set_korean_font()

# 🟢 설정 정보 (형님 깃허브 정보)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" # main 또는 master
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_PLAN_XLSX = "사업계획최종.xlsx"
DEFAULT_TEMP_CSV = "기온.csv"

# 🟢 용도 매핑
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용", 
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
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
# 1. 데이터 로드 및 전처리 (여기에 깃허브 기능을 심었습니다)
# ─────────────────────────────────────────────────────────
def _clean_base(df):
    out = df.copy()
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
    if '연' in out.columns: out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    if '월' in out.columns: out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long(plan_df, actual_df):
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            clean_col = col.strip()
            if clean_col in ["연", "월"]: continue
            group = USE_COL_TO_GROUP.get(clean_col)
            if not group: continue
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = clean_col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

def make_long_supply(df):
    """사업계획 데이터 변환"""
    df = _clean_base(df)
    records = []
    for col in df.columns:
        clean_col = col.strip()
        if clean_col in ["연", "월", "소계", "합계"]: continue
        group = USE_COL_TO_GROUP.get(clean_col)
        if not group: continue
        
        base = df[["연", "월"]].copy()
        base["그룹"] = group
        base["용도"] = clean_col
        base["계획/실적"] = "확정계획" # 공급량 분석 시 실적처럼 사용
        base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append(base)
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

# 🟢 [수정됨] 깃허브 우선 로드 -> 로컬 로드 -> 실패 순서
@st.cache_data(ttl=600)
def load_data_smart(filename, file_type='xlsx'):
    # 1. 깃허브 시도
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if file_type == 'xlsx':
                return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
            else:
                try: return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
                except: return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except: pass

    # 2. 로컬 파일 시도
    if Path(filename).exists():
        if file_type == 'xlsx':
            return pd.ExcelFile(filename, engine='openpyxl')
        else:
            try: return pd.read_csv(filename, encoding='utf-8-sig')
            except: return pd.read_csv(filename, encoding='cp949')
            
    return None

def preprocess_temp(df):
    if df is None: return None
    if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    cols = [c for c in df.columns if "기온" in c]
    if not cols: return None
    target = cols[0]
    monthly = df.groupby(['연', '월'])[target].mean().reset_index()
    monthly.rename(columns={target: '평균기온'}, inplace=True)
    return monthly

# ─────────────────────────────────────────────────────────
# 2. 분석 함수들
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label, title_prefix=""):
    st.subheader(f"📊 {title_prefix} 실적 분석 ({unit_label})")
    if long_df.empty: st.warning("데이터가 없습니다."); return
    
    all_years = sorted(long_df['연'].unique())
    selected_years = st.multiselect("그래프 표시 연도", all_years, default=all_years, key=f"viz_{title_prefix}", label_visibility="collapsed")
    if not selected_years: return

    df_filtered = long_df[long_df['연'].isin(selected_years)]
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 📈 월별 추이")
        df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
        fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown(f"#### 🧱 용도별 구성비")
        df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)

def render_prediction_2035(long_df, unit_label, start_pred_year):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    train_years = sorted(long_df['연'].unique())
    st.info(f"ℹ️ **학습 데이터:** {train_years[0]}~{train_years[-1]}년 (선택된 연도만 사용)")
    
    method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)

    df_train = long_df.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in method:
            try: model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y); pred = model.predict(np.log(np.arange(len(X)+1, len(X)+1+len(future_years))).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in method:
             pred = np.array([y[-1] + (j+1)*(y[-1]-y[0])/len(y) for j in range(len(future_years))])
        else: # CAGR
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(len(future_years))]
            except: pred = [y[-1]]*len(future_years)
                
        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': v, '구분': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    st.markdown("#### 📈 장기 전망")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 상세 예측")
    df_f = df_res[df_res['구분']=='예측']
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

def render_household(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석")
    if df_temp is None: st.error("🚨 기온 데이터 없음"); return

    df_home = long_df[long_df['그룹'] == '가정용'].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    if df_merged.empty: st.warning("데이터 기간 불일치"); return

    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도", years, default=years, key="house_years", label_visibility="collapsed")
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{df_final['평균기온'].corr(df_final['값']):.2f}")

# ─────────────────────────────────────────────────────────
# 3. 메인 실행 (통합)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량/공급량 통합 분석")
    
    # 1. 깃허브 자동 로드 (Smart Load)
    xls_sales = load_data_smart(DEFAULT_SALES_XLSX, 'xlsx')
    xls_plan = load_data_smart(DEFAULT_PLAN_XLSX, 'xlsx')
    
    is_sales_ok = xls_sales is not None
    is_plan_ok = xls_plan is not None
    
    df_sales_long = pd.DataFrame()
    df_plan_long = pd.DataFrame()
    unit_label = "천m³"

    with st.sidebar:
        st.header("설정")
        main_cat = st.radio("📂 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        st.markdown("---")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
        
        # 파일 상태 표시 & 업로더 (실패 시에만)
        if is_sales_ok: st.success("✅ 판매량 로드 완료")
        else: 
            st.error("❌ 판매량 실패")
            up_s = st.file_uploader("판매량(.xlsx)", type="xlsx", key="us")
            if up_s: xls_sales = pd.ExcelFile(up_s, engine='openpyxl'); is_sales_ok = True
            
        if is_plan_ok: st.success("✅ 사업계획(공급량) 로드 완료")
        else:
            st.warning("⚠️ 사업계획 실패")
            up_p = st.file_uploader("사업계획(.xlsx)", type="xlsx", key="up")
            if up_p: xls_plan = pd.ExcelFile(up_p, engine='openpyxl'); is_plan_ok = True
            
        up_t = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])
        
        # 🟢 [학습 기간 선택] - 데이터가 로드되면 즉시 표시
        st.markdown("---")
        st.markdown("**📅 학습/분석 대상 연도 설정**")
        
        # A. 판매량 데이터 준비
        if is_sales_ok:
            try:
                s_p = "계획_부피" if unit.startswith("부피") else "계획_열량"
                s_a = "실적_부피" if unit.startswith("부피") else "실적_열량"
                df_sales_long = make_long(xls_sales.parse(s_p), xls_sales.parse(s_a))
            except: pass
            
        # B. 공급량 데이터 준비
        if is_plan_ok:
            try:
                # 사업계획 파일은 보통 '데이터' 시트
                p_sheet = "데이터" if "데이터" in xls_plan.sheet_names else 0
                df_plan_long = make_long_supply(xls_plan.parse(p_sheet))
            except: pass

        # 현재 카테고리에 맞는 연도 필터 제공
        if main_cat == "1. 판매량 예측":
            if not df_sales_long.empty:
                avail_years = sorted([y for y in df_sales_long['연'].unique() if y <= 2025])
                train_years = st.multiselect("연도(판매량)", avail_years, default=avail_years, label_visibility="collapsed")
                if train_years:
                    df_sales_long = df_sales_long[df_sales_long['연'].isin(train_years) | (df_sales_long['구분']=='계획')]
            else: st.info("판매량 데이터 로드 필요")
            
        else: # 2. 공급량 예측
            if not df_plan_long.empty:
                # 공급량은 2026~2028이 확정계획(실적 역할)
                avail_years = sorted(df_plan_long['연'].unique())
                target_years = st.multiselect("연도(공급량)", avail_years, default=avail_years, label_visibility="collapsed")
                if target_years:
                    df_plan_long = df_plan_long[df_plan_long['연'].isin(target_years)]
            else: st.info("사업계획 데이터 로드 필요")

    # ── 메인 화면 출력 ──
    # 기온 데이터 로드 (업로드 or 깃허브)
    if up_t:
        if up_t.name.endswith('.csv'): df_temp = preprocess_temp(pd.read_csv(up_t))
        else: df_temp = preprocess_temp(pd.read_excel(up_t))
    else:
        # 깃허브에서 기온 파일 시도
        temp_raw = load_data_smart(DEFAULT_TEMP_CSV, 'csv')
        df_temp = preprocess_temp(temp_raw) if temp_raw is not None else None

    if main_cat == "1. 판매량 예측":
        if df_sales_long.empty: st.info("👈 좌측에서 판매량 데이터를 확인해주세요."); return
        
        df_target = df_sales_long[df_sales_long['구분'] == '실적']
        
        if "실적분석" in sub_mode:
            render_analysis_dashboard(df_target, unit_label, "판매량")
        elif "2035 예측" in sub_mode:
            render_prediction_2035(df_target, unit_label, 2026)
        elif "가정용" in sub_mode:
            render_household(df_target, df_temp, unit_label)
            
    else: # 2. 공급량 예측
        if df_plan_long.empty: st.info("👈 좌측에서 사업계획 데이터를 확인해주세요."); return
        
        # 공급량은 확정계획(2026~2028) 데이터를 분석 대상으로 함
        if "실적분석" in sub_mode:
            st.info("💡 2026~2028년 확정 계획 데이터를 분석합니다.")
            render_analysis_dashboard(df_plan_long, unit_label, "공급량(확정계획)")
        elif "2035 예측" in sub_mode:
            st.info("💡 2026~2028년 확정 계획을 바탕으로 미래를 예측합니다.")
            render_prediction_2035(df_plan_long, unit_label, 2029)
        elif "가정용" in sub_mode:
            render_household(df_plan_long, df_temp, unit_label)

if __name__ == "__main__":
    main()
