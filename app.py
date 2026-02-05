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
        # 폰트 파일 없을 경우 시스템 기본 폰트 사용 (깨짐 방지)
        try:
            import matplotlib as mpl
            mpl.rcParams['axes.unicode_minus'] = False
            mpl.rc('font', family='Malgun Gothic') 
        except: pass

set_korean_font()

# 🟢 설정 정보
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" # main 또는 master
SALES_FILE = "판매량(계획_실적).xlsx"
PLAN_FILE = "사업계획최종.xlsx"

# 🟢 용도 매핑 (판매량/공급량 통합)
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
# 1. 데이터 로드 및 전처리 (수정됨: GitHub 연결 기능 추가)
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
        if clean_col in ["연", "월", "소계", "합계", "가정용소계"]: continue
        group = USE_COL_TO_GROUP.get(clean_col)
        if not group: continue
        
        base = df[["연", "월"]].copy()
        base["그룹"] = group
        base["용도"] = clean_col
        base["계획/실적"] = "확정계획"
        base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append(base)
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

# 🟢 [수정된 로더] 1.업로드 -> 2.로컬 -> 3.깃허브 순서로 시도
@st.cache_data(ttl=600)
def load_data_robust(filename, uploaded_file=None):
    # 1. 사용자가 직접 업로드한 경우
    if uploaded_file is not None:
        return pd.ExcelFile(uploaded_file, engine='openpyxl')

    # 2. 로컬 파일 (형님 컴퓨터)
    if Path(filename).exists():
        return pd.ExcelFile(filename, engine='openpyxl')

    # 3. 깃허브 URL (웹 배포용)
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
        response = requests.get(url)
        if response.status_code == 200:
            return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
    except: pass
    
    return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (공통 화면)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    # 모드에 따른 제목 변경
    mode_name = "실적" if "실적" in long_df['계획/실적'].unique() else "확정계획(26~28년)"
    st.subheader(f"📊 {mode_name} 분석 ({unit_label})")
    
    # 데이터 필터링 (실적 또는 확정계획)
    target = '실적' if '실적' in long_df['계획/실적'].unique() else '확정계획'
    df_act = long_df[long_df['계획/실적'] == target].copy()
    
    if df_act.empty: st.warning("데이터가 없습니다."); return
    
    all_years = sorted(df_act['연'].unique())
    selected_years = st.multiselect("연도 선택", options=all_years, default=all_years, label_visibility="collapsed")
    
    if not selected_years: return
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    
    st.markdown("---")

    # 그래프 1
    st.markdown(f"#### 📈 월별 추이")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2
    st.markdown(f"#### 🧱 용도별 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    piv = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

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
    
    train_years = sorted(long_df['연'].unique())
    if not train_years: st.warning("학습 데이터 없음"); return
    
    # 예측 시작점 설정 (마지막 데이터 다음 해부터)
    last_year = train_years[-1]
    future_years = np.arange(last_year + 1, 2036).reshape(-1, 1)
    
    st.markdown("##### 📊 추세 분석 모델 선택")
    pred_method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)

    df_train = long_df.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        if "선형" in pred_method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in pred_method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in pred_method:
            try: model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y); pred = model.predict(np.log(np.arange(len(X)+1, len(X)+1+len(future_years))).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in pred_method: 
            pred = np.array([y[-1] + (j+1)*(y[-1]-y[0])/len(y) for j in range(len(future_years))])
        else: # CAGR
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(len(future_years))]
            except: pred = [y[-1]]*len(future_years)
                
        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': v, '구분': '실적(계획)'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    st.markdown("#### 📈 전체 장기 전망")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 상세 예측")
    df_f = df_res[df_res['구분']=='예측']
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량/공급량 통합 분석")
    
    # 1. 파일 상태 변수
    df_sales_long = pd.DataFrame()
    df_plan_long = pd.DataFrame()
    unit_label = "천m³"
    
    with st.sidebar:
        st.header("설정")
        main_cat = st.radio("📂 분석 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측"]) # 가정용은 일단 제외
        st.markdown("---")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
        
        # 🟢 파일 로드 (깃허브 실패 시 업로드 버튼 자동 노출)
        st.caption("데이터 연결 상태")
        
        # A. 판매량 파일
        up_s = None
        xls_sales = load_data_robust(SALES_FILE, None)
        if xls_sales: 
            st.success("✅ 판매량 연결됨")
        else:
            st.error("❌ 판매량 연결 실패")
            up_s = st.file_uploader("판매량(.xlsx) 업로드", type="xlsx")
            if up_s: xls_sales = load_data_robust(SALES_FILE, up_s)

        # B. 사업계획 파일 (공급량용)
        up_p = None
        xls_plan = load_data_robust(PLAN_FILE, None)
        if xls_plan: 
            st.success("✅ 사업계획 연결됨")
        else:
            st.warning("⚠️ 사업계획 연결 실패")
            up_p = st.file_uploader("사업계획(.xlsx) 업로드", type="xlsx")
            if up_p: xls_plan = load_data_robust(PLAN_FILE, up_p)

        # 🟢 [학습 기간 선택] - 핵심 기능
        st.markdown("---")
        st.markdown("**📅 학습/분석 대상 연도 설정**")
        
        # 데이터 처리 및 연도 추출
        try:
            if xls_sales:
                s_p = "계획_부피" if unit.startswith("부피") else "계획_열량"
                s_a = "실적_부피" if unit.startswith("부피") else "실적_열량"
                df_sales_long = make_long(xls_sales.parse(s_p), xls_sales.parse(s_a))
                
            if xls_plan:
                # 사업계획은 보통 '데이터' 시트
                p_sheet = "데이터" if "데이터" in xls_plan.sheet_names else 0
                df_plan_long = make_long_supply(xls_plan.parse(p_sheet))
        except: pass

        # 현재 모드에 따른 연도 필터
        if main_cat == "1. 판매량 예측":
            if not df_sales_long.empty:
                # 판매량: 2025년 이하만 실적으로 간주
                avail_years = sorted([y for y in df_sales_long['연'].unique() if y <= 2025])
                train_years = st.multiselect("연도(판매량)", avail_years, default=avail_years, label_visibility="collapsed")
                
                if train_years:
                    # 선택된 연도만 남기고 나머지는 제거 (계획 데이터는 유지)
                    df_sales_long = df_sales_long[df_sales_long['연'].isin(train_years) | (df_sales_long['구분']=='계획')]
            else: st.info("판매량 파일 필요")
            
        else: # 2. 공급량 예측
            if not df_plan_long.empty:
                # 공급량: 2026~2028 확정계획이 실적 역할
                avail_years = sorted(df_plan_long['연'].unique())
                target_years = st.multiselect("연도(공급량)", avail_years, default=avail_years, label_visibility="collapsed")
                
                if target_years:
                    df_plan_long = df_plan_long[df_plan_long['연'].isin(target_years)]
            else: st.info("사업계획 파일 필요")

    # ── 메인 화면 ──
    if main_cat == "1. 판매량 예측":
        if df_sales_long.empty: st.info("👈 좌측에서 판매량 파일을 확인해주세요."); return
        
        if "실적분석" in sub_mode:
            render_analysis_dashboard(df_sales_long, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(df_sales_long, unit_label)
            
    else: # 2. 공급량 예측
        if df_plan_long.empty: st.info("👈 좌측에서 사업계획 파일을 확인해주세요."); return
        
        if "실적분석" in sub_mode:
            st.info("💡 2026~2028년 확정 계획을 분석합니다.")
            render_analysis_dashboard(df_plan_long, unit_label)
        elif "2035 예측" in sub_mode:
            st.info("💡 2026~2028년 데이터를 바탕으로 미래를 예측합니다.")
            render_prediction_2035(df_plan_long, unit_label)

if __name__ == "__main__":
    main()
