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
# 🟢 1. 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석 시스템", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 깃허브 설정
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" 

# 파일명 상수
FILE_SALES = "판매량(계획_실적).xlsx"       # 판매량 분석용
FILE_SUPPLY_HIST = "상품별공급량_MJ.xlsx"   # 공급량 과거 실적 (New!)
FILE_SUPPLY_PLAN = "사업계획최종.xlsx"      # 공급량 중기 계획
FILE_TEMP = "기온.csv"

# 🟢 용도 매핑 (모든 파일의 컬럼명을 표준 그룹으로 통합)
USE_COL_TO_GROUP = {
    # 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    
    # 영업용
    "일반용": "영업용", "일반용(1)": "영업용", "일반용(2)": "영업용", 
    "영업용_일반용1": "영업용", "영업용_일반용2": "영업용", "일반용1(영업)": "영업용",
    
    # 업무용
    "업무난방용": "업무용", "냉방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", 
    "업무용_냉난방": "업무용", "업무용_주한미군": "업무용", "일반용1(업무)": "업무용",
    
    # 산업용
    "산업용": "산업용",
    
    # 수송용
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용", "CNG": "수송용", "BIO": "수송용",
    
    # 발전/기타
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용", "열전용설비용(주택외)": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 🟢 2. 데이터 로드 및 전처리
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data_robust(filename, uploaded_file=None):
    """1.업로드 -> 2.로컬 -> 3.깃허브 순서로 로드"""
    # 1. 업로드 파일
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try: return pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except: return pd.read_csv(uploaded_file, encoding='cp949')
            else:
                return pd.ExcelFile(uploaded_file, engine='openpyxl')
        except: return None

    # 2. 로컬 파일
    if Path(filename).exists():
        if filename.endswith('.xlsx'): return pd.ExcelFile(filename, engine='openpyxl')
        else:
            try: return pd.read_csv(filename, encoding='utf-8-sig')
            except: return pd.read_csv(filename, encoding='cp949')

    # 3. 깃허브
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
        response = requests.get(url)
        if response.status_code == 200:
            if filename.endswith('.xlsx'):
                return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
            else:
                try: return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
                except: return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except: pass
    
    return None

def _clean_base(df):
    out = df.copy()
    # 쓰레기 컬럼 제거
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
    if '연' in out.columns: out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    if '월' in out.columns: out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long_basic(df, default_label="실적"):
    """일반적인 와이드 데이터를 롱 포맷으로 변환"""
    df = _clean_base(df)
    records = []
    
    # 엑셀 파일일 경우 'ExcelFile' 객체가 들어올 수 있으므로 처리
    if isinstance(df, pd.ExcelFile):
        # 첫 번째 시트 사용
        df = df.parse(0)
        df = _clean_base(df)

    for col in df.columns:
        clean_col = col.strip()
        # 그룹 매핑 확인
        group = USE_COL_TO_GROUP.get(clean_col)
        if not group: continue
        
        base = df[["연", "월"]].copy()
        base["그룹"] = group
        base["용도"] = clean_col
        base["구분"] = default_label
        base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append(base)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True).dropna(subset=["연", "월"])

def make_long_sales(plan_df, actual_df):
    """판매량 전용 변환"""
    df1 = make_long_basic(plan_df, "계획")
    df2 = make_long_basic(actual_df, "실적")
    return pd.concat([df1, df2], ignore_index=True)

def preprocess_temp(df):
    if df is None: return None
    # DataFrame인지 확인
    if isinstance(df, pd.ExcelFile): df = df.parse(0)
    
    if '날짜' not in df.columns: 
        # 첫번째 컬럼을 날짜로 가정
        df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        
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
# 🟢 3. 화면 렌더링 함수
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label, title=""):
    st.subheader(f"📊 {title} 현황 분석")
    if long_df.empty: st.warning("데이터가 없습니다."); return
    
    all_years = sorted(long_df['연'].unique())
    selected_years = st.multiselect("연도 선택", all_years, default=all_years[-3:] if len(all_years)>3 else all_years, label_visibility="collapsed")
    if not selected_years: return

    df_viz = long_df[long_df['연'].isin(selected_years)]
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 월별 추이")
        df_mon = df_viz.groupby(['연', '월'])['값'].sum().reset_index()
        fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("#### 🧱 용도별 구성")
        df_yr = df_viz.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)

def render_prediction_2035(long_df, unit_label, start_pred_year):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 학습 데이터 확인
    train_df = long_df[long_df['연'] < start_pred_year]
    if train_df.empty: st.warning("과거 실적 데이터가 부족합니다."); return
    
    train_years = sorted(train_df['연'].unique())
    st.info(f"ℹ️ **학습 구간:** {train_years[0]}~{train_years[-1]}년 (총 {len(train_years)}년)")
    
    method = st.radio("예측 방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)

    df_train_grp = train_df.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train_grp['그룹'].unique()
    
    # 예측 구간: 예측 시작연도 ~ 2035년
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train_grp[df_train_grp['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        # 알고리즘 적용
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
        
        # 결과: 실적(과거) + 예측(미래)
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': v, '구분': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    # 2026~2028 확정계획이 있다면 그것과 비교 가능하도록 표시
    # (여기서는 간단히 예측 결과만 표시)
    
    st.markdown("#### 📈 장기 전망 그래프")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    # 예측 시작 지점에 수직선
    fig.add_vline(x=start_pred_year-0.5, line_width=1, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 상세 예측 데이터")
    df_f = df_res[df_res['구분']=='예측']
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 실행 로직
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석 시스템")
    
    with st.sidebar:
        st.header("1. 분석 모드 설정")
        main_cat = st.radio("📂 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측"])
        st.markdown("---")
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        
        st.header("2. 데이터 파일 연결")
        
        # 🟢 파일 로드 로직 분기
        df_final = pd.DataFrame()
        
        if main_cat == "1. 판매량 예측":
            st.caption("판매량 데이터 (계획 vs 실적)")
            # 1. 깃허브/로컬 로드 시도
            xls = load_data_robust(FILE_SALES)
            up = None
            if xls is None:
                st.error("GitHub 로드 실패")
                up = st.file_uploader("판매량 파일(.xlsx) 업로드", type="xlsx")
                if up: xls = load_data_robust(FILE_SALES, up)
            else:
                st.success("✅ 판매량 파일 연결됨")
            
            # 데이터 처리
            if xls:
                try:
                    # 시트명 결정
                    s_p = "계획_부피" if unit.startswith("부피") else "계획_열량"
                    s_a = "실적_부피" if unit.startswith("부피") else "실적_열량"
                    df_final = make_long_sales(xls.parse(s_p), xls.parse(s_a))
                except Exception as e: st.error(f"데이터 처리 오류: {e}")

        else: # 2. 공급량 예측
            st.caption("필요 파일: 1)과거실적, 2)중기계획")
            
            # A. 과거 실적 (상품별공급량_MJ.xlsx)
            xls_hist = load_data_robust(FILE_SUPPLY_HIST)
            up_h = None
            if xls_hist is None:
                st.warning("⚠️ 과거 실적(상품별공급량) 없음")
                up_h = st.file_uploader("상품별공급량(.xlsx/csv) 업로드", type=["xlsx", "csv"])
                if up_h: xls_hist = load_data_robust(FILE_SUPPLY_HIST, up_h)
            else: st.success("✅ 과거 실적 데이터 연결됨")
            
            # B. 중기 계획 (사업계획최종.xlsx)
            xls_plan = load_data_robust(FILE_SUPPLY_PLAN)
            up_p = None
            if xls_plan is None:
                st.warning("⚠️ 중기 계획(사업계획) 없음")
                up_p = st.file_uploader("사업계획(.xlsx) 업로드", type="xlsx")
                if up_p: xls_plan = load_data_robust(FILE_SUPPLY_PLAN, up_p)
            else: st.success("✅ 중기 계획 데이터 연결됨")
            
            # 데이터 병합 처리
            try:
                df_list = []
                # 1. 과거 실적 변환 (2013~2025)
                if xls_hist:
                    # CSV 또는 Excel 시트 0번 로드
                    raw_hist = xls_hist if isinstance(xls_hist, pd.DataFrame) else xls_hist.parse(0)
                    df_h = make_long_basic(raw_hist, "실적")
                    df_list.append(df_h)
                
                # 2. 중기 계획 변환 (2026~2028)
                if xls_plan:
                    # '데이터' 시트 또는 0번 시트
                    sheet = "데이터" if "데이터" in xls_plan.sheet_names else 0
                    df_p = make_long_basic(xls_plan.parse(sheet), "확정계획")
                    df_list.append(df_p)
                
                if df_list:
                    df_final = pd.concat(df_list, ignore_index=True)
            except Exception as e: st.error(f"공급량 데이터 병합 오류: {e}")

        # 🟢 [학습 기간 선택] - 데이터가 있을 때만
        if not df_final.empty:
            st.markdown("---")
            st.markdown("**📅 학습/분석 대상 연도**")
            all_years = sorted(df_final['연'].unique())
            # 디폴트: 2025년 이하 (과거 실적만 학습용으로 기본 선택)
            default_train = [y for y in all_years if y <= 2025]
            if not default_train: default_train = all_years # 데이터가 미래뿐이면 전체 선택
            
            train_years = st.multiselect("학습 연도 선택", all_years, default=default_train, label_visibility="collapsed")
            
            # 🔴 여기서 최종 필터링!
            # 분석용 데이터: 선택된 연도 + (공급량인 경우 확정계획 포함)
            # 여기서는 단순하게 선택된 연도만 남기되, 2035 예측 시에는 이걸 학습 데이터로 씀
            df_final = df_final[df_final['연'].isin(train_years) | (df_final['구분'] == '확정계획')]

    # ── 메인 화면 ──
    if df_final.empty:
        st.info("👈 좌측 사이드바에서 데이터를 연결해주세요.")
        return

    # 기온 데이터 (선택)
    # df_temp = load_temp_universal(...) # 생략 (코드 길이상)

    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(df_final, unit_label, "판매량")
        elif "2035 예측" in sub_mode:
            # 판매량 예측은 2026년부터 시작
            render_prediction_2035(df_final, unit_label, 2026)
            
    else: # 2. 공급량 예측
        if "실적분석" in sub_mode:
            st.info("💡 과거 실적(2013~)과 확정 계획(2026~2028)을 함께 분석합니다.")
            render_analysis_dashboard(df_final, unit_label, "공급량")
        elif "2035 예측" in sub_mode:
            st.info("💡 과거 데이터와 2026~2028 확정 계획을 모두 고려하여 2035년까지 예측합니다.")
            # 공급량은 2028년까지 계획이 있으므로, 예측은 2029년부터 시작
            render_prediction_2035(df_final, unit_label, 2029)

if __name__ == "__main__":
    main()
