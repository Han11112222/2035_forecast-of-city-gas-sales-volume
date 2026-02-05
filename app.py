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
FILE_SALES = "판매량(계획_실적).xlsx"       
FILE_SUPPLY_HIST = "상품별공급량_MJ.xlsx"   
FILE_SUPPLY_PLAN = "사업계획최종.xlsx"      
FILE_TEMP = "기온.csv"

# 🟢 [핵심 수정] MJ 파일에 있는 특이한 컬럼명들 완벽 대응
USE_COL_TO_GROUP = {
    # 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    
    # 영업용 (MJ 파일 특이 명칭 추가)
    "일반용": "영업용", "일반용(1)": "영업용", "일반용(2)": "영업용", 
    "영업용_일반용1": "영업용", "영업용_일반용2": "영업용", 
    "일반용1(영업)": "영업용", "일반용2(영업)": "영업용",
    
    # 업무용 (MJ 파일 특이 명칭 추가)
    "업무난방용": "업무용", "냉방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", 
    "업무용_냉난방": "업무용", "업무용_주한미군": "업무용", 
    "일반용1(업무)": "업무용", "일반용2(업무)": "업무용",
    
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
# 🟢 2. 데이터 로드 (형식 무관하게 읽기)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data_robust(filename, uploaded_file=None):
    """1.업로드 -> 2.로컬 -> 3.깃허브 순서로 로드"""
    
    # 내부 함수: 파일 객체를 받아서 엑셀/CSV 자동 판별 후 읽기
    def try_read(file_obj):
        # 1. 엑셀로 먼저 시도
        try: 
            return pd.ExcelFile(file_obj, engine='openpyxl')
        except:
            # 2. 실패하면 CSV (utf-8) 시도
            try:
                if hasattr(file_obj, 'seek'): file_obj.seek(0)
                return pd.read_csv(file_obj, encoding='utf-8-sig')
            except:
                # 3. 실패하면 CSV (cp949) 시도
                try:
                    if hasattr(file_obj, 'seek'): file_obj.seek(0)
                    return pd.read_csv(file_obj, encoding='cp949')
                except: return None

    # 1. 업로드 파일이 있는 경우
    if uploaded_file:
        return try_read(uploaded_file)

    # 2. 로컬 파일이 있는 경우
    if Path(filename).exists():
        return try_read(filename)

    # 3. 깃허브 URL 시도
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
        response = requests.get(url)
        if response.status_code == 200:
            return try_read(io.BytesIO(response.content))
    except: pass
    
    return None

def _clean_base(df):
    """데이터프레임 정리 (연, 월 컬럼 확보)"""
    # ExcelFile 객체가 들어오면 첫 번째 시트 파싱
    if isinstance(df, pd.ExcelFile):
        df = df.parse(0)
        
    out = df.copy()
    # 이상한 컬럼 제거
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
    
    # 컬럼명 공백 제거 (매핑 매칭률 높이기 위해)
    out.columns = out.columns.str.strip()
    
    # 연/월 컬럼 숫자 변환
    if '연' in out.columns: out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    if '월' in out.columns: out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    
    return out.dropna(subset=['연', '월'])

def make_long_basic(df, default_label="실적"):
    """와이드 -> 롱 변환 (매핑 테이블 기준)"""
    df = _clean_base(df)
    records = []
    
    for col in df.columns:
        clean_col = col.strip()
        # 매핑표에 있는 컬럼만 처리
        group = USE_COL_TO_GROUP.get(clean_col)
        if not group: continue
        
        base = df[["연", "월"]].copy()
        base["그룹"] = group
        base["용도"] = clean_col
        base["구분"] = default_label
        base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        records.append(base)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def make_long_sales(xls_obj):
    """판매량 (계획/실적 시트) 변환"""
    if not isinstance(xls_obj, pd.ExcelFile):
        # CSV 등으로 들어왔으면 그냥 변환
        return make_long_basic(xls_obj, "실적")

    records = []
    # 시트 이름에 '계획', '실적'이 포함된 것 찾기
    sheet_p = [s for s in xls_obj.sheet_names if "계획" in s]
    sheet_a = [s for s in xls_obj.sheet_names if "실적" in s]
    
    if sheet_p:
        df_p = make_long_basic(xls_obj.parse(sheet_p[0]), "계획")
        records.append(df_p)
    if sheet_a:
        df_a = make_long_basic(xls_obj.parse(sheet_a[0]), "실적")
        records.append(df_a)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def preprocess_temp(df):
    if df is None: return None
    if isinstance(df, pd.ExcelFile): df = df.parse(0)
    
    # 날짜 컬럼 찾기
    date_cols = [c for c in df.columns if "일자" in c or "날짜" in c or "date" in c.lower()]
    if date_cols:
        df.rename(columns={date_cols[0]: '날짜'}, inplace=True)
    elif '연' not in df.columns: 
        df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
    
    cols = [c for c in df.columns if "기온" in c]
    if not cols: return None
    target = cols[0]
    
    monthly = df.groupby(['연', '월'])[target].mean().reset_index()
    monthly.rename(columns={target: '평균기온'}, inplace=True)
    return monthly

def load_temp_universal(uploaded_file):
    if uploaded_file: return load_data_robust(FILE_TEMP, uploaded_file)
    return load_data_robust(FILE_TEMP)

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 화면 함수
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
    
    # 학습 데이터 (예측 시작년도 이전 데이터)
    train_df = long_df[long_df['연'] < start_pred_year]
    if train_df.empty: st.warning("과거 실적 데이터가 부족합니다."); return
    
    train_years = sorted(train_df['연'].unique())
    st.info(f"ℹ️ **학습 구간:** {train_years[0]}~{train_years[-1]}년 (총 {len(train_years)}년)")
    
    method = st.radio("예측 방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)

    df_train_grp = train_df.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train_grp['그룹'].unique()
    
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train_grp[df_train_grp['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        # 알고리즘
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
        
        # 병합
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': v, '구분': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 장기 전망 그래프")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    fig.add_vline(x=start_pred_year-0.5, line_width=1, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 상세 예측 데이터")
    df_f = df_res[df_res['구분']=='예측']
    st.dataframe(df_f.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

def render_household(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석")
    temp_df = preprocess_temp(df_temp)
    if temp_df is None: st.error("🚨 기온 데이터 없음"); return

    df_home = long_df[long_df['그룹'] == '가정용'].copy()
    df_merged = pd.merge(df_home, temp_df, on=['연', '월'], how='inner')
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
        
        df_final = pd.DataFrame()
        
        if main_cat == "1. 판매량 예측":
            st.caption("판매량 데이터 (계획 vs 실적)")
            xls = load_data_robust(FILE_SALES)
            up = None
            if xls is None:
                st.error("GitHub 로드 실패")
                up = st.file_uploader("판매량 파일(.xlsx) 업로드", type="xlsx")
                if up: xls = load_data_robust(FILE_SALES, up)
            else: st.success("✅ 판매량 파일 연결됨")
            
            if xls:
                try: df_final = make_long_sales(xls)
                except Exception as e: st.error(f"판매량 데이터 처리 오류: {e}")

        else: # 2. 공급량 예측
            st.caption("필요 파일: 1)과거실적(MJ), 2)중기계획")
            
            # A. 과거 실적
            xls_hist = load_data_robust(FILE_SUPPLY_HIST)
            up_h = None
            if xls_hist is None:
                st.warning("⚠️ 과거 실적(상품별공급량) 없음")
                up_h = st.file_uploader("상품별공급량(.xlsx/csv) 업로드", type=["xlsx", "csv"])
                if up_h: xls_hist = load_data_robust(FILE_SUPPLY_HIST, up_h)
            else: st.success("✅ 과거 실적 데이터 연결됨")
            
            # B. 중기 계획
            xls_plan = load_data_robust(FILE_SUPPLY_PLAN)
            up_p = None
            if xls_plan is None:
                st.warning("⚠️ 중기 계획(사업계획) 없음")
                up_p = st.file_uploader("사업계획(.xlsx) 업로드", type="xlsx")
                if up_p: xls_plan = load_data_robust(FILE_SUPPLY_PLAN, up_p)
            else: st.success("✅ 중기 계획 데이터 연결됨")
            
            # 데이터 병합
            try:
                df_list = []
                # 1. 과거 실적 (MJ 파일)
                if xls_hist:
                    raw_hist = xls_hist if isinstance(xls_hist, pd.DataFrame) else xls_hist.parse(0)
                    df_h = make_long_basic(raw_hist, "실적")
                    df_list.append(df_h)
                
                # 2. 중기 계획 (사업계획 파일)
                if xls_plan:
                    sheet = "데이터" if "데이터" in xls_plan.sheet_names else 0
                    df_p = make_long_basic(xls_plan.parse(sheet), "확정계획")
                    df_list.append(df_p)
                
                if df_list:
                    df_final = pd.concat(df_list, ignore_index=True)
            except Exception as e: st.error(f"공급량 데이터 병합 오류: {e}")

        # 🟢 [학습 기간 선택]
        if not df_final.empty:
            st.markdown("---")
            st.markdown("**📅 학습/분석 대상 연도**")
            all_years = sorted(df_final['연'].unique())
            default_train = [y for y in all_years if y <= 2025]
            if not default_train: default_train = all_years
            
            train_years = st.multiselect("학습 연도 선택", all_years, default=default_train, label_visibility="collapsed")
            
            # 최종 필터링: 선택된 연도 + (공급량인 경우 확정계획 포함)
            df_final = df_final[df_final['연'].isin(train_years) | (df_final['구분'] == '확정계획')]
            
        up_t = st.file_uploader("기온(.csv, .xlsx)", type=["csv", "xlsx"])

    # ── 메인 화면 ──
    if df_final.empty:
        st.info("👈 좌측 사이드바에서 데이터를 연결해주세요.")
        return

    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(df_final, unit_label, "판매량")
        elif "2035 예측" in sub_mode:
            render_prediction_2035(df_final, unit_label, 2026)
            
    else: # 2. 공급량 예측
        if "실적분석" in sub_mode:
            st.info("💡 과거 실적(2013~)과 확정 계획(2026~2028)을 함께 분석합니다.")
            render_analysis_dashboard(df_final, unit_label, "공급량")
        elif "2035 예측" in sub_mode:
            st.info("💡 과거 데이터와 2026~2028 확정 계획을 모두 고려하여 2035년까지 예측합니다.")
            render_prediction_2035(df_final, unit_label, 2029)

if __name__ == "__main__":
    main()
