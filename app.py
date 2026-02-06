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

# 🟢 깃허브 설정 (백업용)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" 

# 파일명 상수
FILE_SALES = "판매량(계획_실적).xlsx"       
FILE_SUPPLY_MJ = "공급량실적_계획_실적_MJ.xlsx" # 공급량 통합 파일
FILE_TEMP = "기온.csv"

# 🟢 [매핑 테이블] 모든 파일의 컬럼명을 표준 그룹으로 통합
USE_COL_TO_GROUP = {
    # 🏠 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    
    # 🏪 영업용
    "일반용": "영업용", "일반용(1)": "영업용", "일반용(2)": "영업용", 
    "영업용_일반용1": "영업용", "영업용_일반용2": "영업용", 
    "일반용1(영업)": "영업용", "일반용2(영업)": "영업용", "일반용1": "영업용",
    
    # 🏢 업무용
    "업무난방용": "업무용", "냉방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", 
    "업무용_냉난방": "업무용", "업무용_주한미군": "업무용", 
    "일반용1(업무)": "업무용", "일반용2(업무)": "업무용",
    
    # 🏭 산업용
    "산업용": "산업용",
    
    # 🚌 수송용
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용", "CNG": "수송용", "BIO": "수송용",
    
    # ⚡ 발전/기타
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용", "열전용설비용(주택외)": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 🟢 2. 만능 데이터 로더
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data_super_robust(filename, uploaded_file=None):
    """업로드 > 로컬 > 깃허브 순으로 로드 (확장자 무시)"""
    
    def try_read_stream(file_obj):
        # 1. Excel로 시도
        try: return pd.ExcelFile(file_obj, engine='openpyxl')
        except: pass
        
        # 2. CSV (utf-8) 시도
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        try: return pd.read_csv(file_obj, encoding='utf-8-sig')
        except: pass
        
        # 3. CSV (cp949) 시도
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        try: return pd.read_csv(file_obj, encoding='cp949')
        except: pass
        return None

    if uploaded_file: return try_read_stream(uploaded_file)
    if Path(filename).exists():
        with open(filename, 'rb') as f: return try_read_stream(io.BytesIO(f.read()))
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
        r = requests.get(url)
        if r.status_code == 200: return try_read_stream(io.BytesIO(r.content))
    except: pass
    
    return None

def standardize_df(df_or_excel, sheet_name=None):
    """데이터프레임 표준화 (시트 지정 가능)"""
    if df_or_excel is None: return None
    
    df = None
    if isinstance(df_or_excel, pd.ExcelFile):
        # 시트 이름이 지정되었고 존재하면 그 시트 사용
        if sheet_name and sheet_name in df_or_excel.sheet_names:
            df = df_or_excel.parse(sheet_name)
        elif sheet_name is None: # 시트 지정 안했으면 첫번째
            df = df_or_excel.parse(0)
        else: # 시트가 없으면 None
            return None
    else:
        df = df_or_excel # CSV인 경우
        
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip() # 공백 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 날짜 처리
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def process_data_to_long(df, label_name):
    """매핑 테이블을 이용해 분석용 데이터로 변환"""
    if df is None or df.empty: return pd.DataFrame()
    
    if '연' not in df.columns or '월' not in df.columns: return pd.DataFrame() 
        
    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    for col in df.columns:
        group = USE_COL_TO_GROUP.get(col)
        if not group: continue
        
        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label_name
        sub['값'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def load_temp_universal(uploaded_file):
    raw = load_data_super_robust(FILE_TEMP, uploaded_file)
    if raw is None: return None
    
    df = standardize_df(raw)
    cols = [c for c in df.columns if "기온" in c]
    if not cols: return None
    
    monthly = df.groupby(['연', '월'])[cols[0]].mean().reset_index()
    monthly.rename(columns={cols[0]: '평균기온'}, inplace=True)
    return monthly

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 및 시각화
# ─────────────────────────────────────────────────────────
def render_analysis(df, unit):
    st.subheader(f"📊 실적 현황 분석 ({unit})")
    if df.empty: st.warning("데이터가 없습니다."); return

    all_years = sorted(df['연'].unique())
    sel_years = st.multiselect("연도 선택", all_years, default=all_years[-3:] if len(all_years)>3 else all_years)
    
    df_viz = df[df['연'].isin(sel_years)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 월별 추이")
        mon_grp = df_viz.groupby(['연', '월'])['값'].sum().reset_index()
        fig = px.line(mon_grp, x='월', y='값', color='연', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### 🧱 용도별 구성")
        yr_grp = df_viz.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(yr_grp, x='연', y='값', color='그룹')
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df_viz.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"), use_container_width=True)

def render_prediction(df, unit, start_year):
    st.subheader(f"🔮 2035 장기 예측 ({unit})")
    
    # 학습 데이터: 예측 시작년도 이전 데이터 (실적 + 확정계획)
    train_df = df[df['연'] < start_year]
    if train_df.empty: st.warning("학습할 과거 데이터가 부족합니다."); return
        
    st.info(f"학습 데이터 구간: {int(train_df['연'].min())}년 ~ {int(train_df['연'].max())}년 (실적 + 확정계획)")
    
    method = st.radio("예측 방법", ["선형 회귀", "2차 곡선", "로그 추세", "지수 평활", "CAGR"], horizontal=True)
    
    train_grp = train_df.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = train_grp['그룹'].unique()
    future_years = np.arange(start_year, 2036).reshape(-1, 1)
    
    results = []
    
    for grp in groups:
        sub = train_grp[train_grp['그룹'] == grp]
        if len(sub) < 2: continue
        
        X = sub['연'].values
        y = sub['값'].values
        pred = []
        
        if method == "선형 회귀":
            model = LinearRegression(); model.fit(X.reshape(-1,1), y)
            pred = model.predict(future_years)
        elif method == "2차 곡선":
            try: z = np.polyfit(X, y, 2); p = np.poly1d(z); pred = p(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif method == "로그 추세":
            try: model = LinearRegression(); model.fit(np.log(X.reshape(-1,1)), y); pred = model.predict(np.log(future_years))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif method == "지수 평활":
            pred = [y[-1]] * len(future_years)
        else: # CAGR
            try: cagr = (y[-1]/y[0])**(1/len(y)) - 1
            except: cagr = 0
            pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
            
        pred = [max(0, p) for p in pred]
        
        for yr, val in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': val, '구분': '실적(계획포함)'})
        for yr, val in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': val, '구분': '예측'})
        
    res_df = pd.DataFrame(results)
    
    st.markdown("#### 📈 장기 전망")
    fig = px.line(res_df, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    fig.add_vline(x=start_year-0.5, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📋 연도별 예측 데이터")
    pred_only = res_df[res_df['구분'] == '예측']
    st.dataframe(pred_only.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"])
        func = st.radio("기능", ["실적분석", "2035 예측"])
        unit = st.radio("단위", ["부피(천m³)", "열량(GJ)"])
        st.markdown("---")
        
    df_final = pd.DataFrame()
    
    # 🟢 1. 판매량 예측 모드 (파일 1개 필요)
    if mode.startswith("1"):
        with st.sidebar:
            st.warning("📂 **[판매량(계획_실적).xlsx]** 파일 업로드")
            up = st.file_uploader("판매량 파일", type=["xlsx", "csv"], key="sales_up")
            
        raw = load_data_super_robust(FILE_SALES, up)
        
        if raw is not None:
            try:
                # 엑셀 파일인 경우 시트 분리
                if isinstance(raw, pd.ExcelFile):
                    s_p = [s for s in raw.sheet_names if "계획" in s]
                    s_a = [s for s in raw.sheet_names if "실적" in s]
                    df_p = standardize_df(raw, s_p[0]) if s_p else pd.DataFrame()
                    df_a = standardize_df(raw, s_a[0]) if s_a else pd.DataFrame()
                    
                    final_p = process_data_to_long(df_p, "계획")
                    final_a = process_data_to_long(df_a, "실적")
                    df_final = pd.concat([final_p, final_a], ignore_index=True)
                else:
                    # CSV인 경우
                    df_std = standardize_df(raw)
                    df_final = process_data_to_long(df_std, "실적")
            except:
                st.error("데이터 처리 중 오류가 발생했습니다.")

    # 🟢 2. 공급량 예측 모드 (파일 1개 필요)
    else:
        with st.sidebar:
            st.warning("📂 **[공급량실적_계획_실적_MJ.xlsx]** 파일 업로드")
            st.caption("시트: 공급량_실적, 공급량_계획 포함")
            up_mj = st.file_uploader("공급량 통합 파일", type=["xlsx", "csv"], key="mj_up")
            
        # 로드
        raw = load_data_super_robust(FILE_SUPPLY_MJ, up_mj)
        
        if raw is not None:
            st.sidebar.success("✅ 파일 연결됨")
            
            # 시트별 데이터 로드
            df_hist = standardize_df(raw, "공급량_실적")
            df_plan = standardize_df(raw, "공급량_계획")
            
            # CSV로 들어와서 시트 구분이 안 될 경우 (예외처리)
            if df_hist is None and not isinstance(raw, pd.ExcelFile):
                df_hist = standardize_df(raw)
            
            # 병합
            long_h = process_data_to_long(df_hist, "실적")
            long_p = process_data_to_long(df_plan, "확정계획")
            
            df_final = pd.concat([long_h, long_p], ignore_index=True)
            
            if df_final.empty:
                st.error("🚨 데이터가 비어있습니다. '공급량_실적' 또는 '공급량_계획' 시트명을 확인해주세요.")
        else:
            st.info("👈 좌측에서 '공급량실적_계획_실적_MJ' 파일을 업로드하세요.")

    # 🟢 메인 화면 렌더링
    if not df_final.empty:
        with st.sidebar:
            st.markdown("---")
            all_years = sorted(df_final['연'].unique())
            default_yrs = [y for y in all_years if y <= 2025]
            if not default_yrs: default_yrs = all_years
            
            st.markdown("**📅 분석 대상 연도**")
            train_years = st.multiselect("연도 선택", all_years, default=default_yrs, label_visibility="collapsed")
            
            # 필터링
            df_final = df_final[df_final['연'].isin(train_years) | (df_final['구분'] == '확정계획')]

        if "실적분석" in func:
            render_analysis(df_final, unit)
        else:
            # 공급량인 경우 2029년부터 예측
            start_year = 2029 if mode.startswith("2") else 2026
            render_prediction(df_final, unit, start_year)

if __name__ == "__main__":
    main()
