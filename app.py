import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정 & 한글 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 [매핑 테이블] 형님 파일의 모든 컬럼명 변수 대응 (공백 있어도 처리됨)
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
# 🟢 2. 파일 로딩 & 전처리 (강력해진 기능)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_excel_file(uploaded_file):
    """업로드된 파일을 엑셀(시트별) 또는 CSV로 읽어오는 만능 함수"""
    if uploaded_file is None: return None
    
    # 1. 엑셀로 시도
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        sheets = {}
        for name in excel.sheet_names:
            sheets[name] = excel.parse(name)
        return sheets # 딕셔너리 반환 {'시트명': 데이터프레임}
    except:
        # 2. CSV로 시도 (엑셀 파일 형식이지만 실제론 텍스트인 경우 대응)
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            return {"default": df}
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='cp949')
                return {"default": df}
            except:
                return None

def standardize_dataframe(df):
    """데이터프레임의 컬럼명을 깨끗하게 청소 (공백 제거 등)"""
    if df is None: return pd.DataFrame()
    
    df = df.copy()
    
    # 1. 컬럼명 앞뒤 공백 제거 (이게 핵심!)
    df.columns = df.columns.astype(str).str.strip()
    
    # 2. 불필요한 컬럼 삭제
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 3. 날짜 처리 (MJ 파일 대응)
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def convert_to_long_format(df, label_name):
    """분석하기 좋은 형태(Long Format)로 변환"""
    df = standardize_dataframe(df)
    
    if df.empty or '연' not in df.columns or '월' not in df.columns:
        return pd.DataFrame()
        
    records = []
    # 숫자형 변환
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    for col in df.columns:
        # 매핑 확인
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

def find_sheet(data_dict, keywords):
    """시트 이름에 특정 단어(keywords)가 포함된 데이터를 찾음"""
    if not data_dict: return None
    
    for sheet_name, df in data_dict.items():
        # 시트 이름의 공백을 제거하고 비교
        clean_name = sheet_name.replace(" ", "")
        for key in keywords:
            if key in clean_name:
                return df
    
    # 못 찾았는데 시트가 하나뿐이면 그거라도 반환 (CSV인 경우)
    if len(data_dict) == 1:
        return list(data_dict.values())[0]
        
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 및 시각화 함수들
# ─────────────────────────────────────────────────────────
def render_analysis_tab(df):
    st.markdown("#### 📈 월별 추이")
    # 월별 합계
    mon_grp = df.groupby(['연', '월'])['값'].sum().reset_index()
    fig = px.line(mon_grp, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 용도별 구성비")
    yr_grp = df.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(yr_grp, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📋 상세 데이터 보기"):
        st.dataframe(df.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"))

def render_prediction_tab(df, start_year):
    st.markdown(f"#### 🔮 2035 장기 예측 (기준: {start_year}년부터 예측)")
    
    # 학습 데이터: 예측 시작년도 이전 (실적 + 확정계획 포함)
    train_df = df[df['연'] < start_year]
    
    if train_df.empty:
        st.warning("예측을 위한 과거 데이터가 부족합니다.")
        return
        
    method = st.radio("예측 방법", ["선형 회귀", "2차 곡선", "로그 추세", "지수 평활", "CAGR"], horizontal=True)
    
    # 그룹별 예측
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
        
        # 모델링
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
            
        pred = [max(0, p) for p in pred] # 음수 제거
        
        # 데이터 합치기
        for yr, val in zip(sub['연'], sub['값']): 
            results.append({'연': yr, '그룹': grp, '값': val, '구분': '실적/계획'})
        for yr, val in zip(future_years.flatten(), pred): 
            results.append({'연': yr, '그룹': grp, '값': val, '구분': '예측(AI)'})
            
    res_df = pd.DataFrame(results)
    
    fig = px.line(res_df, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    fig.add_vline(x=start_year-0.5, line_dash="dash", line_color="green", annotation_text="예측 시작")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 예측 데이터 다운로드"):
        pred_only = res_df[res_df['구분'] == '예측(AI)']
        st.dataframe(pred_only.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"))

def render_household_tab(df, temp_file):
    st.markdown("#### 🏠 가정용 정밀 분석 (기온 영향)")
    
    if temp_file is None:
        st.warning("⚠️ 기온 데이터 파일(.csv)을 업로드해야 분석이 가능합니다.")
        return
        
    # 기온 데이터 로드
    temp_dict = load_excel_file(temp_file)
    if temp_dict:
        # 첫 번째 시트나 default 사용
        df_temp = list(temp_dict.values())[0]
        df_temp = standardize_dataframe(df_temp)
        
        # 기온 컬럼 찾기
        cols = [c for c in df_temp.columns if "기온" in c]
        if not cols:
            st.error("기온 파일에 '기온' 컬럼이 없습니다.")
            return
            
        # 월별 평균 기온
        monthly_temp = df_temp.groupby(['연', '월'])[cols[0]].mean().reset_index()
        monthly_temp.rename(columns={cols[0]: '평균기온'}, inplace=True)
        
        # 가정용 데이터 필터링
        df_home = df[df['그룹'] == '가정용'].groupby(['연', '월'])['값'].sum().reset_index()
        
        # 병합
        df_merged = pd.merge(df_home, monthly_temp, on=['연', '월'], how='inner')
        
        if not df_merged.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.scatter(df_merged, x='평균기온', y='값', color='연', trendline="ols", 
                               title=f"기온에 따른 가정용 판매량 변화")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                corr = df_merged['평균기온'].corr(df_merged['값'])
                st.metric("기온과의 상관계수", f"{corr:.2f}")
                st.caption("-1에 가까울수록 기온이 낮으면 사용량이 늘어남을 의미")
        else:
            st.warning("기온 데이터와 판매량 데이터의 날짜가 일치하지 않습니다.")

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 실행 (UI 및 로직 통합)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    # ── 사이드바 설정 (기능 원상복구) ──
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"])
        
        # 🟢 형님이 원하시던 모든 기능 탭 부활
        func = st.radio("기능 선택", ["실적분석", "2035 예측", "가정용 정밀 분석"])
        
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
    
    df_final = pd.DataFrame()
    
    # 🟢 1. 판매량 모드
    if mode.startswith("1"):
        with st.sidebar:
            st.info("📂 **[판매량(계획_실적).xlsx]** 업로드")
            up_sales = st.file_uploader("판매량 파일", type=["xlsx", "csv"], key="sales")
            
        if up_sales:
            data = load_excel_file(up_sales)
            if data:
                # '계획'이 들어간 시트와 '실적'이 들어간 시트 찾기
                df_plan = find_sheet(data, ["계획"])
                df_act = find_sheet(data, ["실적"])
                
                # 변환
                long_p = convert_to_long_format(df_plan, "계획")
                long_a = convert_to_long_format(df_act, "실적")
                
                df_final = pd.concat([long_p, long_a], ignore_index=True)
                
                if df_final.empty:
                    st.error("데이터를 읽었지만 비어있습니다. 컬럼명을 확인해주세요.")

    # 🟢 2. 공급량 모드 (파일 1개로 통합)
    else:
        with st.sidebar:
            st.info("📂 **[공급량실적_계획_실적_MJ.xlsx]** 업로드")
            st.caption("※ 이 파일 하나만 올리면 됩니다.")
            up_mj = st.file_uploader("공급량 통합 파일", type=["xlsx", "csv"], key="supply")
            
        if up_mj:
            data = load_excel_file(up_mj)
            if data:
                # 1) 공급량_실적 (과거)
                df_hist = find_sheet(data, ["공급량_실적", "실적"])
                # 2) 공급량_계획 (2026~2028)
                df_plan = find_sheet(data, ["공급량_계획", "공급량_계", "계획"])
                
                long_h = convert_to_long_format(df_hist, "실적")
                long_p = convert_to_long_format(df_plan, "확정계획")
                
                df_final = pd.concat([long_h, long_p], ignore_index=True)
                
                if df_final.empty:
                    st.error("데이터가 없습니다. 시트명('공급량_실적', '공급량_계획')을 확인해주세요.")

    # ── 메인 화면 렌더링 ──
    if not df_final.empty:
        # 연도 필터링
        with st.sidebar:
            st.markdown("---")
            all_years = sorted(df_final['연'].unique())
            # 기본 선택: 2025년 이하
            default_yrs = [y for y in all_years if y <= 2025]
            if not default_yrs: default_yrs = all_years
            
            st.markdown("**📅 분석 연도 설정**")
            train_years = st.multiselect("연도 선택", all_years, default=default_yrs, label_visibility="collapsed")
            
            # 분석 대상: 선택된 연도 OR 미래 계획 데이터
            df_target = df_final[df_final['연'].isin(train_years) | (df_final['구분'].str.contains('계획'))]

        # 기능별 화면 표시
        if "실적분석" in func:
            render_analysis_tab(df_target)
            
        elif "2035 예측" in func:
            # 공급량이면 2029년부터, 판매량이면 2026년부터 예측 시작
            start_year = 2029 if mode.startswith("2") else 2026
            render_prediction_tab(df_target, start_year)
            
        elif "가정용" in func:
            with st.sidebar:
                st.markdown("---")
                up_temp = st.file_uploader("🌡️ 기온 데이터(.csv) 업로드", type=["csv", "xlsx"])
            render_household_tab(df_target, up_temp)
            
    else:
        st.info("👈 좌측 사이드바에서 파일을 업로드하면 분석이 시작됩니다.")

if __name__ == "__main__":
    main()
