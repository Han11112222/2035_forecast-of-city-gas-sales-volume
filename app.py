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
# 🟢 1. 기본 설정 & 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 깃허브 설정 (백업)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
BRANCH = "main" 

# 파일명 상수
FILE_SALES = "판매량(계획_실적).xlsx"       
FILE_SUPPLY_MJ = "공급량실적_계획_실적_MJ.xlsx"
FILE_TEMP = "기온.csv"

# 🟢 [매핑] 컬럼명 -> 표준 그룹 (모든 케이스 포함)
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
# 🟢 2. 스마트 파일 로더 (단일/다중 파일, 엑셀/CSV 모두 처리)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_files_smart(uploaded_files):
    """
    업로드된 파일 리스트를 받아서 
    {'파일명/시트명': DataFrame} 형태의 딕셔너리로 반환
    """
    if not uploaded_files: return {}
    
    data_dict = {}
    
    # 리스트가 아니라 단일 파일이면 리스트로 감쌈
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    for file in uploaded_files:
        # 1. 엑셀로 시도
        try:
            excel = pd.ExcelFile(file, engine='openpyxl')
            for sheet in excel.sheet_names:
                data_dict[f"{file.name}_{sheet}"] = excel.parse(sheet)
        except:
            # 2. CSV로 시도 (포인터 초기화)
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                data_dict[f"{file.name}_csv"] = df
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp949')
                    data_dict[f"{file.name}_csv"] = df
                except:
                    pass
                    
    return data_dict

def get_github_file(filename):
    """깃허브 백업 로드"""
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
    try:
        r = requests.get(url)
        if r.status_code == 200: return io.BytesIO(r.content)
    except: pass
    return None

def clean_dataframe(df):
    """데이터프레임 전처리"""
    if df is None: return pd.DataFrame()
    df = df.copy()
    
    # 공백 제거
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 날짜 처리
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def convert_to_long(df, label, mapping):
    """분석용 포맷 변환"""
    df = clean_dataframe(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns:
        return pd.DataFrame()
        
    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    for col in df.columns:
        group = mapping.get(col)
        if not group: continue
        
        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label
        sub['값'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def find_df_by_keyword(data_dict, keywords):
    """딕셔너리 키(파일명/시트명)에서 키워드 검색"""
    for key, df in data_dict.items():
        clean_key = key.replace(" ", "")
        for k in keywords:
            if k in clean_key:
                return df
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 및 예측 로직
# ─────────────────────────────────────────────────────────
def render_dashboard(df, unit, mode_type):
    start_pred_year = 2029 if mode_type == "supply" else 2026
    
    st.markdown("---")
    
    # 학습 기간 설정
    all_years = sorted(df['연'].unique())
    default_yrs = [y for y in all_years if y <= 2025]
    if not default_yrs: default_yrs = all_years
    
    st.subheader("1️⃣ 분석 구간 설정")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        train_years = st.multiselect("학습 연도(과거) 선택", all_years, default=default_yrs)
    
    # 필터링: 선택된 연도 + 확정계획
    df_filtered = df[df['연'].isin(train_years) | (df['구분'].str.contains('계획')) | (df['구분'].str.contains('확정'))]
    
    if df_filtered.empty:
        st.warning("선택된 데이터가 없습니다.")
        return

    # 탭 구성
    tab1, tab2 = st.tabs(["📊 실적/계획 분석", "🔮 2035년 예측"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📈 월별 추이")
            mon_grp = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
            fig = px.line(mon_grp, x='월', y='값', color='연', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("##### 🧱 용도별 구성")
            yr_grp = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
            fig2 = px.bar(yr_grp, x='연', y='값', color='그룹')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.dataframe(df_filtered.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"), use_container_width=True)

    with tab2:
        st.info(f"💡 과거 실적과 2026~2028 확정 계획을 합쳐 **{start_pred_year}년부터 2035년까지** 예측합니다.")
        
        train_grp = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
        groups = train_grp['그룹'].unique()
        future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
        results = []
        
        method = st.radio("예측 알고리즘", ["선형 회귀", "2차 곡선", "CAGR"], horizontal=True)
        
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
            else: # CAGR
                try: cagr = (y[-1]/y[0])**(1/len(y)) - 1
                except: cagr = 0
                pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
                
            pred = [max(0, p) for p in pred]
            
            for yr, val in zip(sub['연'], sub['값']): 
                results.append({'연': yr, '그룹': grp, '값': val, '구분': '실적/확정계획'})
            for yr, val in zip(future_years.flatten(), pred): 
                results.append({'연': yr, '그룹': grp, '값': val, '구분': '예측(AI)'})
                
        res_df = pd.DataFrame(results)
        
        fig_pred = px.line(res_df, x='연', y='값', color='그룹', line_dash='구분', markers=True)
        fig_pred.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green", annotation_text="예측 시작")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        with st.expander("📋 연도별 상세 예측값 보기"):
            pred_data = res_df[res_df['구분'] == '예측(AI)']
            st.dataframe(pred_data.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 실행 (기본 모드를 공급량으로 변경)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        # 🟢 [중요] 공급량 예측을 기본값(index=1)으로 설정!
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측(2035)"], index=1)
        unit = st.radio("단위", ["부피(천m³)", "열량(GJ)"])
        st.markdown("---")
        
    df_final = pd.DataFrame()
    
    # 🟢 2. 공급량 모드 (형님의 궁극적 목표)
    if "공급량" in mode:
        with st.sidebar:
            st.warning("📂 **[공급량실적_계획_실적_MJ.xlsx]** 파일을 업로드하세요.")
            st.caption("※ 파일 안에 '공급량_실적', '공급량_계획' 시트가 있거나, 따로 된 2개의 파일을 올려도 됩니다.")
            
            # 여러 파일 업로드 가능하게 설정 (CSV 분리된 경우 대비)
            up_files = st.file_uploader("파일 업로드 (여러 개 가능)", type=["xlsx", "csv"], accept_multiple_files=True)
            
        # 1. 파일 로드 (업로드 or 깃허브)
        if up_files:
            data_dict = load_files_smart(up_files)
        else:
            # 업로드 없으면 깃허브 백업 사용
            backup = get_github_file(FILE_SUPPLY_MJ)
            data_dict = load_files_smart([backup]) if backup else {}
            
        if data_dict:
            # 2. 키워드로 데이터 찾기 ('실적', '계획')
            df_hist = find_df_by_keyword(data_dict, ["공급량_실적", "실적"])
            df_plan = find_df_by_keyword(data_dict, ["공급량_계획", "공급량_계", "계획"])
            
            # 3. 디버깅 및 병합
            if df_hist is None and df_plan is None:
                st.error(f"🚨 시트나 파일을 찾을 수 없습니다. (읽은 목록: {list(data_dict.keys())})")
            else:
                long_h = convert_to_long(df_hist, "실적", USE_COL_TO_GROUP)
                long_p = convert_to_long(df_plan, "확정계획", USE_COL_TO_GROUP) # 26~28년 계획
                
                df_final = pd.concat([long_h, long_p], ignore_index=True)
                
                if df_final.empty:
                    st.error("🚨 데이터를 읽었으나 비어있습니다. 컬럼명을 확인하세요.")
                else:
                    st.sidebar.success(f"✅ 데이터 로드 성공 ({len(df_final)}건)")
        else:
            st.info("👈 좌측에서 공급량 파일을 업로드해주세요.")
            
        # 대시보드 렌더링
        if not df_final.empty:
            render_dashboard(df_final, unit, "supply")

    # 🟢 1. 판매량 모드
    else:
        with st.sidebar:
            st.info("📂 **[판매량(계획_실적).xlsx]** 업로드")
            up_sales = st.file_uploader("판매량 파일", type=["xlsx", "csv"], accept_multiple_files=True)
            
        if up_sales:
            data_dict = load_files_smart(up_sales)
        else:
            backup = get_github_file(FILE_SALES)
            data_dict = load_files_smart([backup]) if backup else {}
            
        if data_dict:
            df_plan = find_df_by_keyword(data_dict, ["계획"])
            df_act = find_df_by_keyword(data_dict, ["실적"])
            
            long_p = convert_to_long(df_plan, "계획", USE_COL_TO_GROUP)
            long_a = convert_to_long(df_act, "실적", USE_COL_TO_GROUP)
            
            df_final = pd.concat([long_p, long_a], ignore_index=True)
            
            if not df_final.empty:
                render_dashboard(df_final, unit, "sales")
            else:
                st.error("데이터 로드 실패.")
        else:
            st.info("👈 좌측에서 판매량 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
