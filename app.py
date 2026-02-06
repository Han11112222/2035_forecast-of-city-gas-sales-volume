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

# 🟢 깃허브 설정 (백업용)
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
# 🟢 2. 핵심: 파일 로더 (시트 자동 찾기 기능 탑재)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_file_smart(file_obj, is_excel=True):
    """
    파일 객체를 받아서:
    1. 엑셀이면 -> 시트별로 쪼개서 딕셔너리 {'시트명': df} 반환
    2. CSV면 -> {'default': df} 반환
    """
    if file_obj is None: return None
    
    # 1. Excel로 시도
    try:
        excel = pd.ExcelFile(file_obj, engine='openpyxl')
        result = {}
        for sheet in excel.sheet_names:
            result[sheet] = excel.parse(sheet)
        return result
    except:
        # 실패하면 포인터 초기화 후 CSV로 시도
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        
    # 2. CSV (utf-8)
    try:
        df = pd.read_csv(file_obj, encoding='utf-8-sig')
        return {"default": df}
    except:
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        
    # 3. CSV (cp949)
    try:
        df = pd.read_csv(file_obj, encoding='cp949')
        return {"default": df}
    except:
        return None

def find_sheet_by_keyword(data_dict, keyword):
    """
    딕셔너리 키(시트명) 중에서 keyword가 포함된 시트의 데이터를 찾음
    예: '공급량_실적'을 찾으면 ' 공급량_실적 (수정)' 같은 시트도 찾아냄
    """
    if data_dict is None: return None
    
    # 1. 정확한 매칭
    if keyword in data_dict:
        return data_dict[keyword]
        
    # 2. 부분 매칭 (공백 제거 후 비교)
    for sheet_name, df in data_dict.items():
        if keyword in sheet_name.replace(" ", ""):
            return df
            
    # 3. 못 찾았으면 None
    return None

def clean_dataframe(df):
    """데이터프레임 전처리 (공통)"""
    if df is None: return pd.DataFrame()
    
    df = df.copy()
    
    # 컬럼명 공백 제거
    df.columns = df.columns.astype(str).str.strip()
    # Unnamed 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 날짜 처리 (MJ 파일 등)
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def convert_to_long(df, label, mapping):
    """분석 가능한 형태(Long Format)로 변환"""
    df = clean_dataframe(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns:
        return pd.DataFrame()
        
    records = []
    # 숫자형 변환
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

# ─────────────────────────────────────────────────────────
# 🟢 3. 깃허브 파일 가져오기 (백업)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def get_github_file(filename):
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/{quote(filename)}"
    try:
        r = requests.get(url)
        if r.status_code == 200: return io.BytesIO(r.content)
    except: pass
    return None

# ─────────────────────────────────────────────────────────
# 🟢 4. 분석 & 시각화
# ─────────────────────────────────────────────────────────
def render_dashboard(df, unit, mode_type):
    # 모드에 따라 예측 시작 연도 설정
    start_pred_year = 2029 if mode_type == "supply" else 2026
    
    st.markdown("---")
    
    # 1. 학습 기간(과거) 설정
    all_years = sorted(df['연'].unique())
    # 기본값: 2025년 이하
    default_yrs = [y for y in all_years if y <= 2025]
    if not default_yrs: default_yrs = all_years
    
    st.subheader("1️⃣ 분석 구간 설정")
    train_years = st.multiselect("학습(과거) 연도 선택", all_years, default=default_yrs)
    
    # 필터링: 선택된 연도 + 확정계획(미래)
    df_filtered = df[df['연'].isin(train_years) | (df['구분'].str.contains('계획'))]
    
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
        st.info(f"💡 선택한 과거 데이터와 확정 계획(2026~2028)을 바탕으로 **{start_pred_year}년부터 2035년까지** 예측합니다.")
        
        # 예측 로직
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
            
            # 과거 데이터
            for yr, val in zip(sub['연'], sub['값']): 
                results.append({'연': yr, '그룹': grp, '값': val, '구분': '실적/계획'})
            # 미래 예측
            for yr, val in zip(future_years.flatten(), pred):
                results.append({'연': yr, '그룹': grp, '값': val, '구분': '예측(AI)'})
                
        res_df = pd.DataFrame(results)
        
        fig_pred = px.line(res_df, x='연', y='값', color='그룹', line_dash='구분', markers=True)
        fig_pred.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green", annotation_text="예측 시작")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("##### 📋 연도별 상세 예측값")
        pred_data = res_df[res_df['구분'] == '예측(AI)']
        st.dataframe(pred_data.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 5. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"])
        unit = st.radio("단위", ["부피(천m³)", "열량(GJ)"])
        st.markdown("---")
        
    df_final = pd.DataFrame()
    
    # 🟢 1. 판매량 모드
    if mode.startswith("1"):
        with st.sidebar:
            st.warning("📂 **판매량(계획_실적).xlsx** 업로드")
            up = st.file_uploader("판매량 파일", type=["xlsx", "csv"], key="sales")
        
        # 파일 확보
        file_obj = up if up else get_github_file(FILE_SALES)
        
        if file_obj:
            # 데이터 로드 (딕셔너리 형태)
            data_dict = load_file_smart(file_obj)
            
            if data_dict:
                # 시트 찾기
                df_p = find_sheet_by_keyword(data_dict, "계획")
                df_a = find_sheet_by_keyword(data_dict, "실적")
                
                # CSV인 경우 'default' 키에 있음
                if df_p is None and df_a is None and "default" in data_dict:
                    df_a = data_dict["default"] # CSV는 보통 실적으로 간주
                
                long_p = convert_to_long(df_p, "계획", USE_COL_TO_GROUP)
                long_a = convert_to_long(df_a, "실적", USE_COL_TO_GROUP)
                df_final = pd.concat([long_p, long_a], ignore_index=True)
            
            if df_final.empty:
                st.error("🚨 데이터를 읽었으나 분석할 내용이 없습니다. (컬럼명 확인 필요)")
        else:
            st.info("👈 좌측에서 판매량 파일을 업로드하세요.")
            
    # 🟢 2. 공급량 모드 (형님 요청 로직)
    else:
        with st.sidebar:
            st.warning("📂 **공급량실적_계획_실적_MJ.xlsx** 업로드")
            st.caption("※ 시트: '공급량_실적', '공급량_계획' 포함 필수")
            up = st.file_uploader("공급량 파일", type=["xlsx", "csv"], key="supply")
            
        file_obj = up if up else get_github_file(FILE_SUPPLY_MJ)
        
        if file_obj:
            data_dict = load_file_smart(file_obj)
            
            if data_dict:
                # 1) 공급량_실적 (과거)
                df_hist = find_sheet_by_keyword(data_dict, "공급량_실적")
                # 2) 공급량_계획 (2026~2028 확정)
                df_plan = find_sheet_by_keyword(data_dict, "공급량_계획")
                
                # 디버깅 정보 표시
                if df_hist is None and df_plan is None:
                    if "default" in data_dict:
                        st.warning("⚠️ 엑셀 시트 구분을 실패했습니다. CSV 파일로 인식하여 '실적'으로 처리합니다.")
                        df_hist = data_dict["default"]
                    else:
                        st.error(f"🚨 시트를 찾을 수 없습니다. (발견된 시트: {list(data_dict.keys())})")
                
                long_h = convert_to_long(df_hist, "실적", USE_COL_TO_GROUP)
                long_p = convert_to_long(df_plan, "확정계획", USE_COL_TO_GROUP)
                
                df_final = pd.concat([long_h, long_p], ignore_index=True)
                
                if df_final.empty and (df_hist is not None or df_plan is not None):
                     st.error("🚨 시트는 찾았으나 컬럼 매핑에 실패했습니다. (컬럼명이 올바른지 확인하세요)")
                     if df_hist is not None: st.write("실적 데이터 컬럼:", df_hist.columns.tolist())
            else:
                st.error("🚨 파일을 읽을 수 없습니다. (암호화되었거나 손상됨)")
        else:
            st.info("👈 좌측에서 공급량 파일을 업로드하세요.")

    # 대시보드 렌더링
    if not df_final.empty:
        mode_key = "supply" if mode.startswith("2") else "sales"
        render_dashboard(df_final, unit, mode_key)

if __name__ == "__main__":
    main()
