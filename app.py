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
def load_file_smart(file_obj):
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

def find_sheet_by_keyword(data_dict, keyword_list):
    """
    딕셔너리 키(시트명) 중에서 keyword가 포함된 시트의 데이터를 찾음
    """
    if data_dict is None: return None
    
    for sheet_name, df in data_dict.items():
        clean_name = sheet_name.replace(" ", "")
        for key in keyword_list:
            if key in clean_name:
                return df
            
    # 시트가 하나뿐인데 못 찾았으면 그거라도 반환 (CSV 대비)
    if len(data_dict) == 1:
        return list(data_dict.values())[0]
        
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
def render_dashboard(df, unit, mode_type, sub_mode, temp_file=None):
    # 모드에 따라 예측 시작 연도 설정
    # 공급량: 2028년까지 확정 계획이 있으므로 2029년부터 예측
    # 판매량: 2025년까지 실적이므로 2026년부터 예측
    start_pred_year = 2029 if mode_type == "supply" else 2026
    
    # 1. 학습 기간(과거) 설정
    all_years = sorted(df['연'].unique())
    # 기본값: 2025년 이하
    default_yrs = [y for y in all_years if y <= 2025]
    if not default_yrs: default_yrs = all_years
    
    if "실적분석" in sub_mode:
        st.subheader("1️⃣ 분석 구간 설정")
        train_years = st.multiselect("분석 연도 선택", all_years, default=default_yrs)
        
        # 필터링
        df_filtered = df[df['연'].isin(train_years) | (df['구분'].str.contains('계획'))]
        
        if df_filtered.empty: st.warning("데이터 없음"); return

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

    elif "2035 예측" in sub_mode:
        st.subheader(f"🔮 2035 장기 예측 (기준: {start_pred_year}년부터)")
        
        # 학습 데이터: 예측 시작년도 이전 (실적 + 확정계획 포함)
        train_df = df[df['연'] < start_pred_year]
        if train_df.empty: st.warning("학습 데이터 부족"); return
        
        st.info(f"ℹ️ 학습 데이터 구간: {int(train_df['연'].min())}년 ~ {int(train_df['연'].max())}년")
        
        method = st.radio("예측 알고리즘", ["선형 회귀", "2차 곡선", "CAGR"], horizontal=True)
        
        train_grp = df.groupby(['연', '그룹'])['값'].sum().reset_index()
        train_grp = train_grp[train_grp['연'] < start_pred_year] # 학습용 다시 필터링
        
        groups = train_grp['그룹'].unique()
        future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
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

    elif "가정용" in sub_mode:
        st.subheader("🏠 가정용 정밀 분석")
        if temp_file is None:
            st.warning("⚠️ 좌측에서 기온 데이터(.csv)를 업로드해주세요.")
            return
            
        # 기온 데이터 처리
        temp_dict = load_file_smart(temp_file)
        if temp_dict:
            df_temp = list(temp_dict.values())[0] # 첫번째 시트
            df_temp = clean_dataframe(df_temp)
            cols = [c for c in df_temp.columns if "기온" in c]
            
            if cols:
                mon_temp = df_temp.groupby(['연', '월'])[cols[0]].mean().reset_index()
                mon_temp.rename(columns={cols[0]: '평균기온'}, inplace=True)
                
                df_home = df[df['그룹'] == '가정용'].groupby(['연', '월'])['값'].sum().reset_index()
                df_merged = pd.merge(df_home, mon_temp, on=['연', '월'], how='inner')
                
                if not df_merged.empty:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fig = px.scatter(df_merged, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("상관계수", f"{df_merged['평균기온'].corr(df_merged['값']):.2f}")
                else: st.warning("날짜가 일치하는 데이터가 없습니다.")
            else: st.error("기온 컬럼을 찾을 수 없습니다.")

# ─────────────────────────────────────────────────────────
# 🟢 5. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        # 1. 메인 모드 선택
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"], index=0)
        
        # 2. 서브 기능 선택 (모든 기능 복구)
        sub_mode = st.radio("기능 선택", ["실적분석", "2035 예측", "가정용 정밀 분석"])
        
        unit = st.radio("단위 선택", ["부피(천m³)", "열량(GJ)"])
        st.markdown("---")
        
    df_final = pd.DataFrame()
    up_temp = None
    
    # 🟢 [판매량 모드] - 기존 그대로 유지
    if mode.startswith("1"):
        with st.sidebar:
            st.info("📂 **[판매량(계획_실적).xlsx]** 업로드")
            up_sales = st.file_uploader("판매량 파일", type=["xlsx", "csv"], key="sales")
            if "가정용" in sub_mode:
                up_temp = st.file_uploader("기온 파일", type=["xlsx", "csv"], key="temp1")
        
        # 파일 확보 (업로드 없으면 깃허브)
        file_obj = up_sales if up_sales else get_github_file(FILE_SALES)
        
        if file_obj:
            data_dict = load_file_smart(file_obj)
            if data_dict:
                df_plan = find_sheet_by_keyword(data_dict, ["계획"])
                df_act = find_sheet_by_keyword(data_dict, ["실적"])
                
                # CSV 예외처리
                if df_plan is None and df_act is None and "default" in data_dict:
                    df_act = data_dict["default"]
                
                long_p = convert_to_long(df_plan, "계획", USE_COL_TO_GROUP)
                long_a = convert_to_long(df_act, "실적", USE_COL_TO_GROUP)
                df_final = pd.concat([long_p, long_a], ignore_index=True)
            
            if df_final.empty: st.error("데이터 로드 실패 (판매량)")
        else:
            st.info("👈 판매량 파일을 업로드하세요.")
            
    # 🟢 [공급량 모드] - MJ 파일 하나로 통합 (형님 요청 반영)
    else:
        with st.sidebar:
            st.info("📂 **[공급량실적_계획_실적_MJ.xlsx]** 업로드")
            st.caption("※ 시트: '공급량_실적'(과거), '공급량_계획'(26~28년)")
            up_mj = st.file_uploader("공급량 통합 파일", type=["xlsx", "csv"], key="supply")
            if "가정용" in sub_mode:
                up_temp = st.file_uploader("기온 파일", type=["xlsx", "csv"], key="temp2")
            
        file_obj = up_mj if up_mj else get_github_file(FILE_SUPPLY_MJ)
        
        if file_obj:
            data_dict = load_file_smart(file_obj)
            if data_dict:
                # 1) 공급량_실적 (과거)
                df_hist = find_sheet_by_keyword(data_dict, ["공급량_실적", "실적"])
                # 2) 공급량_계획 (2026~2028)
                df_plan = find_sheet_by_keyword(data_dict, ["공급량_계획", "공급량_계", "계획"])
                
                # 시트 못 찾았을 때 디버깅
                if df_hist is None and df_plan is None:
                    if "default" in data_dict:
                        # CSV로 들어온 경우 -> 그냥 실적으로 간주
                        df_hist = data_dict["default"]
                        st.caption("⚠️ 시트 구분 없음. 단일 데이터로 처리합니다.")
                    else:
                        st.error(f"🚨 시트 찾기 실패. 발견된 시트: {list(data_dict.keys())}")
                
                long_h = convert_to_long(df_hist, "실적", USE_COL_TO_GROUP)
                long_p = convert_to_long(df_plan, "확정계획", USE_COL_TO_GROUP)
                
                df_final = pd.concat([long_h, long_p], ignore_index=True)
                
                if df_final.empty:
                    st.error("데이터 로드 실패 (공급량). 컬럼명을 확인해주세요.")
        else:
            st.info("👈 공급량 파일을 업로드하세요.")

    # ── 메인 화면 렌더링 ──
    if not df_final.empty:
        mode_key = "supply" if mode.startswith("2") else "sales"
        render_dashboard(df_final, unit, mode_key, sub_mode, up_temp)

if __name__ == "__main__":
    main()
