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

# 🟢 매핑 테이블 (판매량 + 공급량 파일 컬럼 모두 포함)
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
# 🟢 2. 데이터 로드 및 전처리 (만능 로더)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_file_robust(uploaded_file):
    """엑셀/CSV 구분 없이 내용을 읽어서 딕셔너리 반환"""
    if uploaded_file is None: return None
    
    # 1. 엑셀로 시도
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        sheets = {name: excel.parse(name) for name in excel.sheet_names}
        return sheets
    except:
        # 2. CSV로 시도
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            return {"default": df}
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='cp949')
                return {"default": df}
            except: return None

def clean_df(df):
    """데이터프레임 표준화"""
    if df is None: return pd.DataFrame()
    df = df.copy()
    
    # 컬럼명 공백 제거
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 날짜 컬럼 처리 (MJ 파일 대응)
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def make_long_data(df, label):
    """분석용 포맷으로 변환"""
    df = clean_df(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns: return pd.DataFrame()
    
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
        sub['구분'] = label
        sub['값'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def find_sheet(data_dict, keywords):
    """시트 이름 검색"""
    if not data_dict: return None
    for name, df in data_dict.items():
        clean = name.replace(" ", "")
        for k in keywords:
            if k in clean: return df
    
    # 못 찾았는데 시트가 하나면 그거라도 반환 (CSV 대응)
    if len(data_dict) == 1: return list(data_dict.values())[0]
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 화면 (형님이 주신 코드 스타일 적용)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 실적 데이터만 필터링
    df_act = long_df[long_df['구분'].str.contains('실적')].copy()
    
    if df_act.empty: st.error("분석할 실적 데이터가 없습니다."); return
    
    all_years = sorted(df_act['연'].unique())
    selected_years = st.multiselect("연도 선택", options=all_years, default=all_years[-3:] if len(all_years)>=3 else all_years)
    
    if not selected_years: return
    
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")
    
    # [그래프 1] 월별 추이
    st.markdown(f"#### 📈 월별 추이")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### 📋 월별 상세 수치")
    st.dataframe(df_mon.pivot(index='월', columns='연', values='값').style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")
    
    # [그래프 2] 용도별 구성
    st.markdown(f"#### 🧱 용도별 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 용도별 상세 수치")
    piv = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 예측 화면 (형님이 주신 코드 스타일 적용)
# ─────────────────────────────────────────────────────────
def render_prediction_2035(long_df, unit_label, start_pred_year, train_years_selected):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 학습 데이터 필터링 (사용자가 선택한 연도만 사용)
    # 공급량의 경우 '확정계획(26~28)'은 무조건 학습에 포함해야 함
    df_train = long_df[
        (long_df['연'].isin(train_years_selected)) | 
        (long_df['구분'] == '확정계획')
    ].copy()
    
    if df_train.empty: st.warning("학습 데이터가 없습니다."); return
    
    st.markdown("##### 📊 추세 분석 모델 선택")
    pred_method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 로그 추세", "4. 지수 평활", "5. CAGR"], horizontal=True)
    
    # 예측 수행
    df_grp = df_train.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_grp['그룹'].unique()
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
    results = []
    
    for grp in groups:
        sub = df_grp[df_grp['그룹'] == grp]
        if len(sub) < 2: continue
        
        X = sub['연'].values
        y = sub['값'].values
        pred = []
        
        if "선형" in pred_method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in pred_method:
            try: z = np.polyfit(X, y, 2); p = np.poly1d(z); pred = p(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in pred_method:
            try: model = LinearRegression(); model.fit(np.log(X.reshape(-1,1)), y); pred = model.predict(np.log(future_years))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in pred_method:
            pred = [y[-1]] * len(future_years)
        else: # CAGR
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1
            except: cagr = 0
            pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
            
        pred = [max(0, p) for p in pred]
        
        # 결과 합치기
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': v, '구분': '실적/계획'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측'})
        
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 전체 장기 전망")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green", annotation_text="예측 시작")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 상세 예측 데이터")
    df_pred = df_res[df_res['구분'] == '예측']
    st.dataframe(df_pred.pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 5. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    # ── 사이드바 ──
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"])
        sub_mode = st.radio("기능 선택", ["1) 실적 분석", "2) 2035 예측"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        st.markdown("---")
        
        # 🟢 파일 업로더 항상 노출
        st.subheader("파일 업로드")
        up_sales = st.file_uploader("판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s")
        up_supply = st.file_uploader("공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        
        st.markdown("---")
    
    # ── 데이터 로드 및 처리 ──
    df_final = pd.DataFrame()
    start_year = 2026
    
    # [모드 1] 판매량 예측
    if mode.startswith("1"):
        if up_sales:
            data = load_file_robust(up_sales)
            if data:
                df_p = find_sheet(data, ["계획"])
                df_a = find_sheet(data, ["실적"])
                
                # CSV 예외처리
                if df_p is None and df_a is None and len(data) == 1:
                    df_a = list(data.values())[0]
                
                long_p = make_long_data(df_p, "계획")
                long_a = make_long_data(df_a, "실적")
                df_final = pd.concat([long_p, long_a], ignore_index=True)
        else:
            st.info("👈 좌측에서 [판매량 파일]을 업로드해주세요.")
            return

    # [모드 2] 공급량 예측
    else:
        start_year = 2029 # 공급량은 2029년부터 예측
        if up_supply:
            data = load_file_robust(up_supply)
            if data:
                # 1) 공급량_실적 (과거)
                df_hist = find_sheet(data, ["공급량_실적", "실적"])
                # 2) 공급량_계획 (2026~2028)
                df_plan = find_sheet(data, ["공급량_계획", "계획"])
                
                # CSV 예외처리
                if df_hist is None and df_plan is None and len(data) == 1:
                    df_hist = list(data.values())[0]
                
                long_h = make_long_data(df_hist, "실적")
                long_p = make_long_data(df_plan, "확정계획")
                df_final = pd.concat([long_h, long_p], ignore_index=True)
        else:
            st.info("👈 좌측에서 [공급량 파일]을 업로드해주세요.")
            return

    # ── 학습 연도 선택 (사이드바 하단에 배치) ──
    if not df_final.empty:
        with st.sidebar:
            st.markdown("**📅 학습 데이터 연도 설정**")
            all_years = sorted(df_final['연'].unique())
            # 기본적으로 2024년까지만 선택 (2025년 제외)
            default_yrs = [y for y in all_years if y < 2025] 
            
            train_years = st.multiselect(
                "학습 연도 선택 (왜곡 방지용)", 
                options=all_years, 
                default=default_yrs
            )
            st.caption("※ 2025년 데이터가 불완전하면 체크 해제하세요.")

        # ── 기능 실행 ──
        if "실적" in sub_mode:
            render_analysis_dashboard(df_final, unit)
        else:
            render_prediction_2035(df_final, unit, start_year, train_years)

if __name__ == "__main__":
    main()
