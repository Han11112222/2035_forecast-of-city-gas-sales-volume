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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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

# 🟢 [매핑] 컬럼명 -> 표준 그룹
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
# 🟢 2. 파일 로딩 (만능 처리)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_file_robust(uploaded_file):
    if uploaded_file is None: return None
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        sheets = {name: excel.parse(name) for name in excel.sheet_names}
        return sheets
    except:
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
    if df is None: return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
    return df

def make_long_data(df, label):
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
    if not data_dict: return None
    for name, df in data_dict.items():
        clean = name.replace(" ", "")
        for k in keywords:
            if k in clean: return df
    if len(data_dict) == 1: return list(data_dict.values())[0]
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 화면 (최근 5년 디폴트 적용)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['구분'].str.contains('실적')].copy()
    if df_act.empty: st.error("실적 데이터 없음"); return
    
    all_years = sorted(df_act['연'].unique())
    
    # 🔴 [수정됨] 최근 5년치 데이터를 디폴트로 설정
    if len(all_years) >= 5:
        default_years = all_years[-5:]
    else:
        default_years = all_years
        
    selected_years = st.multiselect("연도 선택", options=all_years, default=default_years)
    
    if not selected_years: return
    
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 📈 월별 추이")
        mon_grp = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
        fig1 = px.line(mon_grp, x='월', y='값', color='연', markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown(f"#### 🧱 용도별 구성비")
        yr_grp = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(yr_grp, x='연', y='값', color='그룹', text_auto='.2s')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 상세 수치")
    st.dataframe(df_filtered.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 예측 화면 (2026~2028 계획 반영 + 스택 누적 + 알고리즘 추가)
# ─────────────────────────────────────────────────────────
def render_prediction_2035(long_df, unit_label, start_pred_year, train_years_selected):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 1. 학습 데이터 준비
    # 사용자가 선택한 과거 실적 + '확정계획'(26~28년)
    df_train = long_df[
        (long_df['연'].isin(train_years_selected)) | 
        (long_df['구분'] == '확정계획')
    ].copy()
    
    if df_train.empty: st.warning("학습 데이터가 부족합니다."); return
    
    # 2. 알고리즘 선택
    st.markdown("##### 📊 추세 분석 모델 선택")
    pred_method = st.radio("방법", [
        "1. 선형 회귀 (Linear)", "2. 2차 곡선 (Quadratic)", "3. 3차 곡선 (Cubic)",
        "4. 로그 추세 (Log)", "5. 지수 평활 (Holt)", "6. CAGR (성장률)"
    ], horizontal=True)
    
    # 3. 예측 수행
    df_grp = df_train.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_grp['그룹'].unique()
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1) # 2029~2035 (공급량 기준)
    results = []
    
    for grp in groups:
        sub = df_grp[df_grp['그룹'] == grp]
        if len(sub) < 2: continue
        
        X = sub['연'].values.reshape(-1, 1)
        y = sub['값'].values
        pred = []
        
        try:
            if "선형" in pred_method:
                model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
            elif "2차" in pred_method:
                model = make_pipeline(PolynomialFeatures(2), LinearRegression()); model.fit(X, y); pred = model.predict(future_years)
            elif "3차" in pred_method:
                model = make_pipeline(PolynomialFeatures(3), LinearRegression()); model.fit(X, y); pred = model.predict(future_years)
            elif "로그" in pred_method:
                X_idx = np.arange(1, len(X) + 1).reshape(-1, 1)
                X_future = np.arange(len(X) + 1, len(X) + 1 + len(future_years)).reshape(-1, 1)
                model = LinearRegression(); model.fit(np.log(X_idx), y); pred = model.predict(np.log(X_future))
            elif "지수" in pred_method:
                fit = np.polyfit(X.flatten(), np.log(y + 1), 1)
                pred = np.exp(fit[1] + fit[0] * future_years.flatten())
            else: # CAGR
                cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1
                pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
        except:
            model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
            
        pred = [max(0, p) for p in pred]
        
        # [데이터 합치기]
        # (1) 과거 실적 및 확정계획 (그대로 사용)
        for yr, v in zip(sub['연'], sub['값']): 
            # 구분 라벨링 정교화
            label = '실적'
            if yr >= 2026 and yr <= 2028: label = '확정계획(26~28)'
            elif yr >= start_pred_year: label = '예측'
            
            results.append({'연': yr, '그룹': grp, '값': v, '구분': label})
            
        # (2) 미래 예측 (29년 이후)
        for yr, v in zip(future_years.flatten(), pred): 
            results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측(AI)'})
        
    df_res = pd.DataFrame(results)
    
    # 4. 시각화
    st.markdown("---")
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green", annotation_text="AI 예측 시작")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🧱 연도별 예측량 구성 (누적 스택)")
    # 예측 데이터만? 아니면 전체? -> 흐름을 보려면 전체가 좋음
    fig_stack = px.bar(df_res, x='연', y='값', color='그룹', title="연도별 공급량 구성비 (실적 -> 계획 -> AI예측)", text_auto='.2s')
    st.plotly_chart(fig_stack, use_container_width=True)
    
    with st.expander("📋 연도별 상세 데이터 확인"):
        piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum')
        piv['합계'] = piv.sum(axis=1)
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 5. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        # 1. 명칭 변경 (예측 제거)
        mode = st.radio("분석 모드", ["1. 판매량", "2. 공급량"], index=1) # 공급량 기본
        
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        
        # 2. 단위 기본값 변경 (GJ 우선)
        unit = st.radio("단위 선택", ["열량 (GJ)", "부피 (천m³)"])
        st.markdown("---")
        
        # 파일 업로더
        st.subheader("파일 업로드")
        up_sales = st.file_uploader("1. 판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s")
        up_supply = st.file_uploader("2. 공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        
        st.markdown("---")
    
    df_final = pd.DataFrame()
    start_year = 2026
    
    # [모드 1] 판매량
    if mode.startswith("1"):
        if up_sales:
            data = load_file_robust(up_sales)
            if data:
                df_p = find_sheet(data, ["계획"])
                df_a = find_sheet(data, ["실적"])
                if df_p is None and df_a is None and len(data) == 1: df_a = list(data.values())[0]
                long_p = make_long_data(df_p, "계획")
                long_a = make_long_data(df_a, "실적")
                df_final = pd.concat([long_p, long_a], ignore_index=True)
        else:
            st.info("👈 [판매량 파일]을 업로드하세요.")
            return

    # [모드 2] 공급량
    else:
        # 공급량은 2028년까지 확정 계획이 있으므로 2029년부터 AI 예측
        start_year = 2029 
        if up_supply:
            data = load_file_robust(up_supply)
            if data:
                # 3. 공급량 파일 시트 자동 인식
                df_hist = find_sheet(data, ["공급량_실적", "실적"])
                df_plan = find_sheet(data, ["공급량_계획", "계획"]) # 26~28년 데이터
                
                if df_hist is None and df_plan is None and len(data) == 1:
                    df_hist = list(data.values())[0]
                
                long_h = make_long_data(df_hist, "실적")
                long_p = make_long_data(df_plan, "확정계획")
                df_final = pd.concat([long_h, long_p], ignore_index=True)
        else:
            st.info("👈 [공급량 파일]을 업로드하세요.")
            return

    # ── 학습 연도 선택 (사이드바 하단) ──
    if not df_final.empty:
        with st.sidebar:
            st.markdown("### 📅 데이터 학습 기간 설정")
            all_years = sorted(df_final['연'].unique())
            # 기본값: 2024년까지만 선택 (2025 제외, 26~28은 확정계획이라 자동 포함됨)
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
        elif "2035" in sub_mode:
            render_prediction_2035(df_final, unit, start_year, train_years)
        elif "가정용" in sub_mode:
            with st.sidebar:
                up_t = st.file_uploader("기온 파일(.csv)", type=["csv", "xlsx"])
            st.info("기온 데이터 업로드 시 분석 가능")

if __name__ == "__main__":
    main()
