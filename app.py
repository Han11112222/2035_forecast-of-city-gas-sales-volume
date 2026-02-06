import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 [매핑 테이블]
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    "일반용": "영업용", "일반용(1)": "영업용", "일반용(2)": "영업용", 
    "영업용_일반용1": "영업용", "영업용_일반용2": "영업용", 
    "일반용1(영업)": "영업용", "일반용2(영업)": "영업용", "일반용1": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", 
    "업무용_냉난방": "업무용", "업무용_주한미군": "업무용", 
    "일반용1(업무)": "업무용", "일반용2(업무)": "업무용",
    "산업용": "산업용", "수송용(CNG)": "수송용", "수송용(BIO)": "수송용", "CNG": "수송용", "BIO": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용", "열전용설비용(주택외)": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 🟢 2. 파일 로더 (멀티 파일, 엑셀/CSV 자동 처리)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_files_smart(uploaded_files):
    """
    업로드된 파일(들)을 읽어서 {'파일명_시트명': DataFrame} 형태로 반환
    """
    if not uploaded_files: return {}
    data_dict = {}
    
    if not isinstance(uploaded_files, list): uploaded_files = [uploaded_files]
        
    for file in uploaded_files:
        # 1. 엑셀 시도
        try:
            excel = pd.ExcelFile(file, engine='openpyxl')
            for sheet in excel.sheet_names:
                data_dict[f"{file.name}_{sheet}"] = excel.parse(sheet)
        except:
            # 2. CSV 시도
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                data_dict[f"{file.name}_csv"] = df
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp949')
                    data_dict[f"{file.name}_csv"] = df
                except: pass
    return data_dict

def standardize_df(df):
    if df is None: return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip() # 공백 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # 날짜 처리
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
    return df

def convert_to_long(df, label):
    df = standardize_df(df)
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

def find_df_by_keyword(data_dict, keywords):
    for key, df in data_dict.items():
        clean_key = key.replace(" ", "")
        for k in keywords:
            if k in clean_key: return df
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 및 시각화
# ─────────────────────────────────────────────────────────
def render_dashboard(df, unit, mode_type, sub_mode, temp_file=None):
    # 예측 시작 연도: 공급량은 2029년(28년까지 계획있음), 판매량은 2026년
    start_pred_year = 2029 if mode_type == "supply" else 2026
    
    all_years = sorted(df['연'].unique())
    default_yrs = [y for y in all_years if y <= 2025]
    if not default_yrs: default_yrs = all_years
    
    if "실적분석" in sub_mode:
        st.subheader("📊 실적 및 계획 분석")
        train_years = st.multiselect("분석 연도 선택", all_years, default=default_yrs)
        df_viz = df[df['연'].isin(train_years) | (df['구분'].str.contains('계획'))]
        
        if df_viz.empty: st.warning("데이터 없음"); return
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📈 월별 추이")
            mon_grp = df_viz.groupby(['연', '월'])['값'].sum().reset_index()
            fig = px.line(mon_grp, x='월', y='값', color='연', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("##### 🧱 용도별 구성")
            yr_grp = df_viz.groupby(['연', '그룹'])['값'].sum().reset_index()
            fig2 = px.bar(yr_grp, x='연', y='값', color='그룹')
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_viz.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"), use_container_width=True)

    elif "2035 예측" in sub_mode:
        st.subheader(f"🔮 2035 장기 예측 (기준: {start_pred_year}년부터)")
        st.info(f"ℹ️ **과거 실적 + 확정 계획(2026~2028)** 데이터를 모두 학습하여 2035년까지 예측합니다.")
        
        # 학습 데이터: 예측 시작년도 이전 데이터 전체
        train_df = df[df['연'] < start_pred_year]
        if train_df.empty: st.warning("학습 데이터 부족"); return
        
        method = st.radio("예측 알고리즘", ["선형 회귀", "2차 곡선", "CAGR"], horizontal=True)
        
        train_grp = df.groupby(['연', '그룹'])['값'].sum().reset_index()
        # 학습용 데이터만 다시 필터링
        train_grp = train_grp[train_grp['연'] < start_pred_year]
        
        groups = train_grp['그룹'].unique()
        future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
        results = []
        
        for grp in groups:
            sub = train_grp[train_grp['그룹'] == grp]
            if len(sub) < 2: continue
            X, y = sub['연'].values, sub['값'].values
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
            
            for yr, val in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '값': val, '구분': '실적/계획'})
            for yr, val in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '값': val, '구분': '예측(AI)'})
            
        res_df = pd.DataFrame(results)
        fig = px.line(res_df, x='연', y='값', color='그룹', line_dash='구분', markers=True)
        fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green", annotation_text="예측 시작")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 예측 데이터 보기"):
            st.dataframe(res_df[res_df['구분']=='예측(AI)'].pivot_table(index='연', columns='그룹', values='값').style.format("{:,.0f}"))

    elif "가정용" in sub_mode:
        st.subheader("🏠 가정용 정밀 분석")
        if not temp_file: st.error("🌡️ 기온 파일을 업로드해주세요."); return
        
        t_dict = load_files_smart([temp_file])
        if t_dict:
            df_t = list(t_dict.values())[0]
            df_t = standardize_df(df_t)
            cols = [c for c in df_t.columns if "기온" in c]
            if cols:
                mon_t = df_t.groupby(['연', '월'])[cols[0]].mean().reset_index()
                mon_t.rename(columns={cols[0]: '평균기온'}, inplace=True)
                df_h = df[df['그룹']=='가정용'].groupby(['연', '월'])['값'].sum().reset_index()
                merged = pd.merge(df_h, mon_t, on=['연', '월'])
                if not merged.empty:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fig = px.scatter(merged, x='평균기온', y='값', trendline="ols", title="기온 vs 가정용 판매량")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.metric("상관계수", f"{merged['평균기온'].corr(merged['값']):.2f}")
                else: st.warning("날짜 일치 데이터 없음")

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 실행 (UI 구조 변경)
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    # ── 사이드바: 설정 및 파일 업로드 (항상 보임!) ──
    with st.sidebar:
        st.header("1. 분석 설정")
        mode = st.radio("분석 모드", ["1. 판매량 예측", "2. 공급량 예측"], index=1)
        sub_mode = st.radio("기능 선택", ["실적분석", "2035 예측", "가정용 정밀 분석"])
        unit = st.radio("단위", ["부피(천m?)", "열량(GJ)"])
        
        st.markdown("---")
        st.header("2. 파일 업로드")
        
        # 🟢 파일 업로더들을 항상 노출 (사라지지 않음)
        st.markdown("**(1) 판매량 파일**")
        up_sales = st.file_uploader("판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="sales", accept_multiple_files=True)
        
        st.markdown("**(2) 공급량 파일**")
        st.caption("공급량실적_계획_실적_MJ.xlsx")
        up_supply = st.file_uploader("공급량 통합 파일", type=["xlsx", "csv"], key="supply", accept_multiple_files=True)
        
        st.markdown("**(3) 기온 파일 (선택)**")
        up_temp = st.file_uploader("기온.csv", type=["xlsx", "csv"], key="temp")
    
    # ── 데이터 로드 및 처리 ──
    df_final = pd.DataFrame()
    
    # [모드 1] 판매량 예측
    if mode.startswith("1"):
        if up_sales:
            data = load_files_smart(up_sales)
            df_p = find_df_by_keyword(data, ["계획"])
            df_a = find_df_by_keyword(data, ["실적"])
            # CSV 예외: 파일 하나만 올렸는데 키워드 없으면 실적으로 간주
            if df_p is None and df_a is None and len(data) == 1:
                df_a = list(data.values())[0]
                
            lp = convert_to_long(df_p, "계획")
            la = convert_to_long(df_a, "실적")
            df_final = pd.concat([lp, la], ignore_index=True)
        else:
            st.info("👈 좌측에서 [판매량 파일]을 업로드해주세요.")
            
    # [모드 2] 공급량 예측
    else:
        if up_supply:
            data = load_files_smart(up_supply)
            # 1. 공급량_실적 (과거)
            df_hist = find_df_by_keyword(data, ["공급량_실적", "실적"])
            # 2. 공급량_계획 (2026~2028)
            df_plan = find_df_by_keyword(data, ["공급량_계획", "계획"])
            
            # CSV 예외: 파일 하나만 올렸는데 키워드 없으면 실적으로 간주
            if df_hist is None and df_plan is None and len(data) == 1:
                df_hist = list(data.values())[0]
                st.caption("⚠️ 시트 구분이 없어 전체를 '실적'으로 처리합니다.")

            lh = convert_to_long(df_hist, "실적")
            lp = convert_to_long(df_plan, "확정계획")
            df_final = pd.concat([lh, lp], ignore_index=True)
        else:
            st.info("👈 좌측에서 [공급량 파일]을 업로드해주세요.")

    # ── 대시보드 렌더링 ──
    if not df_final.empty:
        mode_key = "supply" if mode.startswith("2") else "sales"
        render_dashboard(df_final, unit, mode_key, sub_mode, up_temp)
    elif (mode.startswith("1") and up_sales) or (mode.startswith("2") and up_supply):
        st.error("🚨 데이터를 읽었으나 내용이 비어있습니다. 컬럼명이나 시트명을 확인해주세요.")

if __name__ == "__main__":
    main()
