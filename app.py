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
# 🟢 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        # 폰트 설정 시도 (없으면 기본 폰트 사용)
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 설정 정보
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
SALES_FILE_NAME = "판매량(계획_실적).xlsx"
TEMP_FILE_NAME = "기온_198001_202512.xlsx" # 또는 기온.csv

# 🟢 용도 매핑 (이 컬럼명들이 엑셀 헤더에 반드시 있어야 함!)
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 (진단 기능 포함)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_github_file(filename, file_type='xlsx'):
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if file_type == 'xlsx':
            return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
        else: # csv
            try: return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            except: return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
    except Exception as e:
        return None

def _clean_base(df):
    out = df.copy()
    out = out.loc[:, ~out.columns.str.contains('^Unnamed')]
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long(plan_df, actual_df):
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)
    records = []
    
    # 디버깅용: 컬럼명 확인
    # st.sidebar.write(f"계획 시트 컬럼: {list(plan_df.columns)}")
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue
            
            # 공백 제거 후 매핑 시도
            clean_col = col.strip() 
            group = USE_COL_TO_GROUP.get(clean_col)
            
            if not group: continue
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = clean_col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    return long_df.dropna(subset=["연", "월"])

def load_temp_universal(file_obj):
    try:
        if hasattr(file_obj, 'name'):
            if file_obj.name.endswith('.csv'):
                try: df = pd.read_csv(file_obj, encoding='utf-8-sig')
                except: df = pd.read_csv(file_obj, encoding='cp949')
            else: df = pd.read_excel(file_obj, engine='openpyxl').parse(0)
        else: return None # 파일 없음
        
        if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        
        t_col = [c for c in df.columns if "기온" in c][0]
        df_mon = df.groupby(['연', '월'])[t_col].mean().reset_index()
        df_mon.rename(columns={t_col: '평균기온'}, inplace=True)
        return df_mon
    except: return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 데이터 확인
    if long_df.empty:
        st.error("데이터 변환 결과가 비어있습니다. 엑셀 컬럼명(취사용, 업무난방용 등)을 확인해주세요.")
        return

    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    if df_act.empty: 
        st.warning("실적 데이터가 없습니다. (계획 데이터만 있거나 필터링됨)")
        return
    
    all_years = sorted(df_act['연'].unique())
    st.markdown("##### 📅 그래프에 표시할 연도 선택")
    selected_years = st.multiselect("연도 선택", all_years, default=all_years[-3:] if len(all_years)>=3 else all_years, label_visibility="collapsed")
    if not selected_years: return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 표 1
    st.markdown("##### 📋 상세 데이터")
    piv_mon = df_mon.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(piv_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2
    st.markdown(f"#### 🧱 연도별 용도 구성비")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    # 표 2
    st.markdown("##### 📋 상세 데이터")
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측
# ─────────────────────────────────────────────────────────
def holt_linear_trend(y, n_preds):
    if len(y) < 2: return np.full(n_preds, y[0])
    alpha, beta = 0.8, 0.2
    level, trend = y[0], y[1] - y[0]
    for val in y[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + i * trend for i in range(1, n_preds + 1)])

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    st.markdown("##### 🤖 예측 모델 선택")
    method = st.radio("방법", ["1. 선형 회귀", "2. 2차 곡선", "3. 연평균 성장률(CAGR)", "4. 지수 평활", "5. 로그 추세"], 0, horizontal=True)

    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    if df_act.empty: st.warning("학습할 실적 데이터가 없습니다."); return

    train_years = sorted(df_act['연'].unique())
    st.caption(f"ℹ️ 학습 데이터: {train_years[0]}~{train_years[-1]}년 (선택된 연도만 반영)")

    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups, future = df_train['그룹'].unique(), np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        X, y = sub['연'].values, sub['값'].values
        pred = []
        
        if "선형" in method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "2차" in method:
            try: pred = np.poly1d(np.polyfit(X, y, 2))(future.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "로그" in method:
            try: 
                model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y)
                pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future)
        elif "지수" in method: pred = holt_linear_trend(y, 10)
        else: # CAGR
            try:
                start, end = y[0], y[-1]; n = len(y)-1
                cagr = (end/start)**(1/n) - 1 if start>0 and end>0 else 0
                pred = [end * (1+cagr)**(j+1) for j in range(10)]
            except: pred = [y[-1]]*10
                
        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '실적'})
        for yr, v in zip(future.flatten(), pred): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 전체 장기 전망")
    fig = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', markers=True)
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧱 2035 미래 예측 상세")
    df_f = df_res[df_res['Type']=='예측']
    fig2 = px.bar(df_f, x='연', y='판매량', color='그룹', text_auto='.2s')
    st.plotly_chart(fig2, use_container_width=True)
    
    piv = df_f.pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("다운로드", piv.to_csv().encode('utf-8-sig'), "forecast.csv")

# ─────────────────────────────────────────────────────────
# 4. [기능 3] 가정용 정밀 분석
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    if df_temp is None: st.error("🚨 기온 데이터가 없습니다."); return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    if df_merged.empty: st.warning("기간 불일치"); return

    years = sorted(df_merged['연'].unique())
    st.markdown("##### 📅 분석할 연도 선택")
    sel_years = st.multiselect("연도 선택", years, default=years[-5:] if len(years)>=5 else years, label_visibility="collapsed")
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    corr = df_final['평균기온'].corr(df_final['값'])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 판매량")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{corr:.2f}")
        st.caption("음수일수록 반비례")

    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_final['기간'], y=df_final['값'], name="판매량", yaxis='y'))
    fig2.add_trace(go.Scatter(x=df_final['기간'], y=df_final['평균기온'], name="기온", line=dict(color='red'), yaxis='y2'))
    fig2.update_layout(yaxis=dict(title="판매량"), yaxis2=dict(title="기온", overlaying='y', side='right'))
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    # 1. 깃허브 데이터 로드
    xls_sales = load_github_file(SALES_FILE_NAME, 'xlsx')
    
    # 기온 데이터
    xls_temp = load_github_file(TEMP_FILE_NAME, 'xlsx') 
    if xls_temp: df_temp = preprocess_temp(xls_temp.parse(0))
    else:
        csv_temp = load_github_file("기온.csv", 'csv')
        df_temp = preprocess_temp(csv_temp) if csv_temp is not None else None

    # 로드 상태 확인 및 진단
    is_loaded = xls_sales is not None
    long_df = pd.DataFrame()
    unit_label = "천m³"

    with st.sidebar:
        st.header("설정")
        main_cat = st.radio("📂 분석 카테고리", ["1. 판매량 예측", "2. 공급량 예측"])
        st.markdown("---")
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        st.markdown("---")
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        
        st.markdown("---")
        st.caption("데이터 파일 상태")
        
        # 🟢 [진단] 깃허브 로드 성공 여부 표시
        if is_loaded:
            st.success("✅ GitHub 판매량 로드 성공")
            # 시트 이름 확인 (디버깅용)
            # st.write(f"시트 목록: {xls_sales.sheet_names}") 
        else:
            st.error("❌ GitHub 로드 실패")
            uploaded_sales = st.file_uploader("판매량 파일 업로드", type="xlsx")
            if uploaded_sales: 
                xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
                is_loaded = True
        
        uploaded_temp = st.file_uploader("기온 파일 업로드", type=["csv", "xlsx"])
        if uploaded_temp: df_temp = load_temp_universal(uploaded_temp)

        # 🟢 [학습 기간 선택]
        if is_loaded:
            try:
                # 시트 이름 체크 (빈화면 원인 방지)
                sheet_name_p = "계획_부피" if unit.startswith("부피") else "계획_열량"
                sheet_name_a = "실적_부피" if unit.startswith("부피") else "실적_열량"
                
                if unit.startswith("부피"): unit_label = "천m³"
                else: unit_label = "GJ"

                if sheet_name_p in xls_sales.sheet_names and sheet_name_a in xls_sales.sheet_names:
                    df_p = xls_sales.parse(sheet_name_p)
                    df_a = xls_sales.parse(sheet_name_a)
                    long_df = make_long(df_p, df_a)
                    
                    # 2025년까지만 학습 데이터로
                    years_avail = sorted([y for y in long_df['연'].unique() if y <= 2025])
                    
                    st.markdown("---")
                    st.markdown("**📅 학습 대상 연도 설정**")
                    train_years = st.multiselect("학습 연도", years_avail, default=years_avail, label_visibility="collapsed")
                    
                    if train_years: long_df = long_df[long_df['연'].isin(train_years)]
                    else: st.warning("연도를 선택하세요."); long_df = pd.DataFrame()
                else:
                    st.error(f"시트 '{sheet_name_p}' 또는 '{sheet_name_a}' 가 엑셀에 없습니다.")
                    st.write(f"현재 시트 목록: {xls_sales.sheet_names}")

            except Exception as e:
                st.error(f"데이터 처리 오류: {e}")
                long_df = pd.DataFrame()

    # ── 메인 화면 ──
    if not is_loaded: 
        st.info("👈 좌측 사이드바에서 판매량 파일을 업로드해주세요.")
        return
        
    if long_df.empty: 
        return # 에러 메시지는 사이드바에 표시됨

    # 라우팅
    if main_cat == "1. 판매량 예측":
        if "실적분석" in sub_mode:
            render_analysis_dashboard(long_df, unit_label)
        elif "2035 예측" in sub_mode:
            render_prediction_2035(long_df, unit_label)
        elif "가정용" in sub_mode:
            render_household_analysis(long_df, df_temp, unit_label)
    else:
        st.header("🚧 공급량 예측")
        st.warning("공급량 예측 서비스는 준비 중입니다.")
        st.info("현재 '1. 판매량 예측' 메뉴만 활성화되어 있습니다.")

if __name__ == "__main__":
    main()
