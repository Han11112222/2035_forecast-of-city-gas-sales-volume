import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import requests
from pathlib import Path
from sklearn.linear_model import LinearRegression
from urllib.parse import quote

# ─────────────────────────────────────────────────────────
# 🟢 기본 설정 & 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 계획/실적 분석", layout="wide")

def set_korean_font():
    ttf = Path(__file__).parent / "NanumGothic-Regular.ttf"
    if ttf.exists():
        try:
            import matplotlib as mpl
            mpl.font_manager.fontManager.addfont(str(ttf))
            mpl.rcParams["font.family"] = "NanumGothic"
            mpl.rcParams["axes.unicode_minus"] = False
        except: pass

set_korean_font()

# 🟢 설정 정보 (형님 깃허브 정보)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_TEMP_XLSX = "기온_198001_202512.xlsx" # 기온 파일명 (엑셀 기준)

# 🟢 용도 매핑
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
# 1. 데이터 로드 및 전처리 (가장 확실한 방법으로 복구)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_excel_from_github(filename):
    """깃허브의 엑셀 파일을 바이너리로 가져와서 읽습니다 (에러 방지)"""
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
        response = requests.get(url)
        response.raise_for_status()
        # 바이너리 데이터를 BytesIO로 감싸서 엑셀 파일로 인식시킴
        return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
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
    
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue
            group = USE_COL_TO_GROUP.get(col)
            if not group: continue
            
            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            records.append(base)
            
    if not records: return pd.DataFrame()
    long_df = pd.concat(records, ignore_index=True)
    return long_df.dropna(subset=["연", "월"])

# [기온 데이터 처리] 일별 데이터를 월별 평균으로 변환
def process_temp_data(xls_file):
    try:
        # 첫 번째 시트를 읽음
        df = xls_file.parse(0)
        
        # 날짜 컬럼 찾기 (보통 첫 번째 컬럼)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 연/월 추출
        df['연'] = df[date_col].dt.year
        df['월'] = df[date_col].dt.month
        
        # 기온 컬럼 찾기 ('기온' 글자가 포함된 컬럼)
        temp_cols = [c for c in df.columns if "기온" in c]
        if not temp_cols: return None
        target_col = temp_cols[0]
        
        # 월별 평균 집계
        df_monthly = df.groupby(['연', '월'])[target_col].mean().reset_index()
        df_monthly.rename(columns={target_col: '평균기온'}, inplace=True)
        
        return df_monthly
    except:
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (연도 선택 버튼 포함)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 실적 데이터만, 2025년 이하
    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("데이터가 없습니다.")
        return

    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요 (다중 선택)")
    selected_years = st.multiselect("연도 선택", options=all_years, default=default_years, label_visibility="collapsed")
    
    if not selected_years:
        st.warning("연도를 선택해주세요.")
        return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1 (꺾은선)
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 표 1
    st.markdown("##### 📋 상세 데이터")
    piv_mon = df_mon.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(piv_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2 (스택바)
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 표 2
    st.markdown("##### 📋 상세 데이터")
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측 (5가지 모델)
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
    pred_method = st.radio(
        "분석 방법",
        ["1. 선형 추세 (Linear)", "2. 2차 곡선 (Quadratic)", "3. 로그 추세 (Logarithmic)", "4. 지수 평활 (Holt's)", "5. 연평균 성장률 (CAGR)"],
        index=0, horizontal=True
    )

    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        sub_recent = sub.tail(5) # 최근 5년만 반영
        if len(sub_recent) < 2: sub_recent = sub
            
        X = sub_recent['연'].values
        y = sub_recent['값'].values
        pred = []

        if "선형" in pred_method:
            model = LinearRegression()
            model.fit(X.reshape(-1,1), y)
            pred = model.predict(future_years)
        elif "2차" in pred_method:
            try:
                coeffs = np.polyfit(X, y, 2)
                pred = np.poly1d(coeffs)(future_years.flatten())
            except:
                model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in pred_method:
            try:
                X_idx = np.arange(1, len(X) + 1).reshape(-1, 1)
                X_future = np.arange(len(X) + 1, len(X) + 11).reshape(-1, 1)
                model = LinearRegression()
                model.fit(np.log(X_idx), y)
                pred = model.predict(np.log(X_future))
            except:
                model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in pred_method:
            pred = holt_linear_trend(y, 10)
        else: # CAGR
            try:
                start_v, end_v = y[0], y[-1]
                n = len(y) - 1
                cagr = (end_v/start_v)**(1/n) - 1 if start_v > 0 and end_v > 0 else 0
                pred = [end_v * (1+cagr)**(j+1) for j in range(10)]
            except:
                pred = [y[-1]] * 10
                
        pred = [max(0, p) for p in pred] # 음수 제거
        
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '예측'})
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
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("다운로드", piv.to_csv().encode('utf-8-sig'), "forecast.csv")

# ─────────────────────────────────────────────────────────
# 4. [기능 3] 가정용 정밀 분석 (기온 연동)
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None:
        st.error("🚨 기온 데이터가 없습니다. 좌측에서 파일을 로드해주세요.")
        return

    # 데이터 병합
    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty:
        st.warning("기간이 일치하는 데이터가 없습니다.")
        return

    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도", years, default=years[-5:] if len(years)>=5 else years)
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    # 상관관계
    corr = df_final['평균기온'].corr(df_final['값'])
    st.markdown(f"#### 🌡️ 기온 vs 판매량 (상관계수: {corr:.2f})")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        fig_scatter = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        if corr < -0.7: st.success("강한 반비례 (정상)")
        elif corr < -0.3: st.warning("보통 반비례")
        else: st.error("관계 약함")

    st.markdown("---")
    
    # 시계열 비교
    st.markdown("#### 📉 기간별 패턴 비교")
    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_final['기간'], y=df_final['값'], name="판매량", marker_color='blue', yaxis='y'))
    fig.add_trace(go.Scatter(x=df_final['기간'], y=df_final['평균기온'], name="기온(℃)", line=dict(color='red'), yaxis='y2'))
    
    fig.update_layout(
        yaxis=dict(title="판매량"),
        yaxis2=dict(title="기온", overlaying='y', side='right'),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    with st.sidebar:
        st.header("설정")
        
        # 1. 판매량
        st.markdown("**1. 판매량 데이터**")
        sales_src = st.radio("판매량 소스", ["☁️ GitHub", "📂 파일 업로드"], key="s_src")
        uploaded_sales = None
        if sales_src == "📂 파일 업로드":
            uploaded_sales = st.file_uploader("판매량(.xlsx)", type="xlsx", key="up_s")
            
        st.markdown("---")
        
        # 2. 기온
        st.markdown("**2. 기온 데이터**")
        temp_src = st.radio("기온 소스", ["☁️ GitHub", "📂 파일 업로드"], key="t_src")
        uploaded_temp = None
        if temp_src == "📂 파일 업로드":
            uploaded_temp = st.file_uploader("기온(.xlsx, .csv)", type=["xlsx", "csv"], key="up_t")

        st.markdown("---")
        mode = st.radio("메뉴", ["1. 실적 분석", "2. 2035 예측", "3. 가정용 정밀 분석"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    # 로드 프로세스
    # A. 판매량
    xls_sales = None
    if sales_src == "☁️ GitHub":
        sales_bytes = load_excel_from_github(DEFAULT_SALES_XLSX)
        if sales_bytes: xls_sales = sales_bytes
    elif uploaded_sales:
        xls_sales = pd.ExcelFile(uploaded_sales, engine='openpyxl')
        
    if not xls_sales:
        st.info("판매량 데이터를 연결해주세요.")
        return

    # B. 기온
    df_temp = None
    if temp_src == "☁️ GitHub":
        # 깃허브에서 기온 파일(엑셀) 로드 시도
        temp_xls = load_excel_from_github(DEFAULT_TEMP_XLSX)
        if temp_xls: df_temp = process_temp_data(temp_xls)
    elif uploaded_temp:
        # 업로드된 파일이 CSV인지 Excel인지 구분
        try:
            if uploaded_temp.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_temp, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(uploaded_temp, encoding='cp949')
                # CSV 전처리 (날짜 변환 등)
                df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
                df['연'] = df.iloc[:,0].dt.year
                df['월'] = df.iloc[:,0].dt.month
                temp_col = [c for c in df.columns if "기온" in c][0]
                df_temp = df.groupby(['연', '월'])[temp_col].mean().reset_index()
                df_temp.rename(columns={temp_col: '평균기온'}, inplace=True)
            else:
                df_temp = process_temp_data(pd.ExcelFile(uploaded_temp, engine='openpyxl'))
        except:
            st.error("기온 파일 형식을 확인해주세요.")

    # C. 데이터 변환
    try:
        if unit.startswith("부피"):
            df_p = xls_sales.parse("계획_부피")
            df_a = xls_sales.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p = xls_sales.parse("계획_열량")
            df_a = xls_sales.parse("실적_열량")
            unit_label = "GJ"
        long_df = make_long(df_p, df_a)
    except:
        st.error("판매량 시트 이름(계획_부피 등)을 확인해주세요.")
        return

    # 실행
    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    elif mode.startswith("2"):
        render_prediction_2035(long_df, unit_label)
    else:
        render_household_analysis(long_df, df_temp, unit_label)

if __name__ == "__main__":
    main()
