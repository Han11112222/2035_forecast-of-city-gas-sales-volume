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

# 🟢 파일명 설정 (기온 파일 추가!)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_TEMP_XLSX = "기온_198001_202512.xlsx" # 형님이 업로드한 기온 파일명

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
# 1. 데이터 로드 및 전처리
# ─────────────────────────────────────────────────────────
def _clean_base(df):
    out = df.copy()
    if "Unnamed: 0" in out.columns: out = out.drop(columns=["Unnamed: 0"])
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

# [신규] 기온 데이터 전처리 함수 (일별 -> 월별 평균)
def preprocess_temp_data(df_temp):
    # 컬럼명 확인 및 날짜 변환
    if '날짜' not in df_temp.columns: 
        # 혹시 컬럼명이 다를 경우 첫번째 컬럼을 날짜로 가정
        df_temp.rename(columns={df_temp.columns[0]: '날짜'}, inplace=True)
    
    df_temp['날짜'] = pd.to_datetime(df_temp['날짜'])
    df_temp['연'] = df_temp['날짜'].dt.year
    df_temp['월'] = df_temp['날짜'].dt.month
    
    # 월별 평균 기온 집계
    # '평균기온(℃)' 컬럼이 있다고 가정 (형님 파일 기준)
    temp_col = [c for c in df_temp.columns if '기온' in c][0] # '기온' 글자 들어간 컬럼 찾기
    
    df_monthly_temp = df_temp.groupby(['연', '월'])[temp_col].mean().reset_index()
    df_monthly_temp.rename(columns={temp_col: '평균기온'}, inplace=True)
    
    return df_monthly_temp

# 깃허브 로드 함수 (판매량/기온 공용)
@st.cache_data(ttl=600)
def load_bytes_from_github(filename):
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        return None

# 로컬/업로드 파일 로드 (판매량/기온 공용)
def load_excel_file(uploaded_file, default_filename):
    try:
        if uploaded_file:
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        elif Path(default_filename).exists():
            return pd.ExcelFile(default_filename, engine='openpyxl')
        return None
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (기존 유지)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_act = df_act[df_act['연'] <= 2025] 
    
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("분석할 실적 데이터가 없습니다.")
        return

    # [수정됨] 2017년부터 보고 싶으시다면 여기서 필터 기본값을 조정하면 됩니다.
    # 하지만 데이터가 존재하는 한 모두 보여주는 게 원칙이므로 전체 리스트 제공
    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요 (다중 선택)")
    selected_years = st.multiselect(
        "연도 선택",
        options=all_years,
        default=default_years,
        label_visibility="collapsed"
    )
    
    if not selected_years:
        st.warning("연도를 1개 이상 선택해주세요.")
        return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # [그래프 1] 월별 실적 추이
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon_compare = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    
    fig1 = px.line(
        df_mon_compare, x='월', y='값', color='연', markers=True,
        title="월별 실적 추이 비교"
    )
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### 📋 월별 상세 수치")
    pivot_mon = df_mon_compare.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(pivot_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # [그래프 2] 연도별 용도 누적
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr_usage = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    fig2 = px.bar(
        df_yr_usage, x='연', y='값', color='그룹',
        title="연도별 판매량 및 용도 구성", text_auto='.2s'
    )
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 용도별 상세 수치")
    pivot_usage = df_yr_usage.pivot(index='연', columns='그룹', values='값').fillna(0)
    pivot_usage['합계'] = pivot_usage.sum(axis=1)
    st.dataframe(pivot_usage.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측 (기존 유지)
# ─────────────────────────────────────────────────────────
def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    st.markdown("##### 🤖 예측 모델 선택")
    pred_method = st.radio(
        "예측 방법",
        ["1. 선형 회귀 (Linear)", "2. 2차 곡선 (Polynomial)", "3. 연평균 성장률 (CAGR)"],
        index=0, horizontal=True
    )
    
    # ... (기존 예측 로직 동일) ...
    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 최근 5년 보정
        sub_recent = sub.tail(5)
        if len(sub_recent) < 2: sub_recent = sub
            
        X = sub_recent['연'].values.reshape(-1, 1)
        y = sub_recent['값'].values
        
        if "선형" in pred_method:
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(future_years)
        elif "2차" in pred_method:
            try:
                coeffs = np.polyfit(X.flatten(), y, 2)
                p = np.poly1d(coeffs)
                pred = p(future_years.flatten())
            except:
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(future_years)
        else: # CAGR
            try:
                start_v, end_v = y[0], y[-1]
                n = len(y) - 1
                cagr = (end_v/start_v)**(1/n) - 1 if start_v > 0 and end_v > 0 else 0
                pred = [end_v * (1+cagr)**(j+1) for j in range(10)]
            except:
                pred = [y[-1]] * 10
                
        pred = [max(0, p) for p in pred]
        
        for y_val, v in zip(sub['연'], sub['값']):
            results.append({'연': y_val, '그룹': grp, '판매량': v, 'Type': '실적'})
        for y_val, v in zip(future_years.flatten(), pred):
            results.append({'연': y_val, '그룹': grp, '판매량': v, 'Type': '예측'})
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

# ─────────────────────────────────────────────────────────
# 4. [신규 기능] 가정용 정밀 분석 (기온 상관관계)
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None or df_temp.empty:
        st.error("기온 데이터가 없습니다. 기온 파일을 업로드하거나 깃허브를 확인해주세요.")
        return

    # 1. 데이터 준비 (가정용 실적 + 기온 병합)
    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    
    # '연', '월' 기준으로 병합 (Inner Join)
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty:
        st.warning("판매량 데이터와 기온 데이터의 기간(연/월)이 일치하는 구간이 없습니다.")
        return

    # 연도 필터
    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도 선택", years, default=years[-3:] if len(years)>=3 else years)
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    # ---------------------------------------------------------
    # [차트 1] 상관관계 분석 (Scatter Plot)
    # ---------------------------------------------------------
    st.markdown("#### 🌡️ 기온 vs 가정용 판매량 상관관계")
    st.caption("기온이 낮을수록(왼쪽) 판매량이 높아지는(위쪽) 반비례 관계가 나타나야 정상입니다.")
    
    # 상관계수 계산
    corr = df_final['평균기온'].corr(df_final['값'])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_scatter = px.scatter(
            df_final, x='평균기온', y='값', color='연',
            trendline="ols", # 회귀선 추가
            hover_data=['연', '월'],
            title=f"기온에 따른 판매량 분포 (상관계수: {corr:.2f})"
        )
        fig_scatter.update_layout(xaxis_title="평균기온 (℃)", yaxis_title=f"판매량 ({unit_label})")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.metric("상관계수 (Correlation)", f"{corr:.2f}")
        if corr < -0.7:
            st.success("강한 음의 상관관계! (기온 영향 큼)")
        elif corr < -0.3:
            st.warning("보통의 음의 상관관계")
        else:
            st.error("상관관계가 약함 (다른 요인 존재 가능)")

    st.markdown("---")

    # ---------------------------------------------------------
    # [차트 2] 시계열 패턴 비교 (이중축)
    # ---------------------------------------------------------
    st.markdown("#### 📉 판매량과 기온의 시계열 패턴 비교")
    
    # 데이터 정렬
    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    
    fig_dual = go.Figure()
    
    # 막대: 판매량
    fig_dual.add_trace(go.Bar(
        x=df_final['기간'], y=df_final['값'],
        name=f"가정용 판매량 ({unit_label})",
        marker_color='rgba(50, 100, 255, 0.6)',
        yaxis='y'
    ))
    
    # 선: 기온 (우측 축) - 기온은 보통 역축으로 보기도 하지만 여기선 있는 그대로 표시
    fig_dual.add_trace(go.Scatter(
        x=df_final['기간'], y=df_final['평균기온'],
        name="평균기온 (℃)",
        mode='lines+markers',
        line=dict(color='red', width=3),
        yaxis='y2'
    ))
    
    fig_dual.update_layout(
        title="기간별 판매량 및 기온 변화",
        yaxis=dict(title=f"판매량 ({unit_label})"),
        yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_dual, use_container_width=True)

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    with st.sidebar:
        st.header("설정")
        
        # 1. 판매량 파일
        st.markdown("**1. 판매량 데이터**")
        uploaded_sales = None
        sales_src = st.radio("판매량 소스", ["☁️ GitHub", "📂 파일 업로드"], key="src_sales")
        if sales_src == "📂 파일 업로드":
            uploaded_sales = st.file_uploader("판매량 파일(.xlsx)", type="xlsx", key="up_sales")
        
        st.markdown("---")
        
        # 2. 기온 파일
        st.markdown("**2. 기온 데이터 (분석용)**")
        uploaded_temp = None
        temp_src = st.radio("기온 소스", ["☁️ GitHub", "📂 파일 업로드"], key="src_temp")
        if temp_src == "📂 파일 업로드":
            uploaded_temp = st.file_uploader("기온 파일(.xlsx)", type="xlsx", key="up_temp")

        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 실적 분석", "2. 2035 예측", "3. 가정용 정밀 분석"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    # ── 데이터 로드 프로세스 ──
    # A. 판매량 로드
    if sales_src == "☁️ GitHub":
        sales_bytes = load_bytes_from_github(DEFAULT_SALES_XLSX)
        if sales_bytes: xls_sales = pd.ExcelFile(io.BytesIO(sales_bytes), engine='openpyxl')
        else: xls_sales = None
    else:
        xls_sales = load_excel_file(uploaded_sales, DEFAULT_SALES_XLSX)
        
    if xls_sales is None:
        st.info("좌측에서 '판매량 데이터'를 연결해주세요.")
        return

    # B. 기온 로드 (가정용 분석 탭에서만 필수지만, 미리 로드 시도)
    df_temp = None
    if temp_src == "☁️ GitHub":
        temp_bytes = load_bytes_from_github(DEFAULT_TEMP_XLSX)
        if temp_bytes: 
            xls_temp = pd.ExcelFile(io.BytesIO(temp_bytes), engine='openpyxl')
            df_temp = preprocess_temp_data(xls_temp.parse(0)) # 첫번째 시트 사용
    else:
        xls_temp = load_excel_file(uploaded_temp, DEFAULT_TEMP_XLSX)
        if xls_temp:
            df_temp = preprocess_temp_data(xls_temp.parse(0))

    # C. 데이터 처리 (판매량 Wide -> Long)
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
    except Exception as e:
        st.error(f"판매량 데이터 처리 실패: {e}")
        return

    # ── 기능 실행 ──
    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    elif mode.startswith("2"):
        render_prediction_2035(long_df, unit_label)
    else:
        render_household_analysis(long_df, df_temp, unit_label)

if __name__ == "__main__":
    main()
