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

# 🟢 설정
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_TEMP_FILE = "기온.csv" # 기본값

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
# 1. 데이터 로드 및 전처리 (CSV 지원 추가!)
# ─────────────────────────────────────────────────────────
def _clean_base(df):
    out = df.copy()
    # Unnamed 컬럼 제거
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

# [수정] 기온 데이터 로드 (CSV/Excel 자동 감지)
def load_temp_data(file_buffer, filename):
    try:
        if filename.endswith('.csv'):
            # 한글 인코딩 대응
            try:
                df = pd.read_csv(file_buffer, encoding='utf-8-sig')
            except:
                df = pd.read_csv(file_buffer, encoding='cp949')
        else:
            df = pd.read_excel(file_buffer, engine='openpyxl')
            
        # 전처리: 날짜 컬럼 확인
        if '날짜' not in df.columns:
            df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
            
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        
        # 기온 컬럼 찾기 ('기온' 글자 포함된 것)
        temp_cols = [c for c in df.columns if '기온' in c]
        if not temp_cols: return None
        target_col = temp_cols[0]
        
        # 월별 평균 집계
        df_monthly = df.groupby(['연', '월'])[target_col].mean().reset_index()
        df_monthly.rename(columns={target_col: '평균기온'}, inplace=True)
        
        return df_monthly
        
    except Exception as e:
        return None

# 깃허브 로드 (판매량용)
@st.cache_data(ttl=600)
def load_bytes_from_github(filename):
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except:
        return None

# 로컬/업로드 파일 로드 (판매량용)
def load_sales_excel(uploaded_file):
    try:
        if uploaded_file:
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        elif Path(DEFAULT_SALES_XLSX).exists():
            return pd.ExcelFile(DEFAULT_SALES_XLSX, engine='openpyxl')
        return None
    except:
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (기존 유지)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    # 2025년까지만 (데이터가 있는 경우)
    df_act = df_act[df_act['연'] <= 2025]
    
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("분석할 실적 데이터가 없습니다.")
        return

    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요 (다중 선택)")
    selected_years = st.multiselect("연도 선택", options=all_years, default=default_years, label_visibility="collapsed")
    
    if not selected_years:
        st.warning("연도를 1개 이상 선택해주세요.")
        return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon_compare = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon_compare, x='월', y='값', color='연', markers=True, title="월별 실적 추이 비교")
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### 📋 월별 상세 수치")
    pivot_mon = df_mon_compare.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(pivot_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr_usage = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr_usage, x='연', y='값', color='그룹', title="연도별 판매량 및 용도 구성", text_auto='.2s')
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
    pred_method = st.radio("예측 방법", ["1. 선형 회귀 (Linear)", "2. 2차 곡선 (Polynomial)", "3. 연평균 성장률 (CAGR)"], index=0, horizontal=True)
    
    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
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
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("예측 데이터 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv")

# ─────────────────────────────────────────────────────────
# 4. [신규 기능] 가정용 정밀 분석 (기온 연동)
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None or df_temp.empty:
        st.error("🚨 기온 데이터가 로드되지 않았습니다. 좌측에서 '기온 데이터'를 업로드해주세요.")
        return

    # 1. 데이터 병합 (가정용 실적 + 기온)
    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    
    # 타입 맞추기 (병합을 위해)
    df_home['연'] = df_home['연'].astype(int)
    df_home['월'] = df_home['월'].astype(int)
    df_temp['연'] = df_temp['연'].astype(int)
    df_temp['월'] = df_temp['월'].astype(int)
    
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty:
        st.warning("판매량 데이터와 기온 데이터의 기간(연/월)이 일치하는 구간이 없습니다.")
        return

    # 연도 필터
    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석할 연도를 선택하세요", years, default=years[-5:] if len(years)>=5 else years)
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    # 2. 상관관계 분석
    st.markdown("#### 🌡️ 기온 vs 가정용 판매량 상관관계")
    corr = df_final['평균기온'].corr(df_final['값'])
    
    c1, c2 = st.columns([3, 1])
    with c1:
        fig_scatter = px.scatter(
            df_final, x='평균기온', y='값', color='연',
            trendline="ols",
            title=f"기온에 따른 판매량 분포 (Trendline 포함)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        st.metric("상관계수", f"{corr:.2f}")
        st.caption("*-1에 가까울수록 기온이 낮을 때 판매량이 증가함 (반비례)*")

    st.markdown("---")

    # 3. 이중축 그래프 (판매량 & 기온)
    st.markdown("#### 📉 기간별 판매량 및 기온 변화 비교")
    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Bar(x=df_final['기간'], y=df_final['값'], name="가정용 판매량", marker_color='rgba(50, 100, 255, 0.6)', yaxis='y'))
    fig_dual.add_trace(go.Scatter(x=df_final['기간'], y=df_final['평균기온'], name="평균기온 (℃)", line=dict(color='red', width=3), yaxis='y2'))
    
    fig_dual.update_layout(
        yaxis=dict(title=f"판매량 ({unit_label})"),
        yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right'),
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
            uploaded_sales = st.file_uploader("판매량 파일(.xlsx)", type=["xlsx"], key="up_sales")
        
        st.markdown("---")
        
        # 2. 기온 파일 (CSV 지원!)
        st.markdown("**2. 기온 데이터 (분석용)**")
        uploaded_temp = None
        temp_src = st.radio("기온 소스", ["☁️ GitHub", "📂 파일 업로드"], key="src_temp")
        if temp_src == "📂 파일 업로드":
            # [수정] CSV 파일도 허용!
            uploaded_temp = st.file_uploader("기온 파일(.csv, .xlsx)", type=["csv", "xlsx"], key="up_temp")

        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 실적 분석", "2. 2035 예측", "3. 가정용 정밀 분석"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    # A. 판매량 로드
    if sales_src == "☁️ GitHub":
        sales_bytes = load_bytes_from_github(DEFAULT_SALES_XLSX)
        xls_sales = pd.ExcelFile(io.BytesIO(sales_bytes), engine='openpyxl') if sales_bytes else None
    else:
        xls_sales = load_sales_excel(uploaded_sales)
        
    if xls_sales is None:
        st.info("👈 '판매량 데이터'를 연결해주세요.")
        return

    # B. 기온 로드 (가정용 분석 탭에서 사용)
    df_temp = None
    if temp_src == "☁️ GitHub":
        # 깃허브에 기온 파일이 없으면 로드 안됨 (패스)
        pass 
    else:
        if uploaded_temp:
            # 파일명과 함께 로드 함수 호출
            df_temp = load_temp_data(uploaded_temp, uploaded_temp.name)

    # C. 데이터 처리
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

    # D. 기능 실행
    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    elif mode.startswith("2"):
        render_prediction_2035(long_df, unit_label)
    else:
        render_household_analysis(long_df, df_temp, unit_label)

if __name__ == "__main__":
    main()
