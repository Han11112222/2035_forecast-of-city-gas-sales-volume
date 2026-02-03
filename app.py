import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple

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

# 🟢 파일명 설정
DEFAULT_SALES_XLSX = "판매량(계획_실적).xlsx"
DEFAULT_TEMP_XLSX = "기온_198001_202512.xlsx" # 기온 파일

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

def load_data_simple(uploaded_file=None, default_file=DEFAULT_SALES_XLSX):
    try:
        if uploaded_file:
            # CSV 처리 추가
            if uploaded_file.name.endswith('.csv'):
                try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except: df = pd.read_csv(uploaded_file, encoding='cp949')
                return df
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        elif Path(default_file).exists():
            if default_file.endswith('.csv'):
                try: return pd.read_csv(default_file, encoding='utf-8-sig')
                except: return pd.read_csv(default_file, encoding='cp949')
            return pd.ExcelFile(default_file, engine='openpyxl')
        return None
    except Exception as e:
        # st.error(f"파일 읽기 오류: {e}")
        return None

# 기온 데이터 전처리
def preprocess_temp(df):
    if df is None: return None
    # 날짜 컬럼 통일
    if '일자' in df.columns: df.rename(columns={'일자': '날짜'}, inplace=True)
    if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
    
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    
    # 기온 컬럼 찾기
    temp_col = [c for c in df.columns if "기온" in c]
    if not temp_col: return None
    target = temp_col[0]
    
    # 월평균
    monthly = df.groupby(['연', '월'])[target].mean().reset_index()
    monthly.rename(columns={target: '평균기온'}, inplace=True)
    return monthly

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_act = df_act[df_act['연'] <= 2025] 
    
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("분석할 실적 데이터가 없습니다.")
        return

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
# 3. [기능 2] 2035 예측
# ─────────────────────────────────────────────────────────
def holt_linear_trend(y, n_preds):
    if len(y) < 2: return np.full(n_preds, y[0])
    alpha = 0.8; beta = 0.2
    level = y[0]; trend = y[1] - y[0]
    for val in y[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + i * trend for i in range(1, n_preds + 1)])

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    st.markdown("##### 📊 추세 분석 모델 선택")
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
        
        # 최근 5년 데이터 사용
        sub_recent = sub.tail(5)
        if len(sub_recent) < 2: sub_recent = sub
            
        X = sub_recent['연'].values
        y = sub_recent['값'].values
        pred = []
        
        if "선형" in pred_method:
            model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "2차" in pred_method:
            try: coeffs = np.polyfit(X, y, 2); pred = np.poly1d(coeffs)(future_years.flatten())
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "로그" in pred_method:
            try: model = LinearRegression(); model.fit(np.log(np.arange(1, len(X)+1)).reshape(-1,1), y); pred = model.predict(np.log(np.arange(len(X)+1, len(X)+11)).reshape(-1,1))
            except: model = LinearRegression(); model.fit(X.reshape(-1,1), y); pred = model.predict(future_years)
        elif "지수" in pred_method: pred = holt_linear_trend(y, 10)
        else:
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(10)]
            except: pred = [y[-1]]*10

        pred = [max(0, p) for p in pred]
        for yr, v in zip(sub['연'], sub['값']): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '실적'})
        for yr, v in zip(future_years.flatten(), pred): results.append({'연': yr, '그룹': grp, '판매량': v, 'Type': '예측'})
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    fig_line = px.line(df_res, x='연', y='판매량', color='그룹', line_dash='Type', markers=True)
    fig_line.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1)
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 🌟 [요청사항] 가정용 추세 설명 박스 추가
    st.info("""
    **💡 가정용 추세 분석 Insight:**
    
    2025년 가정용 판매량이 반등했음에도 불구하고 향후 추세(2026~)가 하향 또는 완만하게 나타나는 이유는 **'추세 분석(Regression)' 알고리즘의 특성** 때문입니다.
    
    1.  **장기 추세 우선:** AI 모델은 2025년 단일 연도의 급등(Spike)보다는 과거 5년(2021~2025) 간의 **전반적인 기울기**를 더 중요하게 반영합니다.
    2.  **기온 미보정:** 본 예측은 기온 변수 없이 단순 판매량 추이만으로 예측했기에, 2025년의 기온 하락(추위)으로 인한 일시적 증가분이 미래 추세에는 보수적으로 반영된 결과입니다.
    """)

    st.markdown("---")
    st.markdown("#### 🧱 2035년 미래 예측 상세")
    df_f = df_res[df_res['Type']=='예측']
    fig_stack = px.bar(df_f, x='연', y='판매량', color='그룹', text_auto='.2s')
    st.plotly_chart(fig_stack, use_container_width=True)
    
    piv = df_f.pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button("💾 예측 데이터 다운로드", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv", "text/csv")

# ─────────────────────────────────────────────────────────
# 4. [기능 3] 가정용 정밀 분석 (기온 연동)
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, df_temp, unit_label):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향) [{unit_label}]")
    
    if df_temp is None:
        st.error("🚨 기온 데이터가 로드되지 않았습니다. 파일이 있는지 확인해주세요.")
        return

    # 데이터 병합
    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    
    # 병합 전 타입 통일
    df_home['연'] = df_home['연'].astype(int)
    df_home['월'] = df_home['월'].astype(int)
    df_temp['연'] = df_temp['연'].astype(int)
    df_temp['월'] = df_temp['월'].astype(int)
    
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty:
        st.warning("기간이 일치하는 데이터가 없습니다.")
        return

    # 연도 필터
    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석할 연도를 선택하세요", years, default=years[-5:] if len(years)>=5 else years)
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    # 상관관계
    corr = df_final['평균기온'].corr(df_final['값'])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_scatter = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols",
                                 title=f"기온 vs 가정용 판매량 (상관계수: {corr:.2f})")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{corr:.2f}")
        st.caption("*-1에 가까울수록 반비례 (정상)*")

    st.markdown("---")
    
    # 이중축 그래프
    st.markdown("#### 📉 기간별 패턴 비교")
    df_final = df_final.sort_values(['연', '월'])
    df_final['기간'] = df_final['연'].astype(str) + "-" + df_final['월'].astype(str).str.zfill(2)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_final['기간'], y=df_final['값'], name=f"판매량 ({unit_label})", yaxis='y', marker_color='#3182ce'))
    fig.add_trace(go.Scatter(x=df_final['기간'], y=df_final['평균기온'], name="기온(℃)", yaxis='y2', line=dict(color='red')))
    
    fig.update_layout(
        yaxis=dict(title=f"판매량 ({unit_label})"),
        yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right'),
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
        
        # 파일 확인 (로컬 경로에 있는지)
        path_sales = Path(DEFAULT_SALES_XLSX)
        path_temp = Path(DEFAULT_TEMP_XLSX)
        
        # 판매량 소스
        uploaded_sales = None
        if not path_sales.exists():
            st.warning(f"⚠️ '{DEFAULT_SALES_XLSX}' 없음")
            uploaded_sales = st.file_uploader("판매량 파일 업로드", type="xlsx")
        else:
            st.success(f"✅ 판매량 파일 연결됨")
            
        st.markdown("---")
        
        # 기온 소스 (기존 코드에 없던 부분 보완)
        uploaded_temp = None
        if not path_temp.exists() and not Path("기온.csv").exists():
            uploaded_temp = st.file_uploader("기온 파일 업로드 (선택)", type=["xlsx", "csv"])
        else:
            st.success(f"✅ 기온 파일 연결됨")

        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 실적 분석", "2. 2035 예측", "3. 가정용 정밀 분석"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    # 로드
    xls_sales = load_data_simple(uploaded_sales, DEFAULT_SALES_XLSX)
    if xls_sales is None: return

    # 기온 데이터 로드 (우선순위: 업로드 -> xlsx -> csv)
    df_temp = None
    if uploaded_temp:
        if uploaded_temp.name.endswith('.csv'): df_temp = pd.read_csv(uploaded_temp)
        else: df_temp = pd.ExcelFile(uploaded_temp, engine='openpyxl').parse(0)
    elif path_temp.exists():
        df_temp = pd.ExcelFile(path_temp, engine='openpyxl').parse(0)
    elif Path("기온.csv").exists():
        try: df_temp = pd.read_csv("기온.csv", encoding='utf-8-sig')
        except: df_temp = pd.read_csv("기온.csv", encoding='cp949')
        
    if df_temp is not None:
        df_temp = preprocess_temp(df_temp)

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
        st.error(f"데이터 처리 오류: {e}")
        return

    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    elif mode.startswith("2"):
        render_prediction_2035(long_df, unit_label)
    else:
        render_household_analysis(long_df, df_temp, unit_label)

if __name__ == "__main__":
    main()
