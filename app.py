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
    try:
        import matplotlib as mpl
        # 시스템 폰트 중 한글 폰트 찾기 시도 (Mac/Windows)
        from sys import platform
        if platform == "darwin": mpl.rc('font', family='AppleGothic')
        elif platform == "win32": mpl.rc('font', family='Malgun Gothic')
    except: pass

set_korean_font()

# 🟢 깃허브 설정 (여기가 핵심입니다!)
GITHUB_USER = "HanYeop"
REPO_NAME = "GasProject"
SALES_FILE_NAME = "판매량(계획_실적).xlsx"  # 판매량 파일
TEMP_FILE_NAME = "기온_198001_202512.xlsx" # 기온 파일 (혹시 깃허브에 있다면)

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
# 1. 데이터 로드 (깃허브 강제 연결)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_excel_from_github_force(filename):
    """깃허브 Raw 데이터를 강제로 가져옴"""
    # 1. URL 생성 (한글 인코딩)
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(filename)}"
    
    try:
        # 2. 요청
        response = requests.get(url)
        response.raise_for_status() # 404 에러면 즉시 중단
        
        # 3. 바이트로 변환 후 엑셀 읽기
        return pd.ExcelFile(io.BytesIO(response.content), engine='openpyxl')
    
    except Exception as e:
        # 에러 시 URL을 보여줘서 원인 파악
        st.error(f"❌ 깃허브 연결 실패! URL을 확인해주세요: {url}")
        st.error(f"에러 메시지: {e}")
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

# [기온] CSV/Excel 통합 로더
def load_temp_universal(file_obj):
    try:
        # 파일명 확인 (업로드 객체 or 문자열)
        fname = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        
        if fname.endswith('.csv'):
            try:
                df = pd.read_csv(file_obj, encoding='utf-8-sig')
            except:
                df = pd.read_csv(file_obj, encoding='cp949')
        else:
            df = pd.read_excel(file_obj)
            
        # 전처리
        if '날짜' not in df.columns: df.rename(columns={df.columns[0]: '날짜'}, inplace=True)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        
        # 기온 컬럼 찾기
        t_col = [c for c in df.columns if "기온" in c][0]
        df_mon = df.groupby(['연', '월'])[t_col].mean().reset_index()
        df_mon.rename(columns={t_col: '평균기온'}, inplace=True)
        return df_mon
    except:
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (연도 선택)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_act = df_act[df_act['연'] <= 2025] # 2025년 이하
    
    all_years = sorted(df_act['연'].unique())
    if not all_years: return

    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요")
    selected_years = st.multiselect("연도 선택", all_years, default=default_years, label_visibility="collapsed")
    if not selected_years: return

    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")

    # 그래프 1 (꺾은선)
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    df_mon = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    fig1 = px.line(df_mon, x='월', y='값', color='연', markers=True)
    fig1.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis_title=unit_label)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### 📋 상세 데이터")
    piv_mon = df_mon.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(piv_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # 그래프 2 (스택)
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    df_yr = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    fig2 = px.bar(df_yr, x='연', y='값', color='그룹', text_auto='.2s')
    fig2.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 상세 데이터")
    piv_yr = df_yr.pivot(index='연', columns='그룹', values='값').fillna(0)
    piv_yr['합계'] = piv_yr.sum(axis=1)
    st.dataframe(piv_yr.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측 (5가지 모델)
# ─────────────────────────────────────────────────────────
def holt_trend(y, n_preds):
    if len(y) < 2: return np.full(n_preds, y[0])
    alpha, beta = 0.8, 0.2
    level, trend = y[0], y[1] - y[0]
    for val in y[1:]:
        prev, level = level, alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return np.array([level + i * trend for i in range(1, n_preds + 1)])

def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    st.markdown("##### 🤖 예측 모델 선택")
    method = st.radio("방법", ["1. 선형 (Linear)", "2. 2차 곡선 (Poly)", "3. 로그 (Log)", "4. 지수 평활 (Holt)", "5. 성장률 (CAGR)"], 0, horizontal=True)

    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    groups, future = df_train['그룹'].unique(), np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 최근 5년 보정
        sub_r = sub.tail(5) if len(sub) >= 5 else sub
        X, y = sub_r['연'].values, sub_r['값'].values
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
        elif "지수" in method: pred = holt_trend(y, 10)
        else:
            try: cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1; pred = [y[-1]*(1+cagr)**(j+1) for j in range(10)]
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
    st.subheader(f"🏠 가정용 정밀 분석 [{unit_label}]")
    
    if df_temp is None:
        st.error("🚨 기온 데이터가 없습니다. 좌측에서 업로드해주세요.")
        return

    df_home = long_df[(long_df['그룹'] == '가정용') & (long_df['계획/실적'] == '실적')].copy()
    df_merged = pd.merge(df_home, df_temp, on=['연', '월'], how='inner')
    
    if df_merged.empty: st.warning("데이터 기간이 일치하지 않습니다."); return

    years = sorted(df_merged['연'].unique())
    sel_years = st.multiselect("분석 연도", years, default=years[-5:] if len(years)>=5 else years)
    if not sel_years: return
    
    df_final = df_merged[df_merged['연'].isin(sel_years)]
    
    # 상관관계
    corr = df_final['평균기온'].corr(df_final['값'])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(df_final, x='평균기온', y='값', color='연', trendline="ols", title=f"기온 vs 판매량 (상관계수: {corr:.2f})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("상관계수", f"{corr:.2f}")
        st.caption("-1에 가까울수록 반비례")

    # 이중축
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
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("메뉴", ["1. 실적 분석", "2. 2035 예측", "3. 가정용 정밀 분석"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])
        
        st.markdown("---")
        st.markdown("**기온 데이터 업로드 (3번 탭용)**")
        uploaded_temp = st.file_uploader("기온 파일(.csv, .xlsx)", type=["csv", "xlsx"])

    # 1. 판매량 로드 (무조건 깃허브)
    xls_sales = load_excel_from_github_force(SALES_FILE_NAME)
    
    if not xls_sales:
        st.error("🚨 판매량 데이터를 불러오지 못했습니다.")
        return
    else:
        st.success("✅ GitHub 데이터 로드 완료")

    # 2. 기온 로드 (업로드 우선)
    df_temp = None
    if uploaded_temp:
        df_temp = load_temp_universal(uploaded_temp)

    # 3. 데이터 변환
    try:
        if unit.startswith("부피"):
            df_p, df_a = xls_sales.parse("계획_부피"), xls_sales.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p, df_a = xls_sales.parse("계획_열량"), xls_sales.parse("실적_열량")
            unit_label = "GJ"
        long_df = make_long(df_p, df_a)
    except:
        st.error("시트 이름 오류. 파일 내 '계획_부피' 등이 있는지 확인하세요.")
        return

    # 4. 실행
    if mode.startswith("1"): render_analysis_dashboard(long_df, unit_label)
    elif mode.startswith("2"): render_prediction_2035(long_df, unit_label)
    else: render_household_analysis(long_df, df_temp, unit_label)

if __name__ == "__main__":
    main()
