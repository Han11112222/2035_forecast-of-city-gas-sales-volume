import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from urllib.parse import quote
import io

# =============================================================================
# 🟢 [설정] Han형님의 깃허브 정보 입력
# =============================================================================
GITHUB_USER = "HanYeop"      # 형님의 깃허브 아이디
REPO_NAME = "GasProject"     # 저장소 이름
EXCEL_FILE_NAME = "판매량(계획_실적).xlsx"

# 🟢 [매핑] 형님이 주신 용도별 분류 기준 (그대로 적용)
USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

st.set_page_config(page_title="도시가스 판매량 분석 및 예측", page_icon="🔥", layout="wide")

# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 함수
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data(source_type, uploaded_file=None):
    # A. 깃허브 로드
    if source_type == "github":
        try:
            url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(EXCEL_FILE_NAME)}"
            # 깃허브는 read_excel에 url을 바로 넣어도 됩니다 (engine='openpyxl')
            return pd.read_excel(url, sheet_name=None, engine='openpyxl'), True
        except Exception as e:
            return None, False
    # B. 파일 업로드
    elif uploaded_file:
        return pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl'), True
    return None, False

def preprocess_data(df_raw):
    """형님의 코드를 참고하여 데이터를 '그룹'별로 정리하는 함수"""
    if df_raw is None or df_raw.empty: return pd.DataFrame()

    df = df_raw.copy()
    
    # 1. 날짜 처리 (첫 번째 컬럼을 날짜로 가정)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df = df.dropna(subset=['Year', 'Month']) # 날짜 없는 행 제거
    
    # 2. 용도별 그룹 매핑 (Melt: Wide -> Long 변환)
    # 수치형 컬럼만 선택 (연, 월 제외)
    value_vars = [c for c in df.columns if c in USE_COL_TO_GROUP.keys()]
    
    if not value_vars:
        return pd.DataFrame() # 매핑할 컬럼이 없으면 빈 데이터 반환

    # 데이터 구조 변환 (Unpivot)
    df_long = df.melt(id_vars=['Year', 'Month'], value_vars=value_vars, 
                      var_name='상세용도', value_name='판매량')
    
    # 3. 그룹 매핑 적용
    df_long['그룹'] = df_long['상세용도'].map(USE_COL_TO_GROUP)
    
    # 4. 결측치 처리 및 그룹별 집계
    df_long['판매량'] = pd.to_numeric(df_long['판매량'], errors='coerce').fillna(0)
    
    # 최종적으로 [연, 월, 그룹] 기준으로 합침
    df_final = df_long.groupby(['Year', 'Month', '그룹'])['판매량'].sum().reset_index()
    
    return df_final

# -----------------------------------------------------------------------------
# 2. 메인 어플리케이션
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 판매량 실적분석 및 2035 예측")
    st.markdown("**Created by Han (Marketing Planning Team)**")

    # 사이드바 설정
    with st.sidebar:
        st.header("📂 데이터 연결")
        data_source = st.radio("데이터 소스", ["☁️ GitHub (기본)", "📂 파일 업로드"], index=0)
        
        excel_data = None
        if data_source.startswith("📂"):
            uploaded = st.file_uploader("엑셀 파일 업로드", type=['xlsx'])
            if uploaded:
                excel_data, success = load_data("upload", uploaded)
        else:
            excel_data, success = load_data("github")
            if not success:
                st.error("깃허브 연결 실패! 아이디/저장소명을 확인하세요.")
        
        st.markdown("---")
        st.header("📊 분석 옵션")
        # 탭 대신 라디오 버튼으로 기능 구분 (형님 요청: 탭 기능 구현)
        func_mode = st.radio("기능 선택", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit_mode = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])

    if not success or excel_data is None:
        st.info("👈 왼쪽에서 데이터를 연결하면 분석이 시작됩니다.")
        return

    # 단위에 따른 시트 및 데이터 선택
    sheet_name = "실적_부피" if unit_mode.startswith("부피") else "실적_열량"
    unit_label = "천m³" if unit_mode.startswith("부피") else "GJ"

    if sheet_name not in excel_data:
        st.error(f"엑셀 파일에 '{sheet_name}' 시트가 없습니다.")
        return

    # 데이터 전처리 (형님의 매핑 로직 적용)
    df_clean = preprocess_data(excel_data[sheet_name])
    
    if df_clean.empty:
        st.error("데이터 전처리 실패. 컬럼명(취사용, 업무난방용 등)이 엑셀에 있는지 확인해주세요.")
        return

    # -------------------------------------------------------------------------
    # 1. 판매량 실적 분석
    # -------------------------------------------------------------------------
    if func_mode.startswith("1"):
        st.subheader(f"📈 판매량 실적 분석 ({unit_label})")
        
        # 필터링
        all_years = sorted(df_clean['Year'].unique())
        years = st.slider("분석 기간", min(all_years), max(all_years), (min(all_years), max(all_years)))
        df_sub = df_clean[(df_clean['Year'] >= years[0]) & (df_clean['Year'] <= years[1])]
        
        # 요약 KPI
        total_vol = df_sub['판매량'].sum()
        col1, col2 = st.columns(2)
        col1.metric("선택 기간 총 판매량", f"{total_vol:,.0f} {unit_label}")
        col2.metric("데이터 건수", f"{len(df_sub)} 건")
        
        # 1) 연도별/그룹별 누적 막대 그래프
        df_yr_grp = df_sub.groupby(['Year', '그룹'])['판매량'].sum().reset_index()
        fig1 = px.bar(df_yr_grp, x='Year', y='판매량', color='그룹', 
                      title="연도별 용도(그룹) 판매량 추이", text_auto='.2s')
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2) 월별 계절성 패턴
        df_mon_grp = df_sub.groupby(['Month', '그룹'])['판매량'].sum().reset_index()
        fig2 = px.line(df_mon_grp, x='Month', y='판매량', color='그룹', markers=True,
                       title="월별 계절성 패턴 (합계)")
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 2. 판매량 예측 (2035)
    # -------------------------------------------------------------------------
    else:
        st.subheader(f"🔮 2035 장기 판매량 예측 ({unit_label})")
        st.info("💡 과거 데이터를 기반으로 '용도 그룹별' 추세를 분석하여 2035년까지 예측합니다.")
        
        # 예측 설정
        groups = sorted(df_clean['그룹'].unique())
        future_years = np.arange(2026, 2036).reshape(-1, 1)
        
        forecast_results = []
        
        # 그룹별 반복 예측
        progress_bar = st.progress(0)
        for i, grp in enumerate(groups):
            # 학습 데이터 (연도별 합계)
            df_train = df_clean[df_clean['그룹'] == grp].groupby('Year')['판매량'].sum().reset_index()
            
            if len(df_train) < 2: continue # 데이터 너무 적으면 패스
            
            X = df_train['Year'].values.reshape(-1, 1)
            y = df_train['판매량'].values
            
            # 모델링
            model = LinearRegression()
            model.fit(X, y)
            
            # 예측
            pred = model.predict(future_years)
            pred = [max(0, p) for p in pred] # 음수 방지
            
            # 데이터 저장 (실적 + 예측)
            for yr, val in zip(df_train['Year'], df_train['판매량']):
                forecast_results.append({'Year': yr, '그룹': grp, '판매량': val, 'Type': '실적'})
            for yr, val in zip(future_years.flatten(), pred):
                forecast_results.append({'Year': yr, '그룹': grp, '판매량': val, 'Type': '예측'})
                
            progress_bar.progress((i + 1) / len(groups))
            
        progress_bar.empty()
        
        df_forecast = pd.DataFrame(forecast_results)
        
        # 전체 합계 라인 추가 (옵션)
        df_total = df_forecast.groupby(['Year', 'Type'])['판매량'].sum().reset_index()
        df_total['그룹'] = '전체합계'
        df_final_plot = pd.concat([df_forecast, df_total])
        
        # 차트 시각화
        fig3 = px.line(df_final_plot, x='Year', y='판매량', color='그룹', line_dash='Type',
                       markers=True, title="2035년 용도별/전체 장기 전망")
        
        # 예측 구간 표시
        fig3.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측 구간")
        st.plotly_chart(fig3, use_container_width=True)
        
        # 데이터 다운로드
        st.markdown("### 📥 예측 데이터 다운로드")
        df_pivot = df_forecast[df_forecast['Type'] == '예측'].pivot_table(index='Year', columns='그룹', values='판매량')
        df_pivot['총합계'] = df_pivot.sum(axis=1)
        
        st.dataframe(df_pivot.style.format("{:,.0f}"))
        st.download_button("엑셀(CSV) 다운로드", df_pivot.to_csv().encode('utf-8-sig'), "forecast_2035.csv")

if __name__ == "__main__":
    main()
