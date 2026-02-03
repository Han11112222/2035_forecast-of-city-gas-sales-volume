import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="도시가스 판매량 분석 및 예측 시스템",
    page_icon="🔥",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #004d99; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_from_url(url):
    try:
        # 한글 깨짐 방지: utf-8-sig 시도 후 cp949 시도
        try:
            df = pd.read_csv(url, encoding='utf-8-sig')
        except:
            df = pd.read_csv(url, encoding='cp949')
        return df
    except Exception as e:
        return None

def preprocess_data(df):
    if df is None or df.empty:
        return None, [], None
    
    # 날짜 컬럼 자동 인식 (첫 번째 컬럼 가정)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    
    # 분석에 불필요한 컬럼 제외
    exclude_keywords = ['연', '월', '소 계', '합계', date_col, 'Year', 'Month', '주한미군']
    usage_cols = [c for c in df.columns if c not in exclude_keywords and "열병합" not in c]
    
    return df, usage_cols, date_col

# -----------------------------------------------------------------------------
# 3. 메인 로직
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 중장기 판매량 분석 및 예측")
    st.markdown("**Created by Marketing Planning Team (Han)**")

    # 사이드바 설정
    with st.sidebar:
        st.header("📂 데이터 연결")
        st.info("깃허브에서 복사한 **Raw URL**을 아래에 넣어주세요.")
        
        # 여기서 주소를 입력받습니다 (코드를 수정할 필요가 없어요!)
        url_vol = st.text_input("1. 실적_부피.csv 주소 (Raw URL)", placeholder="https://raw.githubusercontent.com/...")
        url_cal = st.text_input("2. 실적_열량.csv 주소 (Raw URL)", placeholder="https://raw.githubusercontent.com/...")

        st.markdown("---")
        st.header("📊 분석 옵션")
        tab_mode = st.radio("메뉴 선택", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit_mode = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])

    # URL이 입력되지 않았을 때 안내
    if not url_vol or not url_cal:
        st.warning("👈 왼쪽 사이드바에 깃허브 파일 주소(Raw URL)를 입력하면 분석이 시작됩니다!")
        st.markdown("""
        **[사용법]**
        1. 깃허브 파일 페이지에서 **Raw** 버튼을 클릭하세요.
        2. 주소창의 URL을 복사해서 왼쪽에 붙여넣으세요.
        """)
        return

    # 데이터 로드
    df_vol_raw = load_data_from_url(url_vol)
    df_cal_raw = load_data_from_url(url_cal)

    if df_vol_raw is None or df_cal_raw is None:
        st.error("❌ 데이터를 불러올 수 없습니다. URL이 정확한지(Raw 버튼 주소인지) 확인해주세요.")
        return

    # 단위에 따른 데이터 선택
    if unit_mode.startswith("부피"):
        df_target, usage_cols, date_col = preprocess_data(df_vol_raw)
        unit_label = "천m³"
    else:
        df_target, usage_cols, date_col = preprocess_data(df_cal_raw)
        unit_label = "GJ"

    # -------------------------------------------------------------------------
    # 탭 1: 판매량 실적분석
    # -------------------------------------------------------------------------
    if tab_mode.startswith("1"):
        st.subheader(f"📈 2015~2025 실적 트렌드 ({unit_label})")
        
        # 연도 슬라이더
        min_year = int(df_target['Year'].min())
        max_year = int(df_target['Year'].max())
        selected_years = st.slider("기간 선택", min_year, max_year, (min_year, max_year))
        
        df_filtered = df_target[(df_target['Year'] >= selected_years[0]) & (df_target['Year'] <= selected_years[1])]
        
        # KPI 및 차트
        total_sum = df_filtered[usage_cols].sum().sum()
        col1, col2 = st.columns(2)
        col1.metric("기간 총 판매량", f"{total_sum:,.0f} {unit_label}")
        
        # 연도별 차트
        df_yearly = df_filtered.groupby('Year')[usage_cols].sum().reset_index().melt(id_vars='Year', var_name='용도', value_name='판매량')
        st.plotly_chart(px.bar(df_yearly, x='Year', y='판매량', color='용도', title="연도별 용도 구성비"), use_container_width=True)

        # 월별 차트
        df_monthly = df_filtered.groupby('Month')[usage_cols].sum().reset_index().melt(id_vars='Month', var_name='용도', value_name='판매량')
        st.plotly_chart(px.line(df_monthly, x='Month', y='판매량', color='용도', markers=True, title="월별 계절성 패턴"), use_container_width=True)

    # -------------------------------------------------------------------------
    # 탭 2: 판매량 예측
    # -------------------------------------------------------------------------
    else:
        st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
        
        forecast_results = []
        future_years = np.arange(2026, 2036).reshape(-1, 1)
        
        for col in usage_cols:
            # 학습
            df_train = df_target.groupby('Year')[col].sum().reset_index()
            X = df_train['Year'].values.reshape(-1, 1)
            y = df_train[col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # 예측
            pred = model.predict(future_years)
            pred = [max(0, p) for p in pred]
            
            # 데이터 병합
            for Y, V in zip(df_train['Year'], df_train[col]):
                forecast_results.append({'Year': Y, '용도': col, '판매량': V, 'Type': '실적'})
            for Y, V in zip(future_years.flatten(), pred):
                forecast_results.append({'Year': Y, '용도': col, '판매량': V, 'Type': '예측'})
                
        df_forecast = pd.DataFrame(forecast_results)
        
        # 차트
        fig = px.line(df_forecast, x='Year', y='판매량', color='용도', line_dash='Type', markers=True, title="2035년까지의 수요 예측")
        fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측구간")
        st.plotly_chart(fig, use_container_width=True)
        
        # 다운로드
        df_pivot = df_forecast[df_forecast['Type']=='예측'].pivot_table(index='Year', columns='용도', values='판매량')
        st.dataframe(df_pivot.style.format("{:,.0f}"))
        st.download_button("예측 데이터 다운로드", df_pivot.to_csv(encoding='utf-8-sig').encode('utf-8-sig'), "forecast.csv")

if __name__ == "__main__":
    main()
