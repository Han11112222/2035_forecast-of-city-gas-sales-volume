import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 스타일
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="도시가스 판매량 분석 및 예측 시스템",
    page_icon="🔥",
    layout="wide"
)

# 한글 폰트 지원 등 스타일 설정 (선택사항)
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #004d99; /* 대성에너지 느낌의 블루 */
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 함수 (GitHub Raw URL 사용)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Han형님! 깃허브에 파일을 올린 후, 'Raw' 버튼을 눌러서 나온 주소를 아래에 넣어주세요.
    # 현재는 예시 주소이므로, 실제 주소로 변경해야 작동합니다.
    base_url = "https://raw.githubusercontent.com/YOUR_GITHUB_ID/YOUR_REPO/main/"
    
    # 파일명 매핑 (실적 데이터만 사용한다고 가정)
    # 인코딩은 엑셀 저장 방식에 따라 'cp949' 또는 'utf-8-sig'를 사용하세요.
    try:
        # 여기서는 업로드된 파일명을 그대로 사용한다고 가정하고 로컬 테스트용 코드로 작성하되
        # 실제 배포시는 pd.read_csv("URL") 형태가 됩니다.
        
        # [실적_부피]
        df_vol = pd.read_csv("판매량(계획_실적).xlsx - 실적_부피.csv") 
        # [실적_열량]
        df_cal = pd.read_csv("판매량(계획_실적).xlsx - 실적_열량.csv")
        
        return df_vol, df_cal
        
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 깃허브 URL이나 파일 경로를 확인해주세요.")
        return pd.DataFrame(), pd.DataFrame()

# 데이터 전처리 함수
def preprocess_data(df):
    if df.empty:
        return df
    
    # 날짜 컬럼 처리 (첫 번째 컬럼이 날짜라고 가정)
    date_col = df.columns[0] 
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    
    # '소 계', '연', '월' 등 집계나 불필요 컬럼 제외하고 '용도'만 남기기
    exclude_cols = ['연', '월', '소 계', date_col]
    usage_cols = [c for c in df.columns if c not in exclude_cols]
    
    return df, usage_cols, date_col

# -----------------------------------------------------------------------------
# 3. 메인 로직
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 중장기 판매량 분석 및 예측")
    st.markdown("**Created by Marketing Planning Team (Han)**")

    # 데이터 로드
    df_vol_raw, df_cal_raw = load_data()

    if df_vol_raw.empty:
        st.warning("데이터를 불러올 수 없습니다. 우측 상단 메뉴에서 깃허브 경로를 확인해주세요.")
        return

    # 사이드바 설정
    with st.sidebar:
        st.header("📊 분석 설정")
        
        # 탭 선택
        tab_mode = st.radio("메뉴 선택", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        
        st.markdown("---")
        
        # 단위 선택
        unit_mode = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])
        
        # 데이터 선택 로직
        if unit_mode == "부피 (천m³)":
            df_target, usage_cols, date_col = preprocess_data(df_vol_raw)
            unit_label = "천m³"
        else:
            df_target, usage_cols, date_col = preprocess_data(df_cal_raw)
            unit_label = "GJ"

    # -------------------------------------------------------------------------
    # 탭 1: 판매량 실적분석
    # -------------------------------------------------------------------------
    if tab_mode == "1. 판매량 실적분석":
        st.subheader(f"📈 2015~2025 판매량 실적 분석 ({unit_label})")
        
        # 연도 필터
        all_years = sorted(df_target['Year'].unique())
        selected_years = st.select_slider("분석 기간 선택", options=all_years, value=(min(all_years), max(all_years)))
        
        # 필터링
        df_filtered = df_target[(df_target['Year'] >= selected_years[0]) & (df_target['Year'] <= selected_years[1])]
        
        # KPI 카드 (전체 합계)
        total_sum = df_filtered[usage_cols].sum().sum()
        last_year_sum = df_filtered[df_filtered['Year'] == selected_years[1]][usage_cols].sum().sum()
        
        col1, col2 = st.columns(2)
        col1.metric(label=f"선택 기간 총 판매량 ({unit_label})", value=f"{total_sum:,.0f}")
        col2.metric(label=f"{selected_years[1]}년 총 판매량", value=f"{last_year_sum:,.0f}")

        st.markdown("---")

        # 차트 1: 연도별 용도별 누적 막대 그래프 (Trend)
        df_yearly = df_filtered.groupby('Year')[usage_cols].sum().reset_index()
        # Wide to Long 변환 (Plotly용)
        df_yearly_melt = df_yearly.melt(id_vars='Year', var_name='용도', value_name='판매량')
        
        fig1 = px.bar(df_yearly_melt, x='Year', y='판매량', color='용도', 
                      title=f"연도별/용도별 판매량 추이 ({unit_label})",
                      text_auto='.2s')
        fig1.update_layout(xaxis_type='category')
        st.plotly_chart(fig1, use_container_width=True)

        # 차트 2: 월별 패턴 (히트맵 또는 라인)
        st.subheader("🗓️ 월별 판매량 패턴")
        df_monthly = df_filtered.groupby('Month')[usage_cols].sum().reset_index()
        df_monthly_melt = df_monthly.melt(id_vars='Month', var_name='용도', value_name='판매량')
        
        fig2 = px.line(df_monthly_melt, x='Month', y='판매량', color='용도', markers=True,
                       title=f"월별 합계 계절성 패턴 ({unit_label})")
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 탭 2: 판매량 예측 (2035)
    # -------------------------------------------------------------------------
    elif tab_mode == "2. 판매량 예측 (2035)":
        st.subheader(f"🔮 2035년 장기 판매량 예측 ({unit_label})")
        st.info("💡 과거 데이터를 기반으로 용도별 선형 추세(Linear Trend)를 분석하여 2035년까지 예측합니다.")

        # 예측 로직 (Linear Regression)
        future_years = np.arange(2026, 2036).reshape(-1, 1) # 2026~2035
        historical_years = df_target['Year'].unique().reshape(-1, 1)
        
        forecast_results = []
        
        # 용도별 반복 예측
        for col in usage_cols:
            # 연도별 합계 데이터 준비
            y_data = df_target.groupby('Year')[col].sum().values
            X_data = df_target.groupby('Year')[col].sum().index.values.reshape(-1, 1)
            
            # 모델 학습
            model = LinearRegression()
            model.fit(X_data, y_data)
            
            # 예측
            predictions = model.predict(future_years)
            
            # 음수 방지 (가스 판매량이 음수가 될 순 없음)
            predictions = [max(0, p) for p in predictions]
            
            # 결과 저장
            for year, pred in zip(future_years.flatten(), predictions):
                forecast_results.append({'Year': year, '용도': col, '판매량': pred, 'Type': 'Forecast'})
            
            # 과거 데이터도 차트용으로 저장
            for year, val in zip(X_data.flatten(), y_data):
                forecast_results.append({'Year': year, '용도': col, '판매량': val, 'Type': 'Actual'})

        df_forecast = pd.DataFrame(forecast_results)
        
        # 차트 시각화 (Line Chart with Dotted Forecast)
        fig3 = px.line(df_forecast, x='Year', y='판매량', color='용도', line_dash='Type',
                       title=f"용도별 장기 수요 예측 (2015~2035) [{unit_label}]",
                       markers=True)
        
        # 예측 구간 강조 (배경색)
        fig3.add_vrect(x0=2025.5, x1=2035.5, annotation_text="예측 구간", annotation_position="top left",
                       fillcolor="green", opacity=0.1, line_width=0)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # 데이터 테이블 표시
        st.subheader("📋 예측 데이터 상세")
        
        # 피벗 테이블로 보기 좋게 변환
        df_pivot = df_forecast[df_forecast['Type']=='Forecast'].pivot_table(index='Year', columns='용도', values='판매량')
        # 합계 컬럼 추가
        df_pivot['전체 합계'] = df_pivot.sum(axis=1)
        
        st.dataframe(df_pivot.style.format("{:,.0f}"))
        
        # 다운로드 버튼
        csv = df_pivot.to_csv().encode('utf-8-sig')
        st.download_button(
            label="예측 데이터 엑셀(CSV) 다운로드",
            data=csv,
            file_name=f'forecast_2035_{unit_label}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
