import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from urllib.parse import quote  # 한글/띄어쓰기 주소 자동 변환용

# =============================================================================
# 🟢 [기본 설정] 형님의 깃허브 정보를 여기에 딱 한 번만 적어주세요!
# =============================================================================
GITHUB_USER = "HanYeop"   # 예: 형님의 깃허브 아이디
REPO_NAME = "GasProject"  # 예: 형님의 저장소(Repository) 이름

# 파일명 (형님이 올리신 파일명 그대로)
FILE_NAME_VOL = "판매량(계획_실적).xlsx - 실적_부피.csv"
FILE_NAME_CAL = "판매량(계획_실적).xlsx - 실적_열량.csv"
# =============================================================================

st.set_page_config(page_title="도시가스 판매량 예측 시스템", page_icon="🔥", layout="wide")

# -----------------------------------------------------------------------------
# 1. 데이터 로드 함수 (GitHub + 업로드 콤보)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)  # 10분마다 갱신
def load_data_from_github(user, repo, filename):
    try:
        # 한글 및 공백이 포함된 파일명을 URL로 변환
        encoded_filename = quote(filename)
        url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{encoded_filename}"
        
        # CSV 읽기 시도 (인코딩 자동 대응)
        try:
            df = pd.read_csv(url, encoding='utf-8-sig')
        except:
            df = pd.read_csv(url, encoding='cp949')
        return df, True # 성공 여부 반환
    except Exception as e:
        return pd.DataFrame(), False

def preprocess_data(df):
    if df is None or df.empty: return df, [], None
    
    # 날짜 처리
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    
    # 불필요 컬럼 제외
    exclude = ['연', '월', '소 계', '합계', date_col, 'Year', 'Month', '주한미군']
    usage_cols = [c for c in df.columns if c not in exclude and "열병합" not in c]
    return df, usage_cols, date_col

# -----------------------------------------------------------------------------
# 2. 메인 로직
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 중장기 판매량 분석 및 예측 (2035)")
    st.markdown(f"**Data Source System: Hybrid (GitHub Default + Manual Backup)**")

    # 사이드바 설정
    with st.sidebar:
        st.header("📂 데이터 소스 설정")
        
        # 데이터 소스 선택 (기본값: GitHub)
        data_source = st.radio("데이터 불러오기 방식", ["☁️ GitHub (기본)", "📂 파일 직접 업로드 (비상용)"])
        
        st.markdown("---")
        st.header("📊 분석 옵션")
        tab_mode = st.radio("메뉴 선택", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit_mode = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])

    # -------------------------------------------------------------------------
    # 데이터 로딩 로직 (핵심!)
    # -------------------------------------------------------------------------
    df_vol = pd.DataFrame()
    df_cal = pd.DataFrame()
    
    # 1) GitHub 모드일 때
    if data_source == "☁️ GitHub (기본)":
        with st.spinner(f"깃허브({GITHUB_USER}/{REPO_NAME})에서 데이터를 가져오는 중입니다..."):
            df_vol, success_vol = load_data_from_github(GITHUB_USER, REPO_NAME, FILE_NAME_VOL)
            df_cal, success_cal = load_data_from_github(GITHUB_USER, REPO_NAME, FILE_NAME_CAL)
            
            if not success_vol or not success_cal:
                st.error("🚨 깃허브에서 파일을 찾을 수 없습니다.")
                st.warning("1. 코드 상단의 'GITHUB_USER'와 'REPO_NAME'이 정확한지 확인해주세요.\n2. 혹은 좌측 메뉴에서 '파일 직접 업로드'를 선택하세요.")
                return
            else:
                st.toast("✅ 깃허브 데이터 로드 완료!", icon="cloud")

    # 2) 직접 업로드 모드일 때
    else:
        st.info("비상용 모드입니다. 가지고 계신 파일을 직접 올려주세요.")
        uploaded_vol = st.file_uploader("실적_부피.csv 파일 업로드", type=['csv', 'xlsx'])
        uploaded_cal = st.file_uploader("실적_열량.csv 파일 업로드", type=['csv', 'xlsx'])
        
        if uploaded_vol and uploaded_cal:
            try:
                df_vol = pd.read_csv(uploaded_vol)
                df_cal = pd.read_csv(uploaded_cal)
            except:
                df_vol = pd.read_excel(uploaded_vol)
                df_cal = pd.read_excel(uploaded_cal)
        else:
            st.warning("👈 파일을 업로드하면 분석이 시작됩니다.")
            return

    # -------------------------------------------------------------------------
    # 단위 선택에 따른 데이터 셋팅
    # -------------------------------------------------------------------------
    if unit_mode.startswith("부피"):
        df_target, usage_cols, date_col = preprocess_data(df_vol)
        unit_label = "천m³"
    else:
        df_target, usage_cols, date_col = preprocess_data(df_cal)
        unit_label = "GJ"

    if df_target.empty: return

    # -------------------------------------------------------------------------
    # 탭 1: 판매량 실적분석
    # -------------------------------------------------------------------------
    if tab_mode.startswith("1"):
        st.subheader(f"📈 2015~2025 판매량 실적 상세분석 ({unit_label})")
        
        # 연도 필터
        all_years = sorted(df_target['Year'].unique())
        years = st.select_slider("분석 기간", options=all_years, value=(min(all_years), max(all_years)))
        
        df_sub = df_target[(df_target['Year'] >= years[0]) & (df_target['Year'] <= years[1])]
        
        # KPI
        total_sum = df_sub[usage_cols].sum().sum()
        col1, col2 = st.columns(2)
        col1.metric("선택 기간 누적 판매량", f"{total_sum:,.0f} {unit_label}")
        col2.metric("데이터 레코드 수", f"{len(df_sub)} 개")
        
        # 차트 1: 연도별 Trend
        df_yr = df_sub.groupby('Year')[usage_cols].sum().reset_index().melt(id_vars='Year', var_name='용도', value_name='Val')
        fig1 = px.bar(df_yr, x='Year', y='Val', color='용도', title="연도별/용도별 판매량 추이", text_auto='.2s')
        st.plotly_chart(fig1, use_container_width=True)
        
        # 차트 2: 월별 Seasonality
        df_mon = df_sub.groupby('Month')[usage_cols].sum().reset_index().melt(id_vars='Month', var_name='용도', value_name='Val')
        fig2 = px.line(df_mon, x='Month', y='Val', color='용도', markers=True, title="월별 패턴 (계절성)")
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 탭 2: 판매량 예측 (2035)
    # -------------------------------------------------------------------------
    else:
        st.subheader(f"🔮 2035 장기 판매량 예측 ({unit_label})")
        st.caption("과거(2015~2025) 패턴을 학습하여 향후 10년(2026~2035)을 전망합니다.")
        
        future_years = np.arange(2026, 2036).reshape(-1, 1)
        res_list = []
        
        # 진행상황 표시
        progress_text = "AI가 용도별 추세를 분석 중입니다..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, col in enumerate(usage_cols):
            # 학습 데이터
            tmp = df_target.groupby('Year')[col].sum().reset_index()
            X = tmp['Year'].values.reshape(-1, 1)
            y = tmp[col].values
            
            # 모델링
            model = LinearRegression()
            model.fit(X, y)
            
            # 예측
            pred = model.predict(future_years)
            pred = [max(0, p) for p in pred] # 음수 제거
            
            # 결과 저장
            for Y, V in zip(tmp['Year'], tmp[col]):
                res_list.append({'Year': Y, '용도': col, '판매량': V, 'Type': '실적(Actual)'})
            for Y, V in zip(future_years.flatten(), pred):
                res_list.append({'Year': Y, '용도': col, '판매량': V, 'Type': '예측(Forecast)'})
            
            my_bar.progress((i + 1) / len(usage_cols), text=progress_text)
            
        my_bar.empty() # 진행바 제거
        
        df_final = pd.DataFrame(res_list)
        
        # 예측 차트
        fig3 = px.line(df_final, x='Year', y='판매량', color='용도', line_dash='Type', markers=True, 
                       title=f"용도별 장기 수요 예측 시뮬레이션 ({unit_label})")
        # 예측 구간 배경색
        fig3.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="Forecast Zone")
        st.plotly_chart(fig3, use_container_width=True)
        
        # 결과표 및 다운로드
        st.markdown("### 📋 상세 예측 데이터")
        df_pivot = df_final[df_final['Type'].str.contains('예측')].pivot_table(index='Year', columns='용도', values='판매량')
        df_pivot['Total'] = df_pivot.sum(axis=1)
        
        st.dataframe(df_pivot.style.format("{:,.0f}"))
        
        csv = df_pivot.to_csv().encode('utf-8-sig')
        st.download_button(
            label="💾 예측 결과 엑셀(CSV) 다운로드",
            data=csv,
            file_name=f'Gas_Sales_Forecast_2035_{unit_label}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
