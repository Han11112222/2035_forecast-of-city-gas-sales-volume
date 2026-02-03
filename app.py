import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from urllib.parse import quote

# =============================================================================
# 🟢 [기본 설정] Han형님의 깃허브 정보를 입력해주세요!
# =============================================================================
GITHUB_USER = "HanYeop"     # 예: 형님의 깃허브 아이디
REPO_NAME = "GasProject"    # 예: 저장소 이름

# 엑셀 파일명 (확장자 .xlsx 포함)
EXCEL_FILE_NAME = "판매량(계획_실적).xlsx"

# 엑셀 내부의 시트(Sheet) 이름 설정
# (형님 엑셀 파일 하단 탭에 적힌 이름과 똑같아야 해요!)
SHEET_VOL = "실적_부피"
SHEET_CAL = "실적_열량"
# =============================================================================

st.set_page_config(page_title="도시가스 판매량 예측 시스템", page_icon="🔥", layout="wide")

# -----------------------------------------------------------------------------
# 1. 데이터 로드 함수 (엑셀 전용)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_excel_from_github(user, repo, filename, sheet_name):
    try:
        # 파일명 URL 인코딩 (한글, 띄어쓰기 처리)
        encoded_filename = quote(filename)
        # 엑셀 파일의 Raw URL 생성
        url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{encoded_filename}"
        
        # 엑셀 읽기 (engine='openpyxl' 필수)
        df = pd.read_excel(url, sheet_name=sheet_name, engine='openpyxl')
        return df, True
    except Exception as e:
        return None, False

def preprocess_data(df):
    if df is None or df.empty: return df, [], None
    
    # 날짜 처리 (첫 번째 컬럼 기준)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    
    # 불필요 컬럼 제거
    exclude = ['연', '월', '소 계', '합계', date_col, 'Year', 'Month', '주한미군']
    # 열병합 등 파생 컬럼 제외하고 순수 용도만 추출
    usage_cols = [c for c in df.columns if c not in exclude and "열병합" not in c]
    
    return df, usage_cols, date_col

# -----------------------------------------------------------------------------
# 2. 메인 로직
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 중장기 판매량 분석 (Excel Ver.)")
    st.markdown(f"**Data Source: GitHub ({EXCEL_FILE_NAME})**")

    # 사이드바 설정
    with st.sidebar:
        st.header("📂 데이터 연결")
        data_source = st.radio("데이터 소스", ["☁️ GitHub (기본)", "📂 엑셀 파일 직접 업로드"])
        
        st.markdown("---")
        st.header("📊 분석 옵션")
        tab_mode = st.radio("메뉴 선택", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit_mode = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])

    # 데이터 담을 변수 초기화
    df_target = pd.DataFrame()
    usage_cols = []
    
    # -------------------------------------------------------------------------
    # A. GitHub에서 가져오기
    # -------------------------------------------------------------------------
    if data_source.startswith("☁️"):
        # 단위 선택에 따라 필요한 시트만 로딩 (속도 최적화)
        target_sheet = SHEET_VOL if unit_mode.startswith("부피") else SHEET_CAL
        
        with st.spinner(f"깃허브에서 '{EXCEL_FILE_NAME}'의 [{target_sheet}] 시트를 가져오는 중..."):
            df_raw, success = load_excel_from_github(GITHUB_USER, REPO_NAME, EXCEL_FILE_NAME, target_sheet)
            
            if not success:
                st.error("🚨 깃허브에서 엑셀 파일을 읽지 못했습니다.")
                st.warning(f"1. 아이디/저장소 이름 확인\n2. 엑셀 파일명({EXCEL_FILE_NAME}) 확인\n3. 시트 이름({target_sheet})이 맞는지 확인해주세요.")
                return
            else:
                st.success(f"✅ GitHub 로드 완료! ({target_sheet})")
                df_target, usage_cols, date_col = preprocess_data(df_raw)

    # -------------------------------------------------------------------------
    # B. 직접 파일 업로드
    # -------------------------------------------------------------------------
    else:
        uploaded_file = st.file_uploader("엑셀 파일(.xlsx)을 업로드해주세요", type=['xlsx'])
        if uploaded_file:
            try:
                target_sheet = SHEET_VOL if unit_mode.startswith("부피") else SHEET_CAL
                df_raw = pd.read_excel(uploaded_file, sheet_name=target_sheet, engine='openpyxl')
                df_target, usage_cols, date_col = preprocess_data(df_raw)
            except ValueError:
                st.error(f"엑셀 파일 안에 '{target_sheet}' 라는 이름의 시트가 없습니다!")
                return
        else:
            st.info("👈 파일을 업로드하면 분석이 시작됩니다.")
            return

    if df_target.empty: return

    # -------------------------------------------------------------------------
    # 탭 1: 실적 분석
    # -------------------------------------------------------------------------
    if tab_mode.startswith("1"):
        unit_label = "천m³" if unit_mode.startswith("부피") else "GJ"
        st.subheader(f"📈 실적 분석 ({unit_label})")
        
        years = st.slider("기간", int(df_target['Year'].min()), int(df_target['Year'].max()), (2015, 2025))
        df_sub = df_target[(df_target['Year'] >= years[0]) & (df_target['Year'] <= years[1])]
        
        # KPI
        total = df_sub[usage_cols].sum().sum()
        st.metric(f"{years[0]}~{years[1]}년 누적 판매량", f"{total:,.0f} {unit_label}")
        
        # 차트
        df_yr = df_sub.groupby('Year')[usage_cols].sum().reset_index().melt(id_vars='Year', var_name='용도', value_name='Val')
        fig1 = px.bar(df_yr, x='Year', y='Val', color='용도', title="연도별 판매량", text_auto='.2s')
        st.plotly_chart(fig1, use_container_width=True)
        
        df_mon = df_sub.groupby('Month')[usage_cols].sum().reset_index().melt(id_vars='Month', var_name='용도', value_name='Val')
        fig2 = px.line(df_mon, x='Month', y='Val', color='용도', markers=True, title="월별 패턴")
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 탭 2: 예측 (2035)
    # -------------------------------------------------------------------------
    else:
        unit_label = "천m³" if unit_mode.startswith("부피") else "GJ"
        st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
        
        future_years = np.arange(2026, 2036).reshape(-1, 1)
        res = []
        
        progress = st.progress(0, text="AI 예측 분석 중...")
        
        for i, col in enumerate(usage_cols):
            tmp = df_target.groupby('Year')[col].sum().reset_index()
            model = LinearRegression()
            model.fit(tmp['Year'].values.reshape(-1, 1), tmp[col].values)
            pred = model.predict(future_years)
            pred = [max(0, p) for p in pred]
            
            for y, v in zip(tmp['Year'], tmp[col]): res.append({'Year':y, 'Type':'실적', 'Val':v, 'Use':col})
            for y, v in zip(future_years.flatten(), pred): res.append({'Year':y, 'Type':'예측', 'Val':v, 'Use':col})
            
            progress.progress((i+1)/len(usage_cols))
            
        progress.empty()
        df_res = pd.DataFrame(res)
        
        fig3 = px.line(df_res, x='Year', y='Val', color='Use', line_dash='Type', markers=True, title="2035 장기 전망")
        fig3.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="Forecast")
        st.plotly_chart(fig3, use_container_width=True)
        
        df_piv = df_res[df_res['Type']=='예측'].pivot_table(index='Year', columns='Use', values='Val')
        st.download_button("예측 결과 다운로드 (CSV)", df_piv.to_csv().encode('utf-8-sig'), "forecast.csv")

if __name__ == "__main__":
    main()
