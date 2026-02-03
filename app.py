import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from urllib.parse import quote
from pathlib import Path

# =============================================================================
# 🟢 [설정] Han형님의 깃허브 정보 (여기에 입력하세요!)
# =============================================================================
GITHUB_USER = "HanYeop"      # 형님의 깃허브 아이디
REPO_NAME = "GasProject"     # 저장소 이름
EXCEL_FILE_NAME = "판매량(계획_실적).xlsx"

# =============================================================================
# 🟢 [기존 로직 유지] 형님이 사용하시던 매핑 및 설정
# =============================================================================
st.set_page_config(page_title="도시가스 판매량 분석 및 예측", page_icon="🔥", layout="wide")

USE_COL_TO_GROUP = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "일반용": "영업용",
    "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "열전용설비용": "열전용설비용"
}

GROUP_OPTIONS = ["총량", "가정용", "영업용", "업무용", "산업용", "수송용", "열병합", "연료전지", "열전용설비용"]

# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 함수 (형님 로직 100% 유지)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data_from_github():
    """깃허브에서 엑셀 파일을 바로 읽어옵니다 (Pandas 기능 사용)"""
    try:
        # 깃허브 Raw URL 생성
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{quote(EXCEL_FILE_NAME)}"
        
        # 엑셀 파일 로드 (시트별로 다 가져옴)
        xls = pd.ExcelFile(url, engine='openpyxl')
        return xls
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def _clean_base(df):
    """형님의 데이터 정제 함수"""
    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    out["연"] = pd.to_numeric(out["연"], errors="coerce").astype("Int64")
    out["월"] = pd.to_numeric(out["월"], errors="coerce").astype("Int64")
    return out

def make_long(plan_df, actual_df):
    """형님의 Wide -> Long 변환 함수 (핵심!)"""
    plan_df = _clean_base(plan_df)
    actual_df = _clean_base(actual_df)

    records = []
    for label, df in [("계획", plan_df), ("실적", actual_df)]:
        for col in df.columns:
            if col in ["연", "월"]: continue

            group = USE_COL_TO_GROUP.get(col)
            # 매핑에 없는 컬럼(합계 등)은 제외
            if group is None: continue 

            base = df[["연", "월"]].copy()
            base["그룹"] = group
            base["용도"] = col
            base["계획/실적"] = label
            base["값"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            records.append(base)

    if not records:
        return pd.DataFrame(columns=["연", "월", "그룹", "용도", "계획/실적", "값"])

    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.dropna(subset=["연", "월"])
    long_df["연"] = long_df["연"].astype(int)
    long_df["월"] = long_df["월"].astype(int)
    return long_df

# -----------------------------------------------------------------------------
# 2. [신규 기능] 2035년 예측 함수 (형님 데이터를 받아서 처리)
# -----------------------------------------------------------------------------
def run_prediction(long_df, unit_label):
    st.markdown(f"### 🔮 2035 장기 판매량 예측 ({unit_label})")
    st.info("💡 기존 실적 데이터를 기반으로 '용도 그룹별' 추세를 분석하여 2035년까지 예측합니다.")

    # 실적 데이터만 추출
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    
    # 연도별/그룹별 합계 (월별 데이터를 연도별로 묶음)
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []

    # 진행률 표시
    progress = st.progress(0)
    
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue # 데이터가 너무 적으면 패스

        # 선형 회귀 학습
        model = LinearRegression()
        X = sub['연'].values.reshape(-1, 1)
        y = sub['값'].values
        model.fit(X, y)

        # 예측
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred] # 음수 방지

        # 결과 저장 (실적)
        for year, val in zip(sub['연'], sub['값']):
            results.append({'연': year, '그룹': grp, '판매량': val, 'Type': '실적'})
        
        # 결과 저장 (예측)
        for year, val in zip(future_years.flatten(), pred):
            results.append({'연': year, '그룹': grp, '판매량': val, 'Type': '예측'})
        
        progress.progress((i + 1) / len(groups))
    
    progress.empty()
    df_res = pd.DataFrame(results)

    # 전체 합계 라인 추가
    df_total = df_res.groupby(['연', 'Type'])['판매량'].sum().reset_index()
    df_total['그룹'] = '전체합계'
    df_final = pd.concat([df_res, df_total])

    # 차트 그리기
    fig = px.line(df_final, x='연', y='판매량', color='그룹', line_dash='Type',
                  markers=True, title=f"2035년까지의 용도별 장기 전망 ({unit_label})",
                  category_orders={"Type": ["실적", "예측"]})
    
    # 예측 구간 배경색
    fig.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측 구간")
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 다운로드
    piv = df_res[df_res['Type']=='예측'].pivot_table(index='연', columns='그룹', values='판매량')
    piv['전체합계'] = piv.sum(axis=1)
    
    st.dataframe(piv.style.format("{:,.0f}"))
    st.download_button("예측 결과 다운로드 (CSV)", piv.to_csv().encode('utf-8-sig'), "forecast_2035.csv")

# -----------------------------------------------------------------------------
# 3. 메인 실행 (통합)
# -----------------------------------------------------------------------------
def main():
    st.title("🔥 도시가스 실적 분석 및 예측 시스템")
    st.markdown("**Created by Han (Marketing Planning Team)**")

    # 사이드바
    with st.sidebar:
        st.header("설정")
        # 데이터 소스 선택 (형님 요청: 깃허브 기본 + 업로드 백업)
        src = st.radio("데이터 소스", ["GitHub (기본)", "엑셀 파일 업로드"])
        
        xls_file = None
        if src == "엑셀 파일 업로드":
            uploaded = st.file_uploader("엑셀 파일", type="xlsx")
            if uploaded: xls_file = pd.ExcelFile(uploaded, engine='openpyxl')
        else:
            xls_file = load_data_from_github()
            if xls_file is None:
                st.error("GitHub 연결 실패. 아이디/저장소/파일명을 확인하세요.")
        
        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 판매량 실적분석", "2. 판매량 예측 (2035)"])
        unit = st.radio("단위 선택", ["부피 (천m³)", "열량 (GJ)"])

    if xls_file is None:
        return

    # 데이터 로드 및 정제 (형님 코드 Logic 적용)
    # 단위에 따라 시트 선택
    try:
        if unit.startswith("부피"):
            df_plan = xls_file.parse("계획_부피")
            df_act = xls_file.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_plan = xls_file.parse("계획_열량")
            df_act = xls_file.parse("실적_열량")
            unit_label = "GJ"
        
        # Wide -> Long 변환 (형님 함수 호출)
        long_df = make_long(df_plan, df_act)
        
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")
        st.warning("엑셀 시트 이름('계획_부피', '실적_부피' 등)이 정확한지 확인해주세요.")
        return

    # 기능 분기
    if mode.startswith("1"):
        # ---------------------------------------------------------------------
        # [기능 1] 기존 실적 분석 (형님이 보시던 화면 스타일)
        # ---------------------------------------------------------------------
        st.subheader(f"📈 판매량 실적 대시보드 ({unit_label})")
        
        # 연도 필터
        years = sorted(long_df['연'].unique())
        sel_year = st.selectbox("기준 연도", years, index=len(years)-1)
        
        # 데이터 필터
        df_sub = long_df[long_df['연'] == sel_year]
        
        # KPI
        plan_sum = df_sub[df_sub['계획/실적']=='계획']['값'].sum()
        act_sum = df_sub[df_sub['계획/실적']=='실적']['값'].sum()
        
        c1, c2 = st.columns(2)
        c1.metric(f"{sel_year}년 계획 합계", f"{plan_sum:,.0f} {unit_label}")
        c2.metric(f"{sel_year}년 실적 합계", f"{act_sum:,.0f} {unit_label}", 
                  delta=f"{act_sum-plan_sum:,.0f} (차이)")
        
        st.markdown("---")
        
        # 1. 월별 추이 그래프
        st.markdown("#### 📅 월별 실적 추이")
        grp_mon = df_sub.groupby(['월', '계획/실적'])['값'].sum().reset_index()
        fig1 = px.line(grp_mon, x='월', y='값', color='계획/실적', markers=True, 
                       color_discrete_map={"계획": "blue", "실적": "green"})
        fig1.update_xaxes(dtick=1)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. 용도별 누적 그래프
        st.markdown("#### 🧱 용도별(그룹) 판매량")
        grp_use = df_sub[df_sub['계획/실적']=='실적'].groupby(['월', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(grp_use, x='월', y='값', color='그룹', title="월별/용도별 실적")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        # ---------------------------------------------------------------------
        # [기능 2] 2035 예측 (신규 추가)
        # ---------------------------------------------------------------------
        run_prediction(long_df, unit_label)

if __name__ == "__main__":
    main()
