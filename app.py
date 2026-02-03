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

def load_data_simple(uploaded_file=None):
    try:
        if uploaded_file:
            return pd.ExcelFile(uploaded_file, engine='openpyxl')
        elif Path(DEFAULT_SALES_XLSX).exists():
            return pd.ExcelFile(DEFAULT_SALES_XLSX, engine='openpyxl')
        return None
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# ─────────────────────────────────────────────────────────
# 2. [기능 1] 실적 분석 (✅ 연도 선택 버튼 수정 완료)
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 🔴 [데이터 준비] 오직 '실적' 데이터만 사용, 2025년까지만
    df_act = long_df[long_df['계획/실적'] == '실적'].copy()
    df_act = df_act[df_act['연'] <= 2025] # 2025년 이하만
    
    # 🔴 [UI 수정] 연도 선택 버튼 (Multi-Select)
    all_years = sorted(df_act['연'].unique())
    if not all_years:
        st.error("분석할 실적 데이터가 없습니다.")
        return

    # 기본값: 최근 3년
    default_years = all_years[-3:] if len(all_years) >= 3 else all_years
    
    st.markdown("##### 📅 분석할 연도를 선택하세요 (다중 선택)")
    selected_years = st.multiselect(
        "연도 선택",
        options=all_years,
        default=default_years,
        label_visibility="collapsed" # 라벨 숨김 (깔끔하게)
    )
    
    if not selected_years:
        st.warning("연도를 1개 이상 선택해주세요.")
        return

    # 🔴 [필터링] 선택된 연도 데이터만 남김
    df_filtered = df_act[df_act['연'].isin(selected_years)]

    st.markdown("---")

    # ---------------------------------------------------------
    # [그래프 1] 월별 실적 추이 (꺾은선 그래프)
    # ---------------------------------------------------------
    st.markdown(f"#### 📈 월별 실적 추이 ({', '.join(map(str, selected_years))})")
    
    # 월별, 연도별 합계 집계
    df_mon_compare = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
    
    fig1 = px.line(
        df_mon_compare, 
        x='월', 
        y='값', 
        color='연', 
        markers=True,
        title="월별 실적 추이 비교"
    )
    fig1.update_layout(
        xaxis=dict(tickmode='linear', dtick=1), # 1~12월 표시
        yaxis_title=unit_label,
        legend_title="연도"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # [표 1]
    st.markdown("##### 📋 월별 상세 수치")
    pivot_mon = df_mon_compare.pivot(index='월', columns='연', values='값').fillna(0)
    st.dataframe(pivot_mon.style.format("{:,.0f}"), use_container_width=True)
    
    st.markdown("---")

    # ---------------------------------------------------------
    # [그래프 2] 연도별 용도 누적 (스택 막대 그래프)
    # ---------------------------------------------------------
    st.markdown(f"#### 🧱 연도별 용도 구성비 ({', '.join(map(str, selected_years))})")
    
    # 연도별, 그룹별 합계 집계
    df_yr_usage = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    fig2 = px.bar(
        df_yr_usage, 
        x='연', 
        y='값', 
        color='그룹', 
        title="연도별 판매량 및 용도 구성",
        text_auto='.2s'
    )
    fig2.update_layout(
        xaxis_type='category', # 연도를 카테고리로
        yaxis_title=unit_label,
        legend_title="용도 그룹"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # [표 2]
    st.markdown("##### 📋 용도별 상세 수치")
    pivot_usage = df_yr_usage.pivot(index='연', columns='그룹', values='값').fillna(0)
    pivot_usage['합계'] = pivot_usage.sum(axis=1)
    st.dataframe(pivot_usage.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 3. [기능 2] 2035 예측 (최근 5년 트렌드 반영 보정)
# ─────────────────────────────────────────────────────────
def render_prediction_2035(long_df, unit_label):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    st.info("💡 **[보정됨]** 최근 5년(2021~2025)의 추세를 반영하여 예측합니다.")
    
    # 실적 데이터만 사용 (2025년 이하)
    df_act = long_df[(long_df['계획/실적'] == '실적') & (long_df['연'] <= 2025)].copy()
    
    df_train = df_act.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_train['그룹'].unique()
    future_years = np.arange(2026, 2036).reshape(-1, 1)
    results = []
    
    progress = st.progress(0)
    
    for i, grp in enumerate(groups):
        sub = df_train[df_train['그룹'] == grp]
        if len(sub) < 2: continue
        
        # 🔴 [보정 로직] 최근 5년 데이터만 사용
        sub_recent = sub.tail(5)
        if len(sub_recent) < 2: sub_recent = sub
            
        model = LinearRegression()
        model.fit(sub_recent['연'].values.reshape(-1, 1), sub_recent['값'].values)
        
        pred = model.predict(future_years)
        pred = [max(0, p) for p in pred] # 음수 방지
        
        for y, v in zip(sub['연'], sub['값']):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '실적'})
        for y, v in zip(future_years.flatten(), pred):
            results.append({'연': y, '그룹': grp, '판매량': v, 'Type': '예측'})
            
        progress.progress((i+1)/len(groups))
    progress.empty()
    
    df_res = pd.DataFrame(results)
    
    # [상단 그래프] 추세선
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    fig_line = px.line(
        df_res, x='연', y='판매량', color='그룹', 
        line_dash='Type', markers=True,
        title=f"용도별 장기 추세 ({unit_label})"
    )
    fig_line.add_vrect(x0=2025.5, x1=2035.5, fillcolor="green", opacity=0.1, annotation_text="예측 구간")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # [하단 그래프] 스택바
    st.markdown("#### 🧱 2035년 미래 예측 상세 (스택)")
    df_forecast_only = df_res[df_res['Type'] == '예측']
    
    fig_stack = px.bar(
        df_forecast_only, x='연', y='판매량', color='그룹',
        title="향후 10년 공급량 예측 구성비", text_auto='.2s'
    )
    fig_stack.update_layout(xaxis_type='category', yaxis_title=unit_label)
    st.plotly_chart(fig_stack, use_container_width=True)
    
    # [표 & 다운로드]
    st.markdown("##### 📋 미래 예측 데이터 상세")
    piv = df_forecast_only.pivot_table(index='연', columns='그룹', values='판매량')
    piv['합계'] = piv.sum(axis=1)
    
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    st.download_button(
        label="💾 예측 데이터 다운로드",
        data=piv.to_csv().encode('utf-8-sig'),
        file_name="forecast_2035.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매량 분석 & 예측")
    
    with st.sidebar:
        st.header("설정")
        uploaded = None
        if not Path(DEFAULT_SALES_XLSX).exists():
            st.warning(f"⚠️ '{DEFAULT_SALES_XLSX}' 파일이 없습니다.")
            uploaded = st.file_uploader("엑셀 파일 업로드", type="xlsx")
        else:
            st.success(f"✅ '{DEFAULT_SALES_XLSX}' 파일 연결됨")
            if st.checkbox("다른 파일 업로드하기"):
                uploaded = st.file_uploader("엑셀 파일 업로드", type="xlsx")

        st.markdown("---")
        mode = st.radio("분석 모드", ["1. 실적 분석", "2. 2035 예측"])
        unit = st.radio("단위", ["부피 (천m³)", "열량 (GJ)"])

    xls = load_data_simple(uploaded)
    if xls is None: return

    try:
        if unit.startswith("부피"):
            df_p = xls.parse("계획_부피")
            df_a = xls.parse("실적_부피")
            unit_label = "천m³"
        else:
            df_p = xls.parse("계획_열량")
            df_a = xls.parse("실적_열량")
            unit_label = "GJ"
            
        long_df = make_long(df_p, df_a)
        
    except Exception as e:
        st.error(f"시트 로드 실패: {e}")
        return

    if mode.startswith("1"):
        render_analysis_dashboard(long_df, unit_label)
    else:
        render_prediction_2035(long_df, unit_label)

if __name__ == "__main__":
    main()
