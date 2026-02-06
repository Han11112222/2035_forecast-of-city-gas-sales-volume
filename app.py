import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정 & 폰트
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# 🟢 [매핑] 컬럼명 -> 표준 그룹
USE_COL_TO_GROUP = {
    # 🏠 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    
    # 🏪 영업용
    "일반용": "영업용", "일반용(1)": "영업용", "일반용(2)": "영업용", 
    "영업용_일반용1": "영업용", "영업용_일반용2": "영업용", 
    "일반용1(영업)": "영업용", "일반용2(영업)": "영업용", "일반용1": "영업용",
    
    # 🏢 업무용
    "업무난방용": "업무용", "냉방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    "업무용_일반용1": "업무용", "업무용_일반용2": "업무용", "업무용_업무난방": "업무용", 
    "업무용_냉난방": "업무용", "업무용_주한미군": "업무용", 
    "일반용1(업무)": "업무용", "일반용2(업무)": "업무용",
    
    # 🏭 산업용
    "산업용": "산업용",
    
    # 🚌 수송용
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용", "CNG": "수송용", "BIO": "수송용",
    
    # ⚡ 발전/기타
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용", "열전용설비용(주택외)": "열전용설비용"
}

# ─────────────────────────────────────────────────────────
# 🟢 2. 파일 로딩 (형님 코드 반영 + 스마트 처리)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_file_robust(uploaded_file):
    if uploaded_file is None: return None
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        sheets = {name: excel.parse(name) for name in excel.sheet_names}
        return sheets
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            return {"default": df}
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='cp949')
                return {"default": df}
            except: return None

def clean_df(df):
    if df is None: return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
    return df

def make_long_data(df, label):
    df = clean_df(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns: return pd.DataFrame()
    
    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    for col in df.columns:
        group = USE_COL_TO_GROUP.get(col)
        if not group: continue
        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label
        sub['값'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def find_sheet(data_dict, keywords, unit_keyword=None):
    if not data_dict: return None
    
    # 1순위: 키워드 + 단위 모두 일치
    if unit_keyword:
        for name, df in data_dict.items():
            clean = name.replace(" ", "")
            if any(k in clean for k in keywords) and (unit_keyword in clean):
                return df
                
    # 2순위: 키워드만 일치
    for name, df in data_dict.items():
        clean = name.replace(" ", "")
        if any(k in clean for k in keywords):
            return df
            
    # 3순위: 단일 파일
    if len(data_dict) == 1: return list(data_dict.values())[0]
    return None

# ─────────────────────────────────────────────────────────
# 🟢 3. 분석 화면
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    df_act = long_df[long_df['구분'].str.contains('실적')].copy()
    if df_act.empty: st.error("실적 데이터 없음"); return
    
    all_years = sorted([int(y) for y in df_act['연'].unique()])
    
    if len(all_years) >= 10: default_years = all_years[-10:]
    else: default_years = all_years
        
    selected_years = st.multiselect("연도 선택", options=all_years, default=default_years)
    if not selected_years: return
    
    df_filtered = df_act[df_act['연'].isin(selected_years)]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 📈 월별 추이")
        mon_grp = df_filtered.groupby(['연', '월'])['값'].sum().reset_index()
        fig1 = px.line(mon_grp, x='월', y='값', color='연', markers=True)
        fig1.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown(f"#### 🧱 용도별 구성비")
        yr_grp = df_filtered.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig2 = px.bar(yr_grp, x='연', y='값', color='그룹', text_auto='.2s')
        fig2.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### 📋 상세 수치")
    st.dataframe(df_filtered.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 예측 화면 (판매량/공급량 로직 분리 및 최적화)
# ─────────────────────────────────────────────────────────
def generate_trend_insight(hist_df, pred_df):
    if hist_df.empty or pred_df.empty: return ""
    hist_yearly = hist_df.groupby('연')['값'].sum().sort_index()
    pred_yearly = pred_df.groupby('연')['값'].sum().sort_index()
    
    diffs = hist_yearly.diff()
    max_up_year = diffs.idxmax() if not diffs.dropna().empty else None
    max_down_year = diffs.idxmin() if not diffs.dropna().empty else None
    
    start_val = pred_yearly.iloc[0]
    end_val = pred_yearly.iloc[-1]
    years = len(pred_yearly)
    if start_val > 0:
        cagr = (end_val / start_val) ** (1 / years) - 1
        trend_str = "지속적인 증가세" if cagr > 0.01 else "감소세" if cagr < -0.01 else "보합세"
    else: trend_str = "변동"

    insight = f"💡 **[AI 분석]** 과거 데이터를 분석한 결과, **{int(max_up_year) if max_up_year else '-'}년의 상승**과 **{int(max_down_year) if max_down_year else '-'}년의 하락/조정**을 종합하여 볼 때, 향후 2035년까지는 **{trend_str}**가 유지될 것으로 전망됩니다."
    return insight

def render_prediction_2035(long_df, unit_label, start_pred_year, train_years_selected, is_supply_mode=False):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 🔴 학습 데이터 필터링
    # 판매량: 선택된 연도만 학습 (과거)
    # 공급량: 선택된 연도 + 확정계획(26~28)까지 학습에 반영
    filter_cond = long_df['연'].isin(train_years_selected)
    if is_supply_mode:
        filter_cond = filter_cond | (long_df['구분'] == '확정계획')
        
    df_train = long_df[filter_cond].copy()
    
    if df_train.empty: st.warning("학습 데이터가 부족합니다."); return
    
    st.markdown("##### 📊 추세 분석 모델 선택")
    pred_method = st.radio("방법", [
        "1. 선형 회귀 (Linear)", "2. 2차 곡선 (Quadratic)", "3. 3차 곡선 (Cubic)",
        "4. 로그 추세 (Log)", "5. 지수 평활 (Holt)", "6. CAGR (성장률)"
    ], horizontal=True)

    desc = ""
    if "선형" in pred_method: desc = "선형 회귀: 매년 일정량씩 꾸준히 변하는 직선 추세"
    elif "2차" in pred_method: desc = "2차 곡선: 성장이 가속화되거나 정점을 찍고 내려오는 곡선 추세"
    elif "3차" in pred_method: desc = "3차 곡선: 상승과 하락 사이클이나 변곡점이 있는 복잡한 추세"
    elif "로그" in pred_method: desc = "로그 추세: 초반 급성장 후 점차 안정화되는(성숙기) 패턴"
    elif "지수" in pred_method: desc = "지수 평활: 최근 실적에 가중치를 두어 최신 트렌드를 민감하게 반영"
    elif "CAGR" in pred_method: desc = "CAGR: 과거의 연평균 성장률이 미래에도 유지된다고 가정"
    st.info(f"ℹ️ **{desc}**")

    # 전체 데이터 (참고용)
    df_grp = long_df.groupby(['연', '그룹', '구분'])['값'].sum().reset_index()
    # 학습 데이터
    df_train_grp = df_train.groupby(['연', '그룹'])['값'].sum().reset_index()
    
    groups = df_grp['그룹'].unique()
    future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
    results = []
    
    total_hist_vals = []
    total_pred_vals = []

    for grp in groups:
        sub_train = df_train_grp[df_train_grp['그룹'] == grp]
        sub_full = df_grp[df_grp['그룹'] == grp]
        
        if len(sub_train) < 2: continue
        
        X = sub_train['연'].values.reshape(-1, 1)
        y = sub_train['값'].values
        pred = []
        
        try:
            if "선형" in pred_method: model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
            elif "2차" in pred_method: model = make_pipeline(PolynomialFeatures(2), LinearRegression()); model.fit(X, y); pred = model.predict(future_years)
            elif "3차" in pred_method: model = make_pipeline(PolynomialFeatures(3), LinearRegression()); model.fit(X, y); pred = model.predict(future_years)
            elif "로그" in pred_method:
                X_idx = np.arange(1, len(X) + 1).reshape(-1, 1)
                X_fut = np.arange(len(X) + 1, len(X) + 1 + len(future_years)).reshape(-1, 1)
                model = LinearRegression(); model.fit(np.log(X_idx), y); pred = model.predict(np.log(X_fut))
            elif "지수" in pred_method:
                fit = np.polyfit(X.flatten(), np.log(y + 1), 1)
                pred = np.exp(fit[1] + fit[0] * future_years.flatten())
            else: 
                cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1
                pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
        except:
            model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
            
        pred = [max(0, p) for p in pred]
        
        # 🔴 데이터 병합 (중복 방지 및 모드별 처리)
        added_years = set()
        
        # 1. 과거 실적 (학습 기간에 해당하는 것만)
        # 공급량 모드: 2026년 미만만 실적 취급
        # 판매량 모드: 전체 실적 취급 (이미 로딩 단계에서 미래 데이터 제거됨)
        hist_mask = sub_full['연'].isin(train_years_selected)
        if is_supply_mode and start_pred_year == 2029:
             hist_mask = hist_mask & (sub_full['연'] < 2026)
        
        hist_data = sub_full[hist_mask]
        
        for _, row in hist_data.iterrows():
            if row['연'] not in added_years:
                results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '실적'})
                total_hist_vals.append({'연': row['연'], '값': row['값']})
                added_years.add(row['연'])
            
        # 2. 확정 계획 (공급량 모드 전용, 2026~2028)
        if is_supply_mode and start_pred_year == 2029:
            plan_data = sub_full[sub_full['연'].between(2026, 2028)]
            for _, row in plan_data.iterrows():
                results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '확정계획'})
                
        # 3. AI 미래 예측
        for yr, v in zip(future_years.flatten(), pred): 
            results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측(AI)'})
            total_pred_vals.append({'연': yr, '값': v})
        
    df_res = pd.DataFrame(results)
    
    insight_text = generate_trend_insight(pd.DataFrame(total_hist_vals), pd.DataFrame(total_pred_vals))
    if insight_text: st.success(insight_text)
    
    st.markdown("---")
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    
    fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green")
    fig.add_vrect(
        x0=start_pred_year-0.5, x1=2035.5, 
        fillcolor="green", opacity=0.05, 
        annotation_text="예측 값", annotation_position="inside top"
    )
    
    if is_supply_mode and start_pred_year == 2029:
        fig.add_vrect(
            x0=2025.5, x1=2028.5, 
            fillcolor="yellow", opacity=0.1, 
            annotation_text="확정계획", annotation_position="inside top"
        )
    
    fig.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🧱 연도별 공급량 구성 (누적 스택)")
    fig_stack = px.bar(df_res, x='연', y='값', color='그룹', title="연도별 용도 구성비", text_auto='.2s')
    fig_stack.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig_stack, use_container_width=True)
    
    with st.expander("📋 연도별 상세 데이터 확인"):
        piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum')
        piv['합계'] = piv.sum(axis=1)
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 5. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량", "2. 공급량"], index=1)
        sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        unit = st.radio("단위 선택", ["열량 (GJ)", "부피 (천m³)"], index=0)
        
        # 단위 키워드 (판매량 파일 찾기용)
        unit_key = "열량" if "열량" in unit else "부피"
        
        st.markdown("---")
        st.subheader("파일 업로드")
        
        up_sales = st.file_uploader("1. 판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s", accept_multiple_files=True)
        up_supply = st.file_uploader("2. 공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        st.markdown("---")
    
    df_final = pd.DataFrame()
    start_year = 2026
    is_supply = False
    
    # 🟢 [모드 1] 판매량: 2025년 이하 실적만 사용 -> 2026년부터 예측
    if mode.startswith("1"):
        start_year = 2026
        if up_sales:
            data = load_file_robust(up_sales) # 스마트 필터 대신 단순 로드 후 정제
            if data:
                # 1. 단위에 맞는 시트 찾기
                df_p = find_sheet(data, ["계획"], unit_key)
                df_a = find_sheet(data, ["실적"], unit_key)
                
                # CSV 예외
                if df_p is None and df_a is None and len(data) == 1: 
                    df_a = list(data.values())[0]
                
                # 2. 데이터 병합 (중복 방지 핵심: 과거는 실적만, 미래 계획은 무시)
                # 형님 요청: 판매량은 2017~2025 실적 반영 -> 2026~2035 추정
                # 따라서 df_p(계획) 파일은 아예 안 쓰는 게 가장 깔끔함.
                
                # 실적 파일 처리
                long_a = make_long_data(df_a, "실적")
                # 2025년 이하 데이터만 남김 (확실하게 하기 위해)
                long_a = long_a[long_a['연'] <= 2025]
                
                df_final = pd.concat([long_a], ignore_index=True)
                
        else: st.info("👈 [판매량 파일]을 업로드하세요."); return

    # 🟢 [모드 2] 공급량: 이전 로직 그대로 유지 (26~28 확정계획 반영)
    else:
        start_year = 2029 
        is_supply = True
        if up_supply:
            data = load_file_robust(up_supply)
            if data:
                df_hist = find_sheet(data, ["공급량_실적", "실적"])
                df_plan = find_sheet(data, ["공급량_계획", "계획"])
                
                if df_hist is None and df_plan is None and len(data) == 1:
                    df_hist = list(data.values())[0]
                
                long_h = make_long_data(df_hist, "실적")
                long_p = make_long_data(df_plan, "확정계획")
                df_final = pd.concat([long_h, long_p], ignore_index=True)
        else: st.info("👈 [공급량 파일]을 업로드하세요."); return

    if not df_final.empty:
        with st.sidebar:
            st.markdown("### 📅 데이터 학습 기간 설정")
            all_years = sorted([int(y) for y in df_final['연'].unique()])
            default_yrs = all_years 
            train_years = st.multiselect("학습 연도 (2025년 포함됨)", options=all_years, default=default_yrs)

        if "실적" in sub_mode:
            render_analysis_dashboard(df_final, unit)
        elif "2035" in sub_mode:
            render_prediction_2035(df_final, unit, start_year, train_years, is_supply)
        elif "가정용" in sub_mode:
            with st.sidebar:
                up_t = st.file_uploader("기온 파일(.csv)", type=["csv", "xlsx"])
            st.info("기온 데이터 업로드 시 분석 가능")

if __name__ == "__main__":
    main()
