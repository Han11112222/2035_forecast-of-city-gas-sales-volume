import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 통합 분석", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# ─────────────────────────────────────────────────────────
# 🟢 2. 용도별 매핑 (형님 지시사항 최종 확정)
# ─────────────────────────────────────────────────────────

# 1) 판매량용 매핑 (기존 유지)
MAPPING_SALES = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    "일반용": "영업용", "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용", "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용"
}

# 2) 공급량용 매핑 (최종 수정: 주한미군 -> 업무용 포함)
MAPPING_SUPPLY_SPECIFIC = {
    # 1. 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", 
    "개별난방": "가정용", "중앙난방": "가정용",
    
    # 2. 영업용
    "영업용": "영업용",
    
    # 3. 업무용 (일반1, 일반2, 업무난방, 냉난방 + 주한미군)
    "일반용(1)": "업무용", "일반용1": "업무용", "일반용1(영업)": "업무용", # 혹시 모를 변형 포함
    "일반용(2)": "업무용", "일반용2": "업무용", "일반용1(업무)": "업무용",
    "업무난방용": "업무용", "냉난방용": "업무용", "냉방용": "업무용", 
    "주한미군": "업무용", # 👈 업무용으로 편입 완료
    
    # 4. 산업용
    "산업용": "산업용",
    
    # 5. 수송용
    "수송용(CNG)": "수송용", "CNG": "수송용",
    "수송용(BIO)": "수송용", "BIO": "수송용"
    
    # 나머지는 매핑하지 않고 원래 이름 그대로 사용 (열병합, 연료전지 등)
}

# ─────────────────────────────────────────────────────────
# 🟢 3. 파일 로딩 및 전처리 (Garbage Cleaning 유지)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_all_sheets(uploaded_file):
    """파일의 모든 시트를 읽어서 딕셔너리로 반환"""
    if uploaded_file is None: return {}
    data_dict = {}
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        for sheet in excel.sheet_names:
            data_dict[sheet] = excel.parse(sheet)
    except:
        # CSV인 경우
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            data_dict["default"] = df
        except:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding='cp949')
                data_dict["default"] = df
            except: pass
    return data_dict

def clean_df(df):
    if df is None: return pd.DataFrame()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    
    # 1. 불필요한 컬럼 삭제 (Unnamed, 열 1, 열 2, 0 등)
    cols = []
    for c in df.columns:
        if "Unnamed" in c: continue
        if re.search(r'^열\s*\d+', c): continue # 열 1, 열 2...
        if c == '0': continue
        cols.append(c)
    df = df[cols]
    
    # 2. 날짜 변환
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def make_long_data(df, label, mode='sales'):
    df = clean_df(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns: return pd.DataFrame()
    
    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    exclude_cols = ['연', '월', '날짜', '평균기온', '총공급량', '총합계', '비교(V-W)', '소 계', '소계']

    for col in df.columns:
        if col in exclude_cols: continue
        
        # 값이 0이면 컬럼 자체를 스킵 (그래프 오염 방지)
        val_series = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if val_series.sum() == 0: continue

        # 그룹 매핑
        if mode == 'sales':
            group = MAPPING_SALES.get(col)
            if not group: continue 
        else:
            if df[col].dtype == object: continue
            # 공급량: 매핑표에 있으면 쓰고, 없으면 원래 이름 사용
            group = MAPPING_SUPPLY_SPECIFIC.get(col, col)

        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label
        sub['값'] = val_series
        
        # 0인 행 제거
        sub = sub[sub['값'] != 0]
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 분석 화면
# ─────────────────────────────────────────────────────────
def render_analysis_dashboard(long_df, unit_label):
    st.subheader(f"📊 실적 분석 ({unit_label})")
    
    # 실적만 필터링
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
    piv = df_filtered.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
    piv['소계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 5. 예측 화면
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

def render_prediction_2035(long_df, unit_label, start_pred_year, train_years_selected, is_supply_mode):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    # 🔴 학습 데이터 필터링
    filter_cond = long_df['연'].isin(train_years_selected)
    if is_supply_mode:
        filter_cond = filter_cond | (long_df['구분'] == '확정계획')
        
    df_train = long_df[filter_cond].copy()
    
    if df_train.empty: st.warning("학습 데이터가 부족합니다."); return
    
    st.markdown("##### 📊 추세 분석 모델 선택")
    pred_method = st.radio("방법", ["선형 회귀", "2차 곡선", "3차 곡선", "로그 추세", "지수 평활", "CAGR"], horizontal=True)
    
    if "선형" in pred_method: st.info("ℹ️ 매년 일정량씩 꾸준히 변하는 직선 추세")
    elif "2차" in pred_method: st.info("ℹ️ 성장이 가속화되거나 둔화되는 곡선 추세")
    
    df_grp = long_df.groupby(['연', '그룹', '구분'])['값'].sum().reset_index()
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
                model = LinearRegression(); model.fit(np.log(X - X.min() + 1), y); pred = model.predict(np.log(future_years - X.min() + 1))
            elif "지수" in pred_method:
                fit = np.polyfit(X.flatten(), np.log(y + 1), 1)
                pred = np.exp(fit[1] + fit[0] * future_years.flatten())
            else: 
                cagr = (y[-1]/y[0])**(1/(len(y)-1)) - 1
                pred = [y[-1] * ((1+cagr)**(i+1)) for i in range(len(future_years))]
        except:
            model = LinearRegression(); model.fit(X, y); pred = model.predict(future_years)
            
        pred = [max(0, p) for p in pred]
        
        # 🔴 [데이터 병합 - 엄격한 규칙 적용]
        added_years = set()
        
        # 1. 과거 실적
        # 공급량 모드(2029 시작) -> 26년 미만만 실적으로 취급
        hist_mask = sub_full['연'].isin(train_years_selected)
        if is_supply_mode and start_pred_year == 2029:
             hist_mask = hist_mask & (sub_full['연'] < 2026)
        # 판매량 모드(2026 시작) -> 2026 미만만 실적 취급
        elif not is_supply_mode:
             hist_mask = hist_mask & (sub_full['연'] < start_pred_year)
        
        hist_data = sub_full[hist_mask]
        for _, row in hist_data.iterrows():
            if row['연'] not in added_years:
                results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '실적'})
                total_hist_vals.append({'연': row['연'], '값': row['값']})
                added_years.add(row['연'])
            
        # 2. 확정 계획 (공급량 전용, 2026~2028)
        if is_supply_mode and start_pred_year == 2029:
            # 파일에서 직접 확정계획 찾아서 넣기
            plan_data = sub_full[sub_full['연'].between(2026, 2028)]
            for _, row in plan_data.iterrows():
                if row['연'] not in added_years:
                    results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '확정계획'})
                    added_years.add(row['연'])
                
        # 3. AI 미래 예측
        for yr, v in zip(future_years.flatten(), pred):
            if yr not in added_years: 
                results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측(AI)'})
                total_pred_vals.append({'연': yr, '값': v})
                added_years.add(yr)
        
    df_res = pd.DataFrame(results)
    
    insight_text = generate_trend_insight(pd.DataFrame(total_hist_vals), pd.DataFrame(total_pred_vals))
    if insight_text: st.success(insight_text)
    
    st.markdown("---")
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
    
    fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green")
    fig.add_vrect(x0=start_pred_year-0.5, x1=2035.5, fillcolor="green", opacity=0.05, annotation_text="예측 값", annotation_position="inside top")
    
    if is_supply_mode and start_pred_year == 2029:
        fig.add_vrect(x0=2025.5, x1=2028.5, fillcolor="yellow", opacity=0.1, annotation_text="확정계획", annotation_position="inside top")
    
    fig.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🧱 연도별 공급량 구성 (누적 스택)")
    fig_stack = px.bar(df_res, x='연', y='값', color='그룹', title="연도별 용도 구성비", text_auto='.2s')
    fig_stack.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig_stack, use_container_width=True)
    
    with st.expander("📋 연도별 상세 데이터 확인"):
        piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
        piv['소계'] = piv.sum(axis=1)
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 6. 기온 분석
# ─────────────────────────────────────────────────────────
def render_household_analysis(long_df, temp_file):
    st.subheader(f"🏠 가정용 정밀 분석 (기온 영향)")
    if temp_file is None:
        st.warning("⚠️ 기온 데이터 파일(.csv)을 업로드해주세요."); return
        
    temp_dict = load_all_sheets(temp_file)
    if not temp_dict: return
    
    df_temp = list(temp_dict.values())[0]
    df_temp = clean_df(df_temp)
    cols = [c for c in df_temp.columns if "기온" in c]
    
    if cols:
        mon_temp = df_temp.groupby(['연', '월'])[cols[0]].mean().reset_index()
        mon_temp.rename(columns={cols[0]: '평균기온'}, inplace=True)
        
        df_home = long_df[long_df['그룹'] == '가정용'].groupby(['연', '월'])['값'].sum().reset_index()
        df_merged = pd.merge(df_home, mon_temp, on=['연', '월'], how='inner')
        
        if not df_merged.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.scatter(df_merged, x='평균기온', y='값', color='연', trendline="ols", title="기온 vs 가정용 사용량")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                corr = df_merged['평균기온'].corr(df_merged['값'])
                st.metric("기온과의 상관계수", f"{corr:.2f}")
                st.caption("(-1에 가까울수록 기온이 낮으면 사용량이 증가)")
        else: st.warning("날짜가 일치하는 데이터가 없습니다.")
    else: st.error("기온 컬럼을 찾을 수 없습니다.")

# ─────────────────────────────────────────────────────────
# 🟢 7. 최종값 확인
# ─────────────────────────────────────────────────────────
def render_final_check(long_df, unit_label):
    st.subheader(f"🏁 최종 확정 데이터 시각화 ({unit_label})")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 📈 연도별 추세")
        yr_grp = long_df.groupby(['연', '그룹'])['값'].sum().reset_index()
        fig1 = px.line(yr_grp, x='연', y='값', color='그룹', markers=True)
        fig1.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown(f"#### 🧱 용도별 구성비")
        fig2 = px.bar(yr_grp, x='연', y='값', color='그룹', text_auto='.2s')
        fig2.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### 📋 최종 데이터 상세")
    piv = long_df.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
    piv['소계'] = piv.sum(axis=1)
    st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 8. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량", "2. 공급량", "3. 최종값 확인"], index=1)
        
        sub_mode = ""
        if not mode.startswith("3"):
            sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        
        idx = 0 
        if mode.startswith("1"): idx = 0 
        unit = st.radio("단위 선택", ["열량 (GJ)", "부피 (천m³)"], index=idx)
        unit_key = "열량" if "열량" in unit else "부피"
        
        st.markdown("---")
        st.subheader("파일 업로드")
        
        up_sales = st.file_uploader("1. 판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s", accept_multiple_files=True)
        up_supply = st.file_uploader("2. 공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        up_final = st.file_uploader("3. 최종값.xlsx", type=["xlsx", "csv"], key="f")
        st.markdown("---")
    
    df_final = pd.DataFrame()
    start_year = 2026
    is_supply = False
    
    # 🟢 [모드 1] 판매량
    if mode.startswith("1"):
        start_year = 2026
        if up_sales:
            data_dict = load_all_sheets(up_sales[0] if isinstance(up_sales, list) else up_sales)
            
            # 시트명에 '실적' + '단위'가 있는 것 찾기
            target_df = None
            for sheet_name, df in data_dict.items():
                if "실적" in sheet_name and unit_key in sheet_name:
                    target_df = df; break
            
            # 없으면 '실적'이라도 찾음
            if target_df is None:
                for sheet_name, df in data_dict.items():
                    if "실적" in sheet_name: target_df = df; break
            
            # 그래도 없으면 CSV 기본값
            if target_df is None and "default" in data_dict:
                target_df = data_dict["default"]
            elif target_df is None and len(data_dict) > 0:
                target_df = list(data_dict.values())[0]

            if target_df is not None:
                long_a = make_long_data(target_df, "실적", 'sales')
                # 2025년 이하만 사용 (과거 실적) -> 2026부터 예측
                long_a = long_a[long_a['연'] <= 2025] 
                df_final = pd.concat([long_a], ignore_index=True)
        else: st.info("👈 [판매량 파일]을 업로드하세요."); return

    # 🟢 [모드 2] 공급량
    elif mode.startswith("2"):
        start_year = 2029 
        is_supply = True
        if up_supply:
            data_dict = load_all_sheets(up_supply)
            
            df_hist = None
            for name, df in data_dict.items():
                if "실적" in name: df_hist = df; break
            
            df_plan = None
            for name, df in data_dict.items():
                if "계획" in name: df_plan = df; break
            
            if df_hist is None and len(data_dict) > 0: df_hist = list(data_dict.values())[0]
            
            if df_hist is not None:
                long_h = make_long_data(df_hist, "실적", 'supply')
                df_final = long_h
                
                if df_plan is not None:
                    long_p = make_long_data(df_plan, "확정계획", 'supply')
                    df_final = pd.concat([long_h, long_p], ignore_index=True)
        else: st.info("👈 [공급량 파일]을 업로드하세요."); return

    # 🟢 [모드 3] 최종값
    elif mode.startswith("3"):
        if up_final:
            data_dict = load_all_sheets(up_final)
            if len(data_dict) > 0:
                df_raw = list(data_dict.values())[0]
                df_final = make_long_data(df_raw, "최종값", 'supply')
        else: st.info("👈 [최종값 파일]을 업로드하세요."); return

    # ── 공통 실행 ──
    if not df_final.empty:
        if not mode.startswith("3"):
            with st.sidebar:
                st.markdown("### 📅 데이터 학습 기간 설정")
                all_years = sorted([int(y) for y in df_final['연'].unique()])
                default_yrs = all_years 
                train_years = st.multiselect("학습 연도 (2025년 포함됨)", options=all_years, default=default_yrs)

        if mode.startswith("3"):
            render_final_check(df_final, unit)
        elif "실적" in sub_mode:
            render_analysis_dashboard(df_final, unit)
        elif "2035" in sub_mode:
            render_prediction_2035(df_final, unit, start_year, train_years, is_supply)
        elif "가정용" in sub_mode:
            with st.sidebar:
                up_t = st.file_uploader("기온 파일(.csv)", type=["csv", "xlsx"])
            render_household_analysis(df_final, up_t)

if __name__ == "__main__":
    main()
