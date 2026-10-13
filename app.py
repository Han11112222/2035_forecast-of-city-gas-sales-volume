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
import os
from copy import copy
from openpyxl import load_workbook
from openpyxl.formula.translate import Translator
from openpyxl.utils import get_column_letter

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
# 🟢 2. 용도별 매핑 및 정렬 순서
# ─────────────────────────────────────────────────────────

# 1) 판매량용 매핑
MAPPING_SALES = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", "자가열전용": "가정용",
    "개별난방": "가정용", "중앙난방": "가정용", "가정용소계": "가정용",
    "일반용": "영업용", "업무난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    "산업용": "산업용", "수송용(CNG)": "수송용", "수송용(BIO)": "수송용",
    "열병합용": "열병합", "열병합용1": "열병합", "열병합용2": "열병합",
    "연료전지용": "연료전지", "연료전지": "연료전지",
    "열전용설비용": "열전용설비용"
}

# 2) 공급량용 매핑
MAPPING_SUPPLY_SPECIFIC = {
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", 
    "개별난방": "가정용", "중앙난방": "가정용",
    "영업용": "영업용",
    "일반용(1)": "업무용", "일반용1": "업무용", "일반용1(영업)": "업무용", "일반용1(업무)": "업무용",
    "일반용(2)": "업무용", "일반용2": "업무용", 
    "업무난방용": "업무난방용", "냉난방용": "업무용", "냉방용": "업무용", 
    "주한미군": "업무용", 
    "산업용": "산업용",
    "수송용(CNG)": "수송용", "CNG": "수송용",
    "수송용(BIO)": "수송용", "BIO": "수송용"
}

# 3) 상품별 상세 매핑
MAPPING_DETAIL = {
    "취사용": "취사용", 
    "개별난방용": "개별난방용", "개별난방": "개별난방용",
    "중앙난방용": "중앙난방용", "중앙난방": "중앙난방용",
    "영업용": "영업용",
    "일반용(1)": "일반용(1)", "일반용1": "일반용(1)", "일반용1(영업)": "일반용(1)",
    "일반용(2)": "일반용(2)", "일반용2": "일반용(2)",
    "업무난방용": "업무난방용", "업무난방": "업무난방용",
    "냉난방용": "냉난방용", "냉방용": "냉난방용",
    "주한미군": "주한미군",
    "산업용": "산업용",
    "열병합용": "열병합용", "열병합용1": "열병합용",
    "연료전지": "연료전지", "연료전지용": "연료전지",
    "자가열전용": "자가열전용",
    "열전용설비용": "열전용설비용(주택외)", "열전용설비용(주택외)": "열전용설비용(주택외)",
    "수송용(CNG)": "수송용(CNG)", "CNG": "수송용(CNG)",
    "수송용(BIO)": "수송용(BIO)", "BIO": "수송용(BIO)"
}

# 🟢 [정렬 순서]
ORDER_LIST_DETAIL = [
    "취사용", "개별난방용", "중앙난방용",       
    "영업용",                                 
    "일반용(1)", "일반용(2)", "업무난방용", "냉난방용", "주한미군", 
    "산업용",                                 
    "열병합용",                               
    "연료전지",                               
    "자가열전용",                             
    "열전용설비용(주택외)",                    
    "수송용(CNG)", "수송용(BIO)"              
]

# 🟢 [추가] 최종값 확인용 그룹핑 매핑 및 순서
MAPPING_FINAL_GROUP = {
    # 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용",
    # 영업용
    "영업용": "영업용",
    # 업무용
    "일반용(1)": "업무용", "일반용(2)": "업무용", "업무난방용": "업무용", "냉난방용": "업무용", "주한미군": "업무용",
    # 산업용
    "산업용": "산업용",
    # 열병합용
    "열병합용": "열병합용",
    # 연료전지
    "연료전지": "연료전지",
    # 자가열전용
    "자가열전용": "자가열전용",
    # 열전용설비용
    "열전용설비용(주택외)": "열전용설비용(주택외)",
    # 수송용
    "수송용(CNG)": "수송용", "수송용(BIO)": "수송용"
}

ORDER_LIST_FINAL_GROUP = [
    "가정용", "영업용", "업무용", "산업용", "열병합용", 
    "연료전지", "자가열전용", "열전용설비용(주택외)", "수송용"
]

# ─────────────────────────────────────────────────────────
# 🟢 3. 파일 로딩 및 전처리
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_all_sheets(uploaded_file):
    if uploaded_file is None: return {}
    data_dict = {}
    try:
        excel = pd.ExcelFile(uploaded_file, engine='openpyxl')
        for sheet in excel.sheet_names:
            data_dict[sheet] = excel.parse(sheet)
    except:
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
    
    if len(df.columns) > 0 and isinstance(df.columns[0], str) and "데이터 학습기간" in df.columns[0]:
        new_header = df.iloc[0] 
        df = df[1:] 
        df.columns = new_header

    df.columns = df.columns.astype(str).str.strip()
    
    cols = []
    for c in df.columns:
        if "Unnamed" in c: continue
        if re.search(r'^열\s*\d+', c): continue 
        if c == '0': continue
        cols.append(c)
    df = df[cols]
    
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        if '연' not in df.columns: df['연'] = df['날짜'].dt.year
        if '월' not in df.columns: df['월'] = df['날짜'].dt.month
        
    return df

def make_long_data(df, label, mode='sales'):
    df = clean_df(df)
    if df.empty or '연' not in df.columns: return pd.DataFrame()
    
    if '월' not in df.columns:
         df['월'] = 1 

    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연'])
    
    exclude_cols = ['연', '월', '날짜', '평균기온', '총공급량', '총합계', '비교(V-W)', '소 계', '소계']

    for col in df.columns:
        if col in exclude_cols: continue
        
        val_series = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if val_series.sum() == 0: continue

        if mode == 'detail':
            group = MAPPING_DETAIL.get(col)
            if not group: group = col 
        elif mode == 'sales':
            group = MAPPING_SALES.get(col)
            if not group: continue 
        else: # supply
            if df[col].dtype == object: continue
            group = MAPPING_SUPPLY_SPECIFIC.get(col, col)

        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label
        sub['값'] = val_series
        
        sub = sub[sub['값'] != 0]
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

# ─────────────────────────────────────────────────────────
# 🟢 4. 분석 화면
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

def render_prediction_2035(long_df, unit_label, start_pred_year, train_years_selected, is_supply_mode, custom_sort_list=None):
    st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
    
    filter_cond = long_df['연'].isin(train_years_selected)
    if is_supply_mode:
        filter_cond = filter_cond | (long_df['구분'] == '확정계획')
        
    df_train = long_df[filter_cond].copy()
    
    if df_train.empty: st.warning("학습 데이터가 부족합니다."); return
    
    st.markdown("##### 📊 추세 분석 모델 선택")
    st.markdown("""
    과거 데이터의 패턴을 분석하여 미래를 전망하는 통계적 기법들입니다. 
    데이터의 특성(지속 성장, 주기적 변동, 안정화 등)에 가장 적합한 모델을 선택해보세요.
    """)
    
    pred_method = st.radio("방법", ["선형 회귀", "2차 곡선", "3차 곡선", "로그 추세", "지수 평활", "CAGR"], horizontal=True)
    
    if "선형" in pred_method: st.info("ℹ️ 매년 일정량씩 꾸준히 변하는 직선 추세")
    elif "2차" in pred_method: st.info("ℹ️ 성장이 가속화되거나 둔화되는 곡선 추세")
    elif "3차" in pred_method: st.info("ℹ️ 상승과 하락 사이클이 있는 복잡한 추세")
    elif "로그" in pred_method: st.info("ℹ️ 초반 급성장 후 점차 안정화되는(성숙기) 패턴")
    elif "지수" in pred_method: st.info("ℹ️ 최근 실적에 가중치를 두어 최신 트렌드를 민감하게 반영")
    elif "CAGR" in pred_method: st.info("ℹ️ 과거의 연평균 성장률이 미래에도 유지된다고 가정")
    
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
        
        added_years = set()
        
        # 1. 과거 실적
        hist_mask = sub_full['연'].isin(train_years_selected)
        if is_supply_mode and start_pred_year == 2029:
             hist_mask = hist_mask & (sub_full['연'] < 2026)
        elif not is_supply_mode:
             hist_mask = hist_mask & (sub_full['연'] < start_pred_year)
        
        hist_data = sub_full[hist_mask]
        for _, row in hist_data.iterrows():
            if row['연'] not in added_years:
                results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '실적'})
                total_hist_vals.append({'연': row['연'], '값': row['값']})
                added_years.add(row['연'])
            
        # 2. 확정 계획 (공급량 전용)
        if is_supply_mode and start_pred_year == 2029:
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
    
    display_order = {} 
    
    if custom_sort_list:
        display_order = {'그룹': custom_sort_list}
        current_groups = df_res['그룹'].unique()
        valid_order = [g for g in custom_sort_list if g in current_groups]
        rest_groups = [g for g in current_groups if g not in valid_order]
        final_sort_order = valid_order + sorted(rest_groups)
        
        df_res['그룹'] = pd.Categorical(df_res['그룹'], categories=final_sort_order, ordered=True)
        df_res = df_res.sort_values(['연', '그룹'])
    else:
        df_res = df_res.sort_values(['연', '그룹'])
    
    insight_text = generate_trend_insight(pd.DataFrame(total_hist_vals), pd.DataFrame(total_pred_vals))
    if insight_text: st.success(insight_text)
    
    st.markdown("---")
    st.markdown("#### 📈 전체 장기 전망 (추세선)")
    
    fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True, 
                  category_orders=display_order)
    
    fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green")
    fig.add_vrect(x0=start_pred_year-0.5, x1=2035.5, fillcolor="green", opacity=0.05, annotation_text="예측 값", annotation_position="inside top")
    
    if is_supply_mode and start_pred_year == 2029:
        fig.add_vrect(x0=2025.5, x1=2028.5, fillcolor="yellow", opacity=0.1, annotation_text="확정계획", annotation_position="inside top")
    
    fig.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🧱 연도별 공급량 구성 (누적 스택)")
    fig_stack = px.bar(df_res, x='연', y='값', color='그룹', title="연도별 용도 구성비", text_auto='.2s',
                       category_orders=display_order)
    fig_stack.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig_stack, use_container_width=True)
    
    with st.expander("📋 연도별 상세 데이터 확인"):
        piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
        
        if custom_sort_list:
            cols_in_piv = piv.columns.tolist()
            sorted_cols = [c for c in custom_sort_list if c in cols_in_piv]
            remaining = [c for c in cols_in_piv if c not in sorted_cols]
            piv = piv[sorted_cols + remaining]
            
        piv['소계'] = piv.sum(axis=1) 
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

        if not piv.empty:
            st.markdown("---")
            if train_years_selected:
                sorted_years = sorted(train_years_selected)
                min_y, max_y = sorted_years[0], sorted_years[-1]
                full_range = set(range(min_y, max_y + 1))
                excluded = sorted(list(full_range - set(sorted_years)))
                exclude_str = ""
                if excluded:
                    exclude_str = f"(학습제외 연도 {', '.join(map(str, excluded))})"
                meta_info = f"데이터 학습기간 {min_y}~{max_y}{exclude_str}"
            else:
                meta_info = "데이터 학습기간 정보 없음"

            csv_buffer = io.StringIO()
            csv_buffer.write(f"{meta_info}\n")
            piv.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue().encode('utf-8-sig')

            st.download_button(
                label="📥 데이터 다운로드 (Excel/CSV)",
                data=csv_data,
                file_name=f"2035_예측결과_{unit_label}.csv",
                mime="text/csv"
            )

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
# 🟢 7. 최종값 확인 (수정됨: 용도별 적용 버튼 추가)
# ─────────────────────────────────────────────────────────
def render_final_check(long_df, unit_label):
    st.subheader(f"🏁 최종 확정 데이터 시각화 ({unit_label})")
    
    # 🟢 [추가] 우측 상단 '용도별 적용' 버튼 (Checkbox)
    col_t1, col_t2 = st.columns([4, 1])
    with col_t2:
        apply_usage_group = st.checkbox("☑️ 용도별 적용")
    
    df_res = long_df.copy()
    
    # 🟢 [로직] 버튼 체크 시 -> 그룹핑 및 정렬 기준 변경
    if apply_usage_group:
        # 매핑 적용 (매핑되지 않는 항목은 원래 이름 유지)
        df_res['New_Group'] = df_res['그룹'].map(MAPPING_FINAL_GROUP).fillna(df_res['그룹'])
        
        # 그룹별 합계 재계산
        df_res = df_res.groupby(['연', 'New_Group'])['값'].sum().reset_index()
        df_res.rename(columns={'New_Group': '그룹'}, inplace=True)
        
        target_order = ORDER_LIST_FINAL_GROUP
    else:
        target_order = ORDER_LIST_DETAIL

    # 정렬 및 범주형 변환
    current_groups = df_res['그룹'].unique()
    valid_order = [g for g in target_order if g in current_groups]
    rest_groups = [g for g in current_groups if g not in valid_order]
    final_sort_order = valid_order + sorted(rest_groups)
    
    df_res['그룹'] = pd.Categorical(df_res['그룹'], categories=final_sort_order, ordered=True)
    df_res = df_res.sort_values(['연', '그룹'])
    
    display_order = {'그룹': final_sort_order}
    
    # 1. Line Chart
    st.markdown("#### 📈 연도별 추세 (Line Chart)")
    fig = px.line(df_res, x='연', y='값', color='그룹', markers=True, category_orders=display_order)
    fig.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Stacked Bar
    st.markdown("#### 🧱 연도별 공급량 구성 (Stacked Bar)")
    fig_stack = px.bar(df_res, x='연', y='값', color='그룹', text_auto='.2s', category_orders=display_order)
    fig_stack.update_xaxes(dtick=1, tickformat="d")
    st.plotly_chart(fig_stack, use_container_width=True)
    
    # 3. Data Table & Download
    with st.expander("📋 최종 데이터 상세 (Click)"):
        piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
        
        cols_in_piv = piv.columns.tolist()
        sorted_cols = [c for c in final_sort_order if c in cols_in_piv]
        remaining = [c for c in cols_in_piv if c not in sorted_cols]
        piv = piv[sorted_cols + remaining]
            
        piv['소계'] = piv.sum(axis=1)
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
        
        if not piv.empty:
            st.markdown("---")
            csv_buffer = io.StringIO()
            piv.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue().encode('utf-8-sig')

            file_prefix = "용도별합산" if apply_usage_group else "상세상품별"
            st.download_button(
                label="📥 최종 데이터 다운로드 (Excel/CSV)",
                data=csv_data,
                file_name=f"최종확정데이터_{file_prefix}_{unit_label}.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────────────────
# 🟢 8-1. 2045 중장기 확장 : 엑셀 구조 인식 / 추세 로직 / 엑셀 쓰기
# ─────────────────────────────────────────────────────────
TARGET_END = 2045
YEAR_RE = re.compile(r"^\s*(\d{4})\s*년?\s*$")
TOTAL_LABELS = {"합계", "총합계", "총계"}

METHODS = {
    "1. 선형 추세": "linear",
    "2. 감쇠 추세 (추천)": "damped",
    "3. 연평균 성장률(CAGR)": "cagr",
    "4. 보합 (마지막 해 유지)": "hold",
    "5. 용도별 직접 입력 증감률": "custom",
}
METHOD_DESC = {
    "linear": "최근 N년의 연간 증감량(기울기)이 2045년까지 그대로 이어진다고 가정합니다. 가장 단순하지만 장기로 갈수록 값이 크게 벌어집니다.",
    "damped": "최근 N년의 기울기로 시작하되 매년 감쇠계수만큼 기울기가 줄어듭니다. 급격한 증가/감소가 점차 안정화되는 시나리오입니다.",
    "cagr": "최근 N년의 연평균 성장률(%)이 매년 유지된다고 가정합니다. 감소 용도는 감소폭이 점점 작아지고 0에 수렴합니다.",
    "hold": "마지막 해(2035년) 값을 2045년까지 유지합니다. 연료전지처럼 설비용량이 고정된 용도에 적합합니다.",
    "custom": "용도별로 직접 입력한 연 증감률(%)을 적용합니다. 정책·시장 전망 등 외부 가정을 반영할 때 사용합니다.",
}
FOLLOW_GLOBAL = "(전체 방법 따름)"


# ─────────────────────────────────────────────────────────
# 1. 엑셀 구조 인식
# ─────────────────────────────────────────────────────────
def _norm(s):
    return re.sub(r"\s+", "", str(s)) if s is not None else ""


def _find_header(ws):
    """연도(2025년 등)가 5개 이상 가로로 나열된 행을 헤더로 인식"""
    best = None
    for r in range(1, min(ws.max_row, 40) + 1):
        cols = {}
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if v is None:
                continue
            m = YEAR_RE.match(str(v))
            if m:
                cols[int(m.group(1))] = c
        if len(cols) >= 5 and (best is None or len(cols) > len(best[1])):
            best = (r, cols)
    return best


def parse_layout(ws, ws_v):
    """시트에서 헤더행/연도열/용도행/합계행을 찾는다. 실패하면 None"""
    h = _find_header(ws)
    if not h:
        return None
    header_row, ycols = h
    first_col = min(ycols.values())

    label_col = None
    for c in range(first_col - 1, 0, -1):
        n = sum(
            isinstance(ws.cell(r, c).value, str) and ws.cell(r, c).value.strip() != ""
            for r in range(header_row + 1, header_row + 16)
        )
        if n >= 3:
            label_col = c
            break
    if label_col is None:
        return None

    total_row, data_rows = None, []
    for r in range(header_row + 1, ws.max_row + 1):
        lab = ws.cell(r, label_col).value
        if lab is None or str(lab).strip() == "":
            continue
        if _norm(lab) in TOTAL_LABELS:
            total_row = r
            break
        data_rows.append(r)
    if total_row is None or not data_rows:
        return None

    years = sorted(ycols)

    def num(r, c):
        v = ws_v.cell(r, c).value
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan

    labels = {r: str(ws.cell(r, label_col).value).strip() for r in data_rows}
    hist = pd.DataFrame(
        {r: [num(r, ycols[y]) for y in years] for r in data_rows}, index=years
    ).T  # 행: 엑셀 행번호, 열: 연도

    filled_years = [y for y in years if hist[y].notna().any()]
    if not filled_years:
        return None
    base_end = max(filled_years)

    return dict(
        header_row=header_row, ycols=ycols, label_col=label_col, labels=labels,
        data_rows=data_rows, total_row=total_row, hist=hist, base_end=base_end,
    )


# ─────────────────────────────────────────────────────────
# 2. 예측 로직 (5가지)
# ─────────────────────────────────────────────────────────
def project(years, vals, n, method, window=7, phi=0.9, rate_pct=0.0):
    """마지막 실적 값을 기준(anchor)으로 n년치를 예측. 음수는 0으로 처리"""
    years = np.asarray(years, float)
    vals = np.asarray(vals, float)
    ok = ~np.isnan(vals)
    if ok.sum() == 0:
        return None
    y_ok, v_ok = years[ok], vals[ok]
    last = v_ok[-1]
    k = np.arange(1, n + 1)

    if method == "custom":
        out = last * (1 + rate_pct / 100.0) ** k
        return np.maximum(0, out)
    if method == "hold" or len(v_ok) < 2:
        return np.full(n, last)

    yw, vw = y_ok[-window:], v_ok[-window:]
    if len(vw) < 2:
        return np.full(n, last)
    slope = np.polyfit(yw, vw, 1)[0]

    if method == "linear":
        out = last + slope * k
    elif method == "damped":
        out, v, st_ = [], last, slope
        for _ in range(n):
            st_ *= phi
            v += st_
            out.append(v)
        out = np.array(out)
    elif method == "cagr":
        span = yw[-1] - yw[0]
        r = (vw[-1] / vw[0]) ** (1 / span) - 1 if vw[0] > 0 and vw[-1] > 0 and span > 0 else 0.0
        out = last * (1 + r) ** k
    else:
        out = np.full(n, last)
    return np.maximum(0, out)


def window_cagr_pct(years, vals, window=7):
    """직접입력 기본값으로 쓸 최근 N년 CAGR(%)"""
    ok = ~np.isnan(vals)
    y, v = np.asarray(years, float)[ok][-window:], np.asarray(vals, float)[ok][-window:]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0 or y[-1] == y[0]:
        return 0.0
    return round(((v[-1] / v[0]) ** (1 / (y[-1] - y[0])) - 1) * 100, 2)


def build_projection(layout, global_method, window, phi, row_cfg):
    """
    row_cfg: {엑셀행번호: (override_method_key or None, rate_pct)}
    반환: (예측 DataFrame[행번호 x 연도], 적용방법 dict)
    """
    hist = layout["hist"]
    base_end = layout["base_end"]
    hy = [y for y in hist.columns if y <= base_end]
    fut = list(range(base_end + 1, TARGET_END + 1))
    n = len(fut)
    rows, used = {}, {}
    for r in layout["data_rows"]:
        vals = hist.loc[r, hy].values.astype(float)
        ov, rate = row_cfg.get(r, (None, 0.0))
        m = ov or global_method
        p = project(hy, vals, n, m, window, phi, rate)
        if p is not None:
            rows[r] = p
            used[r] = m
    return pd.DataFrame(rows, index=fut).T, used


# ─────────────────────────────────────────────────────────
# 3. 엑셀 쓰기 (원본 서식/수식 유지)
# ─────────────────────────────────────────────────────────
def write_workbook(file_bytes, sheet_name, layout, proj, used, global_method, window, phi, row_cfg):
    wb = load_workbook(io.BytesIO(file_bytes))
    ws = wb[sheet_name]
    ycols = dict(layout["ycols"])
    base_end = layout["base_end"]
    src_col = ycols[base_end]
    hr, tr = layout["header_row"], layout["total_row"]
    fut = list(proj.columns)

    # (1) 필요한 연도 열 확보 (없으면 마지막 열 오른쪽에 추가)
    src_hdr = ws.cell(hr, src_col)
    for y in fut:
        if y in ycols:
            continue
        new_c = max(ycols.values()) + 1
        ycols[y] = new_c
        h = ws.cell(hr, new_c)
        h._style = copy(src_hdr._style)
        h.value = f"{y}년" if isinstance(src_hdr.value, str) and "년" in src_hdr.value else y
    for y in fut:
        L, SL = get_column_letter(ycols[y]), get_column_letter(src_col)
        w = ws.column_dimensions[SL].width
        if w:
            ws.column_dimensions[L].width = w

    def style_from_src(cell, src):
        if not cell.has_style:
            cell._style = copy(src._style)
        cell.number_format = src.number_format

    # (2) 용도별 예측값
    for r in layout["data_rows"]:
        if r not in proj.index:
            continue
        src = ws.cell(r, src_col)
        for y in fut:
            c = ws.cell(r, ycols[y])
            c.value = int(round(float(proj.loc[r, y])))
            style_from_src(c, src)

    # (3) 합계 행
    first_r, last_r = min(layout["data_rows"]), max(layout["data_rows"])
    src_t = ws.cell(tr, src_col)
    for y in fut:
        c = ws.cell(tr, ycols[y])
        if isinstance(src_t.value, str) and src_t.value.startswith("="):
            c.value = Translator(src_t.value, origin=src_t.coordinate).translate_formula(c.coordinate)
        else:
            L = get_column_letter(ycols[y])
            c.value = f"=SUM({L}{first_r}:{L}{last_r})"
        style_from_src(c, src_t)

    # (4) 합계 아래의 수식 행(전년대비 증감, 열량/부피 증감 등) 확장
    for r in range(tr + 1, ws.max_row + 1):
        src = ws.cell(r, src_col)
        if isinstance(src.value, str) and src.value.startswith("="):
            for y in fut:
                c = ws.cell(r, ycols[y])
                c.value = Translator(src.value, origin=src.coordinate).translate_formula(c.coordinate)
                style_from_src(c, src)

    # (5) 적용 내역 시트
    memo_name = "2045확장_적용내역"
    if memo_name in wb.sheetnames:
        del wb[memo_name]
    memo = wb.create_sheet(memo_name)
    inv = {v: k for k, v in METHODS.items()}
    memo.append(["항목", "내용"])
    memo.append(["기준(마지막 계획) 연도", base_end])
    memo.append(["전체 적용 방법", inv.get(global_method, global_method)])
    memo.append(["추세 계산 기간(최근 N년)", window])
    memo.append(["감쇠계수", phi])
    memo.append([])
    memo.append(["용도", "적용 방법", "직접입력 증감률(%/년)"])
    for r in layout["data_rows"]:
        if r in used:
            rate = row_cfg.get(r, (None, 0.0))[1] if used[r] == "custom" else ""
            memo.append([layout["labels"][r], inv.get(used[r], used[r]), rate])
    memo.column_dimensions["A"].width = 28
    memo.column_dimensions["B"].width = 32
    memo.column_dimensions["C"].width = 22

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


# ─────────────────────────────────────────────────────────
# 🟢 9. 2045 중장기 확장 화면 (추세 옵션 선택 + 결과 표시 + 엑셀 다운로드)
# ─────────────────────────────────────────────────────────
REPO_PLAN_FILE = "2045중장기사업계획.xlsx"


def _repo_plan_path():
    """저장소(앱과 같은 폴더)에 올려둔 계획 엑셀 경로"""
    try:
        bases = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    except NameError:
        bases = [os.getcwd()]
    for d in bases:
        p = os.path.join(d, REPO_PLAN_FILE)
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def _load_layouts(data: bytes):
    wb_f = load_workbook(io.BytesIO(data))
    wb_v = load_workbook(io.BytesIO(data), data_only=True)
    out = {}
    for name in wb_f.sheetnames:
        lay = parse_layout(wb_f[name], wb_v[name])
        if lay:
            out[name] = lay
    return out


def render_plan_extension_2045():
    st.subheader("🗓️ 중장기 공급량 2036~2045 추정")
    st.caption("2035년까지 작성된 계획을 추세로 2045년까지 연장합니다. 방법을 바꾸면 결과가 바로 바뀌고, 엑셀로 내려받을 수 있습니다. (단위: GJ)")

    # ── 데이터 파일: 저장소 파일 우선 ──
    repo_path = _repo_plan_path()
    opts = (["저장소 파일 사용"] if repo_path else []) + ["직접 업로드"]
    src = st.radio("데이터 파일", opts, horizontal=True, key="p45_src")
    if src == "저장소 파일 사용":
        with open(repo_path, "rb") as f:
            data = f.read()
        st.caption(f"📄 {REPO_PLAN_FILE}")
    else:
        up = st.file_uploader("중장기 공급량 추정 엑셀 (.xlsx)", type=["xlsx"], key="p45_up")
        if up is None:
            st.info("👆 엑셀 파일을 업로드하세요.")
            return
        data = up.getvalue()

    try:
        cands = _load_layouts(data)
    except Exception as e:
        st.error(f"엑셀을 열 수 없습니다: {e}")
        return
    if not cands:
        st.error("연도 헤더(예: 2025년)와 '합계' 행이 있는 표를 찾지 못했습니다.")
        return
    sheet = list(cands)[0] if len(cands) == 1 else st.selectbox("작업할 시트", list(cands), key="p45_sheet")
    layout = cands[sheet]
    base_end = layout["base_end"]
    if base_end >= TARGET_END:
        st.warning(f"이미 {base_end}년까지 값이 있습니다.")
        return
    hy = [y for y in layout["hist"].columns if y <= base_end]

    # ── 추세 옵션 선택 ──
    st.markdown("### ① 추세 옵션 선택")
    mlabel = st.radio("방법", list(METHODS), index=1, horizontal=True, key="p45_m")
    gm = METHODS[mlabel]
    st.info("ℹ️ " + METHOD_DESC[gm])
    c1, c2 = st.columns(2)
    with c1:
        window = st.slider("추세 계산 기간 (최근 N년)", 3, min(15, len(hy)), min(7, len(hy)),
                           help="선형·감쇠·CAGR에서 최근 몇 년의 흐름을 볼지 정합니다. 7년 = 2029~2035", key="p45_win")
    with c2:
        phi = st.slider("감쇠계수 (감쇠 추세용)", 0.5, 1.0, 0.9, 0.05,
                        help="1.0이면 선형과 동일, 낮을수록 빨리 안정화", key="p45_phi")

    seed = []
    for r in layout["data_rows"]:
        lab = layout["labels"][r]
        vals = layout["hist"].loc[r, hy].values.astype(float)
        if np.isnan(vals).all():
            continue
        seed.append({
            "행": r, "용도": lab,
            "방법": "4. 보합 (마지막 해 유지)" if "연료전지" in lab else FOLLOW_GLOBAL,
            "증감률(%/년)": window_cagr_pct(hy, vals, 7),
        })
    with st.expander("🔧 용도별로 방법/증감률 개별 지정 (선택)", expanded=False):
        st.caption("기본값: 연료전지는 설비용량이 고정이라 '보합'. 증감률은 5번(직접입력)일 때만 쓰이며 초기값은 최근 7년 CAGR입니다. "
                   "예) 취사/개별난방을 감소로 보려면 방법을 '5. 용도별 직접 입력'으로, 증감률을 음수로 입력하세요.")
        edited = st.data_editor(
            pd.DataFrame(seed), hide_index=True, use_container_width=True, disabled=["행", "용도"],
            column_config={
                "방법": st.column_config.SelectboxColumn("방법", options=[FOLLOW_GLOBAL] + list(METHODS)),
                "증감률(%/년)": st.column_config.NumberColumn("증감률(%/년)", step=0.1, format="%.2f"),
                "행": st.column_config.NumberColumn("엑셀 행", width="small"),
            },
            key=f"p45_ed_{sheet}",
        )
    row_cfg = {}
    for _, rw in edited.iterrows():
        ov = None if rw["방법"] == FOLLOW_GLOBAL else METHODS[rw["방법"]]
        row_cfg[int(rw["행"])] = (ov, float(rw["증감률(%/년)"] or 0.0))

    proj, used = build_projection(layout, gm, window, phi, row_cfg)
    if proj.empty:
        st.error("예측할 수 있는 용도 행이 없습니다.")
        return

    labels = layout["labels"]
    rows = [r for r in layout["data_rows"] if r in proj.index]
    full = pd.concat([layout["hist"].loc[rows, hy], proj.loc[rows]], axis=1)
    full.index = [labels[r] for r in rows]
    tot = full.sum()
    yoy = tot.pct_change()

    # ── 결과 요약 ──
    st.markdown("### ② 결과")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{base_end}년 합계", f"{tot[base_end]:,.0f}")
    k2.metric("2040년 합계", f"{tot[2040]:,.0f}", f"{tot[2040] / tot[base_end] - 1:+.1%}")
    k3.metric(f"{TARGET_END}년 합계", f"{tot[TARGET_END]:,.0f}", f"{tot[TARGET_END] / tot[base_end] - 1:+.1%}")
    n_yr = TARGET_END - base_end
    k4.metric(f"연평균 증감률 ({base_end}→{TARGET_END})", f"{(tot[TARGET_END] / tot[base_end]) ** (1 / n_yr) - 1:+.2%}")

    t1, t2, t3 = st.tabs(["📈 합계 · 방법 비교", "🧱 용도별 추이", "📋 연도별 상세표"])

    with t1:
        hist_tot = tot[[y for y in tot.index if y <= base_end]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tot.index, y=hist_tot.values, mode="lines+markers",
                                 name="계획/실적", line=dict(color="gray", width=3)))
        comp = {}
        for lab, key in METHODS.items():
            cfg_c = {r: (None, v[1]) for r, v in row_cfg.items()}
            p_c, _ = build_projection(layout, key, window, phi, cfg_c)
            tt = p_c.sum()
            comp[lab] = tt
            fig.add_trace(go.Scatter(x=[base_end] + list(tt.index), y=[hist_tot[base_end]] + list(tt.values),
                                     mode="lines", name=lab, line=dict(width=1.5, dash="dot")))
        sel = tot[[y for y in tot.index if y >= base_end]]
        fig.add_trace(go.Scatter(x=sel.index, y=sel.values, mode="lines+markers",
                                 name="★ 현재 선택(용도별 설정 포함)", line=dict(color="black", width=4)))
        fig.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="green", opacity=0.05,
                      annotation_text="추세 예측", annotation_position="inside top")
        fig.update_xaxes(dtick=2, tickformat="d")
        fig.update_layout(height=460, legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("점선은 '한 가지 방법을 모든 용도에 적용'했을 때의 합계, 굵은 검은 선이 현재 선택(용도별 개별 설정 포함)입니다.")
        cmp_df = pd.DataFrame(
            {k: [v[2036], v[2040], v[TARGET_END]] for k, v in comp.items()},
            index=["2036년", "2040년", f"{TARGET_END}년"]).T
        cmp_df[f"{TARGET_END}년 vs {base_end}년"] = cmp_df[f"{TARGET_END}년"] / tot[base_end] - 1
        st.dataframe(cmp_df.style.format({"2036년": "{:,.0f}", "2040년": "{:,.0f}", f"{TARGET_END}년": "{:,.0f}",
                                           f"{TARGET_END}년 vs {base_end}년": "{:+.1%}"}), use_container_width=True)

    with t2:
        long = full.reset_index().melt(id_vars="index", var_name="연", value_name="값").rename(columns={"index": "용도"})
        long["구분"] = np.where(long["연"] <= base_end, "계획/실적", "추세예측")
        bridge = long[long["연"] == base_end].copy()
        bridge["구분"] = "추세예측"
        long_l = pd.concat([long, bridge], ignore_index=True)
        order = {"용도": list(full.index)}
        as_idx = st.checkbox(f"{base_end}년 = 100 지수로 보기 (용도별 규모 차이가 클 때 추천)", value=True, key="p45_idx")
        if as_idx:
            basev = long[long["연"] == base_end].set_index("용도")["값"]
            long_l["표시값"] = long_l.apply(lambda x: x["값"] / basev[x["용도"]] * 100 if basev[x["용도"]] else np.nan, axis=1)
            ylab = f"{base_end}년=100"
        else:
            long_l["표시값"], ylab = long_l["값"], "GJ"
        figl = px.line(long_l, x="연", y="표시값", color="용도", line_dash="구분", markers=True, category_orders=order,
                       labels={"표시값": ylab})
        figl.add_vrect(x0=base_end + 0.5, x1=TARGET_END + 0.5, fillcolor="green", opacity=0.05)
        figl.update_xaxes(dtick=2, tickformat="d")
        figl.update_layout(height=480)
        st.plotly_chart(figl, use_container_width=True)
        figb = px.bar(long, x="연", y="값", color="용도", category_orders=order, title="연도별 용도 구성 (누적)")
        figb.add_vline(x=base_end + 0.5, line_dash="dash", line_color="green")
        figb.update_xaxes(dtick=2, tickformat="d")
        figb.update_layout(height=460)
        st.plotly_chart(figb, use_container_width=True)

    with t3:
        table = full.copy()
        table.loc["합계"] = tot
        table.loc["전년 대비 증감"] = yoy
        table.columns = [f"{y}년" + ("" if y <= base_end else " ▶") for y in table.columns]
        num_rows = [i for i in table.index if i != "전년 대비 증감"]
        sty = table.style.format("{:,.0f}", subset=pd.IndexSlice[num_rows, :], na_rep="")
        sty = sty.format("{:+.2%}", subset=pd.IndexSlice[["전년 대비 증감"], :], na_rep="")
        st.dataframe(sty, use_container_width=True)
        st.caption(f"▶ 표시는 {base_end}년 이후 추세 예측 값입니다. 계획표의 행 순서와 같습니다.")

    # ── 엑셀 다운로드 ──
    st.markdown("### ③ 엑셀 다운로드")
    out = write_workbook(data, sheet, layout, proj, used, gm, window, phi, row_cfg)
    st.download_button(
        "📥 2045 확장 엑셀 다운로드", data=out,
        file_name=f"중장기공급량추정_{TARGET_END}_{mlabel.split('.')[1].split('(')[0].strip().replace(' ', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    st.caption("원본 서식·수식은 유지되고, 합계/전년대비/증감/비율 행은 수식으로 2045년까지 확장됩니다. "
               "적용한 방법은 엑셀 끝의 '2045확장_적용내역' 시트에 기록됩니다.")


# ─────────────────────────────────────────────────────────
# 🟢 10. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량", "2. 공급량", "3. 2045 중장기 확장"], index=2)

    # 🟢 [모드 3] 2045 중장기 확장 (저장소의 2045중장기사업계획.xlsx 우선 사용)
    if mode.startswith("3"):
        render_plan_extension_2045()
        return

    with st.sidebar:
        sub_mode = ""
        if mode.startswith("2"):
            sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 상품별 예측", "4) 최종값 확인"])
        elif mode.startswith("1"):
            sub_mode = st.radio("기능 선택", ["1) 실적분석"])
        
        idx = 0 
        if mode.startswith("1"): idx = 0 
        
        if mode.startswith("1"): # 판매량
            unit_opts = ["열량 (GJ)", "부피 (천m³)"]
        else: # 공급량, 최종값
            unit_opts = ["열량 (GJ)", "부피 (천m³)"]
            
        unit = st.radio("단위 선택", unit_opts, index=idx)
        unit_key = "열량" if "열량" in unit else "부피"
        
        st.markdown("---")
        st.subheader("파일 업로드")
        
        up_sales = st.file_uploader("1. 판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s", accept_multiple_files=True)
        up_supply = st.file_uploader("2. 공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        up_final = st.file_uploader("3. 최종값.xlsx (결과파일)", type=["xlsx", "csv"], key="f")
        st.markdown("---")
    
    df_final = pd.DataFrame()
    start_year = 2026
    is_supply = False
    
    # 🟢 [모드 1] 판매량
    if mode.startswith("1"):
        start_year = 2026
        if up_sales:
            data_dict = load_all_sheets(up_sales[0] if isinstance(up_sales, list) else up_sales)
            
            target_df = None
            for sheet_name, df in data_dict.items():
                if "실적" in sheet_name and unit_key in sheet_name:
                    target_df = df; break
            
            if target_df is None:
                for sheet_name, df in data_dict.items():
                    if "실적" in sheet_name: target_df = df; break
            
            if target_df is None and len(data_dict) > 0:
                target_df = list(data_dict.values())[0]

            if target_df is not None:
                long_a = make_long_data(target_df, "실적", 'sales')
                long_a = long_a[long_a['연'] <= 2025] 
                df_final = pd.concat([long_a], ignore_index=True)
        else: st.info("👈 [판매량 파일]을 업로드하세요."); return

    # 🟢 [모드 2] 공급량
    elif mode.startswith("2"):
        start_year = 2029 
        is_supply = True
        
        if "최종값" in sub_mode:
            if up_final:
                data_dict = load_all_sheets(up_final)
                if len(data_dict) > 0:
                    df_raw = list(data_dict.values())[0]
                    df_final = make_long_data(df_raw, "최종값", mode='detail')
            else:
                st.info("👈 [최종값 파일]을 업로드하세요."); return
        else:
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

    # ── 공통 실행 ──
    if not df_final.empty:
        # 🔴 [단위 변환 로직 적용]
        if (mode.startswith("2") or mode.startswith("3")) and "GJ" in unit:
            if "최종값" not in sub_mode: 
                df_final['값'] = df_final['값'] / 1000

        if "최종값" not in sub_mode:
            with st.sidebar:
                st.markdown("### 📅 데이터 학습 기간 설정")
                all_years = sorted([int(y) for y in df_final['연'].unique()])
                default_yrs = all_years 
                train_years = st.multiselect("학습 연도 (2025년 포함됨)", options=all_years, default=default_yrs)

        if "최종값" in sub_mode:
            render_final_check(df_final, unit)
        
        elif "실적" in sub_mode:
            render_analysis_dashboard(df_final, unit)
        
        elif "2035" in sub_mode:
            render_prediction_2035(df_final, unit, start_year, train_years, is_supply, custom_sort_list=None)
        
        elif "상품별" in sub_mode:
            df_detail = pd.DataFrame()
            if mode.startswith("1") and up_sales:
                dd = load_all_sheets(up_sales[0] if isinstance(up_sales, list) else up_sales)
                tgt = None
                for sn, d in dd.items():
                    if "실적" in sn and unit_key in sn: tgt = d; break
                if tgt is None:
                    for sn, d in dd.items():
                        if "실적" in sn: tgt = d; break
                if tgt is not None:
                    df_detail = make_long_data(tgt, "실적", mode='detail')
                    df_detail = df_detail[df_detail['연'] <= 2025]

            elif mode.startswith("2") and up_supply:
                dd = load_all_sheets(up_supply)
                dh, dp = None, None
                for n, d in dd.items():
                    if "실적" in n: dh = d; break
                for n, d in dd.items():
                    if "계획" in n: dp = d; break
                if dh is None and len(dd)>0: dh = list(dd.values())[0]

                if dh is not None:
                    ld_h = make_long_data(dh, "실적", mode='detail')
                    df_detail = ld_h
                    if dp is not None:
                        ld_p = make_long_data(dp, "확정계획", mode='detail')
                        df_detail = pd.concat([ld_h, ld_p], ignore_index=True)
            
            if not df_detail.empty:
                if (mode.startswith("2")) and "GJ" in unit:
                    df_detail['값'] = df_detail['값'] / 1000
                
                render_prediction_2035(df_detail, unit, start_year, train_years, is_supply, custom_sort_list=ORDER_LIST_DETAIL)

                # 🟢 [기온 분석 로직]
                st.markdown("---")
                st.subheader("❄️ 동절기(선택 월) 기온 추세 분석 (2035년 예측)")
                
                with st.sidebar:
                    up_t_detail = st.file_uploader("기온 파일(.csv/.xlsx) - 하단 그래프용", type=["csv", "xlsx"], key="temp_detail")
                
                if up_t_detail:
                    temp_dict = load_all_sheets(up_t_detail)
                    if temp_dict:
                        df_temp = list(temp_dict.values())[0]
                        df_temp = clean_df(df_temp)
                        cols = [c for c in df_temp.columns if "기온" in c]
                        
                        if cols and '연' in df_temp.columns and '월' in df_temp.columns:
                            temp_col = cols[0]
                            col_ctrl1, col_ctrl2 = st.columns([1, 1])
                            with col_ctrl1:
                                st.markdown("##### 📅 기온 분석 학습 연도 설정")
                                all_years_temp = sorted(df_temp['연'].unique())
                                default_years_temp = [y for y in all_years_temp if y >= 2010]
                                selected_years_temp = st.multiselect(
                                    "학습에 포함할 연도를 선택하세요 (제외할 연도 제거)",
                                    options=all_years_temp,
                                    default=default_years_temp
                                )
                            with col_ctrl2:
                                st.markdown("##### 📅 월 선택")
                                selected_months = st.multiselect(
                                    "평균을 낼 월을 선택하세요", 
                                    options=list(range(1, 13)), 
                                    default=[12, 1, 2, 3], 
                                    format_func=lambda x: f"{x}월"
                                )
                            
                            if selected_months and selected_years_temp:
                                df_t_filt = df_temp[df_temp['월'].isin(selected_months)]
                                df_t_filt = df_t_filt[df_t_filt['연'].isin(selected_years_temp)]
                                df_t_grp = df_t_filt.groupby('연')[temp_col].mean().reset_index()
                                
                                fig_temp = px.line(df_t_grp, x='연', y=temp_col, markers=True, 
                                                   title=f"선택 월({', '.join(map(str, selected_months))}월) 평균 기온 추세 및 2035년 예측")
                                fig_temp.update_traces(line=dict(color='royalblue', width=2), marker=dict(size=8), name="실측값")
                                
                                if len(df_t_grp) > 1:
                                    X = df_t_grp['연'].values.reshape(-1, 1)
                                    y = df_t_grp[temp_col].values
                                    model = LinearRegression()
                                    model.fit(X, y)
                                    min_y = min(selected_years_temp)
                                    future_years = np.arange(min_y, 2036).reshape(-1, 1)
                                    pred_y = model.predict(future_years)
                                    fig_temp.add_trace(go.Scatter(x=future_years.flatten(), y=pred_y, mode='lines', 
                                                                  name='추세선(2035 예측)', line=dict(dash='dash', color='red', width=2)))
                                
                                fig_temp.update_xaxes(dtick=1, tickformat="d")
                                fig_temp.update_yaxes(title="평균 기온 (℃)")
                                st.plotly_chart(fig_temp, use_container_width=True)
                            else:
                                st.info("👈 상단에서 연도와 월을 선택해주세요.")
                        else:
                            st.error("기온 데이터에 '날짜'나 '기온' 관련 컬럼이 올바르지 않습니다.")
            else:
                st.warning("데이터를 불러올 수 없습니다.")

if __name__ == "__main__":
    main()
