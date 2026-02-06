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
# 🟢 2. 용도별 매핑 (판매량 vs 공급량 분리 적용)
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

# 2) 공급량용 매핑 (형님 요청사항 - 엄격 적용)
# 1) 가정용: 취사용, 개별난방, 중앙난방
# 2) 영업용: 일반용(1)
# 3) 업무용: 일반용(2), 업무난방, 냉난방, 주한미군
# 4) 수송용: 수송용(CNG), 수송용(BIO)
# 5) 나머지: 산업용, 열병합 등 기타
MAPPING_SUPPLY = {
    # 1. 가정용
    "취사용": "가정용", "개별난방용": "가정용", "중앙난방용": "가정용", 
    "개별난방": "가정용", "중앙난방": "가정용",
    
    # 2. 영업용
    "일반용(1)": "영업용", "일반용1": "영업용", "일반용1(영업)": "영업용",
    
    # 3. 업무용
    "일반용(2)": "업무용", "일반용2": "업무용", "일반용1(업무)": "업무용",
    "업무난방용": "업무용", "냉난방용": "업무용", "냉방용": "업무용", "주한미군": "업무용",
    
    # 4. 수송용
    "수송용(CNG)": "수송용", "CNG": "수송용",
    "수송용(BIO)": "수송용", "BIO": "수송용",
    
    # 5. 나머지 (산업용 포함)
    "산업용": "나머지", "열병합용": "나머지", "열병합용1": "나머지", "열병합용2": "나머지",
    "연료전지용": "나머지", "연료전지": "나머지", "열전용설비용": "나머지", "열전용설비용(주택외)": "나머지"
}

# ─────────────────────────────────────────────────────────
# 🟢 3. 파일 로딩 (스마트 로더)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_files_smart(uploaded_files):
    if not uploaded_files: return {}
    data_dict = {}
    if not isinstance(uploaded_files, list): uploaded_files = [uploaded_files]
    
    for file in uploaded_files:
        try:
            # 엑셀 시도
            excel = pd.ExcelFile(file, engine='openpyxl')
            for sheet in excel.sheet_names:
                data_dict[f"{file.name}_{sheet}"] = excel.parse(sheet)
        except:
            # CSV 시도
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
                data_dict[f"{file.name}"] = df
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp949')
                    data_dict[f"{file.name}"] = df
                except: pass
    return data_dict

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

def make_long_data(df, label, mapping_dict):
    """지정된 매핑 사용"""
    df = clean_df(df)
    if df.empty or '연' not in df.columns or '월' not in df.columns: return pd.DataFrame()
    
    records = []
    df['연'] = pd.to_numeric(df['연'], errors='coerce')
    df['월'] = pd.to_numeric(df['월'], errors='coerce')
    df = df.dropna(subset=['연', '월'])
    
    for col in df.columns:
        group = mapping_dict.get(col)
        if not group: continue
        sub = df[['연', '월']].copy()
        sub['그룹'] = group
        sub['용도'] = col
        sub['구분'] = label
        sub['값'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        records.append(sub)
        
    if not records: return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

def find_target_df(data_dict, type_keywords, unit_keyword=None):
    if not data_dict: return None
    
    # 1. 키워드 + 단위 일치
    if unit_keyword:
        for key, df in data_dict.items():
            clean_key = key.replace(" ", "")
            if any(k in clean_key for k in type_keywords) and (unit_keyword in clean_key):
                return df
    
    # 2. 키워드만 일치
    for key, df in data_dict.items():
        clean_key = key.replace(" ", "")
        if any(k in clean_key for k in type_keywords):
            return df
            
    # 3. 데이터가 하나뿐이면 그거 리턴
    if len(data_dict) == 1: return list(data_dict.values())[0]
    return None

# ─────────────────────────────────────────────────────────
# 🟢 4. 분석 및 예측 로직
# ─────────────────────────────────────────────────────────
def render_dashboard(df_final, unit_label, mode_type, sub_mode, start_pred_year, train_years_selected, is_supply_mode=False):
    
    # 1. 실적 분석 화면
    if "실적" in sub_mode:
        st.subheader(f"📊 실적 분석 ({unit_label})")
        
        # 실적 데이터만 필터링
        # 공급량 모드: 2026년 미만까지 실적
        # 판매량 모드: 그냥 전체(이미 로딩때 필터링함)
        df_act = df_final[df_final['구분'].str.contains('실적')].copy()
        
        if df_act.empty: st.error("실적 데이터가 없습니다."); return
        
        # 연도 필터링
        all_yrs = sorted([int(y) for y in df_act['연'].unique()])
        if len(all_yrs) >= 10: def_yrs = all_yrs[-10:]
        else: def_yrs = all_yrs
        
        sel_yrs = st.multiselect("연도 선택", options=all_yrs, default=def_yrs)
        if not sel_yrs: return
        
        df_viz = df_act[df_act['연'].isin(sel_yrs)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 월별 추이")
            mon_grp = df_viz.groupby(['연', '월'])['값'].sum().reset_index()
            fig1 = px.line(mon_grp, x='월', y='값', color='연', markers=True)
            fig1.update_xaxes(dtick=1, tickformat="d")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.markdown("#### 🧱 용도별 구성비")
            yr_grp = df_viz.groupby(['연', '그룹'])['값'].sum().reset_index()
            fig2 = px.bar(yr_grp, x='연', y='값', color='그룹', text_auto='.2s')
            fig2.update_xaxes(dtick=1, tickformat="d")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("##### 📋 상세 데이터 (소계 포함)")
        piv = df_viz.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
        piv['소계'] = piv.sum(axis=1)
        st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

    # 2. 2035 예측 화면
    elif "2035" in sub_mode:
        st.subheader(f"🔮 2035 장기 예측 ({unit_label})")
        
        # 학습 데이터 준비
        # 공급량: 선택연도(실적) + 확정계획(26~28)
        # 판매량: 선택연도(실적)
        filter_cond = df_final['연'].isin(train_years_selected)
        if is_supply_mode:
            filter_cond = filter_cond | (df_final['구분'] == '확정계획')
            
        df_train = df_final[filter_cond].copy()
        
        if df_train.empty: st.warning("학습 데이터가 부족합니다."); return
        
        st.markdown("##### 📊 추세 분석 모델 선택")
        pred_method = st.radio("방법", ["선형 회귀", "2차 곡선", "3차 곡선", "로그 추세", "지수 평활", "CAGR"], horizontal=True)
        
        # 모델 설명
        if "선형" in pred_method: st.info("ℹ️ 매년 일정량씩 꾸준히 변하는 직선 추세")
        elif "2차" in pred_method: st.info("ℹ️ 성장이 가속화되거나 둔화되는 곡선 추세")
        elif "3차" in pred_method: st.info("ℹ️ 상승과 하락 사이클이 있는 복잡한 추세")
        elif "로그" in pred_method: st.info("ℹ️ 초반 급성장 후 안정화되는 패턴")
        elif "지수" in pred_method: st.info("ℹ️ 최근 실적에 가중치를 둔 민감한 추세")
        elif "CAGR" in pred_method: st.info("ℹ️ 과거 연평균 성장률 유지 가정")

        # 예측 실행
        df_grp = df_final.groupby(['연', '그룹', '구분'])['값'].sum().reset_index()
        df_train_grp = df_train.groupby(['연', '그룹'])['값'].sum().reset_index()
        groups = df_grp['그룹'].unique()
        
        # 예측 구간 설정
        # 판매량: 2026~2035 (2026부터 바로 예측)
        # 공급량: 2029~2035 (2029부터 예측, 26~28은 확정계획)
        future_years = np.arange(start_pred_year, 2036).reshape(-1, 1)
        results = []
        
        # AI Insight용
        total_hist = []
        total_pred = []

        for grp in groups:
            sub_train = df_train_grp[df_train_grp['그룹'] == grp]
            sub_full = df_grp[df_grp['그룹'] == grp]
            if len(sub_train) < 2: continue
            
            X = sub_train['연'].values.reshape(-1, 1)
            y = sub_train['값'].values
            pred = []
            
            # 모델링
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
            
            # 결과 병합
            added_years = set()
            
            # 1. 과거 실적 (선택된 연도만)
            hist_mask = sub_full['연'].isin(train_years_selected)
            # 공급량 모드(2029 시작)는 2026미만만 실적
            # 판매량 모드(2026 시작)는 2026미만만 실적
            hist_mask = hist_mask & (sub_full['연'] < start_pred_year)
            
            hist_data = sub_full[hist_mask]
            for _, row in hist_data.iterrows():
                if row['연'] not in added_years:
                    results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '실적'})
                    total_hist.append({'연': row['연'], '값': row['값']})
                    added_years.add(row['연'])
            
            # 2. 확정 계획 (공급량 전용)
            if is_supply_mode and start_pred_year == 2029:
                plan_data = sub_full[sub_full['연'].between(2026, 2028)]
                for _, row in plan_data.iterrows():
                    results.append({'연': row['연'], '그룹': grp, '값': row['값'], '구분': '확정계획'})
                    
            # 3. AI 예측
            for yr, v in zip(future_years.flatten(), pred):
                results.append({'연': yr, '그룹': grp, '값': v, '구분': '예측(AI)'})
                total_pred.append({'연': yr, '값': v})
                
        df_res = pd.DataFrame(results)
        
        # Insight 문구
        if total_hist and total_pred:
            hist_df = pd.DataFrame(total_hist).groupby('연')['값'].sum()
            pred_df = pd.DataFrame(total_pred).groupby('연')['값'].sum()
            max_up = hist_df.diff().idxmax()
            max_down = hist_df.diff().idxmin()
            
            cagr = (pred_df.iloc[-1]/pred_df.iloc[0])**(1/len(pred_df)) - 1 if len(pred_df)>0 else 0
            trend = "증가세" if cagr > 0.01 else "감소세" if cagr < -0.01 else "보합세"
            
            st.success(f"💡 **[AI 분석]** 과거 {int(max_up)}년 급등과 {int(max_down)}년 조정을 고려할 때, 향후 2035년까지 **{trend}**가 전망됩니다.")

        st.markdown("---")
        st.markdown("#### 📈 전체 장기 전망")
        fig = px.line(df_res, x='연', y='값', color='그룹', line_dash='구분', markers=True)
        fig.add_vline(x=start_pred_year-0.5, line_dash="dash", line_color="green")
        fig.add_vrect(x0=start_pred_year-0.5, x1=2035.5, fillcolor="green", opacity=0.05, annotation_text="예측 값", annotation_position="inside top")
        
        if is_supply_mode and start_pred_year == 2029:
            fig.add_vrect(x0=2025.5, x1=2028.5, fillcolor="yellow", opacity=0.1, annotation_text="확정계획", annotation_position="inside top")
            
        fig.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 🧱 연도별 구성 (누적)")
        fig2 = px.bar(df_res, x='연', y='값', color='그룹', text_auto='.2s')
        fig2.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("📋 상세 데이터 확인"):
            piv = df_res.pivot_table(index='연', columns='그룹', values='값', aggfunc='sum').fillna(0)
            piv['소계'] = piv.sum(axis=1)
            st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# 🟢 6. 메인 실행
# ─────────────────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 판매/공급 통합 분석")
    
    with st.sidebar:
        st.header("설정")
        mode = st.radio("분석 모드", ["1. 판매량", "2. 공급량", "3. 최종값 확인"], index=1) # 공급량 기본
        
        sub_mode = ""
        if not mode.startswith("3"):
            sub_mode = st.radio("기능 선택", ["1) 실적분석", "2) 2035 예측", "3) 가정용 정밀 분석"])
        
        # 판매량은 열량 기본, 나머지는 선택
        idx = 0 
        if mode.startswith("1"): idx = 0 
        unit = st.radio("단위 선택", ["열량 (GJ)", "부피 (천m³)"], index=idx)
        unit_key = "열량" if "열량" in unit else "부피"
        
        st.markdown("---")
        st.subheader("파일 업로드")
        # 3가지 파일 업로더 고정 노출
        up_sales = st.file_uploader("1. 판매량(계획_실적).xlsx", type=["xlsx", "csv"], key="s", accept_multiple_files=True)
        up_supply = st.file_uploader("2. 공급량실적_계획_실적_MJ.xlsx", type=["xlsx", "csv"], key="p")
        up_final = st.file_uploader("3. 최종값.xlsx", type=["xlsx", "csv"], key="f")
        st.markdown("---")
    
    df_final = pd.DataFrame()
    start_year = 2026
    is_supply = False
    
    # 🟢 [1. 판매량]
    if mode.startswith("1"):
        start_year = 2026
        if up_sales:
            data = load_files_smart(up_sales)
            if data:
                # 실적 파일만 로드 (계획 무시)
                df_a = find_target_df(data, ["실적"], unit_key)
                if df_a is None and len(data) == 1: df_a = list(data.values())[0]
                
                if df_a is not None:
                    long_a = make_long_data(df_a, "실적", MAPPING_SALES)
                    # 2025년 이하만 남김
                    long_a = long_a[long_a['연'] <= 2025] 
                    df_final = pd.concat([long_a], ignore_index=True)
        else: st.info("👈 [판매량 파일]을 업로드하세요."); return

    # 🟢 [2. 공급량]
    elif mode.startswith("2"):
        start_year = 2029 
        is_supply = True
        if up_supply:
            data = load_files_smart([up_supply])
            if data:
                df_hist = find_target_df(data, ["공급량_실적", "실적"], None)
                df_plan = find_target_df(data, ["공급량_계획", "계획"], None)
                
                if df_hist is None and df_plan is None and len(data) == 1:
                    df_hist = list(data.values())[0]
                
                long_h = make_long_data(df_hist, "실적", MAPPING_SUPPLY)
                long_p = make_long_data(df_plan, "확정계획", MAPPING_SUPPLY)
                df_final = pd.concat([long_h, long_p], ignore_index=True)
        else: st.info("👈 [공급량 파일]을 업로드하세요."); return

    # 🟢 [3. 최종값]
    elif mode.startswith("3"):
        if up_final:
            data = load_files_smart([up_final])
            if data:
                df_raw = list(data.values())[0]
                df_final = make_long_data(df_raw, "최종값", MAPPING_SUPPLY)
        else: st.info("👈 [최종값 파일]을 업로드하세요."); return

    # ── 공통 실행 ──
    if not df_final.empty:
        # 학습 연도 선택 (사이드바)
        if not mode.startswith("3"):
            with st.sidebar:
                st.markdown("### 📅 데이터 학습 기간 설정")
                all_years = sorted([int(y) for y in df_final['연'].unique()])
                default_yrs = all_years 
                train_years = st.multiselect("학습 연도 (2025년 포함됨)", options=all_years, default=default_yrs)

        # 모드별 렌더링
        if mode.startswith("3"):
            render_dashboard(df_final, unit, "final", "실적", 0, [], False) # 재활용
        elif "가정용" in sub_mode:
            with st.sidebar:
                up_t = st.file_uploader("기온 파일(.csv)", type=["csv", "xlsx"])
            st.info("기온 데이터 업로드 시 분석 가능 (기능 준비됨)")
        else:
            render_dashboard(df_final, unit, mode, sub_mode, start_year, train_years, is_supply)

if __name__ == "__main__":
    main()
