import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import io

# 페이지 설정
st.set_page_config(page_title="Bitcoin Prediction Dashboard", layout="wide")

@st.cache_data
def load_and_clean_data(file_path):
    """비트코인 CSV 데이터를 읽고 정제하는 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            cleaned_lines.append(line + "\n")
            
        df = pd.read_csv(io.StringIO("".join(cleaned_lines)), sep=';')
        df.columns = [c.strip() for c in df.columns]
        
        # 데이터 타입 변환
        df['timeOpen'] = pd.to_datetime(df['timeOpen'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['timeOpen', 'close']).sort_values('timeOpen')
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# --- 메인 로직 시작 ---
st.title("📈 비트코인 분석 및 미래 가격 예측 대시보드")
st.markdown("회귀 모델의 복잡도(차수)를 조절하여 향후 가격 추세를 시뮬레이션합니다.")

# 데이터 불러오기
df = load_and_clean_data('coin.csv')

if df is not None:
    # --- 사이드바 설정 ---
    st.sidebar.header("📊 분석 및 예측 설정")
    
    # 1. 날짜 범위 필터 (학습 데이터 범위)
    min_data_date = df['timeOpen'].min().date()
    max_data_date = df['timeOpen'].max().date()
    date_range = st.sidebar.date_input("학습 데이터 기간", [min_data_date, max_data_date], 
                                      min_value=min_data_date, max_value=max_data_date)
    
    # 2. 회귀 모델 설정 (1~20차)
    st.sidebar.subheader("회귀 모델 복잡도")
    poly_degree = st.sidebar.slider("다항 회귀 차수 (Degree)", 1, 20, 3, 
                                     help="차수가 높을수록 과거 데이터에 더 민감하게 반응하지만, 너무 높으면 예측이 부정확해질 수 있습니다.")
    
    # 3. 미래 예측 날짜 선택
    st.sidebar.subheader("🎯 미래 예측")
    default_pred_date = max_data_date + datetime.timedelta(days=1)
    target_date = st.sidebar.date_input("예측하고 싶은 날짜 선택", default_pred_date, min_value=max_data_date)

    # 4. 이동평균선 설정
    sma_period = st.sidebar.number_input("이동평균선(SMA) 기간", min_value=5, max_value=200, value=20)

    # 데이터 필터링 (선택된 기간만큼만 학습에 사용)
    if len(date_range) == 2:
        mask = (df['timeOpen'].dt.date >= date_range[0]) & (df['timeOpen'].dt.date <= date_range[1])
        train_df = df.loc[mask].copy()
    else:
        train_df = df.copy()

    if not train_df.empty:
        # --- 회귀 모델 계산 ---
        train_df['date_numeric'] = train_df['timeOpen'].apply(lambda x: x.toordinal())
        X = train_df['date_numeric'].values.reshape(-1, 1)
        y = train_df['close'].values

        # 다항 회귀 모델 학습
        poly_feat = PolynomialFeatures(degree=poly_degree)
        X_poly = poly_feat.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        train_df['trend_line'] = model.predict(X_poly)
        
        # 미래 날짜 예측 계산
        target_ordinal = np.array([[target_date.toordinal()]])
        target_poly = poly_feat.transform(target_ordinal)
        predicted_price = model.predict(target_poly)[0]
        
        # 현재(최신) 가격과 비교
        current_price = train_df['close'].iloc[-1]
        price_diff = predicted_price - current_price
        percent_diff = (price_diff / current_price) * 100

        # --- 상단 주요 지표 (Metrics) ---
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.metric("현재 종가 (마지막 데이터)", f"{current_price:,.0f} KRW")
        
        with col2:
            color = "normal" if price_diff >= 0 else "inverse"
            st.metric(f"{target_date} 예측가", f"{predicted_price:,.0f} KRW", 
                      f"{price_diff:+,.0f} ({percent_diff:+.2f}%)", delta_color=color)
            
        with col3:
            st.info(f"💡 **분석 결과:** {poly_degree}차 회귀 모델 기준, {target_date}에는 현재보다 약 **{abs(price_diff):,.0f}원 {'상승' if price_diff >= 0 else '하락'}**할 것으로 추정됩니다.")

        # --- 메인 차트 (Plotly) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.1, row_heights=[0.7, 0.3],
                             subplot_titles=("가격 및 회귀 추세 시뮬레이션", "거래량"))

        # 캔들스틱
        fig.add_trace(go.Candlestick(
            x=train_df['timeOpen'],
            open=train_df['open'], high=train_df['high'],
            low=train_df['low'], close=train_df['close'],
            name='실제 시세'
        ), row=1, col=1)

        # 회귀선 (학습 데이터 범위)
        fig.add_trace(go.Scatter(
            x=train_df['timeOpen'], y=train_df['trend_line'],
            line=dict(color='yellow', width=2),
            name=f'{poly_degree}차 추세선'
        ), row=1, col=1)
        
        # 예측 포인트 표시
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(target_date)], y=[predicted_price],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='star'),
            text=[f"예측: {predicted_price:,.0f}"],
            textposition="top center",
            name='미래 예측 점'
        ), row=1, col=1)

        # 이동평균선
        train_df['sma'] = train_df['close'].rolling(window=sma_period).mean()
        fig.add_trace(go.Scatter(
            x=train_df['timeOpen'], y=train_df['sma'],
            line=dict(color='cyan', width=1, dash='dot'),
            name=f'{sma_period}일 이평선'
        ), row=1, col=1)

        # 거래량
        fig.add_trace(go.Bar(
            x=train_df['timeOpen'], y=train_df['volume'],
            name='거래량', marker_color='rgba(150, 150, 150, 0.5)'
        ), row=2, col=1)

        fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False,
                          hovermode='x unified', showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)

        # --- 추가 설명 ---
        with st.expander("🧐 회귀 모델 차수(1~20차)란 무엇인가요?"):
            st.write("""
            - **1차 (선형):** 데이터를 하나의 직선으로 연결합니다. 전체적인 장기 우상향/우하향 흐름을 보기에 좋습니다.
            - **차수가 높아질수록 (2~10차):** 선이 부드러운 곡선이 되어 데이터의 굴곡을 따라갑니다. 최근의 파동을 반영합니다.
            - **고차수 (15~20차):** 과거의 미세한 변동까지 모두 따라가려 합니다. 이 경우 과거 데이터에는 완벽해 보이지만, 미래 예측 시에는 값이 갑자기 튀는 '과적합(Overfitting)' 현상이 발생할 수 있으니 주의 깊게 관찰하세요.
            """)

    else:
        st.warning("선택한 기간에 데이터가 없습니다.")

else:
    st.info("폴더 내의 'coin.csv' 파일을 읽을 수 없습니다.")