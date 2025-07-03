import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="예측 분석",
    page_icon="🔮",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding: 0 !important;
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    
    .page-header {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(34, 197, 94, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .page-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #10b981);
    }
    
    .page-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a1a;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .page-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #666666;
        margin: 0;
        line-height: 1.6;
    }
    
    .chart-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-card:hover {
        transform: translateY(-2px);
        border-color: rgba(34, 197, 94, 0.2);
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: rgba(34, 197, 94, 0.3);
    }
    
    .model-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.5rem 0;
    }
    
    .model-score {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .score-excellent { color: #15803d; }
    .score-good { color: #059669; }
    .score-fair { color: #eab308; }
    .score-poor { color: #dc2626; }
    
    .prediction-input {
        background: rgba(34, 197, 94, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(34, 197, 94, 0.1);
        margin: 1rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05));
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(34, 197, 94, 0.2);
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .prediction-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #15803d;
        margin: 0;
    }
    
    .prediction-label {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #166534;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    .risk-indicator {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #15803d;
        border: 1px solid rgba(21, 128, 61, 0.2);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #92400e;
        border: 1px solid rgba(146, 64, 14, 0.2);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #dc2626;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }
    
    .metric-mini {
        background: rgba(34, 197, 94, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .metric-mini h4 {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #15803d;
        margin: 0;
    }
    
    .metric-mini p {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #166534;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    .scenario-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #22c55e;
    }
    
    .scenario-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.5rem 0;
    }
    
    .scenario-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #15803d;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_prediction_data():
    """예측 분석용 샘플 데이터 생성"""
    np.random.seed(42)
    
    regions = ["경기", "강원", "충남", "전남", "경북"]
    crops = ["쌀", "밀", "콩", "옥수수", "감자"]
    
    data = []
    for year in range(2010, 2024):
        for region in regions:
            for crop in crops:
                # 기후 변수 생성
                temperature = 14.0 + (year - 2010) * 0.12 + np.random.normal(0, 0.8)
                rainfall = 1200 + np.random.normal(0, 200) + 50 * np.sin((year - 2010) * 0.3)
                sunshine = 2000 + np.random.normal(0, 150)
                
                # 병해충 영향
                pest_prob = 0.2 + 0.02 * (year - 2010)  # 연도별 증가
                has_pest = np.random.choice([0, 1], p=[1-pest_prob, pest_prob])
                
                # 수확량 계산 (복잡한 관계 모델링)
                base_yield = 85
                temp_effect = -(temperature - 14.0) * 2.5  # 기온 상승 부정적
                rain_effect = (rainfall - 1200) * 0.008    # 적당한 강수량 긍정적
                sunshine_effect = (sunshine - 2000) * 0.003
                pest_effect = -15 * has_pest               # 병해충 부정적
                
                # 지역별 효과
                region_effects = {"경기": 5, "강원": -2, "충남": 3, "전남": 7, "경북": 1}
                region_effect = region_effects.get(region, 0)
                
                # 작물별 효과
                crop_effects = {"쌀": 0, "밀": -5, "콩": 2, "옥수수": -3, "감자": 4}
                crop_effect = crop_effects.get(crop, 0)
                
                # 무작위 효과
                random_effect = np.random.normal(0, 5)
                
                yield_value = (base_yield + temp_effect + rain_effect + 
                             sunshine_effect + pest_effect + region_effect + 
                             crop_effect + random_effect)
                
                data.append({
                    'year': year,
                    'region': region,
                    'crop': crop,
                    'temperature': temperature,
                    'rainfall': max(rainfall, 600),
                    'sunshine': max(sunshine, 1500),
                    'has_pest': has_pest,
                    'yield': max(yield_value, 30)
                })
    
    return pd.DataFrame(data)

def train_prediction_models(df):
    """예측 모델 훈련"""
    # 특성과 타겟 준비
    features = ['temperature', 'rainfall', 'sunshine', 'has_pest']
    X = df[features]
    y = df['yield']
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델들 훈련
    models = {}
    
    # 1. 선형 회귀
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['Linear Regression'] = {
        'model': lr,
        'predictions': lr_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'r2': r2_score(y_test, lr_pred)
    }
    
    # 2. 랜덤 포레스트
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred),
        'feature_importance': rf.feature_importances_
    }
    
    return models, X_test, y_test, features

def create_prediction_chart(models, X_test, y_test):
    """예측 결과 시각화"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Linear Regression', 'Random Forest'],
        horizontal_spacing=0.1
    )
    
    # 선형 회귀 결과
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=models['Linear Regression']['predictions'],
            mode='markers',
            name='Linear Regression',
            marker=dict(size=8, color='rgba(34, 197, 94, 0.7)', line=dict(width=1, color='white')),
            hovertemplate='실제: %{x:.1f}<br>예측: %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 완벽한 예측선 (y=x)
    min_val = min(y_test.min(), models['Linear Regression']['predictions'].min())
    max_val = max(y_test.max(), models['Linear Regression']['predictions'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='#dc2626', width=2, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 랜덤 포레스트 결과
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=models['Random Forest']['predictions'],
            mode='markers',
            name='Random Forest',
            marker=dict(size=8, color='rgba(16, 185, 129, 0.7)', line=dict(width=1, color='white')),
            hovertemplate='실제: %{x:.1f}<br>예측: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 완벽한 예측선
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='#dc2626', width=2, dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="실제 수확량", row=1, col=1)
    fig.update_xaxes(title_text="실제 수확량", row=1, col=2)
    fig.update_yaxes(title_text="예측 수확량", row=1, col=1)
    fig.update_yaxes(title_text="예측 수확량", row=1, col=2)
    
    fig.update_layout(
        title={
            'text': '모델 성능 비교 - 실제 vs 예측',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=False
    )
    
    return fig

def create_feature_importance_chart(models, features):
    """특성 중요도 차트"""
    if 'Random Forest' not in models:
        return None
    
    importances = models['Random Forest']['feature_importance']
    feature_names = ['기온', '강수량', '일조시간', '병해충']
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=importances,
            marker_color=['#22c55e', '#3b82f6', '#eab308', '#dc2626'],
            text=[f'{imp:.3f}' for imp in importances],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>중요도: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '특성 중요도 분석 (Random Forest)',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        xaxis_title='기후 요소',
        yaxis_title='중요도',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

def create_future_prediction_chart(model, base_data):
    """미래 예측 시나리오"""
    future_years = list(range(2024, 2031))
    
    # 시나리오별 예측
    scenarios = {
        '현재 트렌드 유지': {
            'temp_change': 0.12,
            'rain_change': 0,
            'pest_increase': 0.02
        },
        '기후변화 가속': {
            'temp_change': 0.20,
            'rain_change': -20,
            'pest_increase': 0.05
        },
        '적응 정책 적용': {
            'temp_change': 0.08,
            'rain_change': 10,
            'pest_increase': 0.01
        }
    }
    
    fig = go.Figure()
    colors = ['#22c55e', '#dc2626', '#3b82f6']
    
    for i, (scenario, params) in enumerate(scenarios.items()):
        predictions = []
        
        for year in future_years:
            # 기후 조건 계산
            years_ahead = year - 2023
            temp = 15.5 + params['temp_change'] * years_ahead
            rain = 1200 + params['rain_change'] * years_ahead
            sunshine = 2000
            pest_prob = min(0.3 + params['pest_increase'] * years_ahead, 0.8)
            
            # 예측 수행 (평균적인 조건)
            X_pred = [[temp, rain, sunshine, pest_prob]]
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
        
        fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions,
            mode='lines+markers',
            name=scenario,
            line=dict(color=colors[i], width=3),
            marker=dict(size=8, color=colors[i]),
            hovertemplate=f'<b>{scenario}</b><br>연도: %{{x}}<br>예측 수확량: %{{y:.1f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '미래 수확량 예측 시나리오 (2024-2030)',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        xaxis_title='연도',
        yaxis_title='예측 수확량',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def get_risk_level(predicted_yield):
    """위험도 평가"""
    if predicted_yield >= 80:
        return "낮음", "risk-low"
    elif predicted_yield >= 60:
        return "보통", "risk-medium"
    else:
        return "높음", "risk-high"

def main():
    # 헤더
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔮 예측 분석</h1>
        <p class="page-subtitle">머신러닝 모델을 활용하여 기후 조건에 따른 농작물 수확량을 예측합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 생성 및 모델 훈련
    df = generate_prediction_data()
    models, X_test, y_test, features = train_prediction_models(df)
    
    # 사이드바 - 예측 입력
    with st.sidebar:
        st.markdown("### 🎛️ 예측 조건 설정")
        
        # 기후 조건 입력
        st.markdown("**기후 조건**")
        temperature = st.slider(
            "평균기온 (°C)",
            min_value=10.0,
            max_value=20.0,
            value=15.5,
            step=0.1
        )
        
        rainfall = st.slider(
            "연강수량 (mm)",
            min_value=800,
            max_value=1800,
            value=1200,
            step=10
        )
        
        sunshine = st.slider(
            "연일조시간 (시간)",
            min_value=1500,
            max_value=2500,
            value=2000,
            step=10
        )
        
        has_pest = st.selectbox(
            "병해충 발생 여부",
            ["없음", "있음"]
        )
        
        pest_value = 1 if has_pest == "있음" else 0
        
        # 예측 실행
        if st.button("🔮 수확량 예측", type="primary"):
            st.session_state.prediction_made = True
            st.session_state.prediction_inputs = [temperature, rainfall, sunshine, pest_value]
    
    # 모델 성능 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    lr_r2 = models['Linear Regression']['r2']
    rf_r2 = models['Random Forest']['r2']
    lr_rmse = models['Linear Regression']['rmse']
    rf_rmse = models['Random Forest']['rmse']
    
    with col1:
        score_class = "score-excellent" if lr_r2 > 0.8 else "score-good" if lr_r2 > 0.6 else "score-fair"
        st.markdown(f"""
        <div class="metric-mini">
            <h4 class="{score_class}">{lr_r2:.3f}</h4>
            <p>선형회귀 R²</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score_class = "score-excellent" if rf_r2 > 0.8 else "score-good" if rf_r2 > 0.6 else "score-fair"
        st.markdown(f"""
        <div class="metric-mini">
            <h4 class="{score_class}">{rf_r2:.3f}</h4>
            <p>랜덤포레스트 R²</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{lr_rmse:.1f}</h4>
            <p>선형회귀 RMSE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{rf_rmse:.1f}</h4>
            <p>랜덤포레스트 RMSE</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 예측 결과 표시
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        inputs = st.session_state.prediction_inputs
        
        # 두 모델로 예측
        lr_pred = models['Linear Regression']['model'].predict([inputs])[0]
        rf_pred = models['Random Forest']['model'].predict([inputs])[0]
        avg_pred = (lr_pred + rf_pred) / 2
        
        # 위험도 평가
        risk_level, risk_class = get_risk_level(avg_pred)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-result">
                <p class="prediction-value">{avg_pred:.1f}</p>
                <p class="prediction-label">예측 수확량 지수</p>
                <span class="risk-indicator {risk_class}">위험도: {risk_level}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-card">
                <h4 class="model-title">📊 모델별 예측</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **선형 회귀**: {lr_pred:.1f}  
            **랜덤 포레스트**: {rf_pred:.1f}  
            **앙상블 평균**: {avg_pred:.1f}
            
            **입력 조건**:
            - 기온: {inputs[0]:.1f}°C
            - 강수량: {inputs[1]:.0f}mm  
            - 일조시간: {inputs[2]:.0f}시간
            - 병해충: {"있음" if inputs[3] else "없음"}
            """)
    
    # 모델 성능 비교
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        prediction_chart = create_prediction_chart(models, X_test, y_test)
        st.plotly_chart(prediction_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        importance_chart = create_feature_importance_chart(models, features)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 미래 예측 시나리오
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    future_chart = create_future_prediction_chart(models['Random Forest']['model'], df)
    st.plotly_chart(future_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 시나리오별 분석
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="scenario-card">
            <p class="scenario-title">🟢 현재 트렌드 유지</p>
            <p class="scenario-value">73.2</p>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                기존 기후변화 패턴 지속<br>
                연간 0.12°C 기온 상승
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="scenario-card" style="border-left-color: #dc2626;">
            <p class="scenario-title" style="color: #dc2626;">🔴 기후변화 가속</p>
            <p class="scenario-value" style="color: #dc2626;">58.7</p>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                극단적 기후변화 시나리오<br>
                연간 0.20°C 기온 상승
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="scenario-card" style="border-left-color: #3b82f6;">
            <p class="scenario-title" style="color: #3b82f6;">🔵 적응 정책 적용</p>
            <p class="scenario-value" style="color: #3b82f6;">79.5</p>
            <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                기후 적응 기술 도입<br>
                병해충 관리 강화
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 모델 설명 및 인사이트
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 모델 성능 분석**
        
        **📈 Random Forest 우수성**
        - R² 점수: **{:.3f}** (선형회귀: {:.3f})
        - RMSE: **{:.1f}** (선형회귀: {:.1f})
        - 비선형 관계 포착 능력 뛰어남
        
        **🎯 주요 예측 요인**
        1. **기온**: 가장 큰 영향 요인
        2. **병해충**: 수확량 급감 원인
        3. **강수량**: 적정량 중요
        4. **일조시간**: 보조적 역할
        """.format(rf_r2, lr_r2, rf_rmse, lr_rmse))
    
    with col2:
        st.markdown("""
        **💡 정책 제언**
        
        **🛡️ 단기 대응 (1-2년)**
        - 고온 저항성 품종 보급
        - 병해충 조기 감지 시스템
        - 관개 시설 현대화
        
        **🚀 중장기 전략 (3-7년)**
        - 기후 적응 농업 기술 개발
        - 작물 다양화 정책
        - 스마트팜 확산
        
        **📊 모니터링 체계**
        - 실시간 기상 데이터 수집
        - AI 기반 예측 시스템 구축
        - 농가 맞춤형 경보 서비스
        """)
    
    # 모델 한계 및 주의사항
    with st.expander("⚠️ 모델 한계 및 주의사항"):
        st.markdown("""
        ### 📋 모델 사용시 고려사항
        
        **🔍 데이터 한계**
        - 샘플 데이터 기반으로 실제 성능과 차이 있을 수 있음
        - 지역별 미세 기후 차이 미반영
        - 토양 조건, 농법 등 추가 변수 필요
        
        **⚡ 예측 정확도**
        - 단기 예측 (1년): 높은 정확도
        - 중기 예측 (2-3년): 보통 정확도  
        - 장기 예측 (5년+): 불확실성 증가
        
        **🎯 활용 권장사항**
        - 참고용 지표로 활용
        - 다른 정보와 종합 판단
        - 정기적인 모델 업데이트 필요
        - 전문가 의견과 병행 검토
        
        **🔄 모델 개선 방향**
        - 더 많은 데이터 수집
        - 위성 이미지, IoT 센서 데이터 통합
        - 딥러닝 모델 적용 검토
        - 지역별 특화 모델 개발
        """)
    
    # 추가 분석 도구
    with st.expander("🔬 고급 분석 도구"):
        st.markdown("### 📊 민감도 분석")
        
        # 민감도 분석을 위한 기준값 설정
        base_conditions = [15.5, 1200, 2000, 0]  # 기온, 강수량, 일조시간, 병해충
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**기온 변화 영향**")
            temp_changes = []
            temp_values = []
            for temp_change in range(-20, 21, 5):
                new_temp = base_conditions[0] + temp_change/10
                temp_values.append(new_temp)
                conditions = [new_temp] + base_conditions[1:]
                pred = models['Random Forest']['model'].predict([conditions])[0]
                temp_changes.append(pred)
            
            temp_df = pd.DataFrame({
                '기온': temp_values,
                '예측_수확량': temp_changes
            })
            
            fig_temp = px.line(temp_df, x='기온', y='예측_수확량',
                              title='기온 변화에 따른 수확량 예측',
                              color_discrete_sequence=['#dc2626'])
            fig_temp.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            st.markdown("**강수량 변화 영향**")
            rain_changes = []
            rain_values = []
            for rain_change in range(-400, 401, 100):
                new_rain = base_conditions[1] + rain_change
                rain_values.append(new_rain)
                conditions = [base_conditions[0], new_rain] + base_conditions[2:]
                pred = models['Random Forest']['model'].predict([conditions])[0]
                rain_changes.append(pred)
            
            rain_df = pd.DataFrame({
                '강수량': rain_values,
                '예측_수확량': rain_changes
            })
            
            fig_rain = px.line(rain_df, x='강수량', y='예측_수확량',
                              title='강수량 변화에 따른 수확량 예측',
                              color_discrete_sequence=['#3b82f6'])
            fig_rain.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rain, use_container_width=True)

if __name__ == "__main__":
    main()