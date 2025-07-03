import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(
    page_title="시계열 분석",
    page_icon="📈",
    layout="wide"
)

# CSS 스타일 (메인과 동일한 스타일 적용)
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
    
    .filter-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """샘플 데이터 생성 함수"""
    np.random.seed(42)
    
    # 연도별 데이터 (2010-2023)
    years = list(range(2010, 2024))
    
    # 기온 데이터 (상승 트렌드)
    base_temp = 14.0
    temp_data = []
    for i, year in enumerate(years):
        temp = base_temp + 0.1 * i + np.random.normal(0, 0.5)
        temp_data.append(temp)
    
    # 강수량 데이터 (변동성 있음)
    base_rain = 1200
    rain_data = []
    for i, year in enumerate(years):
        rain = base_rain + np.random.normal(0, 150) + 50 * np.sin(i * 0.5)
        rain_data.append(max(rain, 800))  # 최소값 보장
    
    # 수확량 데이터 (기온과 반비례, 강수량과 어느정도 상관관계)
    yield_data = []
    for i, (temp, rain) in enumerate(zip(temp_data, rain_data)):
        base_yield = 100
        temp_effect = -(temp - 14.0) * 3  # 기온 상승시 수확량 감소
        rain_effect = (rain - 1200) * 0.01  # 강수량 영향
        random_effect = np.random.normal(0, 8)
        
        yield_val = base_yield + temp_effect + rain_effect + random_effect
        yield_data.append(max(yield_val, 50))  # 최소값 보장
    
    return pd.DataFrame({
        'year': years,
        'temperature': temp_data,
        'rainfall': rain_data,
        'crop_yield': yield_data
    })

def create_main_timeseries_chart(df, selected_variables):
    """메인 시계열 차트 생성"""
    fig = make_subplots(
        rows=len(selected_variables), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"{var} 시계열 변화" for var in selected_variables]
    )
    
    colors = {
        '평균기온': '#22c55e',
        '강수량': '#3b82f6', 
        '수확량': '#f59e0b'
    }
    
    var_mapping = {
        '평균기온': 'temperature',
        '강수량': 'rainfall', 
        '수확량': 'crop_yield'
    }
    
    for i, var in enumerate(selected_variables, 1):
        col_name = var_mapping[var]
        
        fig.add_trace(
            go.Scatter(
                x=df['year'],
                y=df[col_name],
                mode='lines+markers',
                name=var,
                line=dict(
                    color=colors[var],
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    color=colors[var],
                    line=dict(width=2, color='white')
                ),
                fill='tonexty' if i == 1 else None,
                fillcolor=f'rgba({",".join(map(str, [int(colors[var][1:3], 16), int(colors[var][3:5], 16), int(colors[var][5:7], 16)]))}, 0.1)',
                hovertemplate=f'<b>{var}</b><br>연도: %{{x}}<br>값: %{{y:.1f}}<extra></extra>'
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(selected_variables) + 100,
        title={
            'text': '농업 기후 시계열 분석',
            'font': {'family': 'Inter', 'size': 18, 'color': '#1a1a1a'},
            'x': 0.5
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        title_text="연도",
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        row=len(selected_variables), col=1
    )
    
    return fig

def create_correlation_chart(df):
    """상관관계 분석 차트"""
    fig = go.Figure()
    
    # 기온 vs 수확량
    fig.add_trace(go.Scatter(
        x=df['temperature'],
        y=df['crop_yield'],
        mode='markers',
        name='기온 vs 수확량',
        marker=dict(
            size=12,
            color=df['year'],
            colorscale='Viridis',
            colorbar=dict(title="연도"),
            line=dict(width=1, color='white')
        ),
        text=df['year'],
        hovertemplate='<b>기온 vs 수확량</b><br>기온: %{x:.1f}°C<br>수확량: %{y:.1f}<br>연도: %{text}<extra></extra>'
    ))
    
    # 추세선 추가
    z = np.polyfit(df['temperature'], df['crop_yield'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df['temperature'],
        y=p(df['temperature']),
        mode='lines',
        name='추세선',
        line=dict(color='#dc2626', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title={
            'text': '기온과 수확량의 상관관계',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        xaxis_title='평균기온 (°C)',
        yaxis_title='수확량 지수',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

def create_trend_analysis(df):
    """트렌드 분석 결과"""
    # 간단한 트렌드 계산
    temp_trend = np.polyfit(df['year'], df['temperature'], 1)[0]
    rain_trend = np.polyfit(df['year'], df['rainfall'], 1)[0]
    yield_trend = np.polyfit(df['year'], df['crop_yield'], 1)[0]
    
    # 상관계수 계산
    temp_yield_corr = np.corrcoef(df['temperature'], df['crop_yield'])[0, 1]
    rain_yield_corr = np.corrcoef(df['rainfall'], df['crop_yield'])[0, 1]
    
    return {
        'temp_trend': temp_trend,
        'rain_trend': rain_trend,
        'yield_trend': yield_trend,
        'temp_yield_corr': temp_yield_corr,
        'rain_yield_corr': rain_yield_corr
    }

def main():
    # 헤더
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📈 시계열 분석</h1>
        <p class="page-subtitle">연도별 기후변화와 농작물 수확량의 변화 추이를 상세히 분석합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 샘플 데이터 생성
    df = generate_sample_data()
    
    # 사이드바 필터
    with st.sidebar:
        st.markdown("### 🎛️ 분석 옵션")
        
        # 분석 기간 선택
        year_range = st.slider(
            "분석 기간",
            min_value=2010,
            max_value=2023,
            value=(2015, 2023),
            step=1
        )
        
        # 분석 변수 선택
        variables = st.multiselect(
            "분석할 변수 선택",
            ['평균기온', '강수량', '수확량'],
            default=['평균기온', '수확량']
        )
        
        # 지역 선택 (추후 실제 데이터 적용시 활용)
        selected_region = st.selectbox(
            "분석 지역",
            ["전국 평균", "서울", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
        )
        
        # 작물 선택
        selected_crop = st.selectbox(
            "분석 작물",
            ["전체 작물", "쌀", "밀", "콩", "옥수수", "감자", "배추", "무"]
        )
    
    # 필터 적용
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # 트렌드 분석 결과
    trend_results = create_trend_analysis(filtered_df)
    
    # 상단 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{trend_results['temp_trend']:.2f}°C/년</h4>
            <p>기온 증가율</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{trend_results['yield_trend']:.1f}/년</h4>
            <p>수확량 변화율</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{trend_results['temp_yield_corr']:.2f}</h4>
            <p>기온-수확량 상관계수</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{len(filtered_df)}년</h4>
            <p>분석 기간</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 차트
    if variables:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        main_chart = create_main_timeseries_chart(filtered_df, variables)
        st.plotly_chart(main_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("분석할 변수를 선택해주세요.")
    
    # 상관관계 분석
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        corr_chart = create_correlation_chart(filtered_df)
        st.plotly_chart(corr_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-card">
            <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">📊 분석 결과</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **🌡️ 기온 변화**
        - 연평균 **{trend_results['temp_trend']:.2f}°C/년** 상승
        - 총 **{trend_results['temp_trend'] * len(filtered_df):.1f}°C** 상승
        
        **🌾 수확량 변화**
        - 연평균 **{trend_results['yield_trend']:.1f}포인트/년** 변화
        - 기온과의 상관계수: **{trend_results['temp_yield_corr']:.2f}**
        
        **☔ 강수량 영향**
        - 수확량과의 상관계수: **{trend_results['rain_yield_corr']:.2f}**
        """)
        
        if trend_results['temp_yield_corr'] < -0.5:
            st.error("🚨 **강한 음의 상관관계**: 기온 상승이 수확량에 부정적 영향")
        elif trend_results['temp_yield_corr'] < -0.3:
            st.warning("⚠️ **중간 음의 상관관계**: 기온 상승 주의 필요")
        else:
            st.info("ℹ️ **약한 상관관계**: 추가 분석 필요")
    
    # 하단 인사이트
    st.markdown("""
    <div class="chart-card">
        <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">💡 주요 인사이트</h3>
    </div>
    """, unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        **📈 관찰된 트렌드**
        - 지속적인 기온 상승 패턴 확인
        - 수확량의 연도별 변동성 증가
        - 극한 기후 이벤트의 영향 가시화
        """)
    
    with insight_col2:
        st.markdown("""
        **🎯 정책 제언**
        - 내열성 작물 품종 개발 필요
        - 기후 적응 농업 기술 도입
        - 조기 경보 시스템 구축
        """)

if __name__ == "__main__":
    main()