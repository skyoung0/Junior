import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="농업 기후 분석 대시보드",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 깔끔하고 현대적인 CSS 스타일
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 전체 배경 */
    .main {
        padding: 0 !important;
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Streamlit 기본 요소 숨기기 */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    
    /* 메인 헤더 - 완전히 새로운 스타일 */
    .dashboard-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 1px solid rgba(34, 197, 94, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #22c55e, #10b981, #059669);
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a1a1a;
        text-align: center;
        margin: 0 0 1rem 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin: 0;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* KPI 카드 - 완전히 새로운 디자인 */
    .kpi-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        border-radius: 20px 20px 0 0;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.95);
        border-color: rgba(34, 197, 94, 0.2);
        box-shadow: 0 20px 60px rgba(34, 197, 94, 0.15);
    }
    
    .kpi-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #888888;
        margin: 0 0 0.8rem 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        color: #1a1a1a;
        letter-spacing: -1px;
    }
    
    .kpi-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #999999;
        margin: 0;
        font-weight: 500;
    }
    
    /* 네비게이션 카드 */
    .nav-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
    }
    
    /* 섹션 - 배경 제거하고 더 깔끔하게 */
    .section-card {
        background: transparent;
        border-radius: 0;
        padding: 2rem 0;
        margin: 2rem 0;
        border: none;
        box-shadow: none;
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 2rem 0;
        letter-spacing: -0.5px;
        position: relative;
        padding-left: 20px;
    }
    
    .section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 2px;
    }
    
    /* 기능 카드 - 완전히 새로운 스타일 */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #22c55e, #16a34a, #15803d);
        border-radius: 24px 24px 0 0;
    }
    
    .feature-card:hover {
        transform: translateY(-12px);
        background: rgba(255, 255, 255, 0.95);
        border-color: rgba(34, 197, 94, 0.2);
        box-shadow: 0 25px 70px rgba(34, 197, 94, 0.2);
    }
    
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 1rem 0;
        letter-spacing: -0.3px;
    }
    
    .feature-desc {
        font-family: 'Inter', sans-serif;
        color: #666666;
        line-height: 1.7;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-list li {
        font-family: 'Inter', sans-serif;
        color: #555555;
        padding: 0.8rem 0;
        position: relative;
        padding-left: 2rem;
        font-size: 0.9rem;
        line-height: 1.5;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }
    
    .feature-list li:before {
        content: '●';
        position: absolute;
        left: 0.5rem;
        color: #22c55e;
        font-size: 0.8rem;
        top: 1rem;
    }
    
    .feature-list li:last-child {
        border-bottom: none;
    }
    
    /* 차트 컨테이너 */
    .chart-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* 인사이트 카드들 */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .insight-card {
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-4px);
    }
    
    /* 푸터 */
    .footer {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 4rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    /* 반응형 */
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .kpi-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def create_trend_chart():
    """트렌드 차트 생성 (세련된 스타일)"""
    years = list(range(2010, 2024))
    
    # 샘플 데이터 생성
    temp_data = [14.2 + 0.08*i + np.random.normal(0, 0.2) for i in range(len(years))]
    yield_data = [100 - 1.5*i + np.random.normal(0, 3) for i in range(len(years))]
    
    fig = go.Figure()
    
    # 기온 데이터
    fig.add_trace(go.Scatter(
        x=years, 
        y=temp_data,
        mode='lines+markers',
        name='평균기온',
        line=dict(
            color='#22c55e', 
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='#22c55e',
            line=dict(width=2, color='white')
        ),
        fill='tonexty',
        fillcolor='rgba(34, 197, 94, 0.1)',
        yaxis='y',
        hovertemplate='<b>기온</b><br>연도: %{x}<br>온도: %{y:.1f}°C<extra></extra>'
    ))
    
    # 수확량 데이터
    fig.add_trace(go.Scatter(
        x=years, 
        y=yield_data,
        mode='lines+markers',
        name='수확량지수',
        line=dict(
            color='#059669', 
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color='#059669',
            line=dict(width=2, color='white')
        ),
        yaxis='y2',
        hovertemplate='<b>수확량</b><br>연도: %{x}<br>지수: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '기온 상승과 수확량 변화 추이',
            'font': {'family': 'Inter', 'size': 18, 'color': '#1f2937'},
            'x': 0.5
        },
        xaxis=dict(
            title={'text': '연도', 'font': {'family': 'Inter', 'color': '#6b7280'}},
            gridcolor='#f3f4f6',
            showgrid=True,
            zeroline=False,
            tickfont={'family': 'Inter', 'color': '#6b7280'}
        ),
        yaxis=dict(
            title={'text': '평균기온(°C)', 'font': {'family': 'Inter', 'color': '#6b7280'}}, 
            side='left',
            gridcolor='#f3f4f6',
            showgrid=True,
            zeroline=False,
            tickfont={'family': 'Inter', 'color': '#6b7280'}
        ),
        yaxis2=dict(
            title={'text': '수확량지수', 'font': {'family': 'Inter', 'color': '#6b7280'}}, 
            side='right', 
            overlaying='y',
            tickfont={'family': 'Inter', 'color': '#6b7280'}
        ),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1
        )
    )
    
    return fig

def create_mini_charts():
    """작은 인터랙티브 차트들 생성"""
    # 지역별 데이터
    regions = ['서울', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    values = np.random.normal(85, 15, len(regions))
    
    fig1 = px.bar(
        x=regions, y=values,
        color=values,
        color_continuous_scale=['#bbf7d0', '#22c55e', '#15803d'],
        title="지역별 수확량 지수"
    )
    fig1.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 12},
        title={'text': "지역별 수확량 지수", 'font': {'size': 14, 'family': 'Inter'}},
        showlegend=False,
        xaxis={'tickfont': {'family': 'Inter', 'size': 10}},
        yaxis={'tickfont': {'family': 'Inter', 'size': 10}}
    )
    fig1.update_traces(hovertemplate='%{x}<br>수확량: %{y:.1f}<extra></extra>')
    
    # 작물별 분포
    crops = ['쌀', '밀', '콩', '옥수수', '감자']
    crop_values = [35, 20, 18, 15, 12]
    
    fig2 = px.pie(
        values=crop_values, names=crops,
        color_discrete_sequence=['#bbf7d0', '#86efac', '#4ade80', '#22c55e', '#16a34a'],
        title="주요 작물 재배 비중"
    )
    fig2.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 12},
        title={'text': "주요 작물 재배 비중", 'font': {'size': 14, 'family': 'Inter'}}
    )
    fig2.update_traces(hovertemplate='%{label}<br>비중: %{percent}<extra></extra>')
    
    return fig1, fig2

def main():
    # 헤더
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="main-title">🌾 농업 기후 분석 대시보드</h1>
        <p class="main-subtitle">기후변화가 농작물 수확량에 미치는 영향을 실시간으로 분석하고 예측합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("""
        <div class="nav-card">
            <h3 style="color: #1f2937; font-family: 'Inter'; margin: 0 0 1rem 0;">📋 분석 메뉴</h3>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">원하는 분석을 선택하세요</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 인터랙티브 필터
        st.markdown("### 🎛️ 데이터 필터")
        
        year_range = st.slider(
            "분석 기간",
            min_value=2010,
            max_value=2023,
            value=(2018, 2023),
            step=1
        )
        
        selected_regions = st.multiselect(
            "분석 지역",
            ["전체", "서울", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"],
            default=["전체"]
        )
        
        selected_crops = st.multiselect(
            "분석 작물",
            ["전체", "쌀", "밀", "콩", "옥수수", "감자", "배추", "무", "토마토"],
            default=["전체"]
        )
        
        st.markdown("---")
        
        # 실시간 상태
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); padding: 1rem; border-radius: 12px; margin: 1rem 0;">
            <h4 style="color: #15803d; margin: 0 0 0.5rem 0; font-family: 'Inter';">📡 실시간 상태</h4>
            <p style="color: #166534; margin: 0; font-size: 0.9rem;">
                ✅ 데이터 연결 정상<br>
                🔄 마지막 업데이트: 2시간 전<br>
                📊 분석 완료율: 98.5%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">평균 기온 변화</div>
            <div class="kpi-value" style="color: #dc2626;">+1.4°C</div>
            <div class="kpi-subtitle">최근 10년 대비 ↗️ 8.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">전체 수확량</div>
            <div class="kpi-value" style="color: #059669;">2.4만톤</div>
            <div class="kpi-subtitle">전년 대비 ↘️ 5.7%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">분석 커버리지</div>
            <div class="kpi-value" style="color: #0ea5e9;">17개 시도</div>
            <div class="kpi-subtitle">전국 데이터 완료 ✅</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">위험 지역</div>
            <div class="kpi-value" style="color: #ea580c;">3개 지역</div>
            <div class="kpi-subtitle">주의 관찰 필요 ⚠️</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 차트
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">📈 핵심 트렌드 분석</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        trend_chart = create_trend_chart()
        st.plotly_chart(trend_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fig1, fig2 = create_mini_charts()
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 기능 소개
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">🛠️ 주요 분석 기능</h2>
        <div class="feature-grid">
            <div class="feature-card">
                <h3 class="feature-title">📈 시계열 분석</h3>
                <p class="feature-desc">연도별 기후변화와 작물 수확량의 변화 추이를 실시간으로 추적합니다.</p>
                <ul class="feature-list">
                    <li>기온/강수량 변화 모니터링</li>
                    <li>수확량 트렌드 예측</li>
                    <li>지역/작물별 맞춤 필터링</li>
                    <li>계절별 패턴 분석</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3 class="feature-title">🗺️ 지역별 영향 분석</h3>
                <p class="feature-desc">전국 시도별 기후변화 영향을 인터랙티브 맵으로 시각화합니다.</p>
                <ul class="feature-list">
                    <li>실시간 히트맵 제공</li>
                    <li>기후요소 상관관계 분석</li>
                    <li>취약지역 조기 감지</li>
                    <li>지역간 비교 분석</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3 class="feature-title">🐛 병해충 영향 분석</h3>
                <p class="feature-desc">병해충 발생 패턴과 수확량 감소의 상관관계를 정밀 분석합니다.</p>
                <ul class="feature-list">
                    <li>실시간 발생 현황</li>
                    <li>피해 강도별 분류</li>
                    <li>통계적 검증 제공</li>
                    <li>예방 전략 제안</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 인사이트 요약
    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">💡 주요 인사이트</h2>
        <div class="insight-grid">
            <div class="insight-card" style="background: linear-gradient(135deg, rgba(255, 243, 199, 0.6), rgba(253, 230, 138, 0.8));">
                <h4 style="color: #92400e; margin: 0 0 0.5rem 0; font-family: 'Inter'; font-weight: 600;">⚠️ 주의 필요</h4>
                <p style="color: #451a03; margin: 0; font-size: 0.9rem;">강원도 지역 감자 수확량 15% 감소 예상</p>
            </div>
            
            <div class="insight-card" style="background: linear-gradient(135deg, rgba(219, 234, 254, 0.6), rgba(191, 219, 254, 0.8));">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0; font-family: 'Inter'; font-weight: 600;">📊 데이터 포인트</h4>
                <p style="color: #1e3a8a; margin: 0; font-size: 0.9rem;">올해 평균기온 1.4°C 상승, 강수량 12% 감소</p>
            </div>
            
            <div class="insight-card" style="background: linear-gradient(135deg, rgba(240, 253, 244, 0.6), rgba(220, 252, 231, 0.8));">
                <h4 style="color: #15803d; margin: 0 0 0.5rem 0; font-family: 'Inter'; font-weight: 600;">✅ 긍정적</h4>
                <p style="color: #14532d; margin: 0; font-size: 0.9rem;">전남 지역 쌀 수확량 전년 대비 8% 증가</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 푸터
    st.markdown("""
    <div class="footer">
        <h4 style="color: #1f2937; margin: 0 0 1rem 0; font-family: 'Inter';">🌾 농업 기후 분석 대시보드</h4>
        <p style="color: #6b7280; margin: 0; font-size: 0.9rem;">
            데이터 기반 농업 정책 지원 시스템 | 실시간 기후 모니터링 및 예측 서비스<br>
            📧 contact@agri-climate.kr | 📞 02-1234-5678 | 🌐 www.agri-climate.kr
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()