import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns

# 페이지 설정
st.set_page_config(
    page_title="병해충 영향 분석",
    page_icon="🐛",
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
    
    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a1a1a;
        margin: 0;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #666666;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    .stat-significant {
        color: #dc2626 !important;
    }
    
    .stat-not-significant {
        color: #059669 !important;
    }
    
    .pest-badge {
        display: inline-block;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #92400e;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid rgba(146, 64, 14, 0.2);
    }
    
    .pest-severe {
        background: linear-gradient(135deg, #fee2e2, #fecaca) !important;
        color: #dc2626 !important;
        border-color: rgba(220, 38, 38, 0.2) !important;
    }
    
    .pest-moderate {
        background: linear-gradient(135deg, #fed7aa, #fdba74) !important;
        color: #ea580c !important;
        border-color: rgba(234, 88, 12, 0.2) !important;
    }
    
    .pest-mild {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0) !important;
        color: #15803d !important;
        border-color: rgba(21, 128, 61, 0.2) !important;
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
    
    .alert-box {
        background: rgba(220, 38, 38, 0.1);
        border: 1px solid rgba(220, 38, 38, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #dc2626;
        margin: 0 0 0.5rem 0;
    }
    
    .alert-text {
        font-family: 'Inter', sans-serif;
        color: #991b1b;
        margin: 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def generate_pest_data():
    """병해충 영향 샘플 데이터 생성"""
    np.random.seed(42)
    
    regions = ["서울", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
    crops = ["쌀", "밀", "콩", "옥수수", "감자", "배추", "무", "토마토"]
    pests = ["진딧물", "응애류", "나방류", "깍지벌레", "선충류", "균류", "바이러스", "세균성병"]
    severity_levels = ["경미", "보통", "심각"]
    
    data = []
    for year in range(2010, 2024):
        for region in regions:
            for crop in crops:
                # 병해충 발생 여부 (30% 확률)
                has_pest = np.random.choice([True, False], p=[0.3, 0.7])
                
                if has_pest:
                    # 병해충 종류 랜덤 선택
                    pest_type = np.random.choice(pests)
                    severity = np.random.choice(severity_levels, p=[0.5, 0.3, 0.2])
                    
                    # 피해 면적 (심각도에 따라)
                    if severity == "심각":
                        damage_area = np.random.uniform(50, 200)
                        yield_impact = np.random.uniform(-40, -20)  # 20-40% 감소
                    elif severity == "보통":
                        damage_area = np.random.uniform(20, 80)
                        yield_impact = np.random.uniform(-25, -10)  # 10-25% 감소
                    else:  # 경미
                        damage_area = np.random.uniform(5, 30)
                        yield_impact = np.random.uniform(-15, -2)   # 2-15% 감소
                    
                    # 기본 수확량에서 피해 적용
                    base_yield = np.random.normal(85, 10)
                    actual_yield = base_yield * (1 + yield_impact/100)
                    
                    data.append({
                        'year': year,
                        'region': region,
                        'crop': crop,
                        'has_pest': True,
                        'pest_type': pest_type,
                        'severity': severity,
                        'damage_area': damage_area,
                        'yield': max(actual_yield, 30),  # 최소값 보장
                        'yield_loss': abs(yield_impact)
                    })
                else:
                    # 병해충 없는 경우
                    base_yield = np.random.normal(90, 8)  # 약간 더 높은 기본 수확량
                    
                    data.append({
                        'year': year,
                        'region': region,
                        'crop': crop,
                        'has_pest': False,
                        'pest_type': None,
                        'severity': None,
                        'damage_area': 0,
                        'yield': base_yield,
                        'yield_loss': 0
                    })
    
    return pd.DataFrame(data)

def create_comparison_boxplot(df, selected_crop, selected_years):
    """병해충 발생/비발생 수확량 비교 박스플롯"""
    filtered_df = df[
        (df['crop'] == selected_crop) & 
        (df['year'].isin(selected_years))
    ]
    
    fig = go.Figure()
    
    # 병해충 미발생 그룹
    no_pest_data = filtered_df[filtered_df['has_pest'] == False]['yield']
    fig.add_trace(go.Box(
        y=no_pest_data,
        name='병해충 미발생',
        boxpoints='outliers',
        marker_color='rgba(34, 197, 94, 0.7)',
        line_color='#22c55e',
        fillcolor='rgba(34, 197, 94, 0.3)',
        marker_size=4
    ))
    
    # 병해충 발생 그룹
    pest_data = filtered_df[filtered_df['has_pest'] == True]['yield']
    fig.add_trace(go.Box(
        y=pest_data,
        name='병해충 발생',
        boxpoints='outliers',
        marker_color='rgba(220, 38, 38, 0.7)',
        line_color='#dc2626',
        fillcolor='rgba(220, 38, 38, 0.3)',
        marker_size=4
    ))
    
    fig.update_layout(
        title={
            'text': f'{selected_crop} 병해충 발생 여부별 수확량 분포',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        yaxis_title='수확량 지수',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=True
    )
    
    return fig

def create_severity_analysis(df, selected_crop, selected_years):
    """병해충 심각도별 영향 분석"""
    filtered_df = df[
        (df['crop'] == selected_crop) & 
        (df['year'].isin(selected_years)) &
        (df['has_pest'] == True)
    ]
    
    if len(filtered_df) == 0:
        return None, {}
    
    fig = go.Figure()
    
    colors = {
        '경미': '#22c55e', 
        '보통': '#eab308', 
        '심각': '#dc2626'
    }
    
    fillcolors = {
        '경미': 'rgba(34, 197, 94, 0.3)',
        '보통': 'rgba(234, 179, 8, 0.3)', 
        '심각': 'rgba(220, 38, 38, 0.3)'
    }
    
    for severity in ['경미', '보통', '심각']:
        severity_data = filtered_df[filtered_df['severity'] == severity]['yield']
        if len(severity_data) > 0:
            fig.add_trace(go.Box(
                y=severity_data,
                name=severity,
                boxpoints='outliers',
                marker_color=colors[severity],
                line_color=colors[severity],
                fillcolor=fillcolors[severity],
                marker_size=4
            ))
    
    fig.update_layout(
        title={
            'text': f'{selected_crop} 병해충 심각도별 수확량 영향',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        yaxis_title='수확량 지수',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    # 심각도별 통계
    severity_stats = {}
    for severity in ['경미', '보통', '심각']:
        severity_data = filtered_df[filtered_df['severity'] == severity]
        if len(severity_data) > 0:
            severity_stats[severity] = {
                'count': len(severity_data),
                'mean_yield': severity_data['yield'].mean(),
                'mean_loss': severity_data['yield_loss'].mean(),
                'damage_area': severity_data['damage_area'].mean()
            }
    
    return fig, severity_stats

def create_pest_type_analysis(df, selected_years):
    """병해충 종류별 발생 빈도 및 피해 분석"""
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['has_pest'] == True)
    ]
    
    # 병해충별 발생 빈도
    pest_counts = filtered_df['pest_type'].value_counts()
    
    # 병해충별 평균 피해율
    pest_damage = filtered_df.groupby('pest_type')['yield_loss'].mean().sort_values(ascending=False)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['병해충별 발생 빈도', '병해충별 평균 피해율'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 발생 빈도 바차트
    fig.add_trace(
        go.Bar(
            x=pest_counts.index,
            y=pest_counts.values,
            name='발생 건수',
            marker_color='rgba(34, 197, 94, 0.7)',
            text=pest_counts.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>발생: %{y}건<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 평균 피해율 바차트
    fig.add_trace(
        go.Bar(
            x=pest_damage.index,
            y=pest_damage.values,
            name='평균 피해율',
            marker_color='rgba(220, 38, 38, 0.7)',
            text=[f'{x:.1f}%' for x in pest_damage.values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>피해율: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="병해충 종류", row=1, col=1)
    fig.update_xaxes(title_text="병해충 종류", row=1, col=2)
    fig.update_yaxes(title_text="발생 건수", row=1, col=1)
    fig.update_yaxes(title_text="평균 피해율 (%)", row=1, col=2)
    
    fig.update_layout(
        title={
            'text': '병해충 종류별 위험도 분석',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=False
    )
    
    return fig, pest_counts, pest_damage

def create_temporal_analysis(df, selected_crop):
    """연도별 병해충 발생 트렌드"""
    crop_df = df[df['crop'] == selected_crop]
    
    # 연도별 병해충 발생률
    yearly_pest_rate = crop_df.groupby('year')['has_pest'].agg(['sum', 'count'])
    yearly_pest_rate['rate'] = (yearly_pest_rate['sum'] / yearly_pest_rate['count']) * 100
    
    # 연도별 평균 피해율 (병해충 발생시에만)
    yearly_damage = crop_df[crop_df['has_pest'] == True].groupby('year')['yield_loss'].mean()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['연도별 병해충 발생률', '연도별 평균 피해율'],
        vertical_spacing=0.1
    )
    
    # 발생률 차트
    fig.add_trace(
        go.Scatter(
            x=yearly_pest_rate.index,
            y=yearly_pest_rate['rate'],
            mode='lines+markers',
            name='발생률',
            line=dict(color='#dc2626', width=3),
            marker=dict(size=8, color='#dc2626'),
            fill='tonexty',
            fillcolor='rgba(220, 38, 38, 0.1)',
            hovertemplate='<b>연도: %{x}</b><br>발생률: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 피해율 차트
    fig.add_trace(
        go.Scatter(
            x=yearly_damage.index,
            y=yearly_damage.values,
            mode='lines+markers',
            name='피해율',
            line=dict(color='#ea580c', width=3),
            marker=dict(size=8, color='#ea580c'),
            fill='tonexty',
            fillcolor='rgba(234, 88, 12, 0.1)',
            hovertemplate='<b>연도: %{x}</b><br>피해율: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="연도", row=2, col=1)
    fig.update_yaxes(title_text="발생률 (%)", row=1, col=1)
    fig.update_yaxes(title_text="평균 피해율 (%)", row=2, col=1)
    
    fig.update_layout(
        title={
            'text': f'{selected_crop} 병해충 발생 추이 분석',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=False
    )
    
    return fig

def perform_statistical_test(df, selected_crop, selected_years):
    """통계적 유의성 검정"""
    filtered_df = df[
        (df['crop'] == selected_crop) & 
        (df['year'].isin(selected_years))
    ]
    
    no_pest_yield = filtered_df[filtered_df['has_pest'] == False]['yield']
    pest_yield = filtered_df[filtered_df['has_pest'] == True]['yield']
    
    if len(no_pest_yield) < 2 or len(pest_yield) < 2:
        return None
    
    # t-검정 수행
    t_stat, p_value = stats.ttest_ind(no_pest_yield, pest_yield)
    
    # 기술통계
    stats_dict = {
        'no_pest': {
            'mean': no_pest_yield.mean(),
            'std': no_pest_yield.std(),
            'count': len(no_pest_yield)
        },
        'pest': {
            'mean': pest_yield.mean(),
            'std': pest_yield.std(),
            'count': len(pest_yield)
        },
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    return stats_dict

def main():
    # 헤더
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🐛 병해충 영향 분석</h1>
        <p class="page-subtitle">병해충 발생이 농작물 수확량에 미치는 영향을 통계적으로 분석하고 검증합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 샘플 데이터 생성
    df = generate_pest_data()
    
    # 사이드바 필터
    with st.sidebar:
        st.markdown("### 🎛️ 분석 옵션")
        
        # 작물 선택
        selected_crop = st.selectbox(
            "분석 작물",
            sorted(df['crop'].unique()),
            index=0
        )
        
        # 분석 기간
        selected_years = st.multiselect(
            "분석 연도",
            sorted(df['year'].unique(), reverse=True),
            default=sorted(df['year'].unique())[-5:]  # 최근 5년
        )
        
        # 지역 선택
        selected_regions = st.multiselect(
            "분석 지역",
            sorted(df['region'].unique()),
            default=sorted(df['region'].unique())
        )
        
        # 병해충 종류 필터
        available_pests = df[df['has_pest'] == True]['pest_type'].unique()
        selected_pests = st.multiselect(
            "특정 병해충 분석",
            sorted(available_pests),
            default=[]
        )
    
    # 데이터 필터링
    filtered_df = df[
        (df['crop'] == selected_crop) & 
        (df['year'].isin(selected_years)) &
        (df['region'].isin(selected_regions))
    ]
    
    if selected_pests:
        pest_filtered = filtered_df[
            (filtered_df['has_pest'] == False) | 
            (filtered_df['pest_type'].isin(selected_pests))
        ]
    else:
        pest_filtered = filtered_df
    
    # 상단 메트릭
    pest_rate = (pest_filtered['has_pest'].sum() / len(pest_filtered)) * 100
    avg_damage = pest_filtered[pest_filtered['has_pest'] == True]['yield_loss'].mean()
    total_damage_area = pest_filtered[pest_filtered['has_pest'] == True]['damage_area'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{pest_rate:.1f}%</h4>
            <p>병해충 발생률</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color_class = "metric-mini"
        if avg_damage > 20:
            color_class = "metric-mini" 
            st.markdown(f"""
            <div class="{color_class}" style="background: rgba(220, 38, 38, 0.1); border-color: rgba(220, 38, 38, 0.2);">
                <h4 style="color: #dc2626;">{avg_damage:.1f}%</h4>
                <p style="color: #dc2626;">평균 수확량 손실</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-mini">
                <h4>{avg_damage:.1f}%</h4>
                <p>평균 수확량 손실</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{total_damage_area:.0f}ha</h4>
            <p>총 피해 면적</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        severe_cases = len(pest_filtered[pest_filtered['severity'] == '심각'])
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{severe_cases}건</h4>
            <p>심각한 피해 사례</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 통계적 검정 결과
    stats_result = perform_statistical_test(pest_filtered, selected_crop, selected_years)
    
    if stats_result:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 박스플롯 비교
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            boxplot_fig = create_comparison_boxplot(pest_filtered, selected_crop, selected_years)
            st.plotly_chart(boxplot_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # 통계 결과
            st.markdown("""
            <div class="stat-card">
                <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">📊 통계적 검정</h3>
            </div>
            """, unsafe_allow_html=True)
            
            significance_class = "stat-significant" if stats_result['significant'] else "stat-not-significant"
            significance_text = "유의함" if stats_result['significant'] else "유의하지 않음"
            
            st.markdown(f"""
            **📈 기술통계**
            - **미발생 평균**: {stats_result['no_pest']['mean']:.1f} ± {stats_result['no_pest']['std']:.1f}
            - **발생 평균**: {stats_result['pest']['mean']:.1f} ± {stats_result['pest']['std']:.1f}
            - **표본 크기**: {stats_result['no_pest']['count']} vs {stats_result['pest']['count']}
            
            **🔬 t-검정 결과**
            - **t-통계량**: {stats_result['t_stat']:.3f}
            - **p-값**: {stats_result['p_value']:.4f}
            - **유의성**: <span class="{significance_class}">{significance_text}</span>
            """, unsafe_allow_html=True)
            
            if stats_result['significant']:
                st.markdown("""
                <div class="alert-box">
                    <p class="alert-title">⚠️ 통계적으로 유의한 차이</p>
                    <p class="alert-text">병해충 발생이 수확량에 유의미한 부정적 영향을 미칩니다 (p < 0.05)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-card">
                    <p style="color: #059669; font-weight: 600;">✅ 통계적으로 유의하지 않음</p>
                    <p style="color: #059669; font-size: 0.9rem;">현재 데이터로는 명확한 영향 확인 어려움</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 심각도별 분석
    severity_fig, severity_stats = create_severity_analysis(pest_filtered, selected_crop, selected_years)
    
    if severity_fig:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(severity_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">🎯 심각도별 영향</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for severity, stats in severity_stats.items():
                badge_class = f"pest-{severity}" if severity in ['mild', 'moderate', 'severe'] else 'pest-badge'
                if severity == '심각':
                    badge_class = 'pest-severe'
                elif severity == '보통':
                    badge_class = 'pest-moderate'
                else:
                    badge_class = 'pest-mild'
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <span class="pest-badge {badge_class}">{severity}</span>
                    <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">
                        • 발생: {stats['count']}건<br>
                        • 평균 수확량: {stats['mean_yield']:.1f}<br>
                        • 평균 손실: {stats['mean_loss']:.1f}%<br>
                        • 피해 면적: {stats['damage_area']:.1f}ha
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # 병해충 종류별 분석
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    pest_type_fig, pest_counts, pest_damage = create_pest_type_analysis(pest_filtered, selected_years)
    st.plotly_chart(pest_type_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 시계열 트렌드 분석
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    temporal_fig = create_temporal_analysis(df, selected_crop)
    st.plotly_chart(temporal_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 하단 인사이트 및 권장사항
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📊 주요 분석 결과**
        
        **🔢 발생 현황**
        - 전체 발생률: **{pest_rate:.1f}%**
        - 평균 수확량 손실: **{avg_damage:.1f}%**
        - 총 피해 면적: **{total_damage_area:.0f}ha**
        
        **🏆 최다 발생 병해충**
        1. **{pest_counts.index[0]}**: {pest_counts.iloc[0]}건
        2. **{pest_counts.index[1]}**: {pest_counts.iloc[1]}건
        3. **{pest_counts.index[2]}**: {pest_counts.iloc[2]}건
        
        **⚠️ 고위험 병해충** (피해율 기준)
        1. **{pest_damage.index[0]}**: {pest_damage.iloc[0]:.1f}%
        2. **{pest_damage.index[1]}**: {pest_damage.iloc[1]:.1f}%
        3. **{pest_damage.index[2]}**: {pest_damage.iloc[2]:.1f}%
        """)
    
    with col2:
        st.markdown("""
        **💡 정책 제언 및 대응방안**
        
        **🛡️ 예방 조치**
        - 조기 감지 시스템 구축
        - 정기적인 작물 모니터링
        - 생물학적 방제 기술 도입
        
        **📋 대응 전략**
        - 고위험 병해충 집중 관리
        - 저항성 품종 개발 및 보급
        - 통합해충관리(IPM) 시스템 적용
        
        **📈 모니터링 강화**
        - 실시간 발생 현황 추적
        - 기상 조건과의 연관성 분석
        - 지역별 맞춤형 대응책 수립
        """)
    
    # 경고 메시지 (조건부)
    if avg_damage > 15:
        st.markdown(f"""
        <div class="alert-box">
            <p class="alert-title">🚨 높은 피해율 경고</p>
            <p class="alert-text">
                {selected_crop}의 평균 수확량 손실이 {avg_damage:.1f}%로 높습니다. 
                긴급한 병해충 관리 대책이 필요합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if severe_cases > 10:
        st.markdown(f"""
        <div class="alert-box">
            <p class="alert-title">⚠️ 심각한 피해 다발</p>
            <p class="alert-text">
                심각한 피해 사례가 {severe_cases}건 발생했습니다. 
                해당 지역에 대한 집중적인 모니터링과 지원이 필요합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 추가 분석 도구
    with st.expander("🔬 상세 통계 분석"):
        if stats_result:
            st.markdown("### 📊 상세 통계 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**병해충 미발생 그룹**")
                st.write({
                    "표본 수": stats_result['no_pest']['count'],
                    "평균": f"{stats_result['no_pest']['mean']:.2f}",
                    "표준편차": f"{stats_result['no_pest']['std']:.2f}",
                    "최솟값": f"{pest_filtered[pest_filtered['has_pest'] == False]['yield'].min():.2f}",
                    "최댓값": f"{pest_filtered[pest_filtered['has_pest'] == False]['yield'].max():.2f}"
                })
            
            with col2:
                st.markdown("**병해충 발생 그룹**")
                st.write({
                    "표본 수": stats_result['pest']['count'],
                    "평균": f"{stats_result['pest']['mean']:.2f}",
                    "표준편차": f"{stats_result['pest']['std']:.2f}",
                    "최솟값": f"{pest_filtered[pest_filtered['has_pest'] == True]['yield'].min():.2f}",
                    "최댓값": f"{pest_filtered[pest_filtered['has_pest'] == True]['yield'].max():.2f}"
                })
            
            # 효과 크기 계산 (Cohen's d)
            pooled_std = np.sqrt(((stats_result['no_pest']['count'] - 1) * stats_result['no_pest']['std']**2 + 
                                 (stats_result['pest']['count'] - 1) * stats_result['pest']['std']**2) /
                                (stats_result['no_pest']['count'] + stats_result['pest']['count'] - 2))
            
            cohens_d = (stats_result['no_pest']['mean'] - stats_result['pest']['mean']) / pooled_std
            
            st.markdown(f"""
            ### 📏 효과 크기 (Cohen's d)
            - **Cohen's d**: {cohens_d:.3f}
            - **효과 크기 해석**: 
            """)
            
            if abs(cohens_d) < 0.2:
                st.write("작은 효과 (Small effect)")
            elif abs(cohens_d) < 0.5:
                st.write("중간 효과 (Medium effect)")
            elif abs(cohens_d) < 0.8:
                st.write("큰 효과 (Large effect)")
            else:
                st.write("매우 큰 효과 (Very large effect)")

if __name__ == "__main__":
    main()