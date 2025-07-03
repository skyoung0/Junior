import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import folium
from streamlit_folium import st_folium

# 페이지 설정
st.set_page_config(
    page_title="지역별 영향 분석",
    page_icon="🗺️",
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
    
    .map-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
        height: 600px;
        overflow: hidden;
    }
    
    .ranking-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    .ranking-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        font-family: 'Inter', sans-serif;
    }
    
    .ranking-item:last-child {
        border-bottom: none;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important; 
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #ea580c, #d97706) !important; 
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #22c55e, #16a34a) !important; 
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

def generate_regional_data():
    """지역별 샘플 데이터 생성"""
    np.random.seed(42)
    
    regions = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
    ]
    
    # 지역별 좌표 (대략적)
    coordinates = {
        "서울": [37.5665, 126.9780], "부산": [35.1796, 129.0756], 
        "대구": [35.8714, 128.6014], "인천": [37.4563, 126.7052],
        "광주": [35.1595, 126.8526], "대전": [36.3504, 127.3845],
        "울산": [35.5384, 129.3114], "세종": [36.4800, 127.2890],
        "경기": [37.4138, 127.5183], "강원": [37.8228, 128.1555],
        "충북": [36.8000, 127.7000], "충남": [36.5184, 126.8000],
        "전북": [35.7175, 127.1530], "전남": [34.8679, 126.9910],
        "경북": [36.4919, 128.8889], "경남": [35.4606, 128.2132],
        "제주": [33.4996, 126.5312]
    }
    
    data = []
    for region in regions:
        # 기본 수확량 (지역별 차이)
        base_yield = np.random.normal(85, 10)
        
        # 연도별 데이터 (2010-2023)
        for year in range(2010, 2024):
            # 연도별 변화 (기후변화 영향)
            year_effect = -(year - 2010) * 0.8 + np.random.normal(0, 3)
            
            # 최종 수확량
            yield_value = max(base_yield + year_effect, 30)
            
            # 기온 (지역별 차이 + 연도별 상승)
            base_temp = 13.5 + np.random.normal(0, 1.5)  # 지역별 기본 기온
            temp_value = base_temp + (year - 2010) * 0.1 + np.random.normal(0, 0.3)
            
            # 강수량
            rain_value = np.random.normal(1200, 200) + 30 * np.sin((year - 2010) * 0.5)
            
            data.append({
                'region': region,
                'year': year,
                'yield': yield_value,
                'temperature': temp_value,
                'rainfall': max(rain_value, 600),
                'lat': coordinates[region][0],
                'lon': coordinates[region][1]
            })
    
    return pd.DataFrame(data)

def create_korea_map(df, selected_year, metric):
    """한국 지도 생성"""
    # 선택된 연도 데이터 필터링
    year_data = df[df['year'] == selected_year].copy()
    
    # 메트릭별 색상 설정
    if metric == '수확량':
        year_data['value'] = year_data['yield']
        colormap = 'RdYlGn'
        value_name = '수확량 지수'
    elif metric == '기온':
        year_data['value'] = year_data['temperature'] 
        colormap = 'Reds'
        value_name = '평균기온 (°C)'
    else:  # 강수량
        year_data['value'] = year_data['rainfall']
        colormap = 'Blues'
        value_name = '연강수량 (mm)'
    
    # 한국 중심 지도 생성
    m = folium.Map(
        location=[36.5, 127.5],
        zoom_start=7,
        tiles='CartoDB positron'
    )
    
    # 마커 추가
    for _, row in year_data.iterrows():
        # 값에 따른 색상 결정
        normalized_value = (row['value'] - year_data['value'].min()) / (year_data['value'].max() - year_data['value'].min())
        
        if metric == '수확량':
            if normalized_value > 0.7:
                color = '#22c55e'
                icon_color = 'green'
            elif normalized_value > 0.3:
                color = '#eab308'
                icon_color = 'orange'
            else:
                color = '#dc2626'
                icon_color = 'red'
        else:
            if normalized_value > 0.7:
                color = '#dc2626'
                icon_color = 'red'
            elif normalized_value > 0.3:
                color = '#eab308'
                icon_color = 'orange'
            else:
                color = '#22c55e'
                icon_color = 'green'
        
        # 마커 추가
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"""
            <div style="width: 200px;">
                <h4>{row['region']}</h4>
                <p><strong>{value_name}:</strong> {row['value']:.1f}</p>
                <p><strong>연도:</strong> {selected_year}</p>
            </div>
            """,
            tooltip=f"{row['region']}: {row['value']:.1f}",
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(m)
        
        # 원형 마커로 크기 표현
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5 + normalized_value * 15,
            popup=f"{row['region']}: {row['value']:.1f}",
            color=color,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
    
    return m

def create_heatmap(df, metric):
    """지역별 히트맵 생성"""
    # 피벗 테이블 생성
    if metric == '수확량':
        pivot_data = df.pivot(index='region', columns='year', values='yield')
        title = '지역별 수확량 변화 히트맵'
        colorscale = 'RdYlGn'
    elif metric == '기온':
        pivot_data = df.pivot(index='region', columns='year', values='temperature')
        title = '지역별 평균기온 변화 히트맵'
        colorscale = 'Reds'
    else:
        pivot_data = df.pivot(index='region', columns='year', values='rainfall')
        title = '지역별 강수량 변화 히트맵'
        colorscale = 'Blues'
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        hovertemplate='<b>%{y}</b><br>연도: %{x}<br>값: %{z:.1f}<extra></extra>',
        colorbar=dict(title=metric)
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        xaxis_title='연도',
        yaxis_title='지역',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

def create_scatter_analysis(df, selected_year):
    """산점도 분석"""
    year_data = df[df['year'] == selected_year]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['기온 vs 수확량', '강수량 vs 수확량'],
        horizontal_spacing=0.1
    )
    
    # 기온 vs 수확량
    fig.add_trace(
        go.Scatter(
            x=year_data['temperature'],
            y=year_data['yield'],
            mode='markers+text',
            text=year_data['region'],
            textposition='top center',
            marker=dict(
                size=12,
                color=year_data['yield'],
                colorscale='RdYlGn',
                line=dict(width=2, color='white'),
                showscale=False
            ),
            name='기온-수확량',
            hovertemplate='<b>%{text}</b><br>기온: %{x:.1f}°C<br>수확량: %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 강수량 vs 수확량
    fig.add_trace(
        go.Scatter(
            x=year_data['rainfall'],
            y=year_data['yield'],
            mode='markers+text',
            text=year_data['region'],
            textposition='top center',
            marker=dict(
                size=12,
                color=year_data['yield'],
                colorscale='RdYlGn',
                line=dict(width=2, color='white')
            ),
            name='강수량-수확량',
            hovertemplate='<b>%{text}</b><br>강수량: %{x:.0f}mm<br>수확량: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 추세선 추가
    temp_corr = np.corrcoef(year_data['temperature'], year_data['yield'])[0, 1]
    rain_corr = np.corrcoef(year_data['rainfall'], year_data['yield'])[0, 1]
    
    # 기온 추세선
    z1 = np.polyfit(year_data['temperature'], year_data['yield'], 1)
    p1 = np.poly1d(z1)
    fig.add_trace(
        go.Scatter(
            x=year_data['temperature'],
            y=p1(year_data['temperature']),
            mode='lines',
            name=f'추세선 (r={temp_corr:.2f})',
            line=dict(color='#dc2626', width=3, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 강수량 추세선
    z2 = np.polyfit(year_data['rainfall'], year_data['yield'], 1)
    p2 = np.poly1d(z2)
    fig.add_trace(
        go.Scatter(
            x=year_data['rainfall'],
            y=p2(year_data['rainfall']),
            mode='lines',
            name=f'추세선 (r={rain_corr:.2f})',
            line=dict(color='#2563eb', width=3, dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="평균기온 (°C)", row=1, col=1)
    fig.update_xaxes(title_text="연강수량 (mm)", row=1, col=2)
    fig.update_yaxes(title_text="수확량 지수", row=1, col=1)
    fig.update_yaxes(title_text="수확량 지수", row=1, col=2)
    
    fig.update_layout(
        title={
            'text': f'{selected_year}년 기후요소와 수확량 상관관계',
            'font': {'family': 'Inter', 'size': 16, 'color': '#1a1a1a'},
            'x': 0.5
        },
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=False
    )
    
    return fig, temp_corr, rain_corr

def create_ranking_analysis(df, selected_year):
    """지역별 순위 분석"""
    year_data = df[df['year'] == selected_year].copy()
    
    # 수확량 기준 정렬
    year_data = year_data.sort_values('yield', ascending=False)
    
    # 위험도 분류
    def get_risk_level(yield_val, temp_val):
        yield_percentile = yield_val / year_data['yield'].max()
        temp_percentile = (temp_val - year_data['temperature'].min()) / (year_data['temperature'].max() - year_data['temperature'].min())
        
        risk_score = (1 - yield_percentile) + temp_percentile
        
        if risk_score > 1.2:
            return "고위험", "risk-high"
        elif risk_score > 0.8:
            return "중위험", "risk-medium"
        else:
            return "저위험", "risk-low"
    
    return year_data, get_risk_level

def main():
    # 헤더
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🗺️ 지역별 영향 분석</h1>
        <p class="page-subtitle">전국 시도별 기후변화 영향을 인터랙티브 지도와 히트맵으로 분석합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 샘플 데이터 생성
    df = generate_regional_data()
    
    # 사이드바 필터
    with st.sidebar:
        st.markdown("### 🎛️ 분석 옵션")
        
        # 분석 연도 선택
        selected_year = st.selectbox(
            "분석 연도",
            sorted(df['year'].unique(), reverse=True),
            index=0
        )
        
        # 지도 표시 메트릭
        map_metric = st.selectbox(
            "지도 표시 항목",
            ['수확량', '기온', '강수량'],
            index=0
        )
        
        # 히트맵 메트릭
        heatmap_metric = st.selectbox(
            "히트맵 분석 항목", 
            ['수확량', '기온', '강수량'],
            index=0
        )
        
        # 지역 선택
        selected_regions = st.multiselect(
            "특정 지역 분석",
            sorted(df['region'].unique()),
            default=[]
        )
    
    # 필터 적용
    if selected_regions:
        filtered_df = df[df['region'].isin(selected_regions)]
    else:
        filtered_df = df
    
    # 상단 메트릭
    year_data = df[df['year'] == selected_year]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_yield = year_data['yield'].mean()
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{avg_yield:.1f}</h4>
            <p>평균 수확량</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_region = year_data.loc[year_data['yield'].idxmax(), 'region']
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{max_region}</h4>
            <p>최고 수확량 지역</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        min_region = year_data.loc[year_data['yield'].idxmin(), 'region']
        st.markdown(f"""
        <div class="metric-mini" style="background: rgba(220, 38, 38, 0.1); border-color: rgba(220, 38, 38, 0.2);">
            <h4 style="color: #dc2626;">{min_region}</h4>
            <p style="color: #dc2626;">최저 수확량 지역</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        temp_yield_corr = np.corrcoef(year_data['temperature'], year_data['yield'])[0, 1]
        st.markdown(f"""
        <div class="metric-mini">
            <h4>{temp_yield_corr:.2f}</h4>
            <p>기온-수확량 상관계수</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 컨텐츠 - 지도와 히트맵
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        <div class="map-card">
            <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">
                🗺️ {selected_year}년 {map_metric} 지역별 분포
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 인터랙티브 지도
        korea_map = create_korea_map(df, selected_year, map_metric)
        map_data = st_folium(korea_map, width=700, height=500)
    
    with col2:
        # 지역별 순위
        year_data_sorted, get_risk_level = create_ranking_analysis(df, selected_year)
        
        st.markdown("""
        <div class="ranking-card">
            <h3 style="font-family: 'Inter'; color: #1a1a1a; margin-bottom: 1rem;">
                📊 지역별 수확량 순위
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 순위 목록
        for idx, (_, row) in enumerate(year_data_sorted.head(10).iterrows()):
            risk_level, risk_class = get_risk_level(row['yield'], row['temperature'])
            
            st.markdown(f"""
            <div class="ranking-item">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-weight: 600; color: #666; width: 20px;">#{idx+1}</span>
                    <span style="font-weight: 600; color: #1a1a1a;">{row['region']}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #666; font-size: 0.9rem;">{row['yield']:.1f}</span>
                    <span class="rank-badge {risk_class}">{risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 히트맵 분석
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    heatmap_chart = create_heatmap(filtered_df, heatmap_metric)
    st.plotly_chart(heatmap_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 상관관계 분석
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    scatter_chart, temp_corr, rain_corr = create_scatter_analysis(df, selected_year)
    st.plotly_chart(scatter_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 하단 인사이트
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📈 {selected_year}년 상관관계 분석**
        - **기온-수확량 상관계수**: {temp_corr:.3f}
        - **강수량-수확량 상관계수**: {rain_corr:.3f}
        
        **🎯 주요 발견**
        - 최고 수확량: **{year_data['yield'].max():.1f}** ({year_data.loc[year_data['yield'].idxmax(), 'region']})
        - 최저 수확량: **{year_data['yield'].min():.1f}** ({year_data.loc[year_data['yield'].idxmin(), 'region']})
        - 수확량 편차: **{year_data['yield'].std():.1f}**
        """)
    
    with col2:
        # 위험 지역 분석
        high_risk_regions = []
        for _, row in year_data.iterrows():
            risk_level, _ = get_risk_level(row['yield'], row['temperature'])
            if risk_level == "고위험":
                high_risk_regions.append(row['region'])
        
        st.markdown(f"""
        **⚠️ 위험 지역 분석**
        - **고위험 지역**: {len(high_risk_regions)}개
        - **위험 지역 목록**: {', '.join(high_risk_regions) if high_risk_regions else '없음'}
        
        **💡 권장사항**
        - 고위험 지역 집중 모니터링
        - 기후 적응 기술 우선 지원
        - 대체 작물 재배 검토
        """)

if __name__ == "__main__":
    main()