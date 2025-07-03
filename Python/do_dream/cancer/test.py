import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# 파스텔 컬러 팔레트 설정
pastel_colors = {
    'cancer': '#FFB6C1',      # 연한 핑크 (악성)
    'benign': '#B6E5D8',      # 연한 민트 (양성)
    'male': '#A8D8EA',        # 연한 파랑 (남성)
    'female': '#FFB3BA',      # 연한 분홍 (여성)
    'high': '#FFDFBA',        # 연한 주황 (높음)
    'medium': '#FFFFBA',      # 연한 노랑 (중간)
    'low': '#BAE1FF',         # 연한 파랑 (낮음)
    'background': '#F8F9FA'   # 연한 회색 (배경)
}

# 한글 폰트 설정 (Mac/Windows 호환)
import platform
import matplotlib.font_manager as fm

def setup_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS에서 사용 가능한 한글 폰트들
        korean_fonts = [
            'AppleGothic',
            'Apple SD Gothic Neo', 
            'AppleMyungjo',
            'Nanum Gothic',
            'NanumBarunGothic',
            'NanumSquare'
        ]
        
        # 사용 가능한 폰트 찾기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                print(f"한글 폰트 설정: {font}")
                break
        else:
            # 기본 폰트가 없으면 시스템 폰트 사용
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
            print("한글 표시 개선을 위해 다음 명령어로 폰트를 설치하세요:")
            print("brew install font-nanum font-nanum-coding font-nanum-gothic-coding")
            
    elif system == 'Windows':  # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
        print("한글 폰트 설정: Malgun Gothic")
        
    else:  # Linux
        korean_fonts = [
            'Nanum Gothic',
            'NanumBarunGothic', 
            'NanumSquare',
            'DejaVu Sans'
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                print(f"한글 폰트 설정: {font}")
                break
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("한글 폰트를 찾지 못했습니다.")
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
setup_korean_font()

def load_and_preprocess_data(file_path):
    """데이터 로드 및 기본 전처리"""
    df = pd.read_csv(file_path)
    
    # 위험 점수 계산
    risk_factors = []
    if 'Family_Background' in df.columns:
        risk_factors.append(df['Family_Background'] == 'Positive')
    if 'Radiation_History' in df.columns:
        risk_factors.append(df['Radiation_History'] == 'Exposed')
    if 'Smoke' in df.columns:
        risk_factors.append(df['Smoke'] == 'Smoker')
    if 'Weight_Risk' in df.columns:
        risk_factors.append(df['Weight_Risk'] == 'Obese')
    if 'Diabetes' in df.columns:
        risk_factors.append(df['Diabetes'] == 'Yes')
    
    if risk_factors:
        df['Risk_Score'] = sum(risk_factors)
    
    # 연령 그룹 생성
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], 
                               bins=[0, 30, 45, 60, 100], 
                               labels=['20-30대', '30-45세', '45-60세', '60세 이상'])
    
    # 결절 크기 그룹 생성
    if 'Nodule_Size' in df.columns:
        df['Nodule_Size_Group'] = pd.cut(df['Nodule_Size'], 
                                       bins=[0, 1, 2, 3, float('inf')], 
                                       labels=['1cm 미만', '1-2cm', '2-3cm', '3cm 이상'])
    
    return df

def plot_categorical_cancer_rates(df):
    """범주형 변수별 갑상선암 발생률 시각화"""
    
    categorical_vars = {
        'Gender': '성별',
        'Family_Background': '가족력',
        'Radiation_History': '방사선 노출',
        'Iodine_Deficiency': '요오드 결핍',
        'Smoke': '흡연',
        'Weight_Risk': '비만 위험도',
        'Diabetes': '당뇨병',
        'Country': '국가',
        'Race': '인종'
    }
    
    # 존재하는 컬럼만 필터링
    existing_vars = {k: v for k, v in categorical_vars.items() if k in df.columns}
    
    # 서브플롯 설정
    n_vars = len(existing_vars)
    cols = 3
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (col, korean_name) in enumerate(existing_vars.items()):
        # 암 발생률 계산
        cancer_rates = df.groupby(col)['Cancer'].agg(['count', 'sum', 'mean']).reset_index()
        cancer_rates['cancer_rate'] = cancer_rates['mean'] * 100
        cancer_rates['benign_rate'] = (1 - cancer_rates['mean']) * 100
        
        # 스택 바 차트
        x_pos = range(len(cancer_rates))
        
        axes[idx].bar(x_pos, cancer_rates['benign_rate'], 
                     color=pastel_colors['benign'], label='양성', alpha=0.8)
        axes[idx].bar(x_pos, cancer_rates['cancer_rate'], 
                     bottom=cancer_rates['benign_rate'],
                     color=pastel_colors['cancer'], label='악성', alpha=0.8)
        
        # 퍼센티지 텍스트 추가
        for i, (idx_val, row) in enumerate(cancer_rates.iterrows()):
            if row['cancer_rate'] > 5:  # 5% 이상일 때만 표시
                axes[idx].text(i, row['benign_rate'] + row['cancer_rate']/2, 
                             f"{row['cancer_rate']:.1f}%", 
                             ha='center', va='center', fontweight='bold')
            
            # 총 건수 표시
            axes[idx].text(i, -8, f"n={int(row['count'])}", 
                         ha='center', va='center', fontsize=10)
        
        axes[idx].set_title(f'{korean_name}별 갑상선암 발생률', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('비율 (%)')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(cancer_rates[col], rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim(-15, 105)
    
    # 빈 서브플롯 제거
    for idx in range(len(existing_vars), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def plot_age_cancer_analysis(df):
    """연령별 갑상선암 발생 분석"""
    
    if 'Age' not in df.columns:
        print("Age 컬럼이 없어 연령 분석을 건너뜁니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 연령 그룹별 암 발생률
    if 'Age_Group' in df.columns:
        age_cancer = df.groupby('Age_Group')['Cancer'].agg(['count', 'sum', 'mean']).reset_index()
        age_cancer['cancer_rate'] = age_cancer['mean'] * 100
        
        bars = axes[0,0].bar(range(len(age_cancer)), age_cancer['cancer_rate'], 
                           color=pastel_colors['cancer'], alpha=0.8)
        
        # 값 표시
        for i, (idx, row) in enumerate(age_cancer.iterrows()):
            axes[0,0].text(i, row['cancer_rate'] + 1, 
                         f"{row['cancer_rate']:.1f}%\n(n={int(row['count'])})", 
                         ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].set_title('연령 그룹별 갑상선암 발생률', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('암 발생률 (%)')
        axes[0,0].set_xticks(range(len(age_cancer)))
        axes[0,0].set_xticklabels(age_cancer['Age_Group'])
        axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. 연령 분포 히스토그램 (암 여부별)
    for cancer_type in [0, 1]:
        subset = df[df['Cancer'] == cancer_type]['Age']
        color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
        label = '양성' if cancer_type == 0 else '악성'
        axes[0,1].hist(subset, alpha=0.7, color=color, label=label, bins=20, edgecolor='white')
    
    axes[0,1].set_title('연령 분포 (진단별)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('나이')
    axes[0,1].set_ylabel('빈도')
    axes[0,1].legend()
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. 연령대별 박스플롯
    cancer_labels = ['양성', '악성']
    age_by_cancer = [df[df['Cancer'] == i]['Age'] for i in [0, 1]]
    colors = [pastel_colors['benign'], pastel_colors['cancer']]
    
    box_plot = axes[1,0].boxplot(age_by_cancer, labels=cancer_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    axes[1,0].set_title('진단별 연령 분포 (박스플롯)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('나이')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. 연령-성별 교차 분석
    if 'Gender' in df.columns:
        pivot_data = df.pivot_table(values='Cancer', index='Age_Group', columns='Gender', aggfunc='mean') * 100
        
        x_pos = np.arange(len(pivot_data.index))
        width = 0.35
        
        if 'M' in pivot_data.columns:
            axes[1,1].bar(x_pos - width/2, pivot_data['M'], width, 
                         label='남성', color=pastel_colors['male'], alpha=0.8)
        if 'F' in pivot_data.columns:
            axes[1,1].bar(x_pos + width/2, pivot_data['F'], width, 
                         label='여성', color=pastel_colors['female'], alpha=0.8)
        
        axes[1,1].set_title('연령-성별 갑상선암 발생률', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('암 발생률 (%)')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(pivot_data.index)
        axes[1,1].legend()
        axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_nodule_size_analysis(df):
    """결절 크기별 갑상선암 발생 분석"""
    
    if 'Nodule_Size' not in df.columns:
        print("Nodule_Size 컬럼이 없어 결절 크기 분석을 건너뜁니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 결절 크기 그룹별 암 발생률
    if 'Nodule_Size_Group' in df.columns:
        nodule_cancer = df.groupby('Nodule_Size_Group')['Cancer'].agg(['count', 'sum', 'mean']).reset_index()
        nodule_cancer['cancer_rate'] = nodule_cancer['mean'] * 100
        
        bars = axes[0,0].bar(range(len(nodule_cancer)), nodule_cancer['cancer_rate'], 
                           color=pastel_colors['cancer'], alpha=0.8)
        
        for i, (idx, row) in enumerate(nodule_cancer.iterrows()):
            axes[0,0].text(i, row['cancer_rate'] + 1, 
                         f"{row['cancer_rate']:.1f}%\n(n={int(row['count'])})", 
                         ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].set_title('결절 크기별 갑상선암 발생률', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('암 발생률 (%)')
        axes[0,0].set_xticks(range(len(nodule_cancer)))
        axes[0,0].set_xticklabels(nodule_cancer['Nodule_Size_Group'])
        axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. 결절 크기 분포 (암 여부별)
    for cancer_type in [0, 1]:
        subset = df[df['Cancer'] == cancer_type]['Nodule_Size']
        color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
        label = '양성' if cancer_type == 0 else '악성'
        axes[0,1].hist(subset, alpha=0.7, color=color, label=label, bins=20, edgecolor='white')
    
    axes[0,1].set_title('결절 크기 분포 (진단별)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('결절 크기 (cm)')
    axes[0,1].set_ylabel('빈도')
    axes[0,1].legend()
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. 결절 크기 박스플롯
    cancer_labels = ['양성', '악성']
    size_by_cancer = [df[df['Cancer'] == i]['Nodule_Size'] for i in [0, 1]]
    colors = [pastel_colors['benign'], pastel_colors['cancer']]
    
    box_plot = axes[1,0].boxplot(size_by_cancer, labels=cancer_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    axes[1,0].set_title('진단별 결절 크기 분포', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('결절 크기 (cm)')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. 결절 크기 vs 연령 산점도
    if 'Age' in df.columns:
        for cancer_type in [0, 1]:
            subset = df[df['Cancer'] == cancer_type]
            color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
            label = '양성' if cancer_type == 0 else '악성'
            axes[1,1].scatter(subset['Age'], subset['Nodule_Size'], 
                            color=color, label=label, alpha=0.7, s=50)
        
        axes[1,1].set_title('연령 vs 결절 크기', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('나이')
        axes[1,1].set_ylabel('결절 크기 (cm)')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_hormone_analysis(df):
    """호르몬 수치별 갑상선암 발생 분석"""
    
    hormone_cols = ['TSH_Result', 'T4_Result', 'T3_Result']
    existing_hormones = [col for col in hormone_cols if col in df.columns]
    
    if not existing_hormones:
        print("호르몬 수치 컬럼이 없어 호르몬 분석을 건너뜁니다.")
        return
    
    n_hormones = len(existing_hormones)
    fig, axes = plt.subplots(2, n_hormones, figsize=(6*n_hormones, 12))
    
    if n_hormones == 1:
        axes = axes.reshape(2, 1)
    
    korean_names = {'TSH_Result': 'TSH', 'T4_Result': 'T4', 'T3_Result': 'T3'}
    
    for idx, hormone in enumerate(existing_hormones):
        # 1. 호르몬 수치 분포 (암 여부별)
        for cancer_type in [0, 1]:
            subset = df[df['Cancer'] == cancer_type][hormone]
            color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
            label = '양성' if cancer_type == 0 else '악성'
            axes[0, idx].hist(subset, alpha=0.7, color=color, label=label, bins=20, edgecolor='white')
        
        axes[0, idx].set_title(f'{korean_names[hormone]} 수치 분포', fontsize=14, fontweight='bold')
        axes[0, idx].set_xlabel(f'{korean_names[hormone]} 수치')
        axes[0, idx].set_ylabel('빈도')
        axes[0, idx].legend()
        axes[0, idx].grid(axis='y', alpha=0.3)
        
        # 2. 호르몬 수치 박스플롯
        cancer_labels = ['양성', '악성']
        hormone_by_cancer = [df[df['Cancer'] == i][hormone] for i in [0, 1]]
        colors = [pastel_colors['benign'], pastel_colors['cancer']]
        
        box_plot = axes[1, idx].boxplot(hormone_by_cancer, labels=cancer_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        axes[1, idx].set_title(f'{korean_names[hormone]} 수치 비교', fontsize=14, fontweight='bold')
        axes[1, idx].set_ylabel(f'{korean_names[hormone]} 수치')
        axes[1, idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_risk_factor_analysis(df):
    """위험 요인 조합 분석"""
    
    if 'Risk_Score' not in df.columns:
        print("Risk_Score가 없어 위험 요인 분석을 건너뜁니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 위험 점수별 암 발생률
    risk_cancer = df.groupby('Risk_Score')['Cancer'].agg(['count', 'sum', 'mean']).reset_index()
    risk_cancer['cancer_rate'] = risk_cancer['mean'] * 100
    
    bars = axes[0,0].bar(risk_cancer['Risk_Score'], risk_cancer['cancer_rate'], 
                       color=pastel_colors['cancer'], alpha=0.8)
    
    for i, row in risk_cancer.iterrows():
        axes[0,0].text(row['Risk_Score'], row['cancer_rate'] + 1, 
                     f"{row['cancer_rate']:.1f}%\n(n={int(row['count'])})", 
                     ha='center', va='bottom', fontweight='bold')
    
    axes[0,0].set_title('위험 요인 개수별 갑상선암 발생률', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('위험 요인 개수')
    axes[0,0].set_ylabel('암 발생률 (%)')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. 개별 위험 요인별 분석
    risk_factors = ['Family_Background', 'Radiation_History', 'Smoke', 'Weight_Risk', 'Diabetes']
    existing_risks = [col for col in risk_factors if col in df.columns]
    
    risk_rates = []
    risk_names = []
    
    for risk in existing_risks:
        positive_cases = df[df[risk].isin(['Positive', 'Exposed', 'Smoker', 'Obese', 'Yes'])]
        if len(positive_cases) > 0:
            cancer_rate = positive_cases['Cancer'].mean() * 100
            risk_rates.append(cancer_rate)
            risk_names.append(risk.replace('_', ' '))
    
    if risk_rates:
        bars = axes[0,1].barh(risk_names, risk_rates, color=pastel_colors['high'], alpha=0.8)
        
        for i, (name, rate) in enumerate(zip(risk_names, risk_rates)):
            axes[0,1].text(rate + 1, i, f"{rate:.1f}%", va='center', fontweight='bold')
        
        axes[0,1].set_title('개별 위험 요인별 암 발생률', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('암 발생률 (%)')
        axes[0,1].grid(axis='x', alpha=0.3)
    
    # 3. 위험 점수 분포
    for cancer_type in [0, 1]:
        subset = df[df['Cancer'] == cancer_type]['Risk_Score']
        color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
        label = '양성' if cancer_type == 0 else '악성'
        axes[1,0].hist(subset, alpha=0.7, color=color, label=label, bins=range(6), edgecolor='white')
    
    axes[1,0].set_title('위험 점수 분포 (진단별)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('위험 점수')
    axes[1,0].set_ylabel('빈도')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. 위험 점수 vs 결절 크기
    if 'Nodule_Size' in df.columns:
        for cancer_type in [0, 1]:
            subset = df[df['Cancer'] == cancer_type]
            color = pastel_colors['benign'] if cancer_type == 0 else pastel_colors['cancer']
            label = '양성' if cancer_type == 0 else '악성'
            axes[1,1].scatter(subset['Risk_Score'], subset['Nodule_Size'], 
                            color=color, label=label, alpha=0.7, s=50)
        
        axes[1,1].set_title('위험 점수 vs 결절 크기', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('위험 점수')
        axes[1,1].set_ylabel('결절 크기 (cm)')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """상관관계 히트맵"""
    
    # 수치형 변수만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("수치형 변수가 부족해 상관관계 분석을 건너뜁니다.")
        return
    
    # 상관관계 계산
    correlation_matrix = df[numeric_cols].corr()
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    
    plt.title('변수 간 상관관계 히트맵', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def create_summary_dashboard(df):
    """종합 대시보드"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 전체 암 발생률
    cancer_dist = df['Cancer'].value_counts()
    cancer_rate = cancer_dist[1] / len(df) * 100
    
    axes[0,0].pie([cancer_dist[0], cancer_dist[1]], 
                 labels=['양성', '악성'], 
                 colors=[pastel_colors['benign'], pastel_colors['cancer']],
                 autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title(f'전체 갑상선암 발생률\n(총 {len(df)}명)', fontsize=14, fontweight='bold')
    
    # 2. 성별 암 발생률
    if 'Gender' in df.columns:
        gender_cancer = pd.crosstab(df['Gender'], df['Cancer'], normalize='index') * 100
        gender_cancer.plot(kind='bar', ax=axes[0,1], 
                          color=[pastel_colors['benign'], pastel_colors['cancer']], rot=0)
        axes[0,1].set_title('성별 암 발생률', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('비율 (%)')
        axes[0,1].legend(['양성', '악성'])
    
    # 3. 연령대별 암 발생률
    if 'Age_Group' in df.columns:
        age_cancer = pd.crosstab(df['Age_Group'], df['Cancer'], normalize='index') * 100
        age_cancer.plot(kind='bar', ax=axes[0,2], 
                       color=[pastel_colors['benign'], pastel_colors['cancer']], rot=45)
        axes[0,2].set_title('연령대별 암 발생률', fontsize=14, fontweight='bold')
        axes[0,2].set_ylabel('비율 (%)')
        axes[0,2].legend(['양성', '악성'])
    
    # 4. 위험 요인별 분포
    if 'Risk_Score' in df.columns:
        risk_dist = df['Risk_Score'].value_counts().sort_index()
        axes[1,0].bar(risk_dist.index, risk_dist.values, color=pastel_colors['medium'], alpha=0.8)
        axes[1,0].set_title('위험 요인 점수 분포', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('위험 요인 개수')
        axes[1,0].set_ylabel('환자 수')
    
    # 5. 결절 크기 분포
    if 'Nodule_Size' in df.columns:
        axes[1,1].hist(df['Nodule_Size'], bins=20, color=pastel_colors['background'], 
                      edgecolor='black', alpha=0.8)
        axes[1,1].set_title('결절 크기 분포', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('결절 크기 (cm)')
        axes[1,1].set_ylabel('빈도')
    
    # 6. 주요 통계
    axes[1,2].axis('off')
    stats_text = f"""
    📊 주요 통계
    
    총 환자 수: {len(df):,}명
    악성 케이스: {cancer_dist.get(1, 0):,}명 ({cancer_rate:.1f}%)
    양성 케이스: {cancer_dist.get(0, 0):,}명 ({100-cancer_rate:.1f}%)
    
    평균 나이: {df['Age'].mean():.1f}세
    평균 결절 크기: {df['Nodule_Size'].mean():.2f}cm
    
    위험 요인 분포:
    """
    
    if 'Risk_Score' in df.columns:
        for score in sorted(df['Risk_Score'].unique()):
            count = (df['Risk_Score'] == score).sum()
            pct = count / len(df) * 100
            stats_text += f"    {score}개: {count}명 ({pct:.1f}%)\n"
    
    axes[1,2].text(0.1, 0.9, stats_text, transform=axes[1,2].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=pastel_colors['background']))
    
    plt.tight_layout()
    plt.show()

def main_analysis(file_path):
    """전체 분석 실행"""
    
    print("🔍 갑상선암 발생 비율 분석을 시작합니다...")
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data(file_path)
    
    print(f"📊 데이터 정보:")
    print(f"   - 총 환자 수: {len(df):,}명")
    print(f"   - 악성 케이스: {df['Cancer'].sum():,}명 ({df['Cancer'].mean()*100:.1f}%)")
    print(f"   - 양성 케이스: {(df['Cancer']==0).sum():,}명 ({(1-df['Cancer'].mean())*100:.1f}%)")
    
    # 2. 종합 대시보드
    print("\n📈 종합 대시보드 생성 중...")
    create_summary_dashboard(df)
    
    # 3. 범주형 변수별 분석
    print("\n📊 범주형 변수별 암 발생률 분석...")
    plot_categorical_cancer_rates(df)
    
    # 4. 연령별 분석
    print("\n👥 연령별 분석...")
    plot_age_cancer_analysis(df)
    
    # 5. 결절 크기별 분석
    print("\n🔬 결절 크기별 분석...")
    plot_nodule_size_analysis(df)
    
    # 6. 호르몬 수치별 분석
    print("\n🧪 호르몬 수치별 분석...")
    plot_hormone_analysis(df)
    
    # 7. 위험 요인 분석
    print("\n⚠️ 위험 요인 분석...")
    plot_risk_factor_analysis(df)
    
    # 8. 상관관계 분석
    print("\n🔗 변수 간 상관관계 분석...")
    plot_correlation_heatmap(df)
    
    print("\n✅ 모든 분석이 완료되었습니다!")
    
    # 핵심 인사이트 출력
    print_key_insights(df)
    
    return df

def print_key_insights(df):
    """핵심 인사이트 출력"""
    
    print("\n" + "="*60)
    print("🎯 주요 발견사항 및 인사이트")
    print("="*60)
    
    # 1. 전체 발생률
    cancer_rate = df['Cancer'].mean() * 100
    print(f"📊 전체 갑상선암 발생률: {cancer_rate:.1f}%")
    
    # 2. 성별 차이
    if 'Gender' in df.columns:
        gender_rates = df.groupby('Gender')['Cancer'].mean() * 100
        if 'M' in gender_rates.index and 'F' in gender_rates.index:
            print(f"👨‍👩‍👧‍👦 성별 발생률: 남성 {gender_rates['M']:.1f}%, 여성 {gender_rates['F']:.1f}%")
            if gender_rates['M'] > gender_rates['F']:
                print(f"   → 남성이 여성보다 {gender_rates['M']/gender_rates['F']:.1f}배 높음")
    
    # 3. 방사선 노출 영향
    if 'Radiation_History' in df.columns:
        radiation_rates = df.groupby('Radiation_History')['Cancer'].mean() * 100
        if 'Exposed' in radiation_rates.index and 'Unexposed' in radiation_rates.index:
            print(f"☢️ 방사선 노출: 노출군 {radiation_rates['Exposed']:.1f}%, 비노출군 {radiation_rates['Unexposed']:.1f}%")
    
    # 4. 가족력 영향
    if 'Family_Background' in df.columns:
        family_rates = df.groupby('Family_Background')['Cancer'].mean() * 100
        if 'Positive' in family_rates.index and 'Negative' in family_rates.index:
            print(f"👨‍👩‍👧‍👦 가족력: 양성 {family_rates['Positive']:.1f}%, 음성 {family_rates['Negative']:.1f}%")
    
    # 5. 결절 크기 차이
    if 'Nodule_Size' in df.columns:
        malignant_size = df[df['Cancer'] == 1]['Nodule_Size'].mean()
        benign_size = df[df['Cancer'] == 0]['Nodule_Size'].mean()
        print(f"🔬 평균 결절 크기: 악성 {malignant_size:.2f}cm, 양성 {benign_size:.2f}cm")
        print(f"   → 악성이 양성보다 {malignant_size - benign_size:.2f}cm 더 큼")
    
    # 6. 위험 요인 개수별 발생률
    if 'Risk_Score' in df.columns:
        print(f"\n⚠️ 위험 요인 개수별 발생률:")
        risk_rates = df.groupby('Risk_Score')['Cancer'].mean() * 100
        for score in sorted(risk_rates.index):
            count = (df['Risk_Score'] == score).sum()
            print(f"   {score}개 요인: {risk_rates[score]:.1f}% (n={count})")
    
    # 7. 고위험군 식별
    high_risk_conditions = []
    
    if 'Radiation_History' in df.columns:
        exposed_rate = df[df['Radiation_History'] == 'Exposed']['Cancer'].mean() * 100
        if exposed_rate > cancer_rate * 1.5:
            high_risk_conditions.append(f"방사선 노출 ({exposed_rate:.1f}%)")
    
    if 'Gender' in df.columns and 'M' in df['Gender'].values:
        male_rate = df[df['Gender'] == 'M']['Cancer'].mean() * 100
        if male_rate > cancer_rate * 1.2:
            high_risk_conditions.append(f"남성 ({male_rate:.1f}%)")
    
    if 'Nodule_Size' in df.columns:
        large_nodule_rate = df[df['Nodule_Size'] > df['Nodule_Size'].quantile(0.75)]['Cancer'].mean() * 100
        if large_nodule_rate > cancer_rate * 1.2:
            large_size_threshold = df['Nodule_Size'].quantile(0.75)
            high_risk_conditions.append(f"대형 결절 (>{large_size_threshold:.1f}cm, {large_nodule_rate:.1f}%)")
    
    if high_risk_conditions:
        print(f"\n🚨 고위험군 (평균 대비 높은 발생률):")
        for condition in high_risk_conditions:
            print(f"   - {condition}")
    
    print("\n💡 권장사항:")
    print("   1. 방사선 노출 이력이 있는 환자는 정기적인 모니터링 필요")
    print("   2. 남성 환자에 대한 적극적인 스크리닝 고려")
    print("   3. 결절 크기가 큰 경우 조기 정밀 검사 권장")
    print("   4. 다중 위험 요인 보유 환자에 대한 개별화된 관리 필요")
    
    print("="*60)

# 실행 예시
if __name__ == "__main__":
    # 파일 경로 설정 (실제 파일 경로로 변경하세요)
    file_path = "/Users/ayaan/Git/Junior/Python/do_dream/cancer/train.csv"
    
    try:
        # 전체 분석 실행
        df = main_analysis(file_path)
        
        print(f"\n✅ 분석 완료! 총 {len(df):,}명의 데이터를 분석했습니다.")
        
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        print("파일 경로 예시: 'train.csv' 또는 '/path/to/train.csv'")
    
    except Exception as e:
        print(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")

# 개별 함수 사용 예시
"""
# 특정 분석만 실행하고 싶은 경우:

df = load_and_preprocess_data("train.csv")

# 1. 범주형 변수 분석만
plot_categorical_cancer_rates(df)

# 2. 연령 분석만
plot_age_cancer_analysis(df)

# 3. 결절 크기 분석만
plot_nodule_size_analysis(df)

# 4. 호르몬 분석만
plot_hormone_analysis(df)

# 5. 위험 요인 분석만
plot_risk_factor_analysis(df)

# 6. 상관관계 분석만
plot_correlation_heatmap(df)
"""