import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

print("🔍 애니메이션 감정 데이터셋 상세 분석")
print("=" * 50)

# 경로 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'
val_dir = 'ani/val'

def analyze_dataset_structure():
    """데이터셋 구조 분석"""
    
    print("📁 데이터셋 구조:")
    print(f"훈련 데이터: {train_dir}")
    print(f"검증 데이터: {val_dir}")
    
    # 클래스 확인
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    val_classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    
    print(f"\n📋 감정 클래스 ({len(train_classes)}개):")
    for i, class_name in enumerate(train_classes, 1):
        print(f"{i}. {class_name}")
    
    return train_classes, val_classes

def count_files_per_class():
    """클래스별 파일 개수 계산"""
    
    train_classes, val_classes = analyze_dataset_structure()
    
    # 훈련 데이터 카운트
    train_counts = {}
    for class_name in train_classes:
        class_path = os.path.join(train_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_counts[class_name] = len(files)
    
    # 검증 데이터 카운트
    val_counts = {}
    for class_name in val_classes:
        class_path = os.path.join(val_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        val_counts[class_name] = len(files)
    
    return train_counts, val_counts

def analyze_data_distribution():
    """데이터 분포 분석"""
    
    train_counts, val_counts = count_files_per_class()
    
    print("\n📊 클래스별 데이터 분포:")
    print("-" * 60)
    print(f"{'클래스명':<15} {'훈련':<8} {'검증':<8} {'총합':<8} {'비율':<8}")
    print("-" * 60)
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_all = total_train + total_val
    
    distribution_data = []
    
    for class_name in sorted(train_counts.keys()):
        train_cnt = train_counts.get(class_name, 0)
        val_cnt = val_counts.get(class_name, 0)
        total_cnt = train_cnt + val_cnt
        ratio = (total_cnt / total_all) * 100
        
        print(f"{class_name:<15} {train_cnt:<8} {val_cnt:<8} {total_cnt:<8} {ratio:<7.1f}%")
        
        distribution_data.append({
            'class': class_name,
            'train': train_cnt,
            'val': val_cnt,
            'total': total_cnt,
            'ratio': ratio
        })
    
    print("-" * 60)
    print(f"{'총합':<15} {total_train:<8} {total_val:<8} {total_all:<8} {'100.0%':<8}")
    
    return distribution_data

def analyze_class_imbalance(distribution_data):
    """클래스 불균형 분석"""
    
    print("\n⚖️ 클래스 불균형 분석:")
    
    totals = [d['total'] for d in distribution_data]
    max_samples = max(totals)
    min_samples = min(totals)
    mean_samples = np.mean(totals)
    std_samples = np.std(totals)
    
    # 불균형 비율
    imbalance_ratio = max_samples / min_samples
    
    print(f"최대 샘플 수: {max_samples}")
    print(f"최소 샘플 수: {min_samples}")
    print(f"평균 샘플 수: {mean_samples:.1f}")
    print(f"표준편차: {std_samples:.1f}")
    print(f"불균형 비율: {imbalance_ratio:.2f}:1")
    
    # 불균형 정도 평가
    if imbalance_ratio > 5:
        print("🚨 심각한 클래스 불균형 (5:1 이상)")
        print("   → 클래스 가중치 또는 오버샘플링 필요")
    elif imbalance_ratio > 3:
        print("⚠️ 중간 정도 클래스 불균형 (3:1~5:1)")
        print("   → 클래스 가중치 적용 권장")
    elif imbalance_ratio > 2:
        print("⚡ 경미한 클래스 불균형 (2:1~3:1)")
        print("   → 데이터 증강으로 해결 가능")
    else:
        print("✅ 균형잡힌 데이터셋")
    
    return imbalance_ratio

def analyze_data_sufficiency(distribution_data):
    """데이터 충분성 분석"""
    
    print("\n📈 데이터 충분성 분석:")
    
    total_samples = sum(d['total'] for d in distribution_data)
    num_classes = len(distribution_data)
    
    print(f"총 샘플 수: {total_samples}")
    print(f"클래스 수: {num_classes}")
    print(f"클래스당 평균: {total_samples/num_classes:.1f}")
    
    # 데이터 충분성 평가 (일반적인 기준)
    min_samples = min(d['total'] for d in distribution_data)
    
    if min_samples < 50:
        print("🚨 데이터 부족 (클래스당 50개 미만)")
        print("   → 심각한 과적합 위험, 강력한 정규화 필요")
    elif min_samples < 100:
        print("⚠️ 데이터 제한적 (클래스당 50~100개)")
        print("   → 전이학습 + 강력한 데이터 증강 필요")
    elif min_samples < 500:
        print("⚡ 소규모 데이터 (클래스당 100~500개)")
        print("   → 전이학습 + 적절한 정규화")
    else:
        print("✅ 충분한 데이터")

def identify_problem_causes(imbalance_ratio, distribution_data):
    """0.49 정확도 문제 원인 분석"""
    
    print("\n🔍 0.49 정확도 문제 원인 분석:")
    print("-" * 40)
    
    causes = []
    
    # 1. 클래스 불균형 체크
    if imbalance_ratio > 3:
        causes.append("심각한 클래스 불균형")
        print("❌ 클래스 불균형이 주요 원인")
    
    # 2. 데이터 부족 체크
    min_samples = min(d['total'] for d in distribution_data)
    if min_samples < 100:
        causes.append("데이터 부족")
        print("❌ 데이터 부족이 주요 원인")
    
    # 3. 7클래스 랜덤 확률 계산
    random_acc = 1/7
    current_acc = 0.49
    print(f"📊 7클래스 랜덤 정확도: {random_acc:.3f} (14.3%)")
    print(f"📊 현재 달성 정확도: {current_acc:.3f} (49.0%)")
    print(f"📊 개선 정도: {current_acc/random_acc:.1f}배")
    
    if current_acc < 0.6:
        causes.append("기본적인 패턴 학습 실패")
        print("❌ 모델이 기본 패턴을 제대로 학습하지 못함")
    
    return causes

def suggest_solutions(causes, distribution_data):
    """해결방안 제시"""
    
    print("\n💡 해결방안 제시:")
    print("-" * 30)
    
    if "심각한 클래스 불균형" in causes:
        print("🎯 클래스 불균형 해결:")
        print("   1. 클래스 가중치 적용")
        print("   2. 과소 클래스 오버샘플링")
        print("   3. 과다 클래스 언더샘플링")
        
        # 가중치 계산 예시
        totals = [d['total'] for d in distribution_data]
        max_samples = max(totals)
        print("\n   권장 클래스 가중치:")
        for data in distribution_data:
            weight = max_samples / data['total']
            print(f"   {data['class']}: {weight:.2f}")
    
    if "데이터 부족" in causes:
        print("\n📈 데이터 부족 해결:")
        print("   1. 강력한 데이터 증강")
        print("   2. 전이학습 적극 활용")
        print("   3. 추가 데이터 수집")
        print("   4. 합성 데이터 생성")
    
    if "기본적인 패턴 학습 실패" in causes:
        print("\n🔧 학습 개선:")
        print("   1. 학습률 조정 (1e-4 → 5e-5)")
        print("   2. 더 간단한 모델 시작")
        print("   3. 배치 크기 조정")
        print("   4. 정규화 강도 조정")

def visualize_distribution(distribution_data):
    """데이터 분포 시각화"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. 클래스별 총 데이터 수
    plt.subplot(2, 2, 1)
    classes = [d['class'] for d in distribution_data]
    totals = [d['total'] for d in distribution_data]
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(classes, totals, color=colors)
    plt.title('클래스별 총 데이터 수', fontsize=14, fontweight='bold')
    plt.xlabel('감정 클래스')
    plt.ylabel('샘플 수')
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, total in zip(bars, totals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(total), ha='center', va='bottom', fontweight='bold')
    
    # 2. 훈련/검증 분할
    plt.subplot(2, 2, 2)
    train_counts = [d['train'] for d in distribution_data]
    val_counts = [d['val'] for d in distribution_data]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='훈련', alpha=0.8)
    plt.bar(x + width/2, val_counts, width, label='검증', alpha=0.8)
    
    plt.title('훈련/검증 데이터 분할', fontsize=14, fontweight='bold')
    plt.xlabel('감정 클래스')
    plt.ylabel('샘플 수')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    
    # 3. 비율 파이차트
    plt.subplot(2, 2, 3)
    ratios = [d['ratio'] for d in distribution_data]
    plt.pie(ratios, labels=classes, autopct='%1.1f%%', colors=colors)
    plt.title('클래스별 데이터 비율', fontsize=14, fontweight='bold')
    
    # 4. 불균형 시각화
    plt.subplot(2, 2, 4)
    max_total = max(totals)
    normalized = [t/max_total for t in totals]
    
    bars = plt.bar(classes, normalized, color='lightcoral', alpha=0.7)
    plt.axhline(y=0.5, color='orange', linestyle='--', label='50% 기준선')
    plt.title('클래스 불균형 정도', fontsize=14, fontweight='bold')
    plt.xlabel('감정 클래스')
    plt.ylabel('최대 클래스 대비 비율')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('anime_emotion_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 🚀 메인 분석 실행
def main_analysis():
    print("🚀 데이터셋 분석 시작!")
    
    try:
        # 1. 데이터 분포 분석
        distribution_data = analyze_data_distribution()
        
        # 2. 클래스 불균형 분석
        imbalance_ratio = analyze_class_imbalance(distribution_data)
        
        # 3. 데이터 충분성 분석
        analyze_data_sufficiency(distribution_data)
        
        # 4. 문제 원인 분석
        causes = identify_problem_causes(imbalance_ratio, distribution_data)
        
        # 5. 해결방안 제시
        suggest_solutions(causes, distribution_data)
        
        # 6. 시각화
        visualize_distribution(distribution_data)
        
        print(f"\n🎯 결론:")
        print(f"주요 문제: {', '.join(causes)}")
        print(f"우선 해결책: 클래스 가중치 적용 + 강력한 데이터 증강")
        
        return distribution_data, imbalance_ratio, causes
        
    except Exception as e:
        print(f"❌ 분석 중 오류: {e}")
        return None, None, None

# 실행
if __name__ == "__main__":
    distribution_data, imbalance_ratio, causes = main_analysis()