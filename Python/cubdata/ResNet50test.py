import numpy as np
import pandas as pd
import os

# GPU 설정
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 설정 완료: RTX 3060 사용")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 경로 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'
val_dir = 'ani/val'

# 🎯 과소적합 상태에서 아주 조금만 완화
data_generator_with_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=15,         # 과소적합 때보다 살짝만 감소 (20→15)
    width_shift_range=0.15,    # 과소적합 때보다 살짝만 감소 (0.2→0.15)
    height_shift_range=0.15,   # 과소적합 때보다 살짝만 감소 (0.2→0.15)
    zoom_range=0.15,          # 과소적합 때보다 살짝만 감소 (0.2→0.15)
    shear_range=0.05,         # 과소적합 때보다 살짝만 감소 (0.1→0.05)
    brightness_range=[0.9, 1.1],  # 과소적합 때보다 살짝만 감소
    fill_mode='nearest'
)

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224
batch_size = 32

train_generator = data_generator_with_aug.flow_from_directory(
    directory=train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = data_generator_no_aug.flow_from_directory(
    directory=val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"클래스 수: {num_classes}")
print(f"훈련 샘플 수: {train_generator.samples}")
print(f"검증 샘플 수: {validation_generator.samples}")

# 🎯 과소적합에서 아주 조금만 완화된 모델
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 🔓 과소적합 때보다 아주 조금만 더 많은 레이어 훈련
    for layer in base_model.layers[:-15]:  # 과소적합 때 10개 → 15개 (조금만 증가)
        layer.trainable = False
    
    cnn = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        
        # 🎯 정규화 아주 조금만 완화
        Dropout(0.6),                    # 과소적합 때 0.7 → 0.6 (조금만 감소)
        Dense(384,                       # 과소적합 때 256 → 384 (조금만 증가)
              activation='relu', 
              kernel_regularizer=l2(0.008)),  # 과소적합 때 0.01 → 0.008 (조금만 감소)
        BatchNormalization(),
        Dropout(0.4),                    # 과소적합 때 0.5 → 0.4 (조금만 감소)
        
        Dense(num_classes, activation='softmax')
    ])

# 🎯 학습률도 아주 조금만 증가
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.00007),  # 과소적합 때 0.00005 → 0.00007 (조금만 증가)
    metrics=['accuracy']
)

print("세밀하게 조정된 모델 구조:")
cnn.summary()

# 모델 저장 디렉토리 생성
os.makedirs('model', exist_ok=True)

# 🎯 콜백도 조금 덜 엄격하게
callbacks_list = [
    ModelCheckpoint(
        "model/fine_tuned_balanced_model.h5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    ),
    # Early Stopping을 조금 덜 엄격하게
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,          # 과소적합 때 5 → 7 (조금만 증가)
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001
    ),
    # 학습률 감소도 조금 덜 적극적으로
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,          # 과소적합 때 0.5 → 0.6 (덜 급격하게)
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"🎯 세밀한 균형 조정 훈련 시작 - 배치 크기: {batch_size}")

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    hist = cnn.fit(
        train_generator,
        epochs=35,           # 적당한 에포크 수
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1
    )

print("🎉 세밀한 조정 완료!")

# 📊 이전 결과들과 비교하는 시각화
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

epochs_range = range(1, len(hist.history['accuracy']) + 1)

# 1. 정확도 그래프
ax1.plot(epochs_range, hist.history['accuracy'], 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=4)
ax1.plot(epochs_range, hist.history['val_accuracy'], 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=4)

# 목표 구간 표시
ax1.axhspan(0.6, 0.8, alpha=0.2, color='green', label='목표 검증 정확도 구간')
ax1.axhspan(0.7, 0.9, alpha=0.1, color='blue', label='목표 훈련 정확도 구간')

ax1.set_title('Fine-Tuned Balance Model', fontsize=16, fontweight='bold', color='darkblue')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.0)

# 2. 손실 그래프
ax2.plot(epochs_range, hist.history['loss'], 'b-', linewidth=3, label='Training Loss', marker='o', markersize=4)
ax2.plot(epochs_range, hist.history['val_loss'], 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=4)
ax2.set_title('Loss Progression', fontsize=16, fontweight='bold', color='darkgreen')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. 과적합 분석 (세밀하게)
train_acc = np.array(hist.history['accuracy'])
val_acc = np.array(hist.history['val_accuracy'])
gap = train_acc - val_acc

ax3.plot(epochs_range, gap, 'purple', linewidth=3, marker='d', markersize=4)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='완벽한 균형')
ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='허용 가능한 과적합 (+10%)')
ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='과적합 경고 (+20%)')
ax3.axhline(y=-0.05, color='blue', linestyle='--', alpha=0.7, label='과소적합 경계 (-5%)')

# 최적 구간 강조
ax3.fill_between(epochs_range, -0.05, 0.1, alpha=0.3, color='lightgreen', label='최적 구간')

ax3.set_title('Overfitting Analysis (Fine-Tuned)', fontsize=16, fontweight='bold', color='purple')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Accuracy Gap (Train - Val)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 이전 시도들과의 비교
ax4.clear()
attempts = ['1st\n(Overfit)', '2nd\n(Underfit)', '3rd\n(Overfit Again)', '4th\n(Fine-Tuned)']
train_accs = [0.95, 0.18, 0.95, hist.history['accuracy'][-1]]
val_accs = [0.65, 0.34, 0.65, hist.history['val_accuracy'][-1]]

x = np.arange(len(attempts))
width = 0.35

bars1 = ax4.bar(x - width/2, train_accs, width, label='Training Acc', color='lightblue', alpha=0.8)
bars2 = ax4.bar(x + width/2, val_accs, width, label='Validation Acc', color='lightcoral', alpha=0.8)

# 막대 위에 값 표시
for i, (train, val) in enumerate(zip(train_accs, val_accs)):
    ax4.text(i - width/2, train + 0.02, f'{train:.2f}', ha='center', va='bottom', fontweight='bold')
    ax4.text(i + width/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

ax4.set_title('Progress Comparison', fontsize=16, fontweight='bold', color='darkred')
ax4.set_ylabel('Accuracy', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(attempts)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('model/fine_tuned_progress.png', dpi=300, bbox_inches='tight')
plt.show()

# 🎯 세밀한 분석
final_train_acc = hist.history['accuracy'][-1]
final_val_acc = hist.history['val_accuracy'][-1]
best_val_acc = max(hist.history['val_accuracy'])
best_epoch = hist.history['val_accuracy'].index(best_val_acc) + 1
final_gap = final_train_acc - final_val_acc

print(f"\n🎯 === 세밀한 조정 결과 ===")
print(f"최종 훈련 정확도: {final_train_acc:.4f}")
print(f"최종 검증 정확도: {final_val_acc:.4f}")
print(f"🏆 최고 검증 정확도: {best_val_acc:.4f} (에포크 {best_epoch})")
print(f"📊 최종 격차: {final_gap:.4f}")

print(f"\n📈 === 진행 상황 비교 ===")
print(f"1차 시도 (과적합):     훈련 95.0% vs 검증 65.0% (격차: 30.0%)")
print(f"2차 시도 (과소적합):   훈련 18.0% vs 검증 34.0% (격차: -16.0%)")
print(f"3차 시도 (다시과적합): 훈련 95.0% vs 검증 65.0% (격차: 30.0%)")
print(f"4차 시도 (세밀조정):   훈련 {final_train_acc*100:.1f}% vs 검증 {final_val_acc*100:.1f}% (격차: {final_gap*100:.1f}%)")

# 🎯 개선도 평가
if 0.05 <= final_gap <= 0.15:
    print("🎉 드디어 적절한 균형을 찾았습니다!")
    status = "BALANCED"
elif final_gap > 0.2:
    print("😅 아직도 과적합입니다. 정규화를 더 강화해야 합니다.")
    status = "STILL_OVERFIT"
elif final_gap < -0.05:
    print("🔵 아직 과소적합입니다. 정규화를 조금 더 완화해야 합니다.")
    status = "STILL_UNDERFIT"
else:
    print("✅ 균형에 근접했습니다!")
    status = "NEAR_BALANCED"

print(f"\n🔍 === 다음 단계 가이드 ===")
if status == "BALANCED":
    print("🎯 완벽! 이제 데이터를 더 수집하거나 앙상블 기법을 시도해보세요.")
elif status == "STILL_OVERFIT":
    print("🔧 Dropout을 0.1 더 늘리고, L2 정규화를 0.002 더 늘려보세요.")
elif status == "STILL_UNDERFIT":
    print("🔧 Dropout을 0.05 줄이고, 훈련 가능한 레이어를 5개 더 늘려보세요.")
else:
    print("🎯 한 번 더 미세 조정하면 완벽한 균형을 찾을 수 있을 것입니다!")

print(f"\n💾 저장된 파일: model/fine_tuned_balanced_model.h5")
print(f"💾 진행상황 그래프: model/fine_tuned_progress.png")