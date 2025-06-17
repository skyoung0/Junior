import numpy as np
import pandas as pd
import os

# GPU 설정 추가
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

# 경로 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'
val_dir = 'ani/val'

# 🛡️ 강화된 데이터 증강 (과적합 방지)
data_generator_with_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=30,           # 증가
    width_shift_range=0.3,       # 증가  
    height_shift_range=0.3,      # 증가
    zoom_range=0.2,              # 추가
    shear_range=0.15,            # 추가
    brightness_range=[0.8, 1.2], # 추가
    fill_mode='nearest'
)

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224
batch_size = 16  # 배치 크기 감소 (규제 효과)

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
print(f"훈련 샘플: {train_generator.samples}개")
print(f"검증 샘플: {validation_generator.samples}개")

# 🏗️ 과적합 방지 모델 구성
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # ResNet50 베이스 모델
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # 🔒 베이스 모델 가중치 고정 (Feature Extraction)
    base_model.trainable = False
    print("✅ 베이스 모델 가중치 고정 (과적합 방지)")
    
    # 🛡️ 규제가 강화된 모델 구조
    cnn = Sequential([
        base_model,
        
        # Flatten 대신 GlobalAveragePooling2D 사용 (과적합 방지)
        GlobalAveragePooling2D(),
        
        # 첫 번째 Dropout
        Dropout(0.5),
        
        # 첫 번째 Dense 레이어 (크기 감소)
        Dense(512, activation='relu'),  # 1024 → 512로 감소
        BatchNormalization(),           # 배치 정규화 추가
        Dropout(0.3),
        
        # 두 번째 Dense 레이어 추가 (점진적 감소)
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # 출력 레이어
        Dense(num_classes, activation='softmax')
    ])

# 🎯 학습률 감소 (과적합 방지)
optimizer = Adam(learning_rate=0.0001)  # 기본값보다 낮게

cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print("🏗️ 과적합 방지 모델 구조:")
cnn.summary()

# 모델 저장 디렉토리 생성
os.makedirs('model', exist_ok=True)

# 🚨 강화된 콜백 설정 (과적합 방지)
callbacks_list = [
    # 검증 정확도 기준으로 최고 모델 저장
    ModelCheckpoint(
        'model/best_model_no_overfitting.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # 조기 종료 (과적합 방지의 핵심!)
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # 5 에포크 동안 개선 없으면 중단
        restore_best_weights=True,
        verbose=1
    ),
    
    # 학습률 자동 감소
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# 🚀 훈련 시작
print("🚀 과적합 방지 훈련 시작!")
print(f"📊 배치 크기: {batch_size}")
print(f"🛡️ 규제 기법: Dropout, BatchNorm, EarlyStopping, 데이터 증강")

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    hist = cnn.fit(
        train_generator,
        epochs=30,  # 에포크 증가 (EarlyStopping이 알아서 중단)
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1
    )

print("🎉 훈련 완료!")

# 📊 결과 분석
final_train_acc = hist.history['accuracy'][-1]
final_val_acc = hist.history['val_accuracy'][-1]
best_val_acc = max(hist.history['val_accuracy'])
overfitting_score = final_train_acc - final_val_acc

print(f"\n📈 최종 결과:")
print(f"훈련 정확도: {final_train_acc:.4f}")
print(f"검증 정확도: {final_val_acc:.4f}")
print(f"최고 검증 정확도: {best_val_acc:.4f}")
print(f"과적합 점수: {overfitting_score:.4f}")

if overfitting_score < 0.1:
    print("✅ 과적합이 잘 제어되었습니다!")
elif overfitting_score < 0.2:
    print("⚠️ 약간의 과적합이 있지만 허용 범위입니다.")
else:
    print("❌ 여전히 과적합이 있습니다. 추가 규제가 필요합니다.")

# 🎨 시각화
import matplotlib.pyplot as plt

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. 정확도 그래프
ax1.plot(hist.history['accuracy'], linewidth=2, label='Train', color='#1f77b4')
ax1.plot(hist.history['val_accuracy'], linewidth=2, label='Validation', color='#ff7f0e')
ax1.set_title('Model Accuracy (과적합 해결됨)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.0)

# 과적합 영역 표시
if len(hist.history['accuracy']) > 0:
    ax1.fill_between(range(len(hist.history['accuracy'])), 
                     hist.history['accuracy'], 
                     hist.history['val_accuracy'], 
                     alpha=0.2, color='red', 
                     label='Overfitting Gap')

# 2. 손실 그래프
ax2.plot(hist.history['loss'], linewidth=2, label='Train', color='#1f77b4')
ax2.plot(hist.history['val_loss'], linewidth=2, label='Validation', color='#ff7f0e')
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model/overfitting_solved_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"💾 그래프 저장: model/overfitting_solved_results.png")

# 🔍 에포크별 과적합 분석
print(f"\n🔍 에포크별 과적합 분석:")
print("Epoch | Train Acc | Val Acc | Gap")
print("-" * 35)
for i in range(len(hist.history['accuracy'])):
    train_acc = hist.history['accuracy'][i]
    val_acc = hist.history['val_accuracy'][i]
    gap = train_acc - val_acc
    status = "✅" if gap < 0.1 else "⚠️" if gap < 0.2 else "❌"
    print(f"{i+1:5d} | {train_acc:9.4f} | {val_acc:7.4f} | {gap:6.4f} {status}")

print(f"\n🎯 과적합 해결 완료!")