import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow 버전 확인 및 호환성 설정
print(f"TensorFlow 버전: {tf.__version__}")

# GPU 최적화 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 설정 완료")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 호환성을 위한 import
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input,
    Conv2D, SeparableConv2D, Activation, Multiply, GlobalMaxPooling2D, 
    Concatenate, Add, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# 옵티마이저 호환성 처리
try:
    from tensorflow.keras.optimizers import AdamW
    ADAMW_AVAILABLE = True
    print("✅ AdamW 사용 가능")
except ImportError:
    try:
        import tensorflow_addons as tfa
        AdamW = tfa.optimizers.AdamW
        ADAMW_AVAILABLE = True
        print("✅ TensorFlow Addons AdamW 사용")
    except ImportError:
        from tensorflow.keras.optimizers import Adam
        ADAMW_AVAILABLE = False
        print("⚠️ AdamW 없음, Adam 사용")

# 경로 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'
val_dir = 'ani/val'

print("🚀 호환성 개선된 애니메이션 감정 인식 모델")
print("=" * 60)

# 🎯 강화된 데이터 증강
def create_advanced_augmentation():
    return ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode='nearest'
    )

def create_validation_generator():
    return ImageDataGenerator(rescale=1./255)

# 📊 데이터 로딩
def load_data(image_size, batch_size):
    train_datagen = create_advanced_augmentation()
    val_datagen = create_validation_generator()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

# 🏗️ 하이브리드 모델 (호환성 개선)
def create_hybrid_attention_model(num_classes=7):
    """Attention 메커니즘이 포함된 하이브리드 모델"""
    inputs = Input(shape=(224, 224, 3))
    
    # EfficientNetB0 백본
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Fine-tuning 설정
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until
    
    x = base_model.output
    
    # Global Average와 Max Pooling 결합
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    
    # Attention 메커니즘 (간단 버전)
    attention_weights = Dense(gap.shape[-1], activation='sigmoid', name='attention')(gap)
    attended_features = Multiply()([gap, attention_weights])
    
    # Feature 융합
    combined_features = Concatenate()([attended_features, gmp])
    
    # 분류 헤드
    x = BatchNormalization()(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_enhanced_efficientnet_model(num_classes=7):
    """강화된 EfficientNet 모델"""
    inputs = Input(shape=(224, 224, 3))
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Fine-tuning
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until
    
    x = base_model.output
    
    # SE (Squeeze-and-Excitation) 블록
    se = GlobalAveragePooling2D()(x)
    se = Dense(se.shape[-1] // 16, activation='relu')(se)
    se = Dense(x.shape[-1], activation='sigmoid')(se)
    
    # Reshape for multiplication
    se = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(se)
    x = Multiply()([x, se])
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # 분류 헤드
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='swish', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_densenet_ensemble_model(num_classes=7):
    """DenseNet 기반 앙상블 모델"""
    inputs = Input(shape=(224, 224, 3))
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Fine-tuning
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until
    
    x = base_model.output
    
    # Multi-scale feature extraction
    x = GlobalAveragePooling2D()(x)
    
    # 분류 헤드
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

# 🚀 호환 가능한 컴파일 함수
def compile_model_compatible(model, learning_rate=1e-4):
    """호환성을 고려한 모델 컴파일"""
    
    if ADAMW_AVAILABLE:
        if hasattr(AdamW, '__module__') and 'addons' in AdamW.__module__:
            # TensorFlow Addons의 AdamW
            optimizer = AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-5
            )
        else:
            # Keras의 AdamW
            optimizer = AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-5
            )
        print(f"✅ AdamW 옵티마이저 사용 (lr={learning_rate})")
    else:
        optimizer = Adam(learning_rate=learning_rate)
        print(f"✅ Adam 옵티마이저 사용 (lr={learning_rate})")
    
    # Focal Loss 대신 Label Smoothing 적용
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 🎯 콜백 함수
def get_callbacks(model_name):
    """모델별 콜백 설정"""
    
    callbacks = [
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

# 🏆 모델 훈련 함수
def train_compatible_model(model_func, model_name):
    """호환성 개선된 모델 훈련"""
    print(f"\n🧪 {model_name} 훈련 중...")
    
    try:
        # 모델 생성
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model = model_func()
        
        # 데이터 로딩
        train_gen, val_gen = load_data(image_size=224, batch_size=16)
        
        # 클래스 정보 출력
        print(f"클래스 개수: {train_gen.num_classes}")
        print(f"클래스 라벨: {list(train_gen.class_indices.keys())}")
        
        # 모델 컴파일
        model = compile_model_compatible(model)
        
        # 콜백 설정
        callbacks = get_callbacks(model_name)
        
        # 모델 요약
        print(f"\n📊 {model_name} 아키텍처:")
        print(f"총 파라미터: {model.count_params():,}")
        
        # 훈련 시작
        start_time = time.time()
        
        history = model.fit(
            train_gen,
            epochs=50,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # 결과 계산
        best_val_acc = max(history.history['val_accuracy'])
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_acc = history.history['accuracy'][-1]
        overfitting = final_train_acc - final_val_acc
        
        print(f"✅ {model_name} 완료!")
        print(f"   최고 검증 정확도: {best_val_acc:.4f}")
        print(f"   과적합 점수: {overfitting:.4f}")
        print(f"   훈련 시간: {train_time:.1f}초")
        
        # 90% 달성 체크
        if best_val_acc >= 0.9:
            print(f"🎉 {model_name}이 90% 달성!")
        elif best_val_acc >= 0.85:
            print(f"🚀 {model_name}이 85% 달성!")
        
        return {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'final_val_acc': final_val_acc,
            'final_train_acc': final_train_acc,
            'overfitting_score': overfitting,
            'train_time': train_time,
            'epochs': len(history.history['accuracy'])
        }, history
        
    except Exception as e:
        print(f"❌ {model_name} 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 🚀 메인 실험 실행
print("🚀 호환성 개선된 애니메이션 감정 인식 실험 시작!")

models_to_test = [
    (create_hybrid_attention_model, 'HybridAttention_EfficientNet'),
    (create_enhanced_efficientnet_model, 'Enhanced_EfficientNet'),
    (create_densenet_ensemble_model, 'DenseNet_Ensemble')
]

results = []
histories = {}

for model_func, model_name in models_to_test:
    result, history = train_compatible_model(model_func, model_name)
    
    if result:
        results.append(result)
        histories[model_name] = history

# 📊 결과 분석
print("\n" + "="*70)
print("🏆 호환성 개선된 애니메이션 감정 인식 결과")
print("="*70)

if results:
    # 성능순 정렬
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n🥇 최종 순위:")
    for i, result in enumerate(results, 1):
        status = "🎉 90%+" if result['best_val_acc'] >= 0.9 else "🚀 85%+" if result['best_val_acc'] >= 0.85 else "✅ 80%+" if result['best_val_acc'] >= 0.8 else "⚠️"
        print(f"{i}. {status} {result['model_name']:<30} | {result['best_val_acc']:.4f} | 과적합: {result['overfitting_score']:.4f}")
    
    # 90% 달성 모델
    success_models = [r for r in results if r['best_val_acc'] >= 0.9]
    
    if success_models:
        print(f"\n🎉 90% 달성 모델들:")
        for model in success_models:
            print(f"  🏆 {model['model_name']}: {model['best_val_acc']:.4f}")
    else:
        best_model = results[0]
        print(f"\n🎯 최고 성능 모델:")
        print(f"  모델: {best_model['model_name']}")
        print(f"  정확도: {best_model['best_val_acc']:.4f}")
        print(f"  90%까지: {0.9 - best_model['best_val_acc']:.4f} 부족")
    
    # 시각화
    if histories:
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # 정확도 그래프
        plt.subplot(2, 2, 1)
        for i, (model_name, history) in enumerate(histories.items()):
            plt.plot(history.history['val_accuracy'], 
                    label=f'{model_name}', color=colors[i], linewidth=2)
        plt.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% 목표')
        plt.title('검증 정확도 비교', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 손실 그래프
        plt.subplot(2, 2, 2)
        for i, (model_name, history) in enumerate(histories.items()):
            plt.plot(history.history['val_loss'], 
                    label=f'{model_name}', color=colors[i], linewidth=2)
        plt.title('검증 손실 비교', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 최종 성능 막대 그래프
        plt.subplot(2, 2, 3)
        model_names = [r['model_name'] for r in results]
        accuracies = [r['best_val_acc'] for r in results]
        bars = plt.bar(model_names, accuracies, color=['gold', 'silver', 'bronze'][:len(results)])
        plt.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.title('최고 성능 비교', fontsize=14, fontweight='bold')
        plt.ylabel('Best Validation Accuracy')
        plt.xticks(rotation=45)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 과적합 분석
        plt.subplot(2, 2, 4)
        overfitting_scores = [r['overfitting_score'] for r in results]
        plt.bar(model_names, overfitting_scores, color='lightcoral', alpha=0.7)
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='과적합 경계')
        plt.title('과적합 분석', fontsize=14, fontweight='bold')
        plt.ylabel('Overfitting Score')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('compatible_anime_emotion_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 최종 결론
    max_accuracy = max(r['best_val_acc'] for r in results)
    
    print(f"\n🎯 최종 결과:")
    print(f"최고 달성 정확도: {max_accuracy:.4f}")
    
    if max_accuracy >= 0.9:
        print("🎉🎉🎉 90% 목표 달성! 대성공!")
    elif max_accuracy >= 0.85:
        print("🎯 85% 이상 달성! 훌륭한 결과!")
    elif max_accuracy >= 0.8:
        print("✅ 80% 이상 달성! 좋은 성능!")
    else:
        print("⚠️ 추가 최적화 필요")

else:
    print("❌ 모든 실험이 실패했습니다.")

print(f"\n🏁 호환성 개선된 애니메이션 감정 인식 실험 완료!")

# 💡 성능 향상 팁
print(f"\n💡 성능 향상을 위한 추가 제안:")
print("=" * 50)
print("1. 📈 데이터 증가:")
print("   - 클래스별 데이터 밸런싱")
print("   - 온라인 데이터 크롤링")
print("   - 기존 이미지 회전/확대 등")

print("\n2. 🔧 하이퍼파라미터 튜닝:")
print("   - Learning Rate 스케줄링")
print("   - Batch Size 최적화")
print("   - Dropout 비율 조정")

print("\n3. 🎯 앙상블 기법:")
print("   - 여러 모델 조합")
print("   - Voting 또는 Averaging")
print("   - Stacking 기법")

print("\n4. 📊 전이학습 활용:")
print("   - 더 큰 데이터셋으로 pre-train")
print("   - Domain adaptation")
print("   - Fine-tuning 전략 개선")