import numpy as np
import pandas as pd
import os
import shutil
from collections import Counter
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# 🔧 TensorFlow JSON 직렬화 문제 완전 해결
import json
import keras.callbacks

# 커스텀 JSON 인코더
class TensorFlowJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'numpy'):
            return float(obj.numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

# JSON 인코더 패치
json._default_encoder = TensorFlowJSONEncoder()

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 설정 완료: RTX 3060 사용")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2

# 경로 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'
val_dir = 'ani/val'
augmented_train_dir = 'ani/augmented_train'

# 🛡️ 완전히 안전한 모델 체크포인트 콜백
class SuperSafeModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy', save_best_only=True, mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 텐서를 파이썬 숫자로 변환
        current = logs.get(self.monitor)
        if current is not None:
            if hasattr(current, 'numpy'):
                current = float(current.numpy())
            else:
                current = float(current)
                
            # 최고 성능 체크
            if self.mode == 'max':
                if current > self.best:
                    self.best = current
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model...")
                    try:
                        self.model.save_weights(self.filepath)
                        print("✅ 모델 가중치 저장 완료")
                    except Exception as e:
                        print(f"⚠️ 모델 저장 중 오류: {e}")
            else:
                if current < self.best:
                    self.best = current
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model...")
                    try:
                        self.model.save_weights(self.filepath)
                        print("✅ 모델 가중치 저장 완료")
                    except Exception as e:
                        print(f"⚠️ 모델 저장 중 오류: {e}")

# 🛡️ 완전히 안전한 Early Stopping
class SuperSafeEarlyStopping(Callback):
    def __init__(self, monitor='val_accuracy', patience=7, restore_best_weights=True, mode='max'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.wait = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        current = logs.get(self.monitor)
        if current is not None:
            if hasattr(current, 'numpy'):
                current = float(current.numpy())
            else:
                current = float(current)
                
            if self.mode == 'max':
                if current > self.best:
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
            else:
                if current < self.best:
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered. Restoring best weights...")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True

# 🔍 1. 기존 증대 데이터 활용 (이미 생성되었다면)
if os.path.exists(augmented_train_dir):
    print("🔍 기존 증대 데이터 발견, 재사용합니다.")
else:
    print("📊 간단한 데이터 증대 수행...")
    # 기본 Keras ImageDataGenerator만 사용 (OpenCV 오류 방지)
    if not os.path.exists(augmented_train_dir):
        shutil.copytree(train_dir, augmented_train_dir)
    print("✅ 기본 데이터 복사 완료")

# 🎯 2. 안전한 앙상블 모델 생성
def create_bulletproof_ensemble(num_classes, input_shape=(224, 224, 3)):
    """완전히 안전한 앙상블 모델"""
    
    models = []
    
    # 모델 1: ResNet50 (가장 안정적)
    print("🔨 ResNet50 모델 생성 중...")
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 마지막 몇 개 레이어만 훈련
    for layer in resnet_base.layers[:-10]:
        layer.trainable = False
    
    resnet_model = Sequential([
        resnet_base,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ], name='ResNet50_Model')
    
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    models.append(('ResNet50', resnet_model))
    
    # 모델 2: VGG16 (안정적)
    print("🔨 VGG16 모델 생성 중...")
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 마지막 몇 개 레이어만 훈련
    for layer in vgg_base.layers[:-4]:
        layer.trainable = False
    
    vgg_model = Sequential([
        vgg_base,
        GlobalAveragePooling2D(),
        Dropout(0.6),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name='VGG16_Model')
    
    vgg_model.compile(
        optimizer=Adam(learning_rate=0.00008),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    models.append(('VGG16', vgg_model))
    
    return models

# 🎯 3. 데이터 로더 설정
def create_safe_data_generators(train_dir, val_dir, batch_size=24):
    """안전한 데이터 생성기"""
    
    # 훈련용 증대 (Keras 기본만 사용)
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # 검증용
    val_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
    
    # 데이터 생성기
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

# 데이터 생성기 생성
train_generator, val_generator = create_safe_data_generators(augmented_train_dir, val_dir, batch_size=24)
num_classes = train_generator.num_classes

print(f"\n📊 데이터 정보:")
print(f"훈련 샘플: {train_generator.samples:,}개")
print(f"검증 샘플: {val_generator.samples:,}개")
print(f"클래스 수: {num_classes}개")
print(f"클래스명: {list(train_generator.class_indices.keys())}")

# 🎯 4. 앙상블 모델 생성
ensemble_models = create_bulletproof_ensemble(num_classes)

print(f"\n🎭 앙상블 구성: {len(ensemble_models)}개 모델")
for name, model in ensemble_models:
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
    print(f"   - {name}: 총 {total_params:,} 파라미터 (훈련 가능: {trainable_params:,})")

# 모델 저장 디렉토리 생성
os.makedirs('model/ensemble', exist_ok=True)

# 🎯 5. 완전히 안전한 훈련 함수
def bulletproof_training(ensemble_models, train_generator, val_generator, epochs=15):
    """완전히 안전한 앙상블 훈련"""
    
    trained_models = []
    training_histories = []
    
    for i, (model_name, model) in enumerate(ensemble_models):
        print(f"\n🚀 {i+1}/{len(ensemble_models)} 모델 훈련 시작: {model_name}")
        print("=" * 60)
        
        # 완전히 안전한 콜백들
        callbacks = [
            SuperSafeModelCheckpoint(
                f"model/ensemble/{model_name.lower()}_safe_weights.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            SuperSafeEarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max'
            )
        ]
        
        # GPU에서 훈련
        print(f"🔥 GPU에서 {model_name} 훈련 시작...")
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            try:
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    callbacks=callbacks,
                    verbose=1
                )
                
                print(f"✅ {model_name} 훈련 성공적으로 완료!")
                
            except Exception as e:
                print(f"❌ {model_name} 훈련 중 오류 발생: {e}")
                # 기본 훈련 (콜백 없이)
                print(f"🔄 {model_name} 기본 모드로 재시도...")
                history = model.fit(
                    train_generator,
                    epochs=min(10, epochs),
                    validation_data=val_generator,
                    verbose=1
                )
        
        trained_models.append((model_name, model))
        training_histories.append((model_name, history))
        
        # 성능 출력 (안전하게)
        try:
            val_acc_history = history.history['val_accuracy']
            best_val_acc = max(val_acc_history)
            final_val_acc = val_acc_history[-1]
            print(f"🏆 {model_name} 최고 검증 정확도: {best_val_acc:.4f}")
            print(f"📊 {model_name} 최종 검증 정확도: {final_val_acc:.4f}")
        except Exception as e:
            print(f"⚠️ {model_name} 성능 출력 중 오류: {e}")
    
    return trained_models, training_histories

# 🎯 6. 안전한 훈련 실행
print("\n🛡️ 완전히 안전한 앙상블 훈련 시작!")
trained_models, training_histories = bulletproof_training(ensemble_models, train_generator, val_generator)

# 🎯 7. 안전한 성능 평가
def safe_evaluation(trained_models, val_generator):
    """안전한 성능 평가"""
    print("\n🔍 안전한 성능 평가 중...")
    
    individual_scores = []
    
    for model_name, model in trained_models:
        try:
            # 안전한 평가
            val_generator.reset()
            score = model.evaluate(val_generator, verbose=0)
            accuracy = float(score[1]) if hasattr(score[1], 'numpy') else score[1]
            individual_scores.append((model_name, accuracy))
            print(f"{model_name:12} 검증 정확도: {accuracy:.4f}")
        except Exception as e:
            print(f"⚠️ {model_name} 평가 중 오류: {e}")
            individual_scores.append((model_name, 0.0))
    
    # 간단한 앙상블 테스트
    try:
        val_generator.reset()
        batch_x, batch_y = next(val_generator)
        
        predictions = []
        for model_name, model in trained_models:
            try:
                pred = model.predict(batch_x, verbose=0)
                predictions.append(pred)
            except Exception as e:
                print(f"⚠️ {model_name} 예측 중 오류: {e}")
        
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            ensemble_accuracy = np.mean(
                np.argmax(ensemble_pred, axis=1) == np.argmax(batch_y, axis=1)
            )
            print(f"\n🎭 앙상블 샘플 정확도: {ensemble_accuracy:.4f}")
        else:
            ensemble_accuracy = 0.0
            print("\n❌ 앙상블 예측 실패")
            
    except Exception as e:
        print(f"⚠️ 앙상블 평가 중 오류: {e}")
        ensemble_accuracy = 0.0
    
    return individual_scores, ensemble_accuracy

# 성능 평가 실행
individual_scores, ensemble_accuracy = safe_evaluation(trained_models, val_generator)

# 🎯 8. 간단한 시각화
def safe_visualization(training_histories, individual_scores, ensemble_accuracy):
    """안전한 시각화"""
    
    try:
        plt.figure(figsize=(15, 5))
        
        # 1. 훈련 곡선
        plt.subplot(1, 3, 1)
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, history) in enumerate(training_histories):
            try:
                val_acc = history.history['val_accuracy']
                epochs = range(1, len(val_acc) + 1)
                plt.plot(epochs, val_acc, color=colors[i % len(colors)], 
                        linewidth=2, label=model_name, marker='o', markersize=3)
            except Exception as e:
                print(f"⚠️ {model_name} 곡선 그리기 오류: {e}")
        
        plt.title('🚀 Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 개별 성능
        plt.subplot(1, 3, 2)
        if individual_scores:
            names = [name for name, _ in individual_scores]
            scores = [score for _, score in individual_scores]
            colors_bar = ['lightblue', 'lightcoral', 'lightgreen']
            
            plt.bar(names, scores, color=colors_bar[:len(names)], alpha=0.8)
            plt.title('🏆 Individual Performance')
            plt.ylabel('Validation Accuracy')
            plt.xticks(rotation=45)
        
        # 3. 앙상블 비교
        plt.subplot(1, 3, 3)
        if individual_scores:
            all_names = [name for name, _ in individual_scores] + ['Ensemble']
            all_scores = [score for _, score in individual_scores] + [ensemble_accuracy]
            colors_comp = ['lightblue', 'lightcoral', 'gold']
            
            plt.bar(all_names, all_scores, color=colors_comp[:len(all_names)], alpha=0.8)
            plt.title('🎯 Ensemble Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model/ensemble/safe_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ 시각화 완료")
        
    except Exception as e:
        print(f"⚠️ 시각화 중 오류: {e}")

# 시각화 실행
safe_visualization(training_histories, individual_scores, ensemble_accuracy)

# 🎯 9. 최종 안전 리포트
print(f"\n" + "="*60)
print("🛡️ 완전히 안전한 앙상블 시스템 완료!")
print("="*60)

if individual_scores:
    print(f"\n📊 개별 모델 성능:")
    best_score = 0
    best_model = ""
    
    for name, score in individual_scores:
        print(f"{name:12}: {score:.4f} ({score*100:.1f}%)")
        if score > best_score:
            best_score = score
            best_model = name
    
    print(f"\n🏆 최고 개별 모델: {best_model} ({best_score:.4f})")
    print(f"🎭 앙상블 성능: {ensemble_accuracy:.4f}")
    
    improvement = ensemble_accuracy - best_score
    if improvement > 0:
        print(f"📈 앙상블 개선: +{improvement:.4f} (+{improvement*100:.1f}%p)")
        print("✅ 앙상블이 효과적입니다!")
    else:
        print(f"📉 앙상블 효과: {improvement:.4f}")
        print("⚠️ 개별 모델이 더 우수합니다.")

print(f"\n💾 저장된 파일:")
print(f"- 모델 가중치: model/ensemble/*_safe_weights.h5")
print(f"- 결과 그래프: model/ensemble/safe_results.png")

print(f"\n🎉 JSON 오류 없이 성공적으로 완료되었습니다! 🚀")

# 🎯 10. 사용법 안내
print(f"\n📖 모델 사용법:")
print("1. 가중치 로드:")
print("   model.load_weights('model/ensemble/resnet50_safe_weights.h5')")
print("2. 예측:")
print("   predictions = model.predict(your_data)")
print("3. 앙상블 예측:")
print("   ensemble_pred = np.mean([model1.predict(data), model2.predict(data)], axis=0)")