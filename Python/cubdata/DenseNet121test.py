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

from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 경로 수정 (Windows용)
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
train_dir = 'ani/train'  # 원래: '/content/drive/MyDrive/AIdata/hymenoptera_data/train'
val_dir = 'ani/val'      # 원래: '/content/drive/MyDrive/AIdata/hymenoptera_data/val'

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True,
                                              width_shift_range = 0.1,
                                              height_shift_range = 0.1)
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224

# 배치 크기 GPU용으로 증가
batch_size = 2  # 원래: 12

train_generator = data_generator_with_aug.flow_from_directory(
       directory = train_dir,
       target_size=(image_size, image_size),
       batch_size=batch_size,  # GPU용 배치 크기
       class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = val_dir,
       target_size=(image_size, image_size),
       batch_size=batch_size,  # GPU용 배치 크기
       class_mode='categorical')

# 클래스 수 자동 감지 (원래: num_classes = 2)
num_classes = train_generator.num_classes

# GPU에서 모델 생성
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    base_model=DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))
    
    cnn = Sequential()
    cnn.add(base_model)
    cnn.add(Flatten())
    cnn.add(Dense(1024,activation='relu'))
    cnn.add(Dense(num_classes,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy'])

# 모델 저장 디렉토리 생성
os.makedirs('model', exist_ok=True)

# define the checkpoint
filepath = "model/model.h5"  # 원래: "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# GPU에서 훈련
print(f"🚀 GPU 훈련 시작 - 배치 크기: {batch_size}")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    hist=cnn.fit(
            train_generator,
            epochs=20,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            verbose=1  # 원래 코드에는 없었지만 추가
            )

print("🎉 훈련 완료!")

# 시각화 추가
import matplotlib.pyplot as plt

# 그래프 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. Model accuracy 그래프
ax1.plot(hist.history['accuracy'], linewidth=2, label='Train', color='#FF0000')
ax1.plot(hist.history['val_accuracy'], linewidth=2, label='Validation', color='#00FF00')
ax1.set_title('Model accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.0)

# 2. Model loss 그래프  
ax2.plot(hist.history['loss'], linewidth=2, label='Train', color='#FF0000')
ax2.plot(hist.history['val_loss'], linewidth=2, label='Validation', color='#00FF00')
ax2.set_title('Model loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 레이아웃 조정
plt.tight_layout()

# 그래프 저장
plt.savefig('model/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 최종 결과 출력
final_train_acc = hist.history['accuracy'][-1]
final_val_acc = hist.history['val_accuracy'][-1]
best_val_acc = max(hist.history['val_accuracy'])

print(f"📊 최종 훈련 정확도: {final_train_acc:.4f}")
print(f"📊 최종 검증 정확도: {final_val_acc:.4f}") 
print(f"🏆 최고 검증 정확도: {best_val_acc:.4f}")
print(f"💾 그래프 저장: model/training_results.png")