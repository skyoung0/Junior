import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time

# GPU 설정
print("=== GPU 설정 확인 ===")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU 설정 완료: {len(gpus)}개")
else:
    print("⚠️ CPU 모드로 실행")

# 모델 imports
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, VGG16, MobileNet, EfficientNetB0
)
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Conv2D, MaxPooling2D, Flatten
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    LearningRateScheduler, CSVLogger
)

# 작업 디렉토리 설정
os.chdir('C:/Users/ayaan/Documents/Git/Junior/Python/Data')
print(f"현재 작업 디렉토리: {os.getcwd()}")

# 데이터 경로
train_dir = 'ani/train'
val_dir = 'ani/val'

# 결과 저장 디렉토리
os.makedirs('experiments', exist_ok=True)
os.makedirs('experiments/models', exist_ok=True)
os.makedirs('experiments/plots', exist_ok=True)
os.makedirs('experiments/logs', exist_ok=True)

class ExperimentLogger:
    def __init__(self):
        self.results = []
        
    def log_experiment(self, experiment_name, model_name, config, history, train_time):
        result = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_name': model_name,
            'config': config,
            'train_time': train_time,
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'best_val_acc': float(max(history.history['val_accuracy'])),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['loss']),
            'overfitting_score': float(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])
        }
        self.results.append(result)
        
        # 개별 실험 결과 저장
        with open(f'experiments/logs/{experiment_name}_{model_name}.json', 'w') as f:
            json.dump(result, f, indent=2)
            
    def save_summary(self):
        df = pd.DataFrame(self.results)
        df.to_csv('experiments/experiment_summary.csv', index=False)
        return df

logger = ExperimentLogger()

# 데이터 로더 함수들
def create_data_generators(augmentation_level='medium'):
    """다양한 강도의 데이터 증강"""
    
    base_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    if augmentation_level == 'none':
        train_gen = base_gen
    elif augmentation_level == 'light':
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
    elif augmentation_level == 'medium':
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.15,
            shear_range=0.1
        )
    elif augmentation_level == 'heavy':
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1
        )
    
    return train_gen, base_gen

def load_data(batch_size=32, augmentation='medium'):
    """데이터 로딩"""
    train_gen, val_gen = create_data_generators(augmentation)
    
    train_generator = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

# 모델 생성 함수들
def create_densenet_model(num_classes, regularization_config):
    """DenseNet121 모델"""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    if regularization_config.get('freeze_base', True):
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-regularization_config.get('unfreeze_layers', 30)]:
            layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # 정규화 레이어 추가
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_1']))
    
    # Dense 레이어
    l1_reg = regularization_config.get('l1_reg', 0)
    l2_reg = regularization_config.get('l2_reg', 0)
    
    if l1_reg > 0 and l2_reg > 0:
        reg = l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = l1(l1_reg)
    elif l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    
    model.add(Dense(regularization_config.get('dense_units', 1024), 
                   activation='relu', kernel_regularizer=reg))
    
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_2']))
    
    if regularization_config.get('second_dense', False):
        model.add(Dense(512, activation='relu', kernel_regularizer=reg))
        if regularization_config.get('dropout_3', 0) > 0:
            model.add(Dropout(regularization_config['dropout_3']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_resnet_model(num_classes, regularization_config):
    """ResNet50 모델"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    if regularization_config.get('freeze_base', True):
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-regularization_config.get('unfreeze_layers', 30)]:
            layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # 정규화 적용 (DenseNet과 동일한 로직)
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_1']))
    
    l1_reg = regularization_config.get('l1_reg', 0)
    l2_reg = regularization_config.get('l2_reg', 0)
    
    if l1_reg > 0 and l2_reg > 0:
        reg = l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = l1(l1_reg)
    elif l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    
    model.add(Dense(regularization_config.get('dense_units', 1024), 
                   activation='relu', kernel_regularizer=reg))
    
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_2']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_vgg_model(num_classes, regularization_config):
    """VGG16 모델"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    if regularization_config.get('freeze_base', True):
        base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # 정규화 적용
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_1']))
    
    l1_reg = regularization_config.get('l1_reg', 0)
    l2_reg = regularization_config.get('l2_reg', 0)
    
    if l1_reg > 0 and l2_reg > 0:
        reg = l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = l1(l1_reg)
    elif l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    
    model.add(Dense(regularization_config.get('dense_units', 1024), 
                   activation='relu', kernel_regularizer=reg))
    
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_2']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_mobilenet_model(num_classes, regularization_config):
    """MobileNet 모델"""
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    if regularization_config.get('freeze_base', True):
        base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # 정규화 적용
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_1']))
    
    l1_reg = regularization_config.get('l1_reg', 0)
    l2_reg = regularization_config.get('l2_reg', 0)
    
    if l1_reg > 0 and l2_reg > 0:
        reg = l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = l1(l1_reg)
    elif l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    
    model.add(Dense(regularization_config.get('dense_units', 512), 
                   activation='relu', kernel_regularizer=reg))
    
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_2']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def create_custom_cnn_model(num_classes, regularization_config):
    """커스텀 CNN 모델"""
    model = Sequential()
    
    # 첫 번째 Conv 블록
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if regularization_config.get('dropout_conv1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_conv1']))
    
    # 두 번째 Conv 블록
    model.add(Conv2D(64, (3, 3), activation='relu'))
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if regularization_config.get('dropout_conv2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_conv2']))
    
    # 세 번째 Conv 블록
    model.add(Conv2D(128, (3, 3), activation='relu'))
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if regularization_config.get('dropout_conv3', 0) > 0:
        model.add(Dropout(regularization_config['dropout_conv3']))
    
    # Dense 레이어
    model.add(GlobalAveragePooling2D())
    
    l1_reg = regularization_config.get('l1_reg', 0)
    l2_reg = regularization_config.get('l2_reg', 0)
    
    if l1_reg > 0 and l2_reg > 0:
        reg = l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = l1(l1_reg)
    elif l2_reg > 0:
        reg = l2(l2_reg)
    else:
        reg = None
    
    if regularization_config.get('dropout_1', 0) > 0:
        model.add(Dropout(regularization_config['dropout_1']))
    
    model.add(Dense(regularization_config.get('dense_units', 256), 
                   activation='relu', kernel_regularizer=reg))
    
    if regularization_config.get('batch_norm', False):
        model.add(BatchNormalization())
    
    if regularization_config.get('dropout_2', 0) > 0:
        model.add(Dropout(regularization_config['dropout_2']))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# 학습률 스케줄러
def create_lr_scheduler(scheduler_type):
    """다양한 학습률 스케줄러"""
    if scheduler_type == 'step':
        def step_decay(epoch):
            initial_lrate = 0.001
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
            return lrate
        return LearningRateScheduler(step_decay)
    
    elif scheduler_type == 'exponential':
        def exp_decay(epoch):
            initial_lrate = 0.001
            k = 0.1
            lrate = initial_lrate * np.exp(-k*epoch)
            return lrate
        return LearningRateScheduler(exp_decay)
    
    elif scheduler_type == 'cosine':
        def cosine_decay(epoch):
            initial_lrate = 0.001
            min_lrate = 0.00001
            epochs = 25
            lrate = min_lrate + (initial_lrate - min_lrate) * (1 + np.cos(np.pi * epoch / epochs)) / 2
            return lrate
        return LearningRateScheduler(cosine_decay)
    
    else:  # plateau
        return ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

def run_experiment(experiment_name, model_func, model_name, config):
    """실험 실행"""
    print(f"\n{'='*50}")
    print(f"실험: {experiment_name}")
    print(f"모델: {model_name}")
    print(f"설정: {config}")
    print('='*50)
    
    # 데이터 로드
    train_gen, val_gen = load_data(
        batch_size=config.get('batch_size', 32),
        augmentation=config.get('augmentation', 'medium')
    )
    
    num_classes = train_gen.num_classes
    
    # 모델 생성
    model = model_func(num_classes, config)
    
    # 옵티마이저 설정
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    
    # 모델 컴파일
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    callbacks = []
    
    # 모델 체크포인트
    checkpoint_path = f'experiments/models/{experiment_name}_{model_name}.h5'
    callbacks.append(ModelCheckpoint(
        checkpoint_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max'
    ))
    
    # 조기 종료
    if config.get('early_stopping', True):
        callbacks.append(EarlyStopping(
            monitor='val_accuracy',
            patience=config.get('patience', 7),
            restore_best_weights=True
        ))
    
    # 학습률 스케줄러
    lr_scheduler = config.get('lr_scheduler', 'plateau')
    callbacks.append(create_lr_scheduler(lr_scheduler))
    
    # CSV 로거
    csv_path = f'experiments/logs/{experiment_name}_{model_name}_log.csv'
    callbacks.append(CSVLogger(csv_path))
    
    # 훈련
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        epochs=config.get('epochs', 25),
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        workers=1,
        use_multiprocessing=False
    )
    
    train_time = time.time() - start_time
    
    # 결과 로깅
    logger.log_experiment(experiment_name, model_name, config, history, train_time)
    
    print(f"✅ 실험 완료! 소요 시간: {train_time:.2f}초")
    print(f"최고 검증 정확도: {max(history.history['val_accuracy']):.4f}")
    
    return history, model

# 실험 설정들
print("\n🚀 종합 실험 시작!")
print("=" * 60)

# 1. 기본 모델들 비교 (규제 없음)
baseline_config = {
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'augmentation': 'medium',
    'freeze_base': True,
    'early_stopping': True,
    'patience': 5
}

models_to_test = [
    (create_densenet_model, 'DenseNet121'),
    (create_resnet_model, 'ResNet50'),
    (create_vgg_model, 'VGG16'),
    (create_mobilenet_model, 'MobileNet'),
    (create_custom_cnn_model, 'CustomCNN')
]

# 기본 모델 비교
print("\n📊 1. 기본 모델 비교 실험")
for model_func, model_name in models_to_test:
    try:
        run_experiment('baseline', model_func, model_name, baseline_config)
    except Exception as e:
        print(f"❌ {model_name} 실험 실패: {e}")
        continue

# 2. 규제 기법 비교 (DenseNet121 기준)
print("\n📊 2. 규제 기법 비교 실험")

regularization_configs = {
    'no_reg': baseline_config,
    
    'dropout_light': {**baseline_config, 'dropout_1': 0.3, 'dropout_2': 0.2},
    
    'dropout_heavy': {**baseline_config, 'dropout_1': 0.5, 'dropout_2': 0.3, 'dropout_3': 0.2, 'second_dense': True},
    
    'l2_reg': {**baseline_config, 'l2_reg': 0.001},
    
    'l1_reg': {**baseline_config, 'l1_reg': 0.001},
    
    'l1_l2_reg': {**baseline_config, 'l1_reg': 0.0005, 'l2_reg': 0.0005},
    
    'batch_norm': {**baseline_config, 'batch_norm': True},
    
    'combined_light': {**baseline_config, 'dropout_1': 0.3, 'dropout_2': 0.2, 'l2_reg': 0.0005, 'batch_norm': True},
    
    'combined_heavy': {**baseline_config, 'dropout_1': 0.5, 'dropout_2': 0.3, 'l1_reg': 0.0005, 'l2_reg': 0.0005, 'batch_norm': True, 'second_dense': True}
}

for reg_name, config in regularization_configs.items():
    try:
        run_experiment(f'regularization_{reg_name}', create_densenet_model, 'DenseNet121', config)
    except Exception as e:
        print(f"❌ 규제 실험 {reg_name} 실패: {e}")
        continue

# 3. 데이터 증강 비교
print("\n📊 3. 데이터 증강 비교 실험")

augmentation_configs = {
    'no_aug': {**baseline_config, 'augmentation': 'none'},
    'light_aug': {**baseline_config, 'augmentation': 'light'},
    'medium_aug': {**baseline_config, 'augmentation': 'medium'},
    'heavy_aug': {**baseline_config, 'augmentation': 'heavy'}
}

for aug_name, config in augmentation_configs.items():
    try:
        run_experiment(f'augmentation_{aug_name}', create_densenet_model, 'DenseNet121', config)
    except Exception as e:
        print(f"❌ 증강 실험 {aug_name} 실패: {e}")
        continue

# 4. 옵티마이저 비교
print("\n📊 4. 옵티마이저 비교 실험")

optimizer_configs = {
    'adam': {**baseline_config, 'optimizer': 'adam'},
    'sgd': {**baseline_config, 'optimizer': 'sgd'},
    'rmsprop': {**baseline_config, 'optimizer': 'rmsprop'}
}

for opt_name, config in optimizer_configs.items():
    try:
        run_experiment(f'optimizer_{opt_name}', create_densenet_model, 'DenseNet121', config)
    except Exception as e:
        print(f"❌ 옵티마이저 실험 {opt_name} 실패: {e}")
        continue

# 5. 학습률 스케줄러 비교
print("\n📊 5. 학습률 스케줄러 비교 실험")

scheduler_configs = {
    'plateau': {**baseline_config, 'lr_scheduler': 'plateau'},
    'step': {**baseline_config, 'lr_scheduler': 'step'},
    'exponential': {**baseline_config, 'lr_scheduler': 'exponential'},
    'cosine': {**baseline_config, 'lr_scheduler': 'cosine'}
}

for sched_name, config in scheduler_configs.items():
    try:
        run_experiment(f'scheduler_{sched_name}', create_densenet_model, 'DenseNet121', config)
    except Exception as e:
        print(f"❌ 스케줄러 실험 {sched_name} 실패: {e}")
        continue

print("\n🎉 모든 실험 완료!")
print("=" * 60)

# 결과 분석 및 시각화
print("\n📈 결과 분석 및 시각화 생성 중...")

try:
    # 실험 결과 요약
    results_df = logger.save_summary()
    
    # 상위 실험 결과 출력
    print("\n🏆 Top 10 실험 결과 (검증 정확도 기준):")
    top_results = results_df.nlargest(10, 'best_val_acc')[
        ['experiment_name', 'model_name', 'best_val_acc', 'overfitting_score', 'train_time']
    ]
    print(top_results.to_string(index=False))
    
    # 시각화
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 모델별 성능 비교
    baseline_results = results_df[results_df['experiment_name'] == 'baseline']
    if len(baseline_results) > 0:
        axes[0, 0].bar(baseline_results['model_name'], baseline_results['best_val_acc'])
        axes[0, 0].set_title('모델별 검증 정확도 비교')
        axes[0, 0].set_ylabel('검증 정확도')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 규제 기법 효과
    reg_results = results_df[results_df['experiment_name'].str.contains('regularization', na=False)]
    if len(reg_results) > 0:
        reg_names = reg_results['experiment_name'].str.replace('regularization_', '')
        axes[0, 1].bar(reg_names, reg_results['best_val_acc'])
        axes[0, 1].set_title('규제 기법별 검증 정확도')
        axes[0, 1].set_ylabel('검증 정확도')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 과적합 점수 비교
    axes[0, 2].scatter(results_df['best_val_acc'], results_df['overfitting_score'], alpha=0.7)
    axes[0, 2].set_xlabel('검증 정확도')
    axes[0, 2].set_ylabel('과적합 점수 (훈련-검증 정확도 차이)')
    axes[0, 2].set_title('과적합 vs 성능 분석')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 4. 데이터 증강 효과
    aug_results = results_df[results_df['experiment_name'].str.contains('augmentation', na=False)]
    if len(aug_results) > 0:
        aug_names = aug_results['experiment_name'].str.replace('augmentation_', '')
        axes[1, 0].bar(aug_names, aug_results['best_val_acc'])
        axes[1, 0].set_title('데이터 증강별 검증 정확도')
        axes[1, 0].set_ylabel('검증 정확도')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 훈련 시간 vs 성능
    axes[1, 1].scatter(results_df['train_time'], results_df['best_val_acc'], alpha=0.7)
    axes[1, 1].set_xlabel('훈련 시간 (초)')
    axes[1, 1].set_ylabel('검증 정확도')
    axes[1, 1].set_title('훈련 시간 vs 성능')
    
    # 6. 옵티마이저 비교
    opt_results = results_df[results_df['experiment_name'].str.contains('optimizer', na=False)]
    if len(opt_results) > 0:
        opt_names = opt_results['experiment_name'].str.replace('optimizer_', '')
        axes[1, 2].bar(opt_names, opt_results['best_val_acc'])
        axes[1, 2].set_title('옵티마이저별 검증 정확도')
        axes[1, 2].set_ylabel('검증 정확도')
    
    plt.tight_layout()
    plt.savefig('experiments/plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 상세 분석 히트맵
    plt.figure(figsize=(15, 10))
    
    # 실험별 주요 지표 히트맵
    metrics_df = results_df.pivot_table(
        index='experiment_name', 
        columns='model_name', 
        values='best_val_acc', 
        fill_value=0
    )
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='viridis')
    plt.title('실험별 검증 정확도 히트맵')
    plt.tight_layout()
    plt.savefig('experiments/plots/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 과적합 분석
    plt.figure(figsize=(12, 8))
    
    # 과적합 정도별 색상 구분
    colors = ['green' if x < 0.05 else 'yellow' if x < 0.1 else 'red' for x in results_df['overfitting_score']]
    
    plt.scatter(results_df['best_val_acc'], results_df['overfitting_score'], 
               c=colors, alpha=0.7, s=100)
    
    # 각 점에 실험명 라벨
    for i, row in results_df.iterrows():
        plt.annotate(f"{row['experiment_name'][:10]}", 
                    (row['best_val_acc'], row['overfitting_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('검증 정확도')
    plt.ylabel('과적합 점수')
    plt.title('과적합 분석 (녹색: 좋음, 노랑: 보통, 빨강: 과적합)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='과적합 경계 (0.05)')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='심각한 과적합 (0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/plots/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 상세 분석 리포트 생성 중...")
    
    # 상세 분석 리포트 생성
    report = f"""
    ================================================================================
    🔬 종합 실험 분석 리포트
    ================================================================================
    
    📅 실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    🔢 총 실험 수: {len(results_df)}
    
    🏆 최고 성능 실험:
    {'-'*50}
    """
    
    best_experiment = results_df.loc[results_df['best_val_acc'].idxmax()]
    report += f"""
    실험명: {best_experiment['experiment_name']}
    모델: {best_experiment['model_name']}
    검증 정확도: {best_experiment['best_val_acc']:.4f}
    과적합 점수: {best_experiment['overfitting_score']:.4f}
    훈련 시간: {best_experiment['train_time']:.2f}초
    """
    
    # 모델별 최고 성능
    report += f"\n\n📈 모델별 최고 성능:\n{'-'*50}\n"
    model_best = results_df.groupby('model_name')['best_val_acc'].max().sort_values(ascending=False)
    for model, acc in model_best.items():
        report += f"{model}: {acc:.4f}\n"
    
    # 규제 기법별 효과
    reg_results = results_df[results_df['experiment_name'].str.contains('regularization', na=False)]
    if len(reg_results) > 0:
        report += f"\n\n🛡️ 규제 기법별 효과:\n{'-'*50}\n"
        reg_best = reg_results.groupby('experiment_name')['best_val_acc'].max().sort_values(ascending=False)
        for exp, acc in reg_best.items():
            reg_name = exp.replace('regularization_', '')
            report += f"{reg_name}: {acc:.4f}\n"
    
    # 과적합 분석
    low_overfitting = results_df[results_df['overfitting_score'] < 0.05]
    medium_overfitting = results_df[(results_df['overfitting_score'] >= 0.05) & (results_df['overfitting_score'] < 0.1)]
    high_overfitting = results_df[results_df['overfitting_score'] >= 0.1]
    
    report += f"""
    
    🎯 과적합 분석:
    {'-'*50}
    과적합 없음 (< 0.05): {len(low_overfitting)}개 실험
    경미한 과적합 (0.05-0.1): {len(medium_overfitting)}개 실험
    심각한 과적합 (> 0.1): {len(high_overfitting)}개 실험
    
    """
    
    if len(low_overfitting) > 0:
        best_balanced = low_overfitting.loc[low_overfitting['best_val_acc'].idxmax()]
        report += f"""
    🎯 최적 균형 실험 (과적합 없으면서 고성능):
    실험명: {best_balanced['experiment_name']}
    모델: {best_balanced['model_name']}
    검증 정확도: {best_balanced['best_val_acc']:.4f}
    과적합 점수: {best_balanced['overfitting_score']:.4f}
    """
    
    # 효율성 분석 (시간 대비 성능)
    results_df['efficiency'] = results_df['best_val_acc'] / (results_df['train_time'] / 60)  # 분당 정확도
    most_efficient = results_df.loc[results_df['efficiency'].idxmax()]
    
    report += f"""
    
    ⚡ 효율성 분석 (시간 대비 성능):
    {'-'*50}
    가장 효율적인 실험:
    실험명: {most_efficient['experiment_name']}
    모델: {most_efficient['model_name']}
    검증 정확도: {most_efficient['best_val_acc']:.4f}
    훈련 시간: {most_efficient['train_time']:.2f}초
    효율성 점수: {most_efficient['efficiency']:.4f}
    """
    
    # 권장 사항
    report += f"""
    
    💡 권장 사항:
    {'-'*50}
    1. 최고 성능: {best_experiment['experiment_name']} ({best_experiment['model_name']})
    2. 최적 균형: {best_balanced['experiment_name'] if len(low_overfitting) > 0 else '과적합 없는 실험 없음'}
    3. 최고 효율: {most_efficient['experiment_name']} ({most_efficient['model_name']})
    
    📈 성능 향상 팁:
    - {'규제 기법 추가 필요' if len(high_overfitting) > len(low_overfitting) else '현재 규제 수준 적절'}
    - {'데이터 증강 강화 고려' if results_df['best_val_acc'].max() < 0.9 else '현재 성능 만족스러움'}
    - {'더 긴 훈련 시간 고려' if results_df['epochs_trained'].mean() < 20 else '적절한 훈련 시간'}
    
    ================================================================================
    """
    
    # 리포트 저장
    with open('experiments/comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    
    # 최종 추천 모델 설정
    print("\n🎯 최종 추천 설정:")
    print("="*60)
    
    if len(low_overfitting) > 0:
        recommended = best_balanced
    else:
        recommended = best_experiment
    
    print(f"추천 실험: {recommended['experiment_name']}")
    print(f"추천 모델: {recommended['model_name']}")
    print(f"예상 성능: {recommended['best_val_acc']:.4f}")
    print(f"과적합 위험: {'낮음' if recommended['overfitting_score'] < 0.05 else '보통' if recommended['overfitting_score'] < 0.1 else '높음'}")
    
    # 추천 설정 저장
    recommended_config = {
        'experiment_name': recommended['experiment_name'],
        'model_name': recommended['model_name'],
        'config': recommended['config'],
        'expected_accuracy': float(recommended['best_val_acc']),
        'overfitting_risk': 'low' if recommended['overfitting_score'] < 0.05 else 'medium' if recommended['overfitting_score'] < 0.1 else 'high'
    }
    
    with open('experiments/recommended_config.json', 'w') as f:
        json.dump(recommended_config, f, indent=2)
    
    print(f"\n💾 모든 결과가 'experiments' 폴더에 저장되었습니다!")
    print("- experiment_summary.csv: 전체 실험 결과")
    print("- comprehensive_analysis_report.txt: 상세 분석 리포트") 
    print("- recommended_config.json: 추천 설정")
    print("- plots/: 모든 시각화 결과")
    print("- models/: 훈련된 모델들")
    print("- logs/: 훈련 로그들")

except Exception as e:
    print(f"❌ 분석 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 종합 분석 완료!")
print("="*60)
print("RTX 3060을 활용한 딥러닝 모델 성능 비교 및 최적화 실험이 모두 완료되었습니다!")
print("결과를 바탕으로 최적의 모델과 설정을 선택하여 사용하시기 바랍니다.")