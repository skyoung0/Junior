import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import warnings
import json
from collections import defaultdict
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# MPS 설정
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
    torch.backends.mps.allow_tf32 = True
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

print(f"Device: {device}")

# 고급 데이터 증강 기법
data_transforms = {
    'train_basic': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_advanced': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def load_data(data_dir, batch_size=16, augmentation='basic'):
    """데이터 로드 함수 - 증강 기법 선택 가능"""
    transform_key = f'train_{augmentation}' if augmentation in ['basic', 'advanced'] else 'train_basic'
    
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms[transform_key]),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, 
                                shuffle=True, num_workers=2, drop_last=True)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def create_model(model_name, num_classes, dropout_rate=0.5, use_pretrained=True):
    """다양한 모델 생성 함수 - 드롭아웃 포함"""
    
    if model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT' if use_pretrained else None)
        # 드롭아웃 추가
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    elif model_name == 'resnet101':
        model = models.resnet101(weights='ResNet101_Weights.DEFAULT' if use_pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT' if use_pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier.in_features, num_classes)
        )
        
    elif model_name == 'densenet169':
        model = models.densenet169(weights='DenseNet169_Weights.DEFAULT' if use_pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier.in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT' if use_pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT' if use_pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == 'vit_b_16':  # Vision Transformer
        model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT' if use_pretrained else None)
        model.heads = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT' if use_pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.classifier[2].in_features, num_classes)
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 전이학습을 위해 특성 추출 레이어 고정
    if use_pretrained:
        for param in model.parameters():
            param.requires_grad = False
        
        # 분류 레이어만 학습 가능하게 설정
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'heads'):
            for param in model.heads.parameters():
                param.requires_grad = True
    
    return model

class EarlyStopping:
    """조기 종료 클래스"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def train_model_advanced(model, dataloaders, dataset_sizes, criterion, optimizer, 
                        scheduler, num_epochs=25, early_stopping=None, use_mixup=False):
    """고급 훈련 함수 - 조기종료, MixUp 등"""
    since = time.time()
    
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
        'lr': []
    }
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    def mixup_data(x, y, alpha=0.2):
        """MixUp 데이터 증강"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and use_mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        preds = outputs.argmax(dim=1)
                        corrects = (lam * (preds == targets_a).float() + 
                                  (1 - lam) * (preds == targets_b).float()).sum()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        corrects = torch.sum(preds == labels.data).item()
                    
                    if phase == 'train':
                        loss.backward()
                        # 그래디언트 클리핑
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                if use_mixup and phase == 'train':
                    running_corrects += corrects
                else:
                    running_corrects += corrects
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 기록 저장
            if phase == 'train':
                history['train_acc'].append(epoch_acc)
                history['train_loss'].append(epoch_loss)
                history['lr'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_acc'].append(epoch_acc)
                history['val_loss'].append(epoch_loss)
                
                # 조기 종료 확인
                if early_stopping and early_stopping(epoch_loss):
                    print(f'Early stopping at epoch {epoch}')
                    break
                    
                # 최고 모델 저장
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
        
        print()
        
        # 조기 종료 체크
        if early_stopping and early_stopping.counter >= early_stopping.patience:
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history

def plot_advanced_results(history, title, save_path=None):
    """고급 결과 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 정확도
    ax1.plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    ax1.plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 손실
    ax2.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    ax2.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 학습률
    ax3.plot(history['lr'], 'g-', linewidth=2)
    ax3.set_title(f'{title} - Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 오버피팅 분석
    train_val_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax4.plot(train_val_diff, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title(f'{title} - Overfitting Analysis')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train Acc - Val Acc')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_model_advanced(model, dataloader, class_names, save_path=None):
    """고급 모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 분류 리포트
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return all_preds, all_labels, all_probs

def main():
    # 스크립트 위치 기준 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets_info = {
        'ani': os.path.join(script_dir, 'ani'),
        'CUB200': os.path.join(script_dir, 'CUB200')
    }
    
    # 테스트할 모델들
    models_to_test = [
        'resnet50', 'resnet101', 'densenet121', 'densenet169',
        'efficientnet_b0', 'efficientnet_b3', 'vit_b_16', 'convnext_tiny'
    ]
    
    # 규제 기법 조합
    regularization_configs = [
        {
            'name': 'baseline',
            'dropout': 0.3,
            'weight_decay': 1e-4,
            'augmentation': 'basic',
            'mixup': False,
            'early_stopping': False
        },
        {
            'name': 'light_reg',
            'dropout': 0.5,
            'weight_decay': 1e-3,
            'augmentation': 'advanced',
            'mixup': False,
            'early_stopping': True
        },
        {
            'name': 'heavy_reg',
            'dropout': 0.7,
            'weight_decay': 1e-2,
            'augmentation': 'advanced',
            'mixup': True,
            'early_stopping': True
        }
    ]
    
    results_summary = defaultdict(dict)
    
    for dataset_name, data_dir in datasets_info.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(data_dir):
            print(f"Dataset not found: {data_dir}")
            continue
        
        # 각 정규화 설정에 대해 테스트
        for reg_config in regularization_configs:
            print(f"\n{'-'*40}")
            print(f"Regularization config: {reg_config['name']}")
            print(f"{'-'*40}")
            
            # 데이터 로드
            dataloaders, dataset_sizes, class_names = load_data(
                data_dir, 
                batch_size=12 if 'vit' in str(models_to_test) else 16,
                augmentation=reg_config['augmentation']
            )
            
            num_classes = len(class_names)
            print(f"Classes ({num_classes}): {class_names}")
            print(f"Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
            
            # 선택된 모델들로 테스트 (시간 절약을 위해 상위 4개만)
            selected_models = ['resnet50', 'densenet121', 'efficientnet_b0', 'vit_b_16']
            
            for model_name in selected_models:
                print(f"\n→ Testing {model_name} with {reg_config['name']}")
                
                try:
                    # 모델 생성
                    model = create_model(
                        model_name, 
                        num_classes, 
                        dropout_rate=reg_config['dropout']
                    )
                    model = model.to(device)
                    
                    # 손실함수와 옵티마이저
                    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 라벨 스무딩
                    
                    # 학습 가능한 파라미터만 최적화
                    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                    optimizer = optim.AdamW(
                        trainable_params, 
                        lr=0.001, 
                        weight_decay=reg_config['weight_decay']
                    )
                    
                    # 코사인 어닐링 스케줄러
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
                    
                    # 조기 종료
                    early_stopping = EarlyStopping(patience=5) if reg_config['early_stopping'] else None
                    
                    # 훈련
                    start_time = time.time()
                    trained_model, history = train_model_advanced(
                        model, dataloaders, dataset_sizes, criterion, optimizer, 
                        scheduler, num_epochs=20, early_stopping=early_stopping,
                        use_mixup=reg_config['mixup']
                    )
                    training_time = time.time() - start_time
                    
                    # 결과 저장
                    config_key = f"{reg_config['name']}_{model_name}"
                    results_summary[dataset_name][config_key] = {
                        'best_val_acc': max(history['val_acc']),
                        'final_val_acc': history['val_acc'][-1],
                        'training_time': training_time,
                        'epochs_trained': len(history['val_acc']),
                        'config': reg_config.copy()
                    }
                    
                    # 시각화
                    plot_title = f"{model_name}_{dataset_name}_{reg_config['name']}"
                    plot_advanced_results(history, plot_title)
                    
                    # 모델 평가
                    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
                    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
                    print(f"Training time: {training_time:.1f}s")
                    
                    evaluate_model_advanced(trained_model, dataloaders['val'], class_names)
                    
                    # 메모리 정리
                    del model, trained_model
                    torch.mps.empty_cache() if device.type == 'mps' else None
                    
                except Exception as e:
                    print(f"Error with {model_name}: {str(e)}")
                    continue
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in results_summary.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print("-" * 40)
        
        # 정확도 순으로 정렬
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['best_val_acc'], 
                              reverse=True)
        
        for config_name, metrics in sorted_results:
            print(f"{config_name:25} | "
                  f"Best: {metrics['best_val_acc']:.4f} | "
                  f"Final: {metrics['final_val_acc']:.4f} | "
                  f"Time: {metrics['training_time']:.1f}s | "
                  f"Epochs: {metrics['epochs_trained']}")
    
    # 결과를 JSON으로 저장
    with open('transfer_learning_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: transfer_learning_results.json")

if __name__ == "__main__":
    main()