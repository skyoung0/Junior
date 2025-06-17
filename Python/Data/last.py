import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with GPU configuration
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, DenseNet121, EfficientNetB0, EfficientNetB3,
    VGG16, VGG19, MobileNetV2, InceptionV3, Xception
)
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Flatten, GlobalAveragePooling2D, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras import mixed_precision

# GPU Configuration
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set up mixed precision for faster training (updated API)
            try:
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print(f'Mixed precision policy: {policy.name}')
                print(f'Compute dtype: {policy.compute_dtype}')
                print(f'Variable dtype: {policy.variable_dtype}')
            except Exception as e:
                print(f"Mixed precision setup failed: {e}")
                print("Continuing without mixed precision...")
            
            print(f"Available GPUs: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu}")
                
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")
    
    return len(gpus) > 0

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TransferLearningExperiment:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = {}
        self.histories = {}
        self.gpu_available = setup_gpu()
        
        # Model configurations
        self.models_config = {
            'ResNet50': ResNet50,
            'DenseNet121': DenseNet121,
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB3': EfficientNetB3,
            'VGG16': VGG16,
            'VGG19': VGG19,
            'MobileNetV2': MobileNetV2,
            'InceptionV3': InceptionV3,
            'Xception': Xception
        }
        
        # Dataset configurations
        self.datasets_config = {
            'ANI': {
                'train_dir': os.path.join(base_path, 'ani', 'train'),
                'val_dir': os.path.join(base_path, 'ani', 'val'),
                'num_classes': 7,
                'batch_size': 64 if self.gpu_available else 16  # Larger batch for GPU
            },
            'CUB200': {
                'train_dir': os.path.join(base_path, 'CUB200', 'train'),
                'val_dir': os.path.join(base_path, 'CUB200', 'val'),
                'num_classes': 200,
                'batch_size': 32 if self.gpu_available else 8   # Adjust for GPU memory
            }
        }
        
        # Architecture configurations with regularization
        self.architectures = {
            'Light': {
                'layers': [
                    ('dense', 256, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.3, None),
                    ('dense', 'num_classes', 'softmax')
                ],
                'regularizer': None
            },
            'Medium': {
                'layers': [
                    ('dense', 512, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.5, None),
                    ('dense', 256, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.3, None),
                    ('dense', 'num_classes', 'softmax')
                ],
                'regularizer': l2(0.01)
            },
            'Heavy': {
                'layers': [
                    ('dense', 1024, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.6, None),
                    ('dense', 512, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.4, None),
                    ('dense', 256, 'relu'),
                    ('batch_norm', None, None),
                    ('dropout', 0.2, None),
                    ('dense', 'num_classes', 'softmax')
                ],
                'regularizer': l1_l2(l1=0.01, l2=0.01)
            }
        }
        
        # Regularization strategies
        self.regularization_configs = {
            'None': {'regularizer': None, 'dropout_rate': 0.0},
            'Light': {'regularizer': l2(0.001), 'dropout_rate': 0.3},
            'Heavy': {'regularizer': l1_l2(l1=0.01, l2=0.01), 'dropout_rate': 0.5}
        }

    def create_data_generators(self, dataset_name):
        """Create data generators with advanced augmentation"""
        config = self.datasets_config[dataset_name]
        
        # Advanced data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
        
        # Only preprocessing for validation
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        train_generator = train_datagen.flow_from_directory(
            config['train_dir'],
            target_size=(224, 224),
            batch_size=config['batch_size'],
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            config['val_dir'],
            target_size=(224, 224),
            batch_size=config['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator

    def create_model(self, model_name, architecture, dataset_name, regularization):
        """Create transfer learning model with specified configuration"""
        config = self.datasets_config[dataset_name]
        num_classes = config['num_classes']
        
        # Get base model
        base_model = self.models_config[model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Create new model
        model = Sequential([base_model])
        model.add(GlobalAveragePooling2D())
        
        # Add architecture layers
        arch_config = self.architectures[architecture]
        reg_config = self.regularization_configs[regularization]
        
        for layer_type, param1, param2 in arch_config['layers']:
            if layer_type == 'dense':
                units = param1 if param1 != 'num_classes' else num_classes
                activation = param2
                
                if activation == 'softmax':
                    # Output layer - ensure float32 for numerical stability
                    model.add(Dense(units, activation=activation, dtype=tf.float32))
                else:
                    model.add(Dense(
                        units, 
                        activation=activation,
                        kernel_regularizer=reg_config['regularizer']
                    ))
            elif layer_type == 'dropout':
                model.add(Dropout(max(param1, reg_config['dropout_rate'])))
            elif layer_type == 'batch_norm':
                model.add(BatchNormalization())
        
        return model

    def get_callbacks(self, model_name, dataset_name, architecture, regularization):
        """Get training callbacks"""
        checkpoint_path = f"models/{model_name}_{dataset_name}_{architecture}_{regularization}.h5"
        os.makedirs("models", exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks

    def train_model(self, model_name, dataset_name, architecture='Medium', regularization='Light', epochs=50):
        """Train a single model configuration"""
        print(f"\n{'='*60}")
        print(f"Training: {model_name} | {dataset_name} | {architecture} | {regularization}")
        print(f"{'='*60}")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(dataset_name)
        
        # Create model
        model = self.create_model(model_name, architecture, dataset_name, regularization)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get callbacks
        callbacks = self.get_callbacks(model_name, dataset_name, architecture, regularization)
        
        print(f"Model summary:")
        model.summary()
        
        # Phase 1: Train classifier only
        print("\n--- Phase 1: Training classifier only ---")
        history1 = model.fit(
            train_gen,
            epochs=min(10, epochs//2),
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune entire model
        print("\n--- Phase 2: Fine-tuning entire model ---")
        model.layers[0].trainable = True  # Unfreeze base model
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            epochs=epochs - min(10, epochs//2),
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        history = self.combine_histories(history1, history2)
        
        # Store results
        key = f"{model_name}_{dataset_name}_{architecture}_{regularization}"
        self.histories[key] = history
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        self.results[key] = {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'model_name': model_name,
            'dataset': dataset_name,
            'architecture': architecture,
            'regularization': regularization
        }
        
        print(f"Final validation accuracy: {val_acc:.4f}")
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
        
        return history

    def combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        return combined

    def run_comprehensive_experiment(self):
        """Run comprehensive transfer learning experiments"""
        # Select models for experiment (reduce for faster execution)
        selected_models = ['ResNet50', 'DenseNet121', 'EfficientNetB0', 'MobileNetV2']
        selected_architectures = ['Light', 'Medium', 'Heavy']
        selected_regularizations = ['None', 'Light', 'Heavy']
        
        total_experiments = len(selected_models) * len(self.datasets_config) * len(selected_architectures) * len(selected_regularizations)
        current_exp = 0
        
        print(f"Starting {total_experiments} experiments...")
        
        for model_name in selected_models:
            for dataset_name in self.datasets_config.keys():
                for architecture in selected_architectures:
                    for regularization in selected_regularizations:
                        current_exp += 1
                        print(f"\nExperiment {current_exp}/{total_experiments}")
                        
                        try:
                            self.train_model(
                                model_name=model_name,
                                dataset_name=dataset_name,
                                architecture=architecture,
                                regularization=regularization,
                                epochs=30  # Reduced for faster execution
                            )
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue

    def plot_training_histories(self):
        """Plot training histories for all experiments"""
        if not self.histories:
            print("No training histories to plot")
            return
        
        # Create subplots
        n_plots = len(self.histories)
        cols = 4
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(2, -1)
        
        plot_idx = 0
        for key, history in self.histories.items():
            if plot_idx >= n_plots:
                break
                
            row = (plot_idx // cols) * 2
            col = plot_idx % cols
            
            # Plot accuracy
            axes[row, col].plot(history['accuracy'], label='Train Accuracy')
            axes[row, col].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[row, col].set_title(f'{key}\nAccuracy')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Accuracy')
            axes[row, col].legend()
            axes[row, col].grid(True)
            
            # Plot loss
            axes[row + 1, col].plot(history['loss'], label='Train Loss')
            axes[row + 1, col].plot(history['val_loss'], label='Validation Loss')
            axes[row + 1, col].set_title(f'{key}\nLoss')
            axes[row + 1, col].set_xlabel('Epoch')
            axes[row + 1, col].set_ylabel('Loss')
            axes[row + 1, col].legend()
            axes[row + 1, col].grid(True)
            
            plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, rows * cols):
            row = (i // cols) * 2
            col = i % cols
            axes[row, col].set_visible(False)
            axes[row + 1, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('training_histories.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_results_summary(self):
        """Create summary of all experimental results"""
        if not self.results:
            print("No results to summarize")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'experiment'}, inplace=True)
        
        # Create summary statistics
        print("=== EXPERIMENT RESULTS SUMMARY ===")
        print(f"Total experiments completed: {len(df)}")
        print(f"Average validation accuracy: {df['val_accuracy'].mean():.4f}")
        print(f"Best validation accuracy: {df['val_accuracy'].max():.4f}")
        print(f"Worst validation accuracy: {df['val_accuracy'].min():.4f}")
        
        # Top 10 results
        print("\n=== TOP 10 RESULTS ===")
        top_results = df.nlargest(10, 'val_accuracy')
        for idx, row in top_results.iterrows():
            print(f"{row['experiment']}: {row['val_accuracy']:.4f}")
        
        # Create comparison plots
        self.plot_results_comparison(df)
        
        return df

    def plot_results_comparison(self, df):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_comparison = df.groupby('model_name')['val_accuracy'].mean().sort_values(ascending=False)
        axes[0, 0].bar(model_comparison.index, model_comparison.values)
        axes[0, 0].set_title('Average Accuracy by Model')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Dataset comparison
        dataset_comparison = df.groupby('dataset')['val_accuracy'].mean().sort_values(ascending=False)
        axes[0, 1].bar(dataset_comparison.index, dataset_comparison.values)
        axes[0, 1].set_title('Average Accuracy by Dataset')
        axes[0, 1].set_ylabel('Validation Accuracy')
        
        # Architecture comparison
        arch_comparison = df.groupby('architecture')['val_accuracy'].mean().sort_values(ascending=False)
        axes[1, 0].bar(arch_comparison.index, arch_comparison.values)
        axes[1, 0].set_title('Average Accuracy by Architecture')
        axes[1, 0].set_ylabel('Validation Accuracy')
        
        # Regularization comparison
        reg_comparison = df.groupby('regularization')['val_accuracy'].mean().sort_values(ascending=False)
        axes[1, 1].bar(reg_comparison.index, reg_comparison.values)
        axes[1, 1].set_title('Average Accuracy by Regularization')
        axes[1, 1].set_ylabel('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize experiment with your data path
    base_path = r"C:\Users\ayaan\Documents\Git\Junior\Python\Data"
    experiment = TransferLearningExperiment(base_path)
    
    # Check if data directories exist
    print("Checking data directories...")
    for dataset_name, config in experiment.datasets_config.items():
        train_dir = config['train_dir']
        val_dir = config['val_dir']
        print(f"{dataset_name}:")
        print(f"  Train dir exists: {os.path.exists(train_dir)} - {train_dir}")
        print(f"  Val dir exists: {os.path.exists(val_dir)} - {val_dir}")
        
        if os.path.exists(train_dir):
            subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"  Classes found: {len(subdirs)} - {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
    
    # Run a single test experiment first
    print("\n" + "="*50)
    print("Running test experiment...")
    experiment.train_model('ResNet50', 'ANI', 'Medium', 'Light', epochs=5)
    
    # Uncomment below to run full experiments
    # experiment.run_comprehensive_experiment()
    
    # Plot results
    experiment.plot_training_histories()
    
    # Create summary
    results_df = experiment.create_results_summary()
    
    print("\nExperiment completed!")
    print("Check the generated plots and model files in the 'models' directory.")