# ============================================================================
# Phase 3: Classifier Implementation & Training
# EEE598 - Speech Signal Processing Final Project
# Keyword Spotting System
# ============================================================================
# This script implements Phase 3 of the project:
# - Trains k-NN and SVM classifiers
# - Compares MFCC-only vs MFCC+Energy+ZCR features
# - Performs cross-validation for hyperparameter tuning
# - Generates confusion matrices and performance plots
# ============================================================================

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# From Phase 3 specifications in 4-phase plan
# ============================================================================

# Directories
PROJECT_IMPLEMENTATION_DIR = Path('/Users/saifelsaady/Documents/EEE598_Speech_Final_Project/Project Implementation')
FEATURES_DIR = PROJECT_IMPLEMENTATION_DIR / 'features'
RESULTS_DIR = PROJECT_IMPLEMENTATION_DIR / 'results'
PLOTS_DIR = PROJECT_IMPLEMENTATION_DIR / 'plots'

# Create output directories
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Classifier parameters from Phase 3 specs
K_VALUES = [3, 5]  # "k-NN: Use fitcknn with k = 3, 5"
SVM_KERNELS = ['linear', 'rbf']  # "SVM: Use fitcsvm with linear and RBF kernels"
CV_FOLDS = 5  # Standard cross-validation

# ============================================================================
# LOAD FEATURES
# ============================================================================

def load_features():
    """
    Load extracted features from Phase 2.
    
    Returns:
        all_features: Dictionary containing features for all splits
        feature_info: Information about feature dimensions
    """
    print("\n" + "="*70)
    print("LOADING FEATURES")
    print("="*70)
    
    # Load features
    features_path = FEATURES_DIR / 'extracted_features.pkl'
    with open(features_path, 'rb') as f:
        all_features = pickle.load(f)
    
    # Load feature info
    info_path = FEATURES_DIR / 'feature_info.pkl'
    with open(info_path, 'rb') as f:
        feature_info = pickle.load(f)
    
    print(f"✓ Loaded features from: {features_path}")
    print(f"  Training samples: {all_features['train']['features'].shape[0]}")
    print(f"  Validation samples: {all_features['val']['features'].shape[0]}")
    print(f"  Test samples: {all_features['test']['features'].shape[0]}")
    print(f"  Feature dimension: {all_features['train']['features'].shape[1]}")
    
    return all_features, feature_info

# ============================================================================
# FEATURE SET PREPARATION
# ============================================================================

def prepare_feature_sets(all_features):
    """
    Prepare two feature sets as specified:
    Set A: MFCC-only
    Set B: MFCC + Energy + ZCR
    
    From Phase 3 specs: "Feature Variants: Set A: MFCC-only. Set B: MFCC + Energy + ZCR."
    
    Args:
        all_features: Dictionary containing all extracted features
    
    Returns:
        feature_sets: Dictionary with Set A and Set B features
    """
    print("\n" + "="*70)
    print("PREPARING FEATURE SETS")
    print("="*70)
    
    feature_sets = {}
    
    # Based on feature extraction from Phase 2:
    # Features 0-7: Energy and ZCR statistics (8 features)
    # Features 8-79: MFCC and derivatives (72 features)
    
    # Set A: MFCC-only (features 8 onwards)
    feature_sets['Set_A_MFCC_only'] = {
        'train': all_features['train']['features'][:, 8:],
        'val': all_features['val']['features'][:, 8:],
        'test': all_features['test']['features'][:, 8:],
        'description': 'MFCC features only (excluding Energy and ZCR)'
    }
    
    # Set B: MFCC + Energy + ZCR (all features)
    feature_sets['Set_B_MFCC_Energy_ZCR'] = {
        'train': all_features['train']['features'],
        'val': all_features['val']['features'],
        'test': all_features['test']['features'],
        'description': 'MFCC + Energy + ZCR features'
    }
    
    print(f"✓ Set A (MFCC-only): {feature_sets['Set_A_MFCC_only']['train'].shape[1]} features")
    print(f"✓ Set B (MFCC+Energy+ZCR): {feature_sets['Set_B_MFCC_Energy_ZCR']['train'].shape[1]} features")
    
    return feature_sets

# ============================================================================
# CLASSIFIER TRAINING
# ============================================================================

def train_knn_classifier(X_train, y_train, X_val, y_val, k_values=K_VALUES):
    """
    Train k-NN classifier with different k values.
    From Phase 3 specs: "k-NN: Use fitcknn with k = 3, 5 (Euclidean distance)"
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        k_values: List of k values to try
    
    Returns:
        best_model: Best k-NN model
        results: Performance results for each k
    """
    results = []
    best_score = 0
    best_model = None
    best_k = None
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for k in k_values:
        # Train k-NN with Euclidean distance
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_score = knn.score(X_train_scaled, y_train)
        
        # Validation accuracy
        val_score = knn.score(X_val_scaled, y_val)
        
        # Cross-validation score
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, 
                                   cv=StratifiedKFold(n_splits=CV_FOLDS), 
                                   scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        results.append({
            'k': k,
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        })
        
        print(f"  k={k}: Train={train_score:.3f}, Val={val_score:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = knn
            best_k = k
    
    return best_model, results, scaler, best_k

def train_svm_classifier(X_train, y_train, X_val, y_val, kernels=SVM_KERNELS):
    """
    Train SVM classifier with different kernels.
    From Phase 3 specs: "SVM: Use fitcsvm with linear and RBF kernels"
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        kernels: List of kernel types to try
    
    Returns:
        best_model: Best SVM model
        results: Performance results for each kernel
    """
    results = []
    best_score = 0
    best_model = None
    best_kernel = None
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for kernel in kernels:
        # Train SVM
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_score = svm.score(X_train_scaled, y_train)
        
        # Validation accuracy
        val_score = svm.score(X_val_scaled, y_val)
        
        # Cross-validation score
        cv_scores = cross_val_score(svm, X_train_scaled, y_train,
                                   cv=StratifiedKFold(n_splits=CV_FOLDS),
                                   scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        results.append({
            'kernel': kernel,
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        })
        
        print(f"  {kernel}: Train={train_score:.3f}, Val={val_score:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = svm
            best_kernel = kernel
    
    return best_model, results, scaler, best_kernel

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================

def evaluate_model(model, scaler, X_test, y_test, model_name):
    """
    Evaluate model performance and generate confusion matrix.
    From Phase 3 specs: "Track training accuracy, validation accuracy, and confusion matrices"
    
    Args:
        model: Trained model
        scaler: Feature scaler
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
    
    Returns:
        accuracy: Test accuracy
        cm: Confusion matrix
    """
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} - Test Accuracy: {accuracy:.3f}")
    
    return accuracy, cm, y_pred

def plot_confusion_matrix(cm, labels, title, save_path):
    """
    Plot confusion matrix.
    From Phase 3 specs: "Visualize confusion matrices"
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_accuracy_comparison(results_dict, save_path):
    """
    Plot accuracy comparison across models and feature sets.
    From Phase 3 specs: "accuracy vs. k/kernel plots"
    
    Args:
        results_dict: Dictionary containing results for all models
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Filter out non-feature-set keys
    feature_sets = {k: v for k, v in results_dict.items() 
                   if k not in ['label_encoder', 'class_names']}
    
    # k-NN comparison for different k values
    for idx, (feature_set_name, feature_results) in enumerate(feature_sets.items()):
        row = idx // 2
        col = idx % 2
        
        if 'knn' in feature_results:
            knn_results = feature_results['knn']['results']
            k_values = [r['k'] for r in knn_results]
            val_accs = [r['val_accuracy'] for r in knn_results]
            train_accs = [r['train_accuracy'] for r in knn_results]
            
            axes[row, col].plot(k_values, train_accs, 'o-', label='Training', linewidth=2)
            axes[row, col].plot(k_values, val_accs, 's-', label='Validation', linewidth=2)
            axes[row, col].set_xlabel('k (number of neighbors)')
            axes[row, col].set_ylabel('Accuracy')
            axes[row, col].set_title(f'k-NN Performance - {feature_set_name}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models(feature_sets, all_features):
    """
    Train all model combinations and evaluate performance.
    
    Args:
        feature_sets: Dictionary with Set A and Set B features
        all_features: Original features dictionary with labels
    
    Returns:
        all_results: Complete results for all experiments
    """
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS")
    print("="*70)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(all_features['train']['labels'])
    y_val = label_encoder.transform(all_features['val']['labels'])
    y_test = label_encoder.transform(all_features['test']['labels'])
    
    # Store all results
    all_results = {}
    
    # Train for each feature set
    for feature_set_name, feature_set in feature_sets.items():
        print(f"\n--- {feature_set_name} ---")
        print(f"Description: {feature_set['description']}")
        
        X_train = feature_set['train']
        X_val = feature_set['val']
        X_test = feature_set['test']
        
        results = {}
        
        # Train k-NN
        print("\nTraining k-NN:")
        knn_model, knn_results, knn_scaler, best_k = train_knn_classifier(
            X_train, y_train, X_val, y_val
        )
        
        # Evaluate k-NN on test set
        knn_accuracy, knn_cm, knn_pred = evaluate_model(
            knn_model, knn_scaler, X_test, y_test, 
            f"k-NN (k={best_k})"
        )
        
        results['knn'] = {
            'model': knn_model,
            'scaler': knn_scaler,
            'results': knn_results,
            'test_accuracy': knn_accuracy,
            'confusion_matrix': knn_cm,
            'predictions': knn_pred,
            'best_k': best_k
        }
        
        # Train SVM
        print("\nTraining SVM:")
        svm_model, svm_results, svm_scaler, best_kernel = train_svm_classifier(
            X_train, y_train, X_val, y_val
        )
        
        # Evaluate SVM on test set
        svm_accuracy, svm_cm, svm_pred = evaluate_model(
            svm_model, svm_scaler, X_test, y_test,
            f"SVM ({best_kernel})"
        )
        
        results['svm'] = {
            'model': svm_model,
            'scaler': svm_scaler,
            'results': svm_results,
            'test_accuracy': svm_accuracy,
            'confusion_matrix': svm_cm,
            'predictions': svm_pred,
            'best_kernel': best_kernel
        }
        
        all_results[feature_set_name] = results
    
    # Store labels for plotting
    all_results['label_encoder'] = label_encoder
    all_results['class_names'] = label_encoder.classes_
    
    return all_results

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_results(all_results):
    """
    Verify that results meet expectations.
    From Phase 3 specs: 
    - "Ensure accuracy improves with added Energy/ZCR"
    - "Confirm confusion matrices show high diagonal dominance"
    
    Args:
        all_results: Complete results dictionary
    """
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Compare Set A (MFCC-only) vs Set B (MFCC+Energy+ZCR)
    set_a_knn = all_results['Set_A_MFCC_only']['knn']['test_accuracy']
    set_b_knn = all_results['Set_B_MFCC_Energy_ZCR']['knn']['test_accuracy']
    
    set_a_svm = all_results['Set_A_MFCC_only']['svm']['test_accuracy']
    set_b_svm = all_results['Set_B_MFCC_Energy_ZCR']['svm']['test_accuracy']
    
    print("\nAccuracy Comparison:")
    print(f"k-NN: Set A (MFCC-only) = {set_a_knn:.3f}")
    print(f"k-NN: Set B (MFCC+Energy+ZCR) = {set_b_knn:.3f}")
    print(f"k-NN Improvement: {(set_b_knn - set_a_knn)*100:.1f}%")
    
    print(f"\nSVM: Set A (MFCC-only) = {set_a_svm:.3f}")
    print(f"SVM: Set B (MFCC+Energy+ZCR) = {set_b_svm:.3f}")
    print(f"SVM Improvement: {(set_b_svm - set_a_svm)*100:.1f}%")
    
    # Check diagonal dominance in confusion matrices
    print("\nConfusion Matrix Diagonal Dominance:")
    for feature_set_name in ['Set_A_MFCC_only', 'Set_B_MFCC_Energy_ZCR']:
        for model_type in ['knn', 'svm']:
            cm = all_results[feature_set_name][model_type]['confusion_matrix']
            diagonal_sum = np.trace(cm)
            total_sum = np.sum(cm)
            dominance = diagonal_sum / total_sum
            print(f"  {feature_set_name} - {model_type}: {dominance:.3f}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy (Best Model):")
    best_model_name = 'Set_B_MFCC_Energy_ZCR'
    best_model_type = 'svm' if set_b_svm > set_b_knn else 'knn'
    cm = all_results[best_model_name][best_model_type]['confusion_matrix']
    class_names = all_results['class_names']
    
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        print(f"  {class_name}: {class_accuracy:.3f}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Execute Phase 3: Classifier Training Pipeline
    """
    print("="*70)
    print("PHASE 3: CLASSIFIER IMPLEMENTATION & TRAINING")
    print("EEE598 Keyword Spotting Final Project")
    print("="*70)
    
    # Load features
    all_features, feature_info = load_features()
    
    # Prepare feature sets
    feature_sets = prepare_feature_sets(all_features)
    
    # Train all models
    all_results = train_all_models(feature_sets, all_features)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Plot confusion matrices for best models
    class_names = all_results['class_names']
    
    for feature_set_name in ['Set_A_MFCC_only', 'Set_B_MFCC_Energy_ZCR']:
        for model_type in ['knn', 'svm']:
            cm = all_results[feature_set_name][model_type]['confusion_matrix']
            title = f"{feature_set_name} - {model_type.upper()}"
            save_path = PLOTS_DIR / f"confusion_matrix_{feature_set_name}_{model_type}.png"
            plot_confusion_matrix(cm, class_names, title, save_path)
            print(f"✓ Saved: {save_path.name}")
    
    # Plot accuracy comparisons
    accuracy_plot_path = PLOTS_DIR / "accuracy_comparison.png"
    plot_accuracy_comparison(all_results, accuracy_plot_path)
    print(f"✓ Saved: {accuracy_plot_path.name}")
    
    # Verify results
    verify_results(all_results)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save trained models and results
    results_path = RESULTS_DIR / 'classifier_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✓ Saved results to: {results_path}")
    
    # Save summary report
    summary_path = RESULTS_DIR / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 3: CLASSIFIER TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for feature_set_name in ['Set_A_MFCC_only', 'Set_B_MFCC_Energy_ZCR']:
            f.write(f"\n{feature_set_name}:\n")
            f.write("-"*40 + "\n")
            
            knn_acc = all_results[feature_set_name]['knn']['test_accuracy']
            svm_acc = all_results[feature_set_name]['svm']['test_accuracy']
            
            f.write(f"k-NN (k={all_results[feature_set_name]['knn']['best_k']}): {knn_acc:.3f}\n")
            f.write(f"SVM ({all_results[feature_set_name]['svm']['best_kernel']}): {svm_acc:.3f}\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE!")
    print("="*70)
    print("\n📊 Best Results:")
    
    best_accuracy = 0
    best_config = ""
    
    for feature_set_name in ['Set_A_MFCC_only', 'Set_B_MFCC_Energy_ZCR']:
        for model_type in ['knn', 'svm']:
            acc = all_results[feature_set_name][model_type]['test_accuracy']
            if acc > best_accuracy:
                best_accuracy = acc
                best_config = f"{feature_set_name} - {model_type.upper()}"
    
    print(f"  Best Configuration: {best_config}")
    print(f"  Test Accuracy: {best_accuracy:.3f}")
    
    # Check against proposal target
    print(f"\n  Target from proposal: 75-90%")
    print(f"  Achieved: {best_accuracy*100:.1f}%")
    if best_accuracy >= 0.75:
        print("  ✓ Target met!")
    
    print("\n✓ Ready for Phase 4: Testing & Report Writing")
    print("="*70 + "\n")

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()