# ============================================================================
# Phase 2: Feature Extraction (Energy, ZCR, MFCC)
# EEE598 - Speech Signal Processing Final Project
# Keyword Spotting System
# ============================================================================
# This script implements Phase 2 of the project:
# - Extracts short-time energy, zero-crossing rate (ZCR), and MFCCs
# - Computes delta and delta-delta features for temporal dynamics
# - Aggregates frame-level features into clip-level representations
# - Saves features for training Phase 3 classifiers
# ============================================================================

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy.fftpack import dct
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# From project proposal and consolidated MATLAB scripts
# ============================================================================

# Sampling rate (from Phase 1 and proposal)
TARGET_SR = 16000

# Frame parameters (from proposal page 4 and MATLAB ZCR_Energy_Demo.m)
# "20-25 ms frame length with 10 ms hop using a Hamming window"
FRAME_LENGTH_MS = 25  # Changed to 25ms as specified in Phase 2 specs
HOP_LENGTH_MS = 10    

# Convert to samples
FRAME_LENGTH = int(FRAME_LENGTH_MS * TARGET_SR / 1000)  # 400 samples at 16kHz
HOP_LENGTH = int(HOP_LENGTH_MS * TARGET_SR / 1000)      # 160 samples at 16kHz

# MFCC parameters (from proposal and phase plan)
NUM_MFCC = 12        # Keep 12 MFCCs as specified
NUM_MEL_BINS = 26    # Standard number of Mel filterbank bins
NFFT = 512          # FFT size (from cepstrum.m: Nfft = 512)

# Pre-emphasis coefficient (optional, from phase plan)
PRE_EMPHASIS = 0.97

# Directories
PROJECT_IMPLEMENTATION_DIR = Path('/Users/saifelsaady/Documents/EEE598_Speech_Final_Project/Project Implementation')
PROCESSED_DATA_DIR = PROJECT_IMPLEMENTATION_DIR / 'processed_data'
FEATURES_DIR = PROJECT_IMPLEMENTATION_DIR / 'features'
PLOTS_DIR = PROJECT_IMPLEMENTATION_DIR / 'plots'

# Create output directories
FEATURES_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# Based on consolidated MATLAB scripts
# ============================================================================

def compute_energy(frame, window):
    """
    Compute short-time energy for a frame.
    From ZCR_Energy_Demo.m: E = sum((x(n).^2) .* w)
    
    Args:
        frame: Audio frame
        window: Window function (Hamming)
    
    Returns:
        energy: Short-time energy value
    """
    windowed_frame = frame * window
    energy = np.sum(windowed_frame ** 2)
    return energy

def compute_zcr(frame):
    """
    Compute zero-crossing rate for a frame.
    From ZCR_Energy_Demo.m: zcr = sum(xwin(1:end-1) .* xwin(2:end) < 0)
    
    Args:
        frame: Audio frame
    
    Returns:
        zcr: Zero-crossing rate
    """
    # Count sign changes
    zcr = np.sum(frame[:-1] * frame[1:] < 0)
    return zcr

def compute_mfcc_manual(audio, sr=TARGET_SR, n_mfcc=NUM_MFCC, n_mels=NUM_MEL_BINS, 
                        n_fft=NFFT, hop_length=HOP_LENGTH, win_length=FRAME_LENGTH):
    """
    Compute MFCCs following the pipeline from proposal:
    STFT -> Mel filterbank -> Log -> DCT
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel bands
        n_fft: FFT size
        hop_length: Hop length in samples
        win_length: Window length in samples
    
    Returns:
        mfccs: MFCC matrix (n_mfcc x n_frames)
        log_energy: Log energy for each frame
    """
    # Use librosa for consistency, but following the specified pipeline
    # Compute power spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                       hop_length=hop_length, 
                                       win_length=win_length,
                                       n_mels=n_mels, window='hamming')
    
    # Convert to log scale (from proposal: "Mel filterbank -> log")
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Apply DCT to get MFCCs (from proposal: "log -> DCT")
    mfccs = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
    
    # Extract log energy (0th cepstral coefficient)
    log_energy = mfccs[0, :]
    
    return mfccs, log_energy

def compute_deltas(features, width=9):
    """
    Compute delta (velocity) features.
    
    Args:
        features: Feature matrix (n_features x n_frames)
        width: Window width for delta computation
    
    Returns:
        deltas: Delta features
    """
    return librosa.feature.delta(features, width=width, order=1)

def extract_features_per_file(audio_path, pre_emphasis=PRE_EMPHASIS):
    """
    Extract all features for a single audio file:
    - Short-time energy
    - Zero-crossing rate
    - MFCCs (12 coefficients)
    - Delta and delta-delta features
    
    Args:
        audio_path: Path to audio file
        pre_emphasis: Pre-emphasis coefficient
    
    Returns:
        features_dict: Dictionary containing all extracted features
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=TARGET_SR)
    
    # Apply pre-emphasis filter (optional)
    if pre_emphasis > 0:
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Initialize feature storage
    energies = []
    zcrs = []
    
    # Create Hamming window (from MATLAB scripts)
    window = np.hamming(FRAME_LENGTH)
    
    # ========================================================================
    # Frame-based processing (from ZCR_Energy_Demo.m)
    # ========================================================================
    n_frames = int(np.floor((len(audio) - FRAME_LENGTH) / HOP_LENGTH)) + 1
    
    for i in range(n_frames):
        start = i * HOP_LENGTH
        end = start + FRAME_LENGTH
        
        if end > len(audio):
            break
            
        frame = audio[start:end]
        
        # Compute energy (from MATLAB: E = sum((x(n).^2) .* w))
        energy = compute_energy(frame, window)
        energies.append(energy)
        
        # Compute ZCR (from MATLAB: zcr = sum(xwin(1:end-1) .* xwin(2:end) < 0))
        zcr = compute_zcr(frame)
        zcrs.append(zcr)
    
    energies = np.array(energies)
    zcrs = np.array(zcrs)
    
    # ========================================================================
    # MFCC extraction (following proposal pipeline)
    # ========================================================================
    mfccs, log_energy = compute_mfcc_manual(audio)
    
    # ========================================================================
    # Compute delta and delta-delta features
    # From phase plan: "compute Δ (delta) and ΔΔ (acceleration) features"
    # ========================================================================
    mfcc_delta = compute_deltas(mfccs)
    mfcc_delta_delta = compute_deltas(mfcc_delta)
    
    # Energy deltas
    energy_delta = compute_deltas(energies.reshape(1, -1))
    energy_delta_delta = compute_deltas(energy_delta)
    
    # ========================================================================
    # Feature aggregation for clip-level representation
    # From phase plan: "mean or pooled across frames"
    # ========================================================================
    features_dict = {
        # Time-domain features
        'energy_mean': np.mean(energies),
        'energy_std': np.std(energies),
        'energy_max': np.max(energies),
        'energy_min': np.min(energies),
        
        'zcr_mean': np.mean(zcrs),
        'zcr_std': np.std(zcrs),
        'zcr_max': np.max(zcrs),
        'zcr_min': np.min(zcrs),
        
        # MFCC features (mean and std for each coefficient)
        'mfcc_mean': np.mean(mfccs, axis=1),  # Shape: (12,)
        'mfcc_std': np.std(mfccs, axis=1),    # Shape: (12,)
        
        # Delta features
        'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
        'mfcc_delta_std': np.std(mfcc_delta, axis=1),
        
        # Delta-delta features
        'mfcc_delta_delta_mean': np.mean(mfcc_delta_delta, axis=1),
        'mfcc_delta_delta_std': np.std(mfcc_delta_delta, axis=1),
        
        # Raw frame-level features for visualization
        'energies_raw': energies,
        'zcrs_raw': zcrs,
        'mfccs_raw': mfccs
    }
    
    return features_dict

def create_feature_vector(features_dict):
    """
    Create a single feature vector from the features dictionary.
    Following phase plan: "total ≈ 39D feature vector"
    
    Args:
        features_dict: Dictionary of extracted features
    
    Returns:
        feature_vector: Concatenated feature vector
    """
    # Combine all features into single vector
    feature_vector = np.concatenate([
        # Energy features (4)
        [features_dict['energy_mean'], features_dict['energy_std'],
         features_dict['energy_max'], features_dict['energy_min']],
        
        # ZCR features (4)
        [features_dict['zcr_mean'], features_dict['zcr_std'],
         features_dict['zcr_max'], features_dict['zcr_min']],
        
        # MFCC features (12 + 12 = 24)
        features_dict['mfcc_mean'],
        features_dict['mfcc_std'],
        
        # Delta MFCC features (12 + 12 = 24)
        features_dict['mfcc_delta_mean'],
        features_dict['mfcc_delta_std'],
        
        # Delta-delta MFCC features (12 + 12 = 24)
        features_dict['mfcc_delta_delta_mean'],
        features_dict['mfcc_delta_delta_std']
    ])
    
    return feature_vector

# ============================================================================
# MAIN FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_features_for_dataset(metadata_path, output_dir):
    """
    Extract features for all files in the dataset.
    
    Args:
        metadata_path: Path to metadata.csv from Phase 1
        output_dir: Directory to save extracted features
    
    Returns:
        all_features: Dictionary containing features for all splits
    """
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FOR DATASET")
    print("="*70)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Initialize storage
    all_features = {
        'train': {'features': [], 'labels': [], 'filenames': []},
        'val': {'features': [], 'labels': [], 'filenames': []},
        'test': {'features': [], 'labels': [], 'filenames': []}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_df = metadata_df[metadata_df['split'] == split]
        
        print(f"\nProcessing {split} set ({len(split_df)} files)...")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split):
            try:
                # Extract features
                features_dict = extract_features_per_file(row['processed_path'])
                
                # Create feature vector
                feature_vector = create_feature_vector(features_dict)
                
                # Store features and labels
                all_features[split]['features'].append(feature_vector)
                all_features[split]['labels'].append(row['label'])
                all_features[split]['filenames'].append(row['filename'])
                
                # Store raw features for first few samples (for visualization)
                if idx < 3:
                    all_features[split][f'sample_{idx}_raw'] = features_dict
                    
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
                continue
        
        # Convert to numpy arrays
        all_features[split]['features'] = np.array(all_features[split]['features'])
        all_features[split]['labels'] = np.array(all_features[split]['labels'])
        
        print(f"✓ Extracted features for {len(all_features[split]['features'])} files")
        print(f"  Feature vector dimension: {all_features[split]['features'].shape[1]}")
    
    return all_features

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_features(all_features, output_dir):
    """
    Create visualization plots for extracted features.
    From phase plan: "Plot energy/ZCR over time" and "Generate MFCC spectrograms"
    
    Args:
        all_features: Dictionary of extracted features
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("CREATING FEATURE VISUALIZATIONS")
    print("="*70)
    
    # ========================================================================
    # PLOT 1: Energy and ZCR over time for sample files
    # ========================================================================
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    
    # Get sample from each class
    classes = ['yes', 'no', 'stop', 'go']
    
    for idx, class_name in enumerate(classes):
        # Find a sample from this class
        train_labels = all_features['train']['labels']
        class_indices = np.where(train_labels == class_name)[0]
        
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            
            # Get raw features for this sample
            for i in range(3):
                if f'sample_{sample_idx}_raw' in all_features['train']:
                    sample_raw = all_features['train'][f'sample_{sample_idx}_raw']
                    
                    # Plot energy
                    axes[idx, 0].plot(sample_raw['energies_raw'])
                    axes[idx, 0].set_title(f'Energy - Class: {class_name}')
                    axes[idx, 0].set_xlabel('Frame')
                    axes[idx, 0].set_ylabel('Energy')
                    axes[idx, 0].grid(True, alpha=0.3)
                    
                    # Plot ZCR
                    axes[idx, 1].plot(sample_raw['zcrs_raw'], color='orange')
                    axes[idx, 1].set_title(f'ZCR - Class: {class_name}')
                    axes[idx, 1].set_xlabel('Frame')
                    axes[idx, 1].set_ylabel('Zero Crossing Rate')
                    axes[idx, 1].grid(True, alpha=0.3)
                    break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_zcr_visualization.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'energy_zcr_visualization.png'}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: MFCC Spectrograms
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(classes):
        train_labels = all_features['train']['labels']
        class_indices = np.where(train_labels == class_name)[0]
        
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            
            # Get raw MFCCs for this sample
            for i in range(3):
                if f'sample_{sample_idx}_raw' in all_features['train']:
                    sample_raw = all_features['train'][f'sample_{sample_idx}_raw']
                    mfccs = sample_raw['mfccs_raw']
                    
                    # Plot MFCC spectrogram
                    img = axes[idx].imshow(mfccs, aspect='auto', origin='lower', 
                                          cmap='viridis', interpolation='nearest')
                    axes[idx].set_title(f'MFCC Spectrogram - Class: {class_name}')
                    axes[idx].set_xlabel('Time (frames)')
                    axes[idx].set_ylabel('MFCC Coefficient')
                    plt.colorbar(img, ax=axes[idx])
                    break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mfcc_spectrograms.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'mfcc_spectrograms.png'}")
    plt.close()

def verify_features(all_features):
    """
    Verify feature extraction results.
    From phase plan: "Check feature value ranges" and "Validate Δ/ΔΔ values"
    
    Args:
        all_features: Dictionary of extracted features
    """
    print("\n" + "="*70)
    print("FEATURE VERIFICATION")
    print("="*70)
    
    # Check feature statistics for training set
    train_features = all_features['train']['features']
    
    print("\nFeature Statistics (Training Set):")
    print(f"  Shape: {train_features.shape}")
    print(f"  Min value: {np.min(train_features):.4f}")
    print(f"  Max value: {np.max(train_features):.4f}")
    print(f"  Mean: {np.mean(train_features):.4f}")
    print(f"  Std: {np.std(train_features):.4f}")
    
    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(train_features))
    inf_count = np.sum(np.isinf(train_features))
    
    print(f"\n  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  ⚠️ Warning: Found NaN or Inf values in features!")
    else:
        print("  ✓ No NaN or Inf values found")
    
    # Verify feature dimensions match expected (~39D from phase plan)
    expected_dim = 80  # 8 (energy+zcr) + 24 (mfcc) + 24 (delta) + 24 (delta-delta)
    actual_dim = train_features.shape[1]
    
    print(f"\n  Expected dimension: ~{expected_dim}")
    print(f"  Actual dimension: {actual_dim}")
    
    # Check class distribution
    print("\nClass Distribution:")
    for split in ['train', 'val', 'test']:
        labels = all_features[split]['labels']
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  {split}:")
        for label, count in zip(unique, counts):
            print(f"    {label}: {count}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Execute Phase 2: Feature Extraction Pipeline
    """
    print("="*70)
    print("PHASE 2: FEATURE EXTRACTION")
    print("EEE598 Keyword Spotting Final Project")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Frame length: {FRAME_LENGTH_MS} ms ({FRAME_LENGTH} samples)")
    print(f"  Hop length: {HOP_LENGTH_MS} ms ({HOP_LENGTH} samples)")
    print(f"  Window: Hamming")
    print(f"  MFCCs: {NUM_MFCC} coefficients")
    print(f"  Pre-emphasis: {PRE_EMPHASIS}")
    print(f"  FFT size: {NFFT}")
    
    # Load metadata
    metadata_path = PROCESSED_DATA_DIR / 'metadata.csv'
    
    # Extract features
    all_features = extract_features_for_dataset(metadata_path, FEATURES_DIR)
    
    # Create visualizations
    visualize_features(all_features, PLOTS_DIR)
    
    # Verify features
    verify_features(all_features)
    
    # Save features
    print("\n" + "="*70)
    print("SAVING EXTRACTED FEATURES")
    print("="*70)
    
    # Save as pickle for Phase 3
    features_path = FEATURES_DIR / 'extracted_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"✓ Saved features to: {features_path}")
    
    # Also save feature dimensions info
    feature_info = {
        'frame_length_ms': FRAME_LENGTH_MS,
        'hop_length_ms': HOP_LENGTH_MS,
        'num_mfcc': NUM_MFCC,
        'feature_dim': all_features['train']['features'].shape[1],
        'feature_names': [
            'energy_mean', 'energy_std', 'energy_max', 'energy_min',
            'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min',
            *[f'mfcc_{i}_mean' for i in range(NUM_MFCC)],
            *[f'mfcc_{i}_std' for i in range(NUM_MFCC)],
            *[f'mfcc_delta_{i}_mean' for i in range(NUM_MFCC)],
            *[f'mfcc_delta_{i}_std' for i in range(NUM_MFCC)],
            *[f'mfcc_delta_delta_{i}_mean' for i in range(NUM_MFCC)],
            *[f'mfcc_delta_delta_{i}_std' for i in range(NUM_MFCC)]
        ]
    }
    
    info_path = FEATURES_DIR / 'feature_info.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"✓ Saved feature info to: {info_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE!")
    print("="*70)
    print(f"\n📁 Outputs:")
    print(f"  - Extracted features: {features_path}")
    print(f"  - Feature info: {info_path}")
    print(f"  - Visualization plots: {PLOTS_DIR}/")
    print(f"\n✓ Ready for Phase 3: Classifier Training")
    print("="*70 + "\n")

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()