# ============================================================================
# Phase 1: Dataset Setup & Preprocessing
# EEE598 - Speech Signal Processing Final Project
# Keyword Spotting System
# ============================================================================
# This script implements Phase 1 of the project:
# - Collects audio files from the Google Speech Commands v2 dataset
# - Resamples to 16 kHz
# - Normalizes amplitude
# - Trims silence
# - Splits into train/val/test sets (70%/15%/15%)
# - Saves organized processed files and metadata.csv
# - Creates verification plots
# ============================================================================

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION PARAMETERS
# From project proposal and 4-phase plan specifications
# ============================================================================

# Sampling rate: 16 kHz (from proposal page 4: "Resample to 16 kHz")
TARGET_SR = 16000

# Frame parameters (from Phase 2 specs and proposal page 4)
# "20-25 ms frame length with 10 ms hop using a Hamming window"
FRAME_LENGTH_MS = 25  
HOP_LENGTH_MS = 10    

# Target keyword classes (from proposal page 2)
# "yes", "no", "stop", "go" plus "unknown/silence"
TARGET_CLASSES = ['yes', 'no', 'stop', 'go']

# Maximum clips per class (from proposal page 8 and Phase 1 specs)
# "~1,000–1,500 clips per target class for training and evaluation"
MAX_CLIPS_PER_CLASS = 1500

# Dataset split ratios (from proposal and Phase 1)
# "training (70%), validation (15%), and testing (15%) sets"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Directories
PROJECT_DATA_DIR = Path('/Users/saifelsaady/Documents/EEE598_Speech_Final_Project/ProjectData')
ALL_WAV_FILES_TXT = PROJECT_DATA_DIR / 'all_wav_files.txt'
PROJECT_IMPLEMENTATION_DIR = Path('/Users/saifelsaady/Documents/EEE598_Speech_Final_Project/Project Implementation')
OUTPUT_DIR = PROJECT_IMPLEMENTATION_DIR / 'processed_data'
PLOTS_DIR = PROJECT_IMPLEMENTATION_DIR / 'plots'

# ============================================================================
# STEP 1: CREATE OUTPUT DIRECTORY STRUCTURE
# ============================================================================

def create_output_directories():
    """
    Create organized folder structure for processed audio files.
    Structure: processed_data/train|val|test/yes|no|stop|go/
    """
    print("\n" + "="*70)
    print("STEP 1: Creating output directory structure")
    print("="*70)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories for each split and each class
    for split in ['train', 'val', 'test']:
        for class_name in TARGET_CLASSES:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directories:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {PLOTS_DIR}")
    print(f"  - Subdirectories for train/val/test × {TARGET_CLASSES}")

# ============================================================================
# STEP 2: COLLECT DATASET FILES
# ============================================================================

def collect_dataset_files(txt_file_path, base_dir, target_classes, max_per_class):
    """
    Collect file paths for target keyword classes from all_wav_files.txt.
    
    Args:
        txt_file_path: Path to all_wav_files.txt
        base_dir: Base directory containing the audio files
        target_classes: List of class names ['yes', 'no', 'stop', 'go']
        max_per_class: Maximum number of files per class (1500)
    
    Returns:
        file_list: List of (file_path, label) tuples
    """
    print("\n" + "="*70)
    print("STEP 2: Collecting dataset files from all_wav_files.txt")
    print("="*70)
    
    # Read all file paths from the text file
    with open(txt_file_path, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(all_paths)} file paths from {txt_file_path}")
    
    # Group files by class
    files_by_class = {class_name: [] for class_name in target_classes}
    
    for path in all_paths:
        # Extract class name from path (e.g., "./go/filename.wav" -> "go")
        # Handle both relative paths like "./go/..." and paths with just "go/..."
        parts = path.replace('./', '').split('/')
        if len(parts) >= 2:
            class_name = parts[0]
            if class_name in target_classes:
                # Convert relative path to absolute path
                full_path = base_dir / path.lstrip('./')
                files_by_class[class_name].append(str(full_path))
    
    # Report counts and sample if needed
    file_list = []
    for class_name in target_classes:
        class_files = files_by_class[class_name]
        print(f"Found {len(class_files)} files for class '{class_name}'")
        
        # Limit to max_per_class for manageable runtime
        if max_per_class is not None and len(class_files) > max_per_class:
            np.random.seed(42)  # For reproducibility
            class_files = list(np.random.choice(class_files, max_per_class, replace=False))
            print(f"  → Randomly sampled {max_per_class} files")
        
        # Add to file list as (file_path, label) tuples
        for file_path in class_files:
            file_list.append((file_path, class_name))
    
    print(f"\n✓ Total files collected: {len(file_list)}")
    return file_list

# ============================================================================
# STEP 3: SPLIT DATASET
# ============================================================================

def split_dataset(file_list, train_ratio, val_ratio, test_ratio):
    """
    Split dataset into train/validation/test sets with stratification.
    Stratification ensures balanced class distribution in each split.
    
    Args:
        file_list: List of (file_path, label) tuples
        train_ratio: 0.70 (70%)
        val_ratio: 0.15 (15%)
        test_ratio: 0.15 (15%)
    
    Returns:
        train_files, val_files, test_files: Split lists
    """
    print("\n" + "="*70)
    print("STEP 3: Splitting dataset (70% train / 15% val / 15% test)")
    print("="*70)
    
    # Separate file paths and labels
    files = [item[0] for item in file_list]
    labels = [item[1] for item in file_list]
    
    # First split: separate test set (15%)
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        files, labels,
        test_size=test_ratio,
        random_state=42,
        stratify=labels  # Keep class balance
    )
    
    # Second split: separate train and validation from remaining 85%
    # Adjust validation ratio: 15% of total = 15/85 of remaining
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_ratio_adjusted,
        random_state=42,
        stratify=train_val_labels
    )
    
    # Recombine into (file, label) tuples
    train_set = list(zip(train_files, train_labels))
    val_set = list(zip(val_files, val_labels))
    test_set = list(zip(test_files, test_labels))
    
    print(f"✓ Training set:   {len(train_set):4d} files ({train_ratio*100:.0f}%)")
    print(f"✓ Validation set: {len(val_set):4d} files ({val_ratio*100:.0f}%)")
    print(f"✓ Test set:       {len(test_set):4d} files ({test_ratio*100:.0f}%)")
    
    return train_set, val_set, test_set

# ============================================================================
# STEP 4: PREPROCESSING FUNCTION
# ============================================================================

def load_and_preprocess_audio(file_path, target_sr=TARGET_SR):
    """
    Load and preprocess audio file according to project specifications:
    1. Resample to 16 kHz (from proposal page 4)
    2. Normalize amplitude to [-1, 1] range
    3. Trim silence from beginning and end
    
    Args:
        file_path: Path to .wav file
        target_sr: Target sampling rate (16000 Hz)
    
    Returns:
        audio: Preprocessed audio as numpy array
        sr: Sampling rate
    """
    try:
        # Load audio - librosa automatically resamples if needed
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Trim silence (top_db=20 means treat audio 20dB below peak as silence)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize to [-1, 1] range (from proposal: "normalize amplitude")
        max_amplitude = np.max(np.abs(audio_trimmed))
        if max_amplitude > 0:
            audio_normalized = audio_trimmed / max_amplitude
        else:
            audio_normalized = audio_trimmed
        
        return audio_normalized, sr
    
    except Exception as e:
        print(f"ERROR processing {file_path}: {e}")
        return None, None

# ============================================================================
# STEP 5: PROCESS AND SAVE ALL FILES
# ============================================================================

def process_and_save_dataset(file_list, split_name, output_dir):
    """
    Process all files in a dataset split and save to organized folders.
    Also collect metadata for each file.
    
    Args:
        file_list: List of (file_path, label) tuples
        split_name: 'train', 'val', or 'test'
        output_dir: Base output directory (processed_data/)
    
    Returns:
        metadata_rows: List of dictionaries with file metadata
    """
    print(f"\nProcessing {split_name} set...")
    
    metadata_rows = []
    
    # Process each file with progress bar
    for file_path, label in tqdm(file_list, desc=f"{split_name}"):
        # Preprocess audio
        audio, sr = load_and_preprocess_audio(file_path, TARGET_SR)
        
        if audio is None:
            continue  # Skip files that failed to load
        
        # Determine output path: processed_data/train|val|test/class/filename.wav
        original_filename = Path(file_path).name
        output_path = output_dir / split_name / label / original_filename
        
        # Save processed audio
        sf.write(output_path, audio, sr)
        
        # Record metadata
        metadata_rows.append({
            'filename': original_filename,
            'label': label,
            'split': split_name,
            'original_path': file_path,
            'processed_path': str(output_path),
            'duration_sec': len(audio) / sr,
            'sampling_rate': sr
        })
    
    print(f"✓ Saved {len(metadata_rows)} processed files")
    return metadata_rows

# ============================================================================
# STEP 6: CREATE METADATA CSV
# ============================================================================

def create_metadata_csv(train_meta, val_meta, test_meta, output_path):
    """
    Combine metadata from all splits and save to CSV.
    This CSV will be used in later phases for feature extraction.
    
    Args:
        train_meta, val_meta, test_meta: Metadata lists
        output_path: Path to save metadata.csv
    
    Returns:
        df: Pandas DataFrame with all metadata
    """
    print("\n" + "="*70)
    print("STEP 6: Creating metadata.csv")
    print("="*70)
    
    # Combine all metadata
    all_metadata = train_meta + val_meta + test_meta
    df = pd.DataFrame(all_metadata)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved metadata to: {output_path}")
    print(f"  Total files: {len(df)}")
    print(f"\nDataset summary by split and class:")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    
    return df

# ============================================================================
# STEP 7: VERIFICATION
# ============================================================================

def verify_preprocessing(metadata_df, plots_dir):
    """
    Verification step: Create plots to confirm preprocessing worked correctly.
    From Phase 1 specs:
    - Plot normalized waveforms to confirm equal duration and amplitude range
    - Verify dataset balance per class using histogram/bar plot
    
    Args:
        metadata_df: DataFrame with metadata
        plots_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("STEP 7: VERIFICATION - Creating plots")
    print("="*70)
    
    # ========================================================================
    # PLOT 1: Sample waveforms from each class (verify normalization)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    print("Creating sample waveform plots...")
    for idx, class_name in enumerate(TARGET_CLASSES):
        # Get one sample file from each class
        sample_row = metadata_df[metadata_df['label'] == class_name].iloc[0]
        audio, sr = librosa.load(sample_row['processed_path'], sr=None)
        
        # Create time axis in seconds
        time = np.arange(len(audio)) / sr
        
        # Plot waveform
        axes[idx].plot(time, audio, linewidth=0.5, color='steelblue')
        axes[idx].set_title(f"Class: '{class_name}' (normalized)", 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (seconds)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_ylim([-1.1, 1.1])  # Should be within [-1, 1]
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'sample_waveforms.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / 'sample_waveforms.png'}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: Dataset balance (class distribution)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    print("Creating dataset balance plots...")
    
    # Overall class counts
    class_counts = metadata_df['label'].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, 
               color='steelblue', edgecolor='black', linewidth=1.5)
    axes[0].set_title('Overall Dataset Balance', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Files', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    # Add count labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    # Balance by split
    split_class_counts = metadata_df.groupby(['split', 'label']).size().unstack(fill_value=0)
    split_class_counts.plot(kind='bar', ax=axes[1], edgecolor='black', linewidth=1.5)
    axes[1].set_title('Dataset Balance by Split', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Split', fontsize=12)
    axes[1].set_ylabel('Number of Files', fontsize=12)
    axes[1].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(['train', 'test', 'val'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'dataset_balance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / 'dataset_balance.png'}")
    plt.close()
    
    # ========================================================================
    # Verify amplitude normalization (print to console)
    # ========================================================================
    print("\nVerifying amplitude normalization (should be within [-1, 1]):")
    for class_name in TARGET_CLASSES:
        class_files = metadata_df[
            metadata_df['label'] == class_name
        ]['processed_path'].tolist()
        
        # Check 3 random files from each class
        for file_path in class_files[:3]:
            audio, _ = librosa.load(file_path, sr=None)
            min_val, max_val = audio.min(), audio.max()
            print(f"  {class_name}: [{min_val:+.4f}, {max_val:+.4f}]")
    
    print("\n✓ Verification complete!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Execute complete Phase 1 pipeline:
    1. Create directories
    2. Collect files
    3. Split dataset
    4. Process and save
    5. Create metadata
    6. Verify
    """
    print("="*70)
    print("PHASE 1: DATASET SETUP & PREPROCESSING")
    print("EEE598 Keyword Spotting Final Project")
    print("="*70)
    print("\nProject Specifications:")
    print(f"  - Target classes: {TARGET_CLASSES}")
    print(f"  - Sampling rate: {TARGET_SR} Hz")
    print(f"  - Max files per class: {MAX_CLIPS_PER_CLASS}")
    print(f"  - Train/Val/Test split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  - Frame length: {FRAME_LENGTH_MS} ms (for Phase 2)")
    print(f"  - Hop length: {HOP_LENGTH_MS} ms (for Phase 2)")
    
    # Execute pipeline
    create_output_directories()
    
    file_list = collect_dataset_files(ALL_WAV_FILES_TXT, PROJECT_DATA_DIR, TARGET_CLASSES, MAX_CLIPS_PER_CLASS)
    
    train_files, val_files, test_files = split_dataset(
        file_list, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    print("\n" + "="*70)
    print("STEP 4: Processing and saving audio files")
    print("="*70)
    train_metadata = process_and_save_dataset(train_files, 'train', OUTPUT_DIR)
    val_metadata = process_and_save_dataset(val_files, 'val', OUTPUT_DIR)
    test_metadata = process_and_save_dataset(test_files, 'test', OUTPUT_DIR)
    
    metadata_df = create_metadata_csv(
        train_metadata, val_metadata, test_metadata,
        OUTPUT_DIR / 'metadata.csv'
    )
    
    verify_preprocessing(metadata_df, PLOTS_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE!")
    print("="*70)
    print(f"\n📁 Outputs:")
    print(f"  - Processed audio: {OUTPUT_DIR}/")
    print(f"  - Metadata CSV: {OUTPUT_DIR / 'metadata.csv'}")
    print(f"  - Verification plots: {PLOTS_DIR}/")
    print(f"\n✓ Ready for Phase 2: Feature Extraction")
    print("="*70 + "\n")

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()