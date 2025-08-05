#!/usr/bin/env python3
"""
VoiceFixer Offline Speech Enhancement Script
============================================

This script uses VoiceFixer for offline speech enhancement, denoising, and repair.
It loads pre-downloaded models from a local directory and processes audio files
to prepare them for ASR models.

Configuration:
- Set the global variables below to configure paths and settings
- Models will be loaded from MODEL_DIR
- Audio will be processed from INPUT_PATH to OUTPUT_PATH

Requirements:
- voicefixer package
- Pre-downloaded model checkpoints

Model Download Instructions:
1. Analysis module: vf.ckpt (place in <model_dir>/analysis_module/checkpoints/)
2. Synthesis module: model.ckpt-1490000_trimed.pt (place in <model_dir>/synthesis_module/44100/)

Download links:
- vf.ckpt: https://zenodo.org/record/5600188/files/vf.ckpt?download=1
- model.ckpt-1490000_trimed.pt: https://zenodo.org/record/5600188/files/model.ckpt-1490000_trimed.pt?download=1
"""

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES - MODIFY THESE PATHS AS NEEDED
# ============================================================================

# Model directory containing VoiceFixer checkpoints
MODEL_DIR = "./models"

# Input audio file or directory path
# For single file: INPUT_PATH = "./input_audio/noisy_speech.wav"
# For batch processing: INPUT_PATH = "./input_audio/"
INPUT_PATH = "./input_audio/noisy_speech.wav"

# Output audio file or directory path
# For single file: OUTPUT_PATH = "./output_audio/clean_speech.wav"
# For batch processing: OUTPUT_PATH = "./output_audio/"
OUTPUT_PATH = "./output_audio/clean_speech.wav"

# Processing mode (0: original, 1: preprocessing, 2: train mode)
ENHANCEMENT_MODE = 0

# Enable GPU acceleration (True/False)
USE_CUDA = False

# Enable batch processing mode (True/False)
# Set to True if INPUT_PATH and OUTPUT_PATH are directories
BATCH_PROCESSING = False

# Audio file extensions to process in batch mode
SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

# ============================================================================
# SCRIPT IMPLEMENTATION - DO NOT MODIFY BELOW UNLESS NECESSARY
# ============================================================================

import os
import sys
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from voicefixer import VoiceFixer, Vocoder
    import torch
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages:")
    print("pip install voicefixer torch librosa soundfile")
    sys.exit(1)


class OfflineVoiceFixerProcessor:
    """
    Offline VoiceFixer processor for speech enhancement and denoising.
    """
    
    def __init__(self, model_dir):
        """
        Initialize the processor with local model directory.
        
        Args:
            model_dir (str): Path to directory containing VoiceFixer models
        """
        self.model_dir = Path(model_dir)
        self.voicefixer = None
        self.vocoder = None
        
        # Expected model paths
        self.analysis_model_path = self.model_dir / "analysis_module" / "checkpoints" / "vf.ckpt"
        self.synthesis_model_path = self.model_dir / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt"
        
        self._validate_model_files()
        self._setup_cache_directory()
        self._initialize_models()
    
    def _validate_model_files(self):
        """Validate that required model files exist."""
        missing_files = []
        
        if not self.analysis_model_path.exists():
            missing_files.append(str(self.analysis_model_path))
        
        if not self.synthesis_model_path.exists():
            missing_files.append(str(self.synthesis_model_path))
        
        if missing_files:
            print("ERROR: Missing model files:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nPlease download the required model files:")
            print("1. vf.ckpt -> analysis_module/checkpoints/")
            print("2. model.ckpt-1490000_trimed.pt -> synthesis_module/44100/")
            print("\nDownload links:")
            print("- https://zenodo.org/record/5600188/files/vf.ckpt?download=1")
            print("- https://zenodo.org/record/5600188/files/model.ckpt-1490000_trimed.pt?download=1")
            sys.exit(1)
        
        print("✓ All required model files found")
    
    def _setup_cache_directory(self):
        """Setup cache directory and copy models if needed."""
        cache_dir = Path.home() / ".cache" / "voicefixer"
        
        # Create cache directory structure
        analysis_cache = cache_dir / "analysis_module" / "checkpoints"
        synthesis_cache = cache_dir / "synthesis_module" / "44100"
        
        analysis_cache.mkdir(parents=True, exist_ok=True)
        synthesis_cache.mkdir(parents=True, exist_ok=True)
        
        # Copy models to cache if they don't exist or are different
        analysis_target = analysis_cache / "vf.ckpt"
        synthesis_target = synthesis_cache / "model.ckpt-1490000_trimed.pt"
        
        if not analysis_target.exists() or not self._files_identical(self.analysis_model_path, analysis_target):
            print("Copying analysis model to cache...")
            shutil.copy2(self.analysis_model_path, analysis_target)
        
        if not synthesis_target.exists() or not self._files_identical(self.synthesis_model_path, synthesis_target):
            print("Copying synthesis model to cache...")
            shutil.copy2(self.synthesis_model_path, synthesis_target)
        
        print("✓ Models prepared in cache directory")
    
    def _files_identical(self, file1, file2):
        """Check if two files are identical by comparing sizes."""
        try:
            return file1.stat().st_size == file2.stat().st_size
        except OSError:
            return False
    
    def _initialize_models(self):
        """Initialize VoiceFixer models."""
        try:
            print("Initializing VoiceFixer models...")
            self.voicefixer = VoiceFixer()
            self.vocoder = Vocoder(sample_rate=44100)
            print("✓ Models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {e}")
            sys.exit(1)
    
    def validate_audio_file(self, audio_path):
        """
        Validate input audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            bool: True if valid, False otherwise
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return False
        
        # Check file extension using global SUPPORTED_EXTENSIONS
        if audio_path.suffix.lower() not in [ext.lower() for ext in SUPPORTED_EXTENSIONS]:
            print(f"Warning: File extension {audio_path.suffix} may not be supported")
            print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        
        try:
            # Try to load audio file to validate
            audio, sr = librosa.load(str(audio_path), sr=None)
            duration = len(audio) / sr
            print(f"✓ Audio file valid: {duration:.2f}s, {sr}Hz, {len(audio)} samples")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def enhance_audio(self, input_path, output_path, mode=0, cuda=False):
        """
        Enhance audio using VoiceFixer.
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to output audio file
            mode (int): Enhancement mode (0: original, 1: preprocessing, 2: train mode)
            cuda (bool): Use GPU acceleration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate input
            if not self.validate_audio_file(input_path):
                return False
            
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing: {input_path}")
            print(f"Mode: {mode} ({'Original' if mode == 0 else 'Preprocessing' if mode == 1 else 'Train mode'})")
            print(f"GPU: {'Enabled' if cuda else 'Disabled'}")
            print("Starting enhancement...")
            
            # Perform enhancement
            self.voicefixer.restore(
                input=str(input_path),
                output=str(output_path),
                cuda=cuda,
                mode=mode
            )
            
            # Validate output
            if output_path.exists():
                # Check output file
                audio, sr = librosa.load(str(output_path), sr=None)
                duration = len(audio) / sr
                print(f"✓ Enhancement completed successfully!")
                print(f"Output: {output_path}")
                print(f"Duration: {duration:.2f}s, Sample rate: {sr}Hz")
                return True
            else:
                print("Error: Output file was not created")
                return False
                
        except Exception as e:
            print(f"Error during enhancement: {e}")
            return False
    
    def batch_enhance(self, input_dir, output_dir, mode=0, cuda=False):
        """
        Enhance multiple audio files in a directory.
        
        Args:
            input_dir (str): Directory containing input audio files
            output_dir (str): Directory for output audio files
            mode (int): Enhancement mode
            cuda (bool): Use GPU acceleration
            
        Returns:
            dict: Results summary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return None
        
        # Find audio files using global SUPPORTED_EXTENSIONS
        audio_files = []
        for ext in SUPPORTED_EXTENSIONS:
            audio_files.extend(input_dir.glob(f"*{ext}"))
            audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
            return None
        
        print(f"Found {len(audio_files)} audio files")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = {"success": 0, "failed": 0, "files": []}
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            output_file = output_dir / f"{audio_file.stem}_enhanced{audio_file.suffix}"
            
            success = self.enhance_audio(
                input_path=str(audio_file),
                output_path=str(output_file),
                mode=mode,
                cuda=cuda
            )
            
            if success:
                results["success"] += 1
                results["files"].append({"input": str(audio_file), "output": str(output_file), "status": "success"})
            else:
                results["failed"] += 1
                results["files"].append({"input": str(audio_file), "output": str(output_file), "status": "failed"})
        
        return results


def validate_global_config():
    """Validate global configuration variables."""
    global MODEL_DIR, INPUT_PATH, OUTPUT_PATH, ENHANCEMENT_MODE, USE_CUDA, BATCH_PROCESSING
    
    # Convert paths to Path objects
    MODEL_DIR = Path(MODEL_DIR)
    INPUT_PATH = Path(INPUT_PATH)
    OUTPUT_PATH = Path(OUTPUT_PATH)
    
    # Validate model directory
    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found: {MODEL_DIR}")
        print("Please ensure MODEL_DIR points to the correct directory containing VoiceFixer models")
        return False
    
    # Validate input path
    if BATCH_PROCESSING:
        if not INPUT_PATH.exists() or not INPUT_PATH.is_dir():
            print(f"Error: Input directory not found: {INPUT_PATH}")
            print("For batch processing, INPUT_PATH must be a valid directory")
            return False
    else:
        if not INPUT_PATH.exists() or INPUT_PATH.is_dir():
            print(f"Error: Input file not found: {INPUT_PATH}")
            print("For single file processing, INPUT_PATH must be a valid audio file")
            return False
    
    # Validate enhancement mode
    if ENHANCEMENT_MODE not in [0, 1, 2]:
        print(f"Error: Invalid enhancement mode: {ENHANCEMENT_MODE}")
        print("ENHANCEMENT_MODE must be 0, 1, or 2")
        return False
    
    return True


def print_config_summary():
    """Print current configuration summary."""
    print("CURRENT CONFIGURATION:")
    print("=" * 40)
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Input Path: {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Enhancement Mode: {ENHANCEMENT_MODE} ({'Original' if ENHANCEMENT_MODE == 0 else 'Preprocessing' if ENHANCEMENT_MODE == 1 else 'Train mode'})")
    print(f"GPU Acceleration: {'Enabled' if USE_CUDA else 'Disabled'}")
    print(f"Processing Mode: {'Batch' if BATCH_PROCESSING else 'Single File'}")
    print("=" * 40)


def main():
    """Main function using global configuration variables."""
    # Print header
    print("=" * 60)
    print("VoiceFixer Offline Speech Enhancement Script")
    print("=" * 60)
    
    # Validate configuration
    if not validate_global_config():
        print("\nPlease check and correct the global configuration variables at the top of the script.")
        sys.exit(1)
    
    # Print configuration summary
    print_config_summary()
    
    # Check CUDA availability
    if USE_CUDA:
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("Warning: CUDA requested but not available, using CPU")
            # Update global variable
            globals()['USE_CUDA'] = False
    
    # Initialize processor
    try:
        processor = OfflineVoiceFixerProcessor(MODEL_DIR)
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process audio
    if BATCH_PROCESSING:
        # Batch processing
        print(f"\nStarting batch processing...")
        results = processor.batch_enhance(
            input_dir=INPUT_PATH,
            output_dir=OUTPUT_PATH,
            mode=ENHANCEMENT_MODE,
            cuda=USE_CUDA
        )
        
        if results:
            print(f"\n" + "=" * 40)
            print("BATCH PROCESSING SUMMARY")
            print("=" * 40)
            print(f"Total files: {len(results['files'])}")
            print(f"Successful: {results['success']}")
            print(f"Failed: {results['failed']}")
            
            if results['failed'] > 0:
                print("\nFailed files:")
                for file_info in results['files']:
                    if file_info['status'] == 'failed':
                        print(f"  - {file_info['input']}")
            
            print(f"\nEnhanced audio files saved to: {OUTPUT_PATH}")
        else:
            print("Batch processing failed!")
            sys.exit(1)
    else:
        # Single file processing
        print(f"\nStarting single file processing...")
        success = processor.enhance_audio(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            mode=ENHANCEMENT_MODE,
            cuda=USE_CUDA
        )
        
        if success:
            print(f"\n✓ Speech enhancement completed successfully!")
            print(f"Enhanced audio saved to: {OUTPUT_PATH}")
            print("Ready for ASR processing.")
        else:
            print(f"\n✗ Speech enhancement failed!")
            sys.exit(1)
    
    print("\nDone!")


def example_configurations():
    """
    Example configurations for different use cases.
    Uncomment and modify the configuration you need.
    """
    
    # Example 1: Single file processing
    # global MODEL_DIR, INPUT_PATH, OUTPUT_PATH, ENHANCEMENT_MODE, USE_CUDA, BATCH_PROCESSING
    # MODEL_DIR = "./models"
    # INPUT_PATH = "./input_audio/noisy_speech.wav"
    # OUTPUT_PATH = "./output_audio/clean_speech.wav"
    # ENHANCEMENT_MODE = 0
    # USE_CUDA = False
    # BATCH_PROCESSING = False
    
    # Example 2: Batch processing with GPU
    # global MODEL_DIR, INPUT_PATH, OUTPUT_PATH, ENHANCEMENT_MODE, USE_CUDA, BATCH_PROCESSING
    # MODEL_DIR = "./models"
    # INPUT_PATH = "./input_audio/"
    # OUTPUT_PATH = "./output_audio/"
    # ENHANCEMENT_MODE = 0
    # USE_CUDA = True
    # BATCH_PROCESSING = True
    
    # Example 3: High-quality enhancement for seriously degraded audio
    # global MODEL_DIR, INPUT_PATH, OUTPUT_PATH, ENHANCEMENT_MODE, USE_CUDA, BATCH_PROCESSING
    # MODEL_DIR = "./models"
    # INPUT_PATH = "./input_audio/very_noisy_speech.wav"
    # OUTPUT_PATH = "./output_audio/enhanced_speech.wav"
    # ENHANCEMENT_MODE = 2  # Train mode for seriously degraded speech
    # USE_CUDA = True
    # BATCH_PROCESSING = False
    
    pass


if __name__ == "__main__":
    main()