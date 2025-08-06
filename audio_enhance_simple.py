#!/usr/bin/env python3
"""
Advanced Speech Enhancement and Denoising System for ASR
=======================================================

A professional-grade audio preprocessing pipeline that transforms noisy call recordings
into clean speech optimized for Automatic Speech Recognition (ASR) models.

Features:
- Multi-stage noise reduction (spectral subtraction, Wiener filtering, adaptive filtering)
- Voice Activity Detection (VAD) with silence trimming
- Dynamic range compression and normalization
- Adaptive resampling for ASR compatibility
- Comprehensive error handling and logging
- Configurable processing parameters

Author: AI Assistant
License: MIT
"""

import os
import sys
import logging
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    import scipy.signal as signal
    from scipy.fft import fft, ifft
    import noisereduce as nr
except ImportError as e:
    print(f"Missing required audio libraries: {e}")
    print("Install with: pip install librosa soundfile scipy noisereduce")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global configuration
AUDIO_FILE_PATH = ""  # Set this to your input audio file path
OUTPUT_DIR = "enhanced_audio"
CONFIG_FILE = "enhancement_config.json"

# Default processing configuration
DEFAULT_CONFIG = {
    "target_sr": 16000,           # ASR-optimized sample rate
    "noise_reduce_strength": 0.8,  # Noise reduction strength (0-1)
    "spectral_subtraction_alpha": 2.0,  # Spectral subtraction factor
    "wiener_filter_noise_power": 0.1,   # Wiener filter noise estimation
    "vad_threshold": 0.02,        # Voice activity detection threshold
    "normalization_target": -23,  # Target LUFS for normalization
    "compression_ratio": 4.0,     # Dynamic range compression ratio
    "high_pass_cutoff": 80,       # High-pass filter cutoff (Hz)
    "low_pass_cutoff": 8000,      # Low-pass filter cutoff (Hz)
    "frame_length": 2048,         # FFT frame length
    "hop_length": 512,            # FFT hop length
    "pre_emphasis": 0.97,         # Pre-emphasis coefficient
    "silence_threshold": 0.01,    # Silence detection threshold
    "min_speech_duration": 0.1    # Minimum speech segment duration (seconds)
}

class AudioEnhancer:
    """
    Advanced audio enhancement system optimized for ASR preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the audio enhancer with configuration."""
        self.config = config or DEFAULT_CONFIG.copy()
        self.logger = self._setup_logging()
        self.original_sr = None
        self.duration = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_enhancement.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with comprehensive error handling.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio with librosa for better compatibility
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            self.original_sr = sr
            self.duration = len(audio) / sr
            
            self.logger.info(f"Loaded audio: {file_path}")
            self.logger.info(f"Duration: {self.duration:.2f}s, Sample Rate: {sr}Hz")
            self.logger.info(f"Audio shape: {audio.shape}, Min: {audio.min():.3f}, Max: {audio.max():.3f}")
            
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file: {e}")
            raise
    
    def pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to enhance high frequencies.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pre-emphasized audio signal
        """
        alpha = self.config['pre_emphasis']
        emphasized = np.append(audio[0], audio[1:] - alpha * audio[:-1])
        return emphasized
    
    def bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply bandpass filter to remove unwanted frequencies.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Filtered audio signal
        """
        nyquist = sr / 2
        low = self.config['high_pass_cutoff'] / nyquist
        high = min(self.config['low_pass_cutoff'] / nyquist, 0.99)
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def spectral_subtraction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Perform spectral subtraction for noise reduction.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Denoised audio signal
        """
        # STFT parameters
        frame_length = self.config['frame_length']
        hop_length = self.config['hop_length']
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from the first 0.5 seconds (assumed to be noise-dominant)
        noise_frames = int(0.5 * sr / hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = self.config['spectral_subtraction_alpha']
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        
        # Prevent negative values
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_audio
    
    def wiener_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Filtered audio signal
        """
        # STFT parameters
        frame_length = self.config['frame_length']
        hop_length = self.config['hop_length']
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate signal and noise power
        signal_power = magnitude ** 2
        noise_power = self.config['wiener_filter_noise_power'] * np.mean(signal_power)
        
        # Wiener filter
        wiener_gain = signal_power / (signal_power + noise_power)
        filtered_magnitude = magnitude * wiener_gain
        
        # Reconstruct signal
        filtered_stft = filtered_magnitude * np.exp(1j * phase)
        filtered_audio = librosa.istft(filtered_stft, hop_length=hop_length)
        
        return filtered_audio
    
    def voice_activity_detection(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect voice activity and remove silence segments.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Voice activity boolean mask
        """
        # Frame-based energy calculation
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        
        # Calculate energy for each frame
        energy = np.sum(frames ** 2, axis=1)
        
        # Adaptive threshold based on energy distribution
        threshold = self.config['vad_threshold'] * np.max(energy)
        
        # Voice activity detection
        vad = energy > threshold
        
        # Smooth VAD decisions (remove isolated frames)
        kernel = np.ones(3) / 3
        vad_smooth = np.convolve(vad.astype(float), kernel, mode='same') > 0.5
        
        # Expand VAD to original audio length
        vad_expanded = np.repeat(vad_smooth, hop_length)
        vad_expanded = vad_expanded[:len(audio)]
        
        return vad_expanded
    
    def trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Trimmed audio signal
        """
        # Use librosa's built-in silence trimming
        trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=20,  # Threshold in dB below peak
            frame_length=self.config['frame_length'],
            hop_length=self.config['hop_length']
        )
        
        return trimmed
    
    def dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to improve speech intelligibility.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Compressed audio signal
        """
        # Compute RMS in sliding windows
        window_size = 1024
        rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='same'))
        
        # Compression parameters
        threshold = 0.1
        ratio = self.config['compression_ratio']
        
        # Apply compression
        compressed = np.copy(audio)
        over_threshold = rms > threshold
        
        if np.any(over_threshold):
            compression_factor = 1 + (ratio - 1) * (rms - threshold) / rms
            compression_factor = np.clip(compression_factor, 0.1, 1.0)
            compressed = audio * compression_factor
        
        return compressed
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to optimal level for ASR.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio signal
        """
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Target RMS level (approximately -23 LUFS)
            target_rms = 10**(self.config['normalization_target']/20)
            audio = audio * (target_rms / rms)
        
        # Peak limiting to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        
        return audio
    
    def adaptive_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply adaptive noise reduction using multiple techniques.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Denoised audio signal
        """
        # Stage 1: Basic noise reduction using noisereduce library
        reduced1 = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=False,  # Non-stationary noise
            prop_decrease=self.config['noise_reduce_strength']
        )
        
        # Stage 2: Spectral subtraction
        reduced2 = self.spectral_subtraction(reduced1, sr)
        
        # Stage 3: Wiener filtering
        reduced3 = self.wiener_filter(reduced2, sr)
        
        return reduced3
    
    def enhance_speech_clarity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance speech clarity and intelligibility.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Enhanced audio signal
        """
        # Apply pre-emphasis to boost high frequencies
        emphasized = self.pre_emphasis(audio)
        
        # Bandpass filter to focus on speech frequencies
        filtered = self.bandpass_filter(emphasized, sr)
        
        # Dynamic range compression
        compressed = self.dynamic_range_compression(filtered)
        
        return compressed
    
    def resample_for_asr(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Resample audio to ASR-optimized sample rate.
        
        Args:
            audio: Input audio signal
            sr: Current sample rate
            
        Returns:
            Tuple of (resampled_audio, target_sample_rate)
        """
        target_sr = self.config['target_sr']
        
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            self.logger.info(f"Resampled from {sr}Hz to {target_sr}Hz")
            return audio_resampled, target_sr
        
        return audio, sr
    
    def remove_silence_segments(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Remove long silence segments while preserving natural pauses.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Audio with silence segments removed
        """
        # Detect voice activity
        vad = self.voice_activity_detection(audio, sr)
        
        # Find speech segments
        speech_samples = audio[vad]
        
        # If too much audio is removed, use original with light trimming
        speech_ratio = len(speech_samples) / len(audio)
        
        if speech_ratio < 0.3:  # If less than 30% is speech, be more conservative
            self.logger.warning("Conservative silence removal applied - too much speech detected as silence")
            return self.trim_silence(audio, sr)
        else:
            self.logger.info(f"Removed {(1-speech_ratio)*100:.1f}% silence")
            return speech_samples
    
    def quality_assessment(self, original: np.ndarray, enhanced: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Assess the quality improvement of the enhanced audio.
        
        Args:
            original: Original audio signal
            enhanced: Enhanced audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal-to-Noise Ratio estimation
        def estimate_snr(signal):
            # Use the difference between signal energy and noise floor
            energy = np.mean(signal**2)
            noise_floor = np.percentile(signal**2, 10)  # Bottom 10% as noise estimate
            return 10 * np.log10(energy / max(noise_floor, 1e-10))
        
        original_snr = estimate_snr(original)
        enhanced_snr = estimate_snr(enhanced)
        
        # RMS levels
        original_rms = np.sqrt(np.mean(original**2))
        enhanced_rms = np.sqrt(np.mean(enhanced**2))
        
        metrics = {
            'original_snr_db': original_snr,
            'enhanced_snr_db': enhanced_snr,
            'snr_improvement_db': enhanced_snr - original_snr,
            'original_rms': original_rms,
            'enhanced_rms': enhanced_rms,
            'dynamic_range_original': np.max(original) - np.min(original),
            'dynamic_range_enhanced': np.max(enhanced) - np.min(enhanced)
        }
        
        return metrics
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str) -> None:
        """
        Save enhanced audio to file.
        
        Args:
            audio: Audio signal to save
            sr: Sample rate
            output_path: Output file path
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as WAV file with 16-bit depth (optimal for ASR)
            sf.write(output_path, audio, sr, subtype='PCM_16')
            
            self.logger.info(f"Enhanced audio saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def process_audio(self, file_path: str) -> str:
        """
        Complete audio enhancement pipeline.
        
        Args:
            file_path: Path to input audio file
            
        Returns:
            Path to enhanced audio file
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING AUDIO ENHANCEMENT PIPELINE")
            self.logger.info("=" * 60)
            
            # Step 1: Load audio
            self.logger.info("Step 1: Loading audio file...")
            audio, sr = self.load_audio(file_path)
            original_audio = audio.copy()
            
            # Step 2: Pre-processing
            self.logger.info("Step 2: Applying bandpass filter...")
            audio = self.bandpass_filter(audio, sr)
            
            # Step 3: Advanced noise reduction
            self.logger.info("Step 3: Performing adaptive noise reduction...")
            audio = self.adaptive_noise_reduction(audio, sr)
            
            # Step 4: Speech enhancement
            self.logger.info("Step 4: Enhancing speech clarity...")
            audio = self.enhance_speech_clarity(audio, sr)
            
            # Step 5: Remove silence segments
            self.logger.info("Step 5: Removing silence segments...")
            audio = self.remove_silence_segments(audio, sr)
            
            # Step 6: Resample for ASR
            self.logger.info("Step 6: Resampling for ASR optimization...")
            audio, sr = self.resample_for_asr(audio, sr)
            
            # Step 7: Final normalization
            self.logger.info("Step 7: Applying final normalization...")
            audio = self.normalize_audio(audio)
            
            # Step 8: Quality assessment
            self.logger.info("Step 8: Assessing quality improvement...")
            if len(original_audio) > 0 and len(audio) > 0:
                # Resample original for fair comparison
                original_resampled = librosa.resample(original_audio, 
                                                    orig_sr=self.original_sr, 
                                                    target_sr=sr)
                
                metrics = self.quality_assessment(original_resampled, audio, sr)
                
                self.logger.info("Quality Metrics:")
                for key, value in metrics.items():
                    self.logger.info(f"  {key}: {value:.3f}")
            
            # Step 9: Save enhanced audio
            input_name = Path(file_path).stem
            output_path = os.path.join(OUTPUT_DIR, f"{input_name}_enhanced.wav")
            
            self.logger.info("Step 9: Saving enhanced audio...")
            self.save_audio(audio, sr, output_path)
            
            self.logger.info("=" * 60)
            self.logger.info("AUDIO ENHANCEMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"Enhanced audio duration: {len(audio)/sr:.2f}s")
            self.logger.info(f"Output file: {output_path}")
            self.logger.info("=" * 60)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio enhancement failed: {e}")
            raise

def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration from JSON file or create default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
    else:
        # Create default configuration file
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"Created default configuration file: {config_path}")
    
    return DEFAULT_CONFIG.copy()

def main():
    """
    Main execution function for speech enhancement.
    """
    global AUDIO_FILE_PATH
    
    # Validate input file path
    if not AUDIO_FILE_PATH or not os.path.exists(AUDIO_FILE_PATH):
        print("Error: Please set AUDIO_FILE_PATH to a valid audio file.")
        print("Supported formats: WAV, MP3, FLAC, M4A, OGG")
        print("\nExample usage:")
        print("AUDIO_FILE_PATH = '/path/to/your/noisy_call_recording.wav'")
        return None
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize enhancer
        enhancer = AudioEnhancer(config)
        
        # Process audio
        enhanced_file_path = enhancer.process_audio(AUDIO_FILE_PATH)
        
        print(f"\n✓ Audio enhancement completed successfully!")
        print(f"✓ Enhanced file: {enhanced_file_path}")
        print(f"✓ Ready for ASR processing")
        
        return enhanced_file_path
        
    except Exception as e:
        print(f"✗ Enhancement failed: {e}")
        return None

def batch_process(input_directory: str, file_extensions: list = None) -> list:
    """
    Process multiple audio files in a directory.
    
    Args:
        input_directory: Directory containing audio files
        file_extensions: List of file extensions to process
        
    Returns:
        List of enhanced file paths
    """
    if file_extensions is None:
        file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    config = load_config()
    enhancer = AudioEnhancer(config)
    
    enhanced_files = []
    audio_files = []
    
    # Find all audio files
    for ext in file_extensions:
        audio_files.extend(Path(input_directory).glob(f"*{ext}"))
        audio_files.extend(Path(input_directory).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"\nProcessing {i}/{len(audio_files)}: {file_path.name}")
        
        try:
            enhanced_path = enhancer.process_audio(str(file_path))
            enhanced_files.append(enhanced_path)
            print(f"✓ Completed: {enhanced_path}")
            
        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")
    
    print(f"\n✓ Batch processing completed: {len(enhanced_files)}/{len(audio_files)} files enhanced")
    return enhanced_files

# Configuration validation
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_keys = [
        'target_sr', 'noise_reduce_strength', 'spectral_subtraction_alpha',
        'wiener_filter_noise_power', 'vad_threshold', 'normalization_target'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate ranges
    if not (8000 <= config['target_sr'] <= 48000):
        print("target_sr must be between 8000 and 48000")
        return False
    
    if not (0 <= config['noise_reduce_strength'] <= 1):
        print("noise_reduce_strength must be between 0 and 1")
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    print("Advanced Speech Enhancement for ASR")
    print("===================================")
    print()
    print("To use this script:")
    print("1. Set AUDIO_FILE_PATH to your input audio file")
    print("2. Run the script: python speech_enhancement.py")
    print("3. Enhanced audio will be saved in the 'enhanced_audio' directory")
    print()
    print("For batch processing:")
    print("enhanced_files = batch_process('/path/to/audio/directory')")
    print()
    
    # Set your audio file path here
    # AUDIO_FILE_PATH = "/path/to/your/noisy_call_recording.wav"
    
    if AUDIO_FILE_PATH:
        main()
    else:
        print("Please set AUDIO_FILE_PATH and run again.")