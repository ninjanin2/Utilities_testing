import os
import numpy as np
import soundfile as sf
import argparse
from scipy import signal
from typing import Tuple, Optional

# Try importing both frameworks
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class UniversalDTLNProcessor:
    """
    Universal DTLN Speech Enhancement Processor
    Supports TensorFlow (.h5, SavedModel) and PyTorch (.pt) models
    """
    
    def __init__(self, model_path: str, sampling_rate: int = 16000):
        """
        Initialize DTLN processor
        
        Args:
            model_path: Path to the model file or directory
            sampling_rate: Audio sampling rate (default: 16000 Hz)
        """
        self.sampling_rate = sampling_rate
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.block_len = 512
        self.block_shift = 128
        self.load_model()
    
    def load_model(self):
        """Load model based on file extension and available frameworks"""
        
        # Determine model type
        if os.path.isfile(self.model_path):
            if self.model_path.endswith('.pt'):
                self.model_type = 'pytorch'
            elif self.model_path.endswith('.h5'):
                self.model_type = 'tensorflow_h5'
            else:
                raise ValueError(f"Unsupported model file format: {self.model_path}")
        elif os.path.isdir(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                self.model_type = 'tensorflow_savedmodel'
            elif any(f.endswith('.h5') for f in os.listdir(self.model_path)):
                self.model_type = 'tensorflow_h5'
                # Update path to point to the .h5 file
                h5_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5')]
                self.model_path = os.path.join(self.model_path, h5_files[0])
            elif any(f.endswith('.pt') for f in os.listdir(self.model_path)):
                self.model_type = 'pytorch'
                # Update path to point to the .pt file
                pt_files = [f for f in os.listdir(self.model_path) if f.endswith('.pt')]
                self.model_path = os.path.join(self.model_path, pt_files)
            else:
                raise FileNotFoundError("No valid model file found in the directory")
        
        # Load model based on type
        try:
            if self.model_type == 'pytorch':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available. Install with: pip install torch")
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                print(f"✓ PyTorch DTLN model loaded successfully from {self.model_path}")
                
            elif self.model_type == 'tensorflow_h5':
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✓ TensorFlow H5 DTLN model loaded successfully from {self.model_path}")
                
            elif self.model_type == 'tensorflow_savedmodel':
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
                self.model = tf.saved_model.load(self.model_path)
                print(f"✓ TensorFlow SavedModel DTLN model loaded successfully from {self.model_path}")
                
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for DTLN input"""
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data.astype(np.float32)
    
    def create_overlapping_blocks(self, audio_data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Create overlapping blocks for DTLN processing"""
        pad_len = self.block_len - (len(audio_data) % self.block_shift)
        if pad_len < self.block_len:
            audio_data = np.concatenate([audio_data, np.zeros(pad_len)])
        
        num_blocks = (len(audio_data) - self.block_len) // self.block_shift + 1
        blocks = np.zeros((num_blocks, self.block_len))
        
        for i in range(num_blocks):
            start_idx = i * self.block_shift
            end_idx = start_idx + self.block_len
            blocks[i] = audio_data[start_idx:end_idx]
        
        return blocks, num_blocks
    
    def reconstruct_audio(self, enhanced_blocks: np.ndarray, original_length: int) -> np.ndarray:
        """Reconstruct audio from enhanced blocks using overlap-add"""
        num_blocks = enhanced_blocks.shape[0]
        output_length = (num_blocks - 1) * self.block_shift + self.block_len
        output_audio = np.zeros(output_length)
        
        for i in range(num_blocks):
            start_idx = i * self.block_shift
            end_idx = start_idx + self.block_len
            output_audio[start_idx:end_idx] += enhanced_blocks[i]
        
        return output_audio[:original_length]
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance audio using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Please check model path.")
        
        original_length = len(audio_data)
        audio_data = self.preprocess_audio(audio_data)
        audio_blocks, num_blocks = self.create_overlapping_blocks(audio_data)
        enhanced_blocks = np.zeros_like(audio_blocks)
        
        print(f"Processing {num_blocks} audio blocks with {self.model_type} model...")
        
        for i, block in enumerate(audio_blocks):
            try:
                if self.model_type == 'pytorch':
                    # PyTorch inference
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(block).unsqueeze(0)
                        enhanced_tensor = self.model(input_tensor)
                        enhanced_block = enhanced_tensor.squeeze(0).numpy()
                
                elif self.model_type in ['tensorflow_h5', 'tensorflow_savedmodel']:
                    # TensorFlow inference
                    input_block = np.expand_dims(block, axis=0)
                    
                    if hasattr(self.model, 'signatures'):
                        # SavedModel format
                        enhanced_block = self.model.signatures['serving_default'](
                            tf.constant(input_block, dtype=tf.float32)
                        )
                        enhanced_block = list(enhanced_block.values())[0].numpy()
                    else:
                        # Keras model format
                        enhanced_block = self.model.predict(input_block, verbose=0)
                    
                    enhanced_block = enhanced_block.squeeze()
                
                enhanced_blocks[i] = enhanced_block
                
            except Exception as e:
                print(f"Warning: Error processing block {i}: {str(e)}")
                enhanced_blocks[i] = block
        
        enhanced_audio = self.reconstruct_audio(enhanced_blocks, original_length)
        return enhanced_audio
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """Process an audio file for speech enhancement"""
        try:
            print(f"Loading audio from: {input_path}")
            audio_data, original_sr = sf.read(input_path)
            
            if original_sr != self.sampling_rate:
                print(f"Resampling from {original_sr} Hz to {self.sampling_rate} Hz")
                num_samples = int(len(audio_data) * self.sampling_rate / original_sr)
                audio_data = signal.resample(audio_data, num_samples)
            
            print(f"Enhancing audio with {self.model_type} DTLN model...")
            enhanced_audio = self.enhance_audio(audio_data)
            
            print(f"Saving enhanced audio to: {output_path}")
            sf.write(output_path, enhanced_audio, self.sampling_rate)
            
            print("✓ Audio enhancement completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error processing file: {str(e)}")
            return False

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Universal DTLN Speech Enhancement Processor')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the DTLN model file (.pt, .h5) or directory (SavedModel)')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input noisy audio file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to save enhanced audio file')
    parser.add_argument('--sampling_rate', type=int, default=16000,
                       help='Audio sampling rate (default: 16000)')
    
    args = parser.parse_args()
    
    processor = UniversalDTLNProcessor(args.model_path, args.sampling_rate)
    success = processor.process_file(args.input_file, args.output_file)
    
    return 0 if success else 1

if __name__ == "__main__":
    main()
