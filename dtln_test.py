import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import argparse
from scipy import signal
from typing import Tuple, Optional

class DTLNProcessor:
    """
    DTLN Speech Enhancement Processor
    Loads pre-trained DTLN model and processes audio files for noise suppression
    """
    
    def __init__(self, model_path: str, sampling_rate: int = 16000):
        """
        Initialize DTLN processor
        
        Args:
            model_path: Path to the saved DTLN model directory
            sampling_rate: Audio sampling rate (default: 16000 Hz)
        """
        self.sampling_rate = sampling_rate
        self.model_path = model_path
        self.model = None
        self.block_len = 512  # Standard DTLN block length
        self.block_shift = 128  # Standard DTLN block shift
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained DTLN model from local directory"""
        try:
            if os.path.exists(self.model_path):
                # Load SavedModel format
                if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                    self.model = tf.saved_model.load(self.model_path)
                    print(f"✓ DTLN model loaded successfully from {self.model_path}")
                # Load .h5 model format
                elif any(f.endswith('.h5') for f in os.listdir(self.model_path)):
                    h5_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5')]
                    model_file = os.path.join(self.model_path, h5_files[0])
                    self.model = tf.keras.models.load_model(model_file)
                    print(f"✓ DTLN model loaded successfully from {model_file}")
                else:
                    raise FileNotFoundError("No valid model file found in the directory")
            else:
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for DTLN input
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Preprocessed audio array
        """
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio to [-1, 1] range
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data.astype(np.float32)
    
    def create_overlapping_blocks(self, audio_data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create overlapping blocks for DTLN processing
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Tuple of (blocked audio data, number of blocks)
        """
        # Pad audio if necessary
        pad_len = self.block_len - (len(audio_data) % self.block_shift)
        if pad_len < self.block_len:
            audio_data = np.concatenate([audio_data, np.zeros(pad_len)])
        
        # Create overlapping blocks
        num_blocks = (len(audio_data) - self.block_len) // self.block_shift + 1
        blocks = np.zeros((num_blocks, self.block_len))
        
        for i in range(num_blocks):
            start_idx = i * self.block_shift
            end_idx = start_idx + self.block_len
            blocks[i] = audio_data[start_idx:end_idx]
        
        return blocks, num_blocks
    
    def reconstruct_audio(self, enhanced_blocks: np.ndarray, original_length: int) -> np.ndarray:
        """
        Reconstruct audio from enhanced blocks using overlap-add
        
        Args:
            enhanced_blocks: Enhanced audio blocks
            original_length: Original audio length
            
        Returns:
            Reconstructed enhanced audio
        """
        num_blocks = enhanced_blocks.shape[0]
        output_length = (num_blocks - 1) * self.block_shift + self.block_len
        output_audio = np.zeros(output_length)
        
        # Overlap-add reconstruction
        for i in range(num_blocks):
            start_idx = i * self.block_shift
            end_idx = start_idx + self.block_len
            output_audio[start_idx:end_idx] += enhanced_blocks[i]
        
        # Trim to original length
        return output_audio[:original_length]
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance audio using DTLN model
        
        Args:
            audio_data: Input noisy audio array
            
        Returns:
            Enhanced audio array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please check model path.")
        
        original_length = len(audio_data)
        
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data)
        
        # Create overlapping blocks
        audio_blocks, num_blocks = self.create_overlapping_blocks(audio_data)
        
        # Process blocks through DTLN model
        enhanced_blocks = np.zeros_like(audio_blocks)
        
        print(f"Processing {num_blocks} audio blocks...")
        
        for i, block in enumerate(audio_blocks):
            # Prepare input for model (add batch dimension)
            input_block = np.expand_dims(block, axis=0)
            
            # Run inference
            try:
                if hasattr(self.model, 'signatures'):
                    # SavedModel format
                    enhanced_block = self.model.signatures['serving_default'](
                        tf.constant(input_block, dtype=tf.float32)
                    )
                    # Extract the output tensor
                    enhanced_block = list(enhanced_block.values())[0].numpy()
                else:
                    # Keras model format
                    enhanced_block = self.model.predict(input_block, verbose=0)
                
                enhanced_blocks[i] = enhanced_block.squeeze()
                
            except Exception as e:
                print(f"Warning: Error processing block {i}: {str(e)}")
                enhanced_blocks[i] = block  # Use original block if enhancement fails
        
        # Reconstruct enhanced audio
        enhanced_audio = self.reconstruct_audio(enhanced_blocks, original_length)
        
        return enhanced_audio
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process an audio file for speech enhancement
        
        Args:
            input_path: Path to input noisy audio file
            output_path: Path to save enhanced audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load input audio
            print(f"Loading audio from: {input_path}")
            audio_data, original_sr = sf.read(input_path)
            
            # Resample if necessary
            if original_sr != self.sampling_rate:
                print(f"Resampling from {original_sr} Hz to {self.sampling_rate} Hz")
                num_samples = int(len(audio_data) * self.sampling_rate / original_sr)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Enhance audio
            print("Enhancing audio with DTLN...")
            enhanced_audio = self.enhance_audio(audio_data)
            
            # Save enhanced audio
            print(f"Saving enhanced audio to: {output_path}")
            sf.write(output_path, enhanced_audio, self.sampling_rate)
            
            print("✓ Audio enhancement completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error processing file: {str(e)}")
            return False

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='DTLN Speech Enhancement Processor')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the DTLN model directory')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input noisy audio file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to save enhanced audio file')
    parser.add_argument('--sampling_rate', type=int, default=16000,
                       help='Audio sampling rate (default: 16000)')
    
    args = parser.parse_args()
    
    # Initialize DTLN processor
    print("Initializing DTLN Speech Enhancement Processor...")
    processor = DTLNProcessor(args.model_path, args.sampling_rate)
    
    # Process audio file
    success = processor.process_file(args.input_file, args.output_file)
    
    if success:
        print("🎉 Speech enhancement completed successfully!")
    else:
        print("❌ Speech enhancement failed!")
        return 1
    
    return 0

# Example usage as a module
def enhance_audio_file(model_path: str, input_file: str, output_file: str, 
                      sampling_rate: int = 16000) -> bool:
    """
    Convenience function for enhancing a single audio file
    
    Args:
        model_path: Path to DTLN model directory
        input_file: Path to input noisy audio
        output_file: Path to save enhanced audio
        sampling_rate: Audio sampling rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        processor = DTLNProcessor(model_path, sampling_rate)
        return processor.process_file(input_file, output_file)
    except Exception as e:
        print(f"Enhancement failed: {str(e)}")
        return False

if __name__ == "__main__":
    main()
