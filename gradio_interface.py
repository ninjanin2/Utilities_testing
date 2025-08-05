import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
from datetime import datetime
import os

# Custom CSS for modern, professional styling
custom_css = """
/* Global Styles */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* Main container styling */
.main-container {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    margin: 20px !important;
    padding: 30px !important;
}

/* Header styling */
.header-text {
    text-align: center !important;
    color: #2c3e50 !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.subtitle-text {
    text-align: center !important;
    color: #7f8c8d !important;
    font-size: 1.1rem !important;
    margin-bottom: 30px !important;
    font-weight: 400 !important;
}

/* Input/Output styling */
.input-audio {
    border-radius: 15px !important;
    border: 2px dashed #667eea !important;
    background: #f8f9ff !important;
    padding: 20px !important;
}

.output-text {
    border-radius: 15px !important;
    border: 1px solid #e0e6ed !important;
    background: #ffffff !important;
    font-family: 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 20px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06) !important;
}

/* Button styling */
.submit-btn {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Progress bar styling */
.progress-bar {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border-radius: 10px !important;
}

/* Footer styling */
.footer-text {
    text-align: center !important;
    color: #95a5a6 !important;
    font-size: 0.9rem !important;
    margin-top: 30px !important;
    padding-top: 20px !important;
    border-top: 1px solid #ecf0f1 !important;
}

/* Loading spinner */
.loading {
    display: inline-block !important;
    width: 20px !important;
    height: 20px !important;
    border: 3px solid #f3f3f3 !important;
    border-top: 3px solid #667eea !important;
    border-radius: 50% !important;
    animation: spin 1s linear infinite !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
    .header-text {
        font-size: 2rem !important;
    }
    
    .main-container {
        margin: 10px !important;
        padding: 20px !important;
    }
}
"""

class ASRProcessor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        
        print(f"Loading ASR model on {self.device}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        print("ASR model loaded successfully!")

# Initialize the ASR processor
asr_processor = ASRProcessor()

def transcribe_audio(audio_file):
    """
    Transcribe audio file to text using Whisper model
    """
    try:
        if audio_file is None:
            return "❌ Please upload an audio file first."
        
        # Load audio file
        audio_path = audio_file.name if hasattr(audio_file, 'name') else audio_file
        
        # Get file info
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
        
        print(f"Processing audio file: {audio_path}")
        print(f"File size: {file_size:.2f} MB")
        
        # Transcribe using the pipeline
        result = asr_processor.pipe(audio_path)
        
        # Extract transcription text
        transcription = result["text"]
        
        # Format the output with metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted_output = f"""
🎵 TRANSCRIPTION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Transcribed Text:
{transcription}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 File Information:
• File Size: {file_size:.2f} MB
• Processed: {timestamp}
• Model: Whisper Large v3
• Device: {asr_processor.device.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip()
        
        return formatted_output
        
    except Exception as e:
        error_msg = f"""
❌ TRANSCRIPTION ERROR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 Error Details:
{str(e)}

💡 Troubleshooting Tips:
• Ensure the audio file is in a supported format (MP3, WAV, M4A, etc.)
• Check that the file is not corrupted
• Try with a smaller file size
• Supported formats: MP3, WAV, FLAC, M4A, OGG, WEBM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip()
        return error_msg

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css, title="Professional ASR Transcription") as demo:
        
        # Header
        gr.HTML("""
            <div class="main-container">
                <h1 class="header-text">🎙️ Professional ASR Transcription</h1>
                <p class="subtitle-text">Upload your audio file and get accurate transcriptions powered by OpenAI Whisper</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Audio input
                audio_input = gr.Audio(
                    label="📁 Upload Audio File",
                    type="filepath",
                    elem_classes=["input-audio"]
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "🚀 Transcribe Audio",
                    variant="primary",
                    elem_classes=["submit-btn"]
                )
                
                # Audio info
                gr.HTML("""
                    <div style="margin-top: 15px; padding: 15px; background: #f8f9ff; border-radius: 10px; border-left: 4px solid #667eea;">
                        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📋 Supported Formats:</h4>
                        <p style="margin: 0; color: #7f8c8d;">MP3, WAV, FLAC, M4A, OGG, WEBM, MP4 (audio track)</p>
                    </div>
                """)
            
            with gr.Column(scale=2):
                # Output text
                output_text = gr.Textbox(
                    label="📄 Transcription Result",
                    placeholder="Your transcribed text will appear here...",
                    lines=20,
                    max_lines=25,
                    elem_classes=["output-text"],
                    show_copy_button=True
                )
        
        # Footer
        gr.HTML("""
            <div class="footer-text">
                <p>🤖 Powered by OpenAI Whisper Large v3 | Built with ❤️ using Gradio</p>
                <p>For best results, use clear audio with minimal background noise</p>
            </div>
        """)
        
        # Event handler
        submit_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output_text],
            show_progress=True
        )
        
        # Also allow transcription on file upload
        audio_input.change(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output_text],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("🚀 Starting Professional ASR Transcription App...")
    print(f"🔧 Device: {asr_processor.device}")
    print(f"🧠 Model: {asr_processor.model_id}")
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        show_error=True,
        show_tips=True,
        enable_queue=True,
        max_threads=4
    )