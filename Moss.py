import os
import math
import tempfile
import warnings
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore")

# Optional librosa fallback for compressed formats not supported by torchaudio build
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# =============== USER SETTINGS ===============
MODEL_PATH = "path/to/your/MossFormer2_SE_48K_checkpoint.pt"  # Update this to your checkpoint
# ============================================

# ----------------- Safe helpers -----------------
def safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def safe_list_get(lst, idx, default=None):
    try:
        if not isinstance(lst, (list, tuple)) or safe_len(lst) == 0:
            return default
        if idx < 0:
            idx = len(lst) + idx
        if 0 <= idx < len(lst):
            return lst[idx]
        return default
    except Exception:
        return default

# ----------------- Model blocks -----------------
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        if not isinstance(input_dimension, (list, tuple)) or len(input_dimension) != 2:
            raise ValueError("input_dimension must be list/tuple of length 2")
        param_size = [1, input_dimension, 1, input_dimension[9]]
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"LayerNormalization4DCF expects 4D input, got {x.ndim}D")
        mu_ = x.mean(dim=(1, 3), keepdim=True)
        std_ = torch.sqrt(x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps)
        return (x - mu_) / std_ * self.gamma + self.beta

class PositionalEncoding(nn.Module):
    # Register only 'pe' as buffer; do NOT register inv_freq to avoid duplicate buffer conflicts
    def __init__(self, d_model: int, max_len: int = 8000):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        self.scale = nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        if d_model >= 2:
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: [B, S, D]
        S = x.size(1)
        if S > self.pe.size(0):
            pe = self._extend_pe(S, x.device)
            pos_emb = pe[:S, :].transpose(0, 1)
        else:
            pos_emb = self.pe[:S, :].transpose(0, 1)
        return x + pos_emb * self.scale

    def _extend_pe(self, seq_len, device):
        d_model = self.pe.size(2)
        pe = torch.zeros(seq_len, d_model, device=device)
        pos = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        if d_model >= 2:
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0).transpose(0, 1)

class MossFormerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class MossFormer_MaskNet(nn.Module):
    """
    Mel(180) -> transformer blocks -> frequency mask with 961 bins (for 48 kHz STFT with n_fft=1920).
    """
    def __init__(self, in_channels=180, d_model=512, out_freq_bins=961, n_layers=18, nhead=8):
        super().__init__()
        self.conv1d_encoder = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.norm = LayerNormalization4DCF([d_model, 1])
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([MossFormerBlock(d_model, nhead, 2048, 0.1) for _ in range(n_layers)])
        self.conv1d_decoder = nn.Conv1d(d_model, out_freq_bins, kernel_size=1)

    def forward(self, x):
        # x: [B, 180, S]
        x = x.transpose(1, 2)              # [B, S, 180]
        x = x.transpose(1, 2)              # [B, 180, S]
        x = self.conv1d_encoder(x)         # [B, d_model, S]
        x = x.unsqueeze(-1)                # [B, d_model, S, 1]
        x = self.norm(x)                   # [B, d_model, S, 1]
        x = x.squeeze(-1)                  # [B, d_model, S]
        x = x.transpose(1, 2)              # [B, S, d_model]
        x = self.pos_enc(x)                # [B, S, d_model]
        for block in self.blocks:
            x = block(x)                   # [B, S, d_model]
        x = x.transpose(1, 2)              # [B, d_model, S]
        mask = self.conv1d_decoder(x)      # [B, 961, S]
        mask = mask.transpose(1, 2)        # [B, S, 961]
        return mask

class MossFormer2_SE_48K(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MossFormer_MaskNet(in_channels=180, d_model=512, out_freq_bins=961, n_layers=18, nhead=8)

    def forward(self, mel):  # mel: [B, 180, S]
        if mel.dim() != 3 or mel.size(1) != 180:
            raise ValueError(f"Expected mel [B, 180, S], got {tuple(mel.shape)}")
        return self.net(mel)  # [B, S, 961]

# --------------- Robust checkpoint loading ---------------
def load_checkpoint_resilient(model: nn.Module, ckpt_path: str, device: str = "cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    last_error = None
    for map_loc in [device, "cpu"]:
        for weights_only in [False, True]:
            try:
                ckpt = torch.load(ckpt_path, map_location=map_loc, weights_only=weights_only)
                if isinstance(ckpt, dict):
                    for key in ["model_state_dict", "state_dict", "model", "net", "network"]:
                        if key in ckpt and isinstance(ckpt[key], dict):
                            state_dict = ckpt[key]
                            break
                    else:
                        state_dict = ckpt
                else:
                    state_dict = ckpt
                # Normalize keys and keep shape-compatible parameters only
                model_sd = model.state_dict()
                model_keys = set(model_sd.keys())
                cleaned = {}
                for k, v in state_dict.items():
                    if not isinstance(k, str):
                        continue
                    candidates = [k]
                    if k.startswith("module."): candidates.append(k[7:])
                    if k.startswith("model."):  candidates.append(k[6:])
                    if k.startswith("net."):    candidates.append(k[4:])
                    matched_key = None
                    for cand in candidates:
                        if cand in model_keys:
                            matched_key = cand
                            break
                    if matched_key is None:
                        # Heuristic: try tail-suffix match
                        parts = k.split(".")
                        for s in range(len(parts)):
                            tail = ".".join(parts[s:])
                            if tail in model_keys and getattr(v, "shape", None) == model_sd[tail].shape:
                                matched_key = tail
                                break
                    if matched_key and getattr(v, "shape", None) == model_sd[matched_key].shape:
                        cleaned[matched_key] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                print(f"Loaded checkpoint with {len(cleaned)} matched keys; missing={len(missing)}, unexpected={len(unexpected)}")
                model.to(device).eval()
                return
            except Exception as e:
                last_error = e
                continue
    raise RuntimeError(f"Failed to load checkpoint robustly: {last_error}")

# ----------------- Enhancement pipeline -----------------
class Enhancer48k:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 48000
        # Choose n_fft so F = n_fft//2 + 1 = 961 for mask alignment
        self.n_fft = 1920
        self.win_length = 1920
        self.hop_length = 480
        self.n_mels = 180

        self.model = MossFormer2_SE_48K().to(self.device).eval()
        load_checkpoint_resilient(self.model, ckpt_path, self.device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            normalized=True,
            mel_scale="htk",
            f_min=0.0,
            f_max=self.sample_rate // 2,
        ).to(self.device)
        self.window = torch.hann_window(self.win_length, device=self.device)

        # Chunking (long-form support)
        self.chunk_sec = 15.0
        self.overlap_sec = 3.0
        self.chunk_samples = int(self.chunk_sec * self.sample_rate)
        self.overlap_samples = int(self.overlap_sec * self.sample_rate)
        self.hop_samples = max(1, self.chunk_samples - self.overlap_samples)
        fade = max(1, self.overlap_samples // 2)
        self.fade_in = torch.linspace(0, 1, fade, device=self.device)
        self.fade_out = torch.linspace(1, 0, fade, device=self.device)

    # ---- IO ----
    def load_audio(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio not found: {path}")
        try:
            wav, sr = torchaudio.load(path, normalize=False)
        except Exception:
            if not LIBROSA_AVAILABLE:
                raise
            arr, sr = librosa.load(path, sr=None, mono=False)
            if arr.ndim == 1:
                arr = arr[None, :]
            wav = torch.from_numpy(arr).float()
        wav = wav.float()
        # Normalize and mono
        m = wav.abs().max()
        if m > 0:
            wav = wav / m
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample to 48k if needed
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)
        # Headroom
        m = wav.abs().max()
        if m > 0:
            wav = wav / m * 0.95
        return wav.to(self.device)  # [1, T]

    # ---- Features ----
    def mel_from_wave(self, wav_1_T: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wav_1_T)  # [1, 180, S]
        mel = torch.log(mel + 1e-8)
        mean = mel.mean(dim=2, keepdim=True)
        std = mel.std(dim=2, keepdim=True).clamp_min(1e-8)
        return (mel - mean) / std  # [1, 180, S]

    # ---- STFT ----
    def stft(self, wav_1_T: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav_1_T,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # [1, F=961, S]

    def istft(self, stft_c: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            stft_c,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )  # [T]

    # ---- Interpolation helpers (safe shapes) ----
    @staticmethod
    def interp_time(mask_F_S: torch.Tensor, target_S: int) -> torch.Tensor:
        # mask_F_S: [F, S] -> [F, target_S]
        x = mask_F_S.unsqueeze(0).unsqueeze(0)      # [1,1,F,S]
        x = F.interpolate(x, size=(mask_F_S.size(0), target_S), mode="bilinear", align_corners=False)
        return x.squeeze(0).squeeze(0)              # [F, target_S]

    @staticmethod
    def interp_freq(mask_F_S: torch.Tensor, target_F: int) -> torch.Tensor:
        # mask_F_S: [F, S] -> [target_F, S]
        x = mask_F_S.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # [1,1,S,F]
        x = F.interpolate(x, size=(mask_F_S.size(1), target_F), mode="bilinear", align_corners=False)
        return x.squeeze(0).squeeze(0).transpose(0, 1)          # [target_F, S]

    # ---- Chunking ----
    def chunk_indices(self, T: int) -> List[Tuple[int, int]]:
        if T <= self.chunk_samples:
            return [(0, T)]
        idxs, start = [], 0
        while start < T:
            end = min(start + self.chunk_samples, T)
            idxs.append((start, end))
            if end >= T:
                break
            start += self.hop_samples
        return idxs

    def crossfade_blend(self, out_1_T: torch.Tensor, chunk_1_t: torch.Tensor, start: int, end: int):
        t_len = end - start
        if t_len <= 0:
            return
        seg = chunk_1_t[:, :t_len]
        fade_n = min(self.fade_in.numel(), t_len // 2)
        # If overlapping previous region, apply crossfade
        if start > 0 and fade_n > 0:
            seg[:, :fade_n] = seg[:, :fade_n] * self.fade_in[:fade_n]
            out_1_T[:, start:start + fade_n] = out_1_T[:, start:start + fade_n] * self.fade_out[:fade_n] + seg[:, :fade_n]
            if start + fade_n < end:
                out_1_T[:, start + fade_n:end] = seg[:, fade_n:]
        else:
            out_1_T[:, start:end] = seg

    # ---- Core ----
    def enhance_chunk(self, wav_1_t: torch.Tensor) -> torch.Tensor:
        mel = self.mel_from_wave(wav_1_t)                  # [1, 180, S]
        with torch.no_grad():
            mask_B_S_F = self.model(mel)                   # [1, S, 961]
        stft_c = self.stft(wav_1_t)                        # [1, 961, S]
        Fbins, Sframes = stft_c.shape[-2], stft_c.shape[-1]
        mask_F_S = mask_B_S_F.transpose(1, 2).squeeze(0)   # [961, S_mask]
        # Time align
        if mask_F_S.size(1) != Sframes:
            mask_F_S = self.interp_time(mask_F_S, Sframes) # [961, S]
        # Freq align (should already match 961)
        if mask_F_S.size(0) != Fbins:
            mask_F_S = self.interp_freq(mask_F_S, Fbins)   # [F, S]
        # Apply mask
        mag = torch.abs(stft_c)
        phase = torch.angle(stft_c)
        m = torch.sigmoid(mask_F_S).unsqueeze(0)           # [1, F, S]
        alpha = 0.1
        enh_mag = alpha * mag + (1 - alpha) * mag * m
        enh_stft = enh_mag * torch.exp(1j * phase)
        enh_wav_T = self.istft(enh_stft.squeeze(0), length=wav_1_t.size(1))  # [t]
        return enh_wav_T.unsqueeze(0)                      # [1, t]

    def enhance(self, input_path: str, output_path: str) -> str:
        wav = self.load_audio(input_path)  # [1, T]
        T = wav.size(1)
        out = torch.zeros_like(wav)
        for (s, e) in self.chunk_indices(T):
            chunk = wav[:, s:e]
            enh = self.enhance_chunk(chunk)               # [1, t]
            self.crossfade_blend(out, enh, s, e)
        # DC removal & normalization
        out = out - out.mean()
        m = out.abs().max()
        if m > 0:
            out = out / m * 0.95
        torchaudio.save(output_path, out.cpu().float(), self.sample_rate, encoding="PCM_S", bits_per_sample=16)
        return output_path

# ----------------- Gradio UI -----------------
import gradio as gr

enhancer_instance: Optional[Enhancer48k] = None

def initialize_enhancer_ui(ckpt_path: str) -> str:
    global enhancer_instance
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhancer_instance = Enhancer48k(ckpt_path, device)
        return f"✅ Model loaded on {device}. Ready for long-form enhancement at 48 kHz."
    except Exception as e:
        return f"❌ Initialization error: {str(e)}"

def process_with_enhancer(input_audio_path: str) -> Tuple[Optional[str], str]:
    global enhancer_instance
    if enhancer_instance is None:
        return None, "❌ Model not initialized. Check checkpoint path and click 'Reload model'."
    if not input_audio_path or not os.path.exists(input_audio_path):
        return None, "❌ Please upload a valid audio file."
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        result = enhancer_instance.enhance(input_audio_path, out_path)
        return result, "✅ Enhancement complete."
    except Exception as e:
        return None, f"❌ Processing error: {str(e)}"

def launch_ui():
    with gr.Blocks(title="MossFormer2_SE_48K - Speech Enhancement (48 kHz)") as demo:
        gr.Markdown("### MossFormer2_SE_48K Speech Enhancement UI — Upload any audio (e.g., 16 kHz WAV/MP3/FLAC) and get an enhanced 48 kHz WAV out.")
        with gr.Row():
            with gr.Column(scale=1):
                ckpt = gr.Textbox(label="Checkpoint path (.pt/.pth)", value=MODEL_PATH)
                status = gr.Textbox(label="Model status", interactive=False)
                reload_btn = gr.Button("Reload model", variant="primary")
                reload_btn.click(fn=initialize_enhancer_ui, inputs=[ckpt], outputs=[status])
                # Initialize once at launch
                status.value = initialize_enhancer_ui(MODEL_PATH)
            with gr.Column(scale=1):
                input_audio = gr.Audio(label="Upload audio (any common format; 16 kHz WAV typical)", type="filepath", sources=["upload"])
                enhance_btn = gr.Button("Enhance")
                output_audio = gr.Audio(label="Enhanced output (48 kHz WAV)", type="filepath")
                log = gr.Textbox(label="Logs", interactive=False)
                enhance_btn.click(fn=process_with_enhancer, inputs=[input_audio], outputs=[output_audio, log])
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True, quiet=False)

if __name__ == "__main__":
    launch_ui()
