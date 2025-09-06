import os
import math
import tempfile
import warnings
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore")

# Optional librosa fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# =========================
# User setting
# =========================
MODEL_PATH = "path/to/your/MossFormer2_SE_48K_checkpoint.pt"  # update this

# =========================
# Safe helpers
# =========================
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

# =========================
# Model blocks
# =========================
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        if not isinstance(input_dimension, (list, tuple)) or len(input_dimension) != 2:
            raise ValueError("input_dimension must be list/tuple of length 2")
        param_size = [1, input_dimension, 1, input_dimension[25]]
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"LayerNormalization4DCF expects 4D, got {x.ndim}D")
        mu_ = x.mean(dim=(1, 3), keepdim=True)
        std_ = torch.sqrt(x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps)
        return (x - mu_) / std_ * self.gamma + self.beta

class PositionalEncoding(nn.Module):
    # Only 'pe' is a buffer; no inv_freq buffer to avoid duplication conflicts
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
        pe = pe.unsqueeze(0).transpose(0, 1)
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
    Predicts a [B, T, Fmask] mask from input mel features, where input is [B, N=180, S] and output Fmask=961 for 48k STFT (n_fft=1920) alignment.
    """
    def __init__(self, in_channels=180, d_model=512, out_freq_bins=961, n_layers=18, nhead=8):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.out_freq_bins = out_freq_bins

        # Encoder to d_model
        self.conv1d_encoder = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.norm = LayerNormalization4DCF([d_model, 1])
        self.pos_enc = PositionalEncoding(d_model)

        # Sequence blocks
        self.blocks = nn.ModuleList([MossFormerBlock(d_model, nhead, 2048, 0.1) for _ in range(n_layers)])

        # Decoder back to frequency bins
        self.conv1d_decoder = nn.Conv1d(d_model, out_freq_bins, kernel_size=1)

    def forward(self, x):
        # x: [B, N=180, S]
        x = x.transpose(1, 2)            # [B, S, N]
        x = x.transpose(1, 2)            # [B, N, S]
        x = self.conv1d_encoder(x)       # [B, d_model, S]
        x = x.unsqueeze(-1)              # [B, d_model, S, 1]
        x = self.norm(x)                 # [B, d_model, S, 1]
        x = x.squeeze(-1)                # [B, d_model, S]
        x = x.transpose(1, 2)            # [B, S, d_model]
        x = self.pos_enc(x)              # [B, S, d_model]
        for block in self.blocks:
            x = block(x)                 # [B, S, d_model]
        x = x.transpose(1, 2)            # [B, d_model, S]
        mask = self.conv1d_decoder(x)    # [B, Fmask, S]
        mask = mask.transpose(1, 2)      # [B, S, Fmask]  where Fmask=961
        return mask

class MossFormer2_SE_48K(nn.Module):
    """
    Wrapper that outputs a mask tensor directly (no lists) to avoid index errors.
    """
    def __init__(self):
        super().__init__()
        self.net = MossFormer_MaskNet(in_channels=180, d_model=512, out_freq_bins=961, n_layers=18, nhead=8)

    def forward(self, mel):  # mel: [B, 180, S]
        if mel.dim() != 3 or mel.size(1) != 180:
            raise ValueError(f"Expected mel [B, 180, S], got {tuple(mel.shape)}")
        return self.net(mel)  # [B, S, 961]

# =========================
# Checkpoint loading (robust)
# =========================
def load_checkpoint_resilient(model: nn.Module, ckpt_path: str, device: str = "cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # Try multiple load patterns
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
                # Normalize keys
                model_keys = set(model.state_dict().keys())
                cleaned = {}
                for k, v in state_dict.items():
                    if not isinstance(k, str):
                        continue
                    candidates = [k]
                    if k.startswith("module."):
                        candidates.append(k[7:])
                    if k.startswith("model."):
                        candidates.append(k[6:])
                    if k.startswith("net."):
                        candidates.append(k[4:])
                    # Try suffix match for stubborn prefixes
                    matched_key = None
                    for cand in candidates:
                        if cand in model_keys:
                            matched_key = cand
                            break
                    if matched_key is None:
                        # Suffix heuristic: match the tail if unique and shapes agree
                        parts = k.split(".")
                        for s in range(len(parts)):
                            tail = ".".join(parts[s:])
                            if tail in model_keys:
                                if tail in model.state_dict() and getattr(v, "shape", None) == model.state_dict()[tail].shape:
                                    matched_key = tail
                                    break
                    if matched_key and matched_key in model.state_dict() and getattr(v, "shape", None) == model.state_dict()[matched_key].shape:
                        cleaned[matched_key] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                print(f"Loaded with {len(cleaned)} matched keys; missing={len(missing)}, unexpected={len(unexpected)}")
                model.to(device).eval()
                return
            except Exception as e:
                last_error = e
                continue
    raise RuntimeError(f"Failed to load checkpoint robustly: {last_error}")

# =========================
# Enhancement pipeline
# =========================
class Enhancer48k:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # STFT aligned to 961 bins (n_fft=1920)
        self.sample_rate = 48000
        self.n_fft = 1920
        self.win_length = 1920
        self.hop_length = 480
        self.n_mels = 180
        self.model = MossFormer2_SE_48K().to(self.device).eval()
        load_checkpoint_resilient(self.model, ckpt_path, self.device)

        # Transforms
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

        # Chunking
        self.chunk_sec = 15.0
        self.overlap_sec = 3.0
        self.chunk_samples = int(self.chunk_sec * self.sample_rate)
        self.overlap_samples = int(self.overlap_sec * self.sample_rate)
        self.hop_samples = max(1, self.chunk_samples - self.overlap_samples)
        fade = self.overlap_samples // 2 if self.overlap_samples > 0 else 1
        self.fade_in = torch.linspace(0, 1, fade, device=self.device)
        self.fade_out = torch.linspace(1, 0, fade, device=self.device)

    # ---------- IO ----------
    def load_audio(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio not found: {path}")
        try:
            wav, sr = torchaudio.load(path, normalize=False)
        except Exception:
            if not LIBROSA_AVAILABLE:
                raise
            arr, sr = librosa.load(path, sr=None, mono=False)
            wav = torch.from_numpy(arr if arr.ndim > 1 else arr[None, :]).float()
        wav = wav.float()
        # Normalize to [-1, 1]
        m = wav.abs().max()
        if m > 0:
            wav = wav / m
        # Mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample to 48k
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
        # Final headroom
        m = wav.abs().max()
        if m > 0:
            wav = wav / m * 0.95
        return wav.to(self.device)  # [1, T]

    # ---------- Features ----------
    def mel_from_wave(self, wav_1_T: torch.Tensor) -> torch.Tensor:
        # wav_1_T: [1, T] on device; return [1, 180, S]
        mel = self.mel_transform(wav_1_T)
        mel = torch.log(mel + 1e-8)
        mean = mel.mean(dim=2, keepdim=True)
        std = mel.std(dim=2, keepdim=True).clamp_min(1e-8)
        mel = (mel - mean) / std
        return mel

    # ---------- STFT ----------
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

    # ---------- Chunking ----------
    def chunk_indices(self, T: int) -> List[Tuple[int, int]]:
        if T <= self.chunk_samples:
            return [(0, T)]
        idxs = []
        start = 0
        while start < T:
            end = min(start + self.chunk_samples, T)
            idxs.append((start, end))
            if end >= T:
                break
            start += self.hop_samples
        return idxs

    def crossfade_blend(self, out_wave_1_T: torch.Tensor, chunk_wave_1_t: torch.Tensor, start: int, end: int):
        # out_wave_1_T is target buffer [1, T]; chunk_wave_1_t is [1, t_len]
        t_len = end - start
        if t_len <= 0:
            return
        seg = chunk_wave_1_t[:, :t_len]
        # crossfade at chunk boundaries
        fade_n = min(self.fade_in.numel(), t_len // 2)
        if fade_n > 0:
            # fade-in at start if overlapping previous
            if start > 0:
                seg[:, :fade_n] = seg[:, :fade_n] * self.fade_in[:fade_n]
                out_wave_1_T[:, start:start + fade_n] = out_wave_1_T[:, start:start + fade_n] * self.fade_out[:fade_n] + seg[:, :fade_n]
                out_wave_1_T[:, start + fade_n:end] = seg[:, fade_n:]
                return
            # fade-out at end if not last
            if end < out_wave_1_T.size(1):
                seg[:, -fade_n:] = seg[:, -fade_n:] * self.fade_out[:fade_n]
        out_wave_1_T[:, start:end] = seg

    # ---------- Core ----------
    def enhance_chunk(self, wav_1_t: torch.Tensor) -> torch.Tensor:
        # wav_1_t: [1, t]
        mel = self.mel_from_wave(wav_1_t)              # [1, 180, S]
        with torch.no_grad():
            mask_B_S_F = self.model(mel)               # [1, S, 961]
        stft_c = self.stft(wav_1_t)                    # [1, 961, S]
        # Align dims: mask [1, S, F] -> [1, F, S]
        mask_F_S = mask_B_S_F.transpose(1, 2).squeeze(0)           # [961, S_mask]
        F, S = stft_c.shape[-2], stft_c.shape[-1]
        # Time align
        if mask_F_S.size(1) != S:
            mask_F_S = F_interpolate_time(mask_F_S, S)             # [961, S]
        # Freq align
        if mask_F_S.size(0) != F:
            mask_F_S = F_interpolate_freq(mask_F_S, F)             # [F, S]
        # Apply sigmoid mask to magnitude with small dry mix
        mag = torch.abs(stft_c)
        phase = torch.angle(stft_c)
        m = torch.sigmoid(mask_F_S).unsqueeze(0)                    # [1, F, S]
        alpha = 0.1
        enh_mag = alpha * mag + (1 - alpha) * mag * m
        enh_stft = enh_mag * torch.exp(1j * phase)
        enh_wav_T = self.istft(enh_stft.squeeze(0), length=wav_1_t.size(1))  # [t]
        return enh_wav_T.unsqueeze(0)  # [1, t]

    def enhance(self, input_path: str, output_path: str) -> str:
        wav = self.load_audio(input_path)    # [1, T]
        T = wav.size(1)
        out = torch.zeros_like(wav)
        for (s, e) in self.chunk_indices(T):
            chunk = wav[:, s:e]
            # pad tail to window length for ISTFT stability
            enh = self.enhance_chunk(chunk)  # [1, t]
            self.crossfade_blend(out, enh, s, e)
        # Final normalization & DC removal
        out = out - out.mean()
        m = out.abs().max()
        if m > 0:
            out = out / m * 0.95
        torchaudio.save(output_path, out.cpu().float(), self.sample_rate, encoding="PCM_S", bits_per_sample=16)
        return output_path

# Helper interpolation for alignment
def F_interpolate_time(mask_F_S: torch.Tensor, target_S: int) -> torch.Tensor:
    # mask_F_S: [F, S] -> interpolate along S
    return F.interpolate(mask_F_S.unsqueeze(0), size=(mask_F_S.size(0), target_S), mode="bilinear", align_corners=False).squeeze(0)

def F_interpolate_freq(mask_F_S: torch.Tensor, target_F: int) -> torch.Tensor:
    # mask_F_S: [F, S] -> interpolate along F
    mask_S_F = mask_F_S.transpose(0, 1).unsqueeze(0)  # [1, S, F]
    mask_S_F = F.interpolate(mask_S_F, size=(mask_S_F.size(1), target_F), mode="bilinear", align_corners=False)
    return mask_S_F.squeeze(0).transpose(0, 1)        # [F, S]

# ================
# Minimal CLI
# ================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=MODEL_PATH, help="Path to MossFormer2_SE_48K checkpoint (.pt/.pth)")
    parser.add_argument("--in", dest="inp", type=str, required=True, help="Input audio path (any common format)")
    parser.add_argument("--out", dest="out", type=str, default=None, help="Output WAV path (48kHz)")
    args = parser.parse_args()

    enhancer = Enhancer48k(args.ckpt)
    out_path = args.out or os.path.splitext(args.inp) + "_enhanced_48k.wav"
    result = enhancer.enhance(args.inp, out_path)
    print(f"Saved: {result}")
