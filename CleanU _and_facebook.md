I'll help you find the official download links for both CleanUNet and Facebook Denoiser models.Based on my research, here are the official download links and installation instructions for both CleanUNet and Facebook Denoiser models:

## **CleanUNet (NVIDIA)**

### **Official Repository & Models**
- **Main Repository**: https://github.com/NVIDIA/CleanUNet
- **Paper**: Speech Denoising in the Waveform Domain with Self-Attention (ICASSP 2022)

### **Pre-trained Models**
Pre-trained models for denoising are provided in `./exp/${EXP}/checkpoint/pretrained.pkl` (each one has size ~177Mb; use git lfs to download)

**Available Models:**
- `DNS-large-full` - Full model variant
- `DNS-large-high` - High-quality variant

### **Installation & Usage**
```bash
# Clone the repository
git clone https://github.com/NVIDIA/CleanUNet
cd CleanUNet

# Install dependencies
pip install numpy scipy torch torchvision torchaudio pesq pystoi tqdm

# Download pre-trained models (requires git lfs)
git lfs pull

# Use pre-trained model for denoising
python denoise.py -c configs/DNS-large-high.json --ckpt_iter pretrained
```

### **Alternative Implementation**
- **Easy-to-use wrapper**: https://github.com/Kabir5296/Audio-Denoiser
```bash
git clone https://github.com/Kabir5296/Speech-Denoiser-System.git
cd Speech-Denoiser-System
pip install -r requirements.txt
python main.py
```

---

## **Facebook Denoiser**

### **Official Repository & Models**
- **Main Repository**: https://github.com/facebookresearch/denoiser
- **Paper**: Real Time Speech Enhancement in the Waveform Domain (Interspeech 2020)
- **Demo Page**: https://facebookresearch.github.io/denoiser/

### **Pre-trained Models (Auto-download)**
The models are automatically downloaded from Facebook's servers when using the official package:

- **DNS48**: https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th
- **DNS64**: https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th
- **Master64**: https://dl.fbaipublicfiles.com/adiyoss/denoiser/master64-8a5dfb4bb92753dd.th

### **Installation & Usage**
```bash
# Install the official package (easiest method)
pip install denoiser

# Use pre-trained models directly
python -m denoiser.enhance --dns64 --noisy_dir /path/to/noisy --out_dir /path/to/output

# OR clone repository for customization
git clone https://github.com/facebookresearch/denoiser
cd denoiser
pip install -e .
```

### **Available Model Variants**
- `--dns48`: Real time H=48 model trained on DNS
- `--dns64`: Real time H=64 model trained on DNS 
- `--master64`: Real time H=64 model trained on DNS and Valentini

### **Real-time Usage**
```bash
# For real-time audio processing
python -m denoiser.live --dns64
```

---

## **Quick Comparison**

| Feature | CleanUNet | Facebook Denoiser |
|---------|-----------|-------------------|
| **Installation** | Manual setup required | `pip install denoiser` |
| **Models** | Manual download with git lfs | Auto-download |
| **Real-time** | Batch processing | ✅ Real-time capable |
| **GPU Memory** | Very GPU hungry (up to 10GB for 3min audio) | Memory efficient |
| **Quality** | High quality denoising | Optimized for speed |

## **Recommendation**

For your scripts:
- **Facebook Denoiser** is easier to get started with - just `pip install denoiser`
- **CleanUNet** requires more setup but may provide higher quality results
- Both have pre-trained models ready to use without training from scratch

The Facebook Denoiser is particularly well-suited for your use case since it's designed for real-time processing and has automatic model downloading built-in!
