# Environment Setup Guide for "Build Your Own GPT"

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **Python**: Version 3.8 or higher

### Required Accounts
- [GitHub Account](https://github.com)
- [OpenAI Account](https://platform.openai.com) (for API access)
- [Hugging Face Account](https://huggingface.co) (optional, for model hosting)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/build-your-own-gpt.git
cd build-your-own-gpt
```

### 2. Python Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda create -n mygpt python=3.9
conda activate mygpt
```

#### Option B: Using venv
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

#### Core Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For GPU support (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Project Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist yet:
```bash
pip install transformers datasets accelerate sentencepiece
pip install jupyter notebook streamlit gradio
pip install scikit-learn pandas numpy matplotlib
pip install tiktoken wandb
```

### 4. Environment Configuration

#### Create `.env` file
```bash
cp .env.example .env
```

#### Edit `.env` with your credentials:
```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face (optional)
HUGGINGFACE_TOKEN=your_hf_token_here
HUGGINGFACE_REPO_ID=your-username/your-model-name

# Training Configuration
MODEL_NAME=gpt2
DATASET_NAME=wikitext
DATASET_CONFIG=wikitext-2-raw-v1

# Hardware Settings
USE_GPU=True
FP16=True
```

## 🔧 Advanced Setup

### GPU Setup (Optional but Recommended)

#### NVIDIA GPU Users
1. Check CUDA compatibility:
```bash
nvidia-smi
```

2. Install CUDA Toolkit (if needed):
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Install corresponding cuDNN version

3. Verify PyTorch GPU support:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### Apple Silicon (M1/M2/M3) Users
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio
```

### Docker Setup (Alternative)

#### Using Docker
```bash
# Build the image
docker build -t mygpt .

# Run the container
docker run -p 8888:8888 -p 8501:8501 mygpt
```

#### Using Docker Compose
```bash
docker-compose up
```

## 📊 Dataset Preparation

### Download Sample Dataset
```bash
python scripts/download_data.py
```

### Prepare Custom Dataset
1. Place your text files in `data/raw/`
2. Run preprocessing:
```bash
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

## 🧪 Verification

### Test Your Setup
```bash
# Run basic tests
python tests/test_environment.py

# Test GPU availability
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Test imports
python scripts/verify_imports.py
```

### Expected Output
```
✓ PyTorch installed: 2.0.1
✓ Transformers installed: 4.30.2
✓ GPU available: True
✓ Environment variables loaded
```

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Problems
```bash
# Reset GPU memory
nvidia-smi --gpu-reset

# Check GPU compatibility
python -c "import torch; print(torch.cuda.get_device_capability())"
```

#### 2. Memory Issues
- Reduce batch size in `config/training_config.yaml`
- Enable gradient accumulation
- Use mixed precision training

#### 3. Dependency Conflicts
```bash
# Create fresh environment
conda env remove -n mygpt
conda create -n mygpt python=3.9
conda activate mygpt
pip install -r requirements.txt
```

### Platform-Specific Issues

#### Windows
```powershell
# If facing SSL errors
conda config --set ssl_verify no

# For path issues
set PYTHONPATH=%cd%
```

#### macOS
```bash
# If facing libomp errors
brew install libomp

# For Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Fix permission issues
sudo chown -R $USER:$USER .
```

## 🔄 Updates

### Keeping Dependencies Updated
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade transformers torch
```

### Pull Latest Code
```bash
git pull origin main
pip install -r requirements.txt
```

## 🎯 Next Steps

After successful setup:

1. **Explore the notebooks**:
```bash
jupyter notebook notebooks/
```

2. **Start training**:
```bash
python train.py --config config/training_config.yaml
```

3. **Launch the web interface**:
```bash
streamlit run app/chat_interface.py
```

4. **Monitor training**:
```bash
# If using Weights & Biases
wandb login
python train.py --with_wandb
```

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/)
- [Project Wiki](../../wiki)

## ❓ Getting Help

If you encounter issues:
1. Check the [Troubleshooting Guide](../../wiki/Troubleshooting)
2. Search [existing issues](../../issues)
3. Create a [new issue](../../issues/new) with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

**Happy Building! 🚀**
