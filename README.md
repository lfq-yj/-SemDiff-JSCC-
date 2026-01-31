# SemDiff-JSCC

<div align="center">

**Wireless Image Transmission via Joint Source-Channel Coding Guided by Diffusion Models and Multimodal Semantics**

[![Paper](https://img.shields.io/badge/Paper-JEI%20(Under%20Review)-blue)](https://github.com/lfq-yj/-SemDiff-JSCC-)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Fuqiang Liu, Lin Ma, Yun Jia***
<br>
*School of Information and Electronic Engineering, Shandong Technology and Business University*

</div>

---

## 📖 Abstract

This repository is the official PyTorch implementation of the paper **"Wireless Image Transmission via Joint Source-Channel Coding Guided by Diffusion Models and Multimodal Semantics"**.

We propose **SemDiff-JSCC**, a novel framework that integrates deep joint source-channel coding (Deep JSCC) with diffusion models to address the "cliff effect" and perceptual quality degradation in low SNR conditions.

**Key Features:**
- **Multimodal Guidance:** Leverages text prompts and edge maps to guide the diffusion decoder.
- **OmniControl Mechanism:** Parameter-efficient adaptation using LoRA-like strategies.
- **Blind Channel Estimation:** A pilot-free mechanism to estimate SNR and adapt to fading channels without explicit CSI.

<div align="center">
  <img src="https://github.com/lfq-yj/-SemDiff-JSCC-/blob/master/figure2.png" width="800"/>
  <br>
  <em>Figure 1: The overall architecture of the proposed SemDiff-JSCC framework.</em>
</div>

---

## 🔧 Installation

### Requirements
- Linux or Windows
- Python 3.8+
- PyTorch 1.13+ (CUDA recommended)
- NVIDIA GPU (for fast inference)

### Setup
```bash
# Clone the repository
git clone https://github.com/lfq-yj/-SemDiff-JSCC-.git
cd -SemDiff-JSCC-

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Data Preparation

We use the MS-COCO 2017 dataset for evaluation. Please download the dataset and organize it as follows:

```
data/
  └── coco/
      ├── val2017/         # Validation images
      └── annotations/     # JSON annotation files
```

For edge extraction, the Canny edge detector is implemented in `models/model_canny.py`.

---

## 🚀 Usage

### Inference / Evaluation

To evaluate the model on the validation set under specific channel conditions (e.g., AWGN, SNR=10dB):

```bash
python inference_one.py --config configs/inference.yaml --snr 10 --channel awgn
```

---

## 📊 Results

### Visual Comparison

<div align="center">
  <img src="https://github.com/lfq-yj/-SemDiff-JSCC-/blob/master/figure8.png" width="800"/>
  <br>
  <em>Figure 2: Visual comparison at SNR = -5/5dB (AWGN).</em>
</div>

---

## 📝 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{jia2026semdiff,
  title={Wireless Image Transmission via Joint Source-Channel Coding Guided by Diffusion Models and Multimodal Semantics},
  author={Liu, Fuqiang and Ma, Lin and Jia, Yun},
  journal={Journal of Electronic Imaging (Under Review)},
  year={2026}
}
```

---

## 📧 Contact

For any questions, please contact Yun Jia at `jiayun0223@sdtbu.edu.cn`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
