# SemDiff-JSCC

<div align="center">

**Wireless Image Transmission via Joint Source-Channel Coding Guided by Diffusion Models and Multimodal Semantics**

[![Paper](https://img.shields.io/badge/Paper-JEI%20(Under%20Review)-blue)](https://github.com/lfq-yj/-SemDiff-JSCC-)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Fuqiang Liu, Lin Ma, Yun Jia\***
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
- **Superior Performance:** Achieves SOTA perceptual quality (FID/LPIPS) compared to traditional JPEG+LDPC and existing Deep JSCC methods.

<div align="center">
  <img src="figures/framework.png" width="800"/>
  <br>
  <em>Figure 1: The overall architecture of the proposed SemDiff-JSCC framework.</em>
</div>

---

## 🔧 Installation

### Requirements
- Linux or Windows
- Python 3.8+
- PyTorch 1.13+ (CUDA recommended)
- NVIDIA GPU (for training and fast inference)

### Setup
```bash
# Clone the repository
git clone [https://github.com/lfq-yj/-SemDiff-JSCC-.git](https://github.com/lfq-yj/-SemDiff-JSCC-.git)
cd -SemDiff-JSCC-

# Install dependencies
pip install -r requirements.txt
