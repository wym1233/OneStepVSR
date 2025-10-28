

# OneStepVSR: Efficient Perceptual Video Super-Resolution via One-Step Diffusion Denoising

Official implementation of  
**"Efficient Perceptual Video Super-Resolution via One-Step Diffusion Denoising"** 

## 🔍 Overview

This project proposes **OneStepVSR**, a diffusion-based video super-resolution (VSR) framework that achieves a new balance among **perceptual quality**, **temporal consistency**, and **efficiency**.

Our key idea:
- Use **SDTurbo** as a one-step diffusion prior.
- Introduce **Temporal Attention Adapter** and **Bidirectional Recurrent Propagation Layer** for inter-frame interaction.
- Apply **LoRA fine-tuning** for efficient adaptation with only 1/8 training parameters.
- 16× faster than multi-step diffusion methods (e.g., StableVSR), while maintaining comparable perceptual quality.

## 🧩 Framework Overview
<img width="928" height="275" alt="framework" src="https://github.com/user-attachments/assets/ad225391-0cde-4d51-9d46-e95aa9472c5f" />

**Pipeline Summary:**

1. A BasicVSR preprocessing module enhances LR frames.
2. SDTurbo is fine-tuned with LoRA adapters.
3. Temporal Attention performs inter-frame fusion.
4. Recurrent Propagation aligns features via optical flow (RAFT).
5. Mixed loss combines L2, LPIPS, GAN, and warping constraints.

## 📈 Results
<img width="786" height="496" alt="image" src="https://github.com/user-attachments/assets/b8a5d716-3023-4fc9-b84d-a656c981be27" />
<img width="691" height="156" alt="image" src="https://github.com/user-attachments/assets/6fa1f847-2fed-432a-b7a2-b015954963d2" />


## 📦 Environment Setup

### 1. Create environment

```bash
git clone https://github.com/wym1233/OneStepVSR.git
cd OneStepVSR
pip install -r requirements.txt
```

## ⚙️ Training / 🚀 Inference / 📊 Evaluation
Please modify the default parameter values in the parse_args() function according to your personal environment.
```bash
python /src/train.py

# Inference
python /src/inference_onestepvsr.py

# Evaluation
python /src/eval.py
```

## 📁 Dataset Preparation

We use REDS/Vimeo benchmarks for training and REDS/UDM10/Vimeo benchmarks for inference:

| Dataset                                                    | Usage                         |
| ---------------------------------------------------------- | ----------------------------- |
| [REDS](https://seungjunnah.github.io/Datasets/reds.html)   | Train / Test (REDS4 / REDS30) |
| [UDM10](https://mmagic.readthedocs.io/en/latest/dataset_zoo/udm10.html) | Test only                     |
| [Vimeo-90K](http://toflow.csail.mit.edu/)                  | Train / Test                  |


Organize as follows:

```
data/
├── REDS/
|   |-- REDS4
|   |   |-- train_sharp
|   |   `-- train_sharp_bicubic
|   |-- train
|   |   |-- train_sharp
|   |   `-- train_sharp_bicubic
|    `-- val
|       |-- val_sharp
|       `-- val_sharp_bicubic
├── UDM10/
|   |-- BIx4
|   `-- GT
└── Vimeo-90K/
|   |-- vimeo_test
|   `-- vimeo_train
```

## 💾 Pretrained Models
Download pretrained checkpoints from:
📂 [google drive](https://drive.google.com/drive/folders/1NG2yhDGJYyH6Af-nkWfnwWP6mw_h0dot?usp=drive_link)

## 🧩 Related Repositories
We thank the following projects for inspiration and code reference:
* [sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
* [S3Diff (Degradation-Guided Diffusion SR)](https://github.com/ArcticHare105/S3Diff)
* [StableVSR](https://github.com/claudiom4sir/StableVSR)
* [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)
* [DOVER](https://github.com/VQAssessment/DOVER)
* [BasicSR](https://github.com/XPixelGroup/BasicSR)
* [BasicVSR-IconVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR)

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wang2025onestepvsr,
  title     = {Efficient Perceptual Video Super-Resolution via One-Step Diffusion Denoising},
  author    = {Wang, Yiming and Lan, Yunwei and Liu, Dong},
  booktitle = {Proceedings of the IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  year      = {2025},
  note      = {to appear}
}

```


