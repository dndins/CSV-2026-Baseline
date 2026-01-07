# 🩺 CSV 2026 Challenge Baseline

[![CSV 2026 Challenge Site](https://img.shields.io/badge/Official-FETUS%20Challenge-red?style=for-the-badge)](http://www.csv-isbi.net/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**🏆 Official Baseline for the CSV 2026 Challenge** —This repository provides the official baseline implementation for the **CSV Challenge**, It implements a semi-supervised UniMatch pipeline for **Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound**, and supports two backbones: a lightweight UNet and the high-performance Echocare encoder (SwinUNETR-based). For official rules, dataset downloads and the evaluation server, visit the [CSV 2026 challenge page](http://www.csv-isbi.net/)

---

## 📁 1. Prepare Data

Place the downloaded 🖼️ training archive (provided by organizers) and unzip it to the 'data/' directory. 
**🤖 Pre-trained Weights**: For Echocare model training, download the pre-trained Echocare encoder weights from [this link](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg) and place the `echocare_encoder.pth` file in the `pretrain/` directory. so the directory structure becomes:
```text
CSV2026_Baseline/
├─ data/
|  └─ train/
|     ├─ images/        # .h5 image files (long_img & trans_img)
|     └─ labels/        # _label.h5 files (long_mask, trans_mask, cls)
└─ pretrain/ 
   └─ echocare_encoder.pth    # Pre-trained Echocare encoder weights
```

## 🧰 2. Quick Start
 We recommend **Python 3.10** and **CUDA 12.1 (cu121)**.
 Minimum recommended PyTorch version: **>= 2.4.1**.

```bash
# 📥 Clone the repository
git clone https://github.com/dndins/CSV-2026-Baseline.git
cd CSV-2026-Baseline

# 🎯 Create Python Environment
conda create -n csv-baseline python=3.10 -y
conda activate csv-baseline

pip install --index-url https://download.pytorch.org/whl/cu121   torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt
```

> ⚠️ **Important:**  
> If your CUDA version differs, install PyTorch following the official instructions first, then run:
> ```bash
> pip install -r requirements.txt --no-deps
> ```


## 🧩 3. Create Local Train / Validation Split

Generate a balanced validation set and JSON splits:

```bash
python split_train_valid_fold.py --root ./data --seed 2026 --val_size 50
```

This creates:
- `train_labeled.json`
- `train_unlabeled.json`
- `valid.json`

under the `data/` directory.


## 🚀 4. Train

### Recommended (High Performance — Echocare)

```bash
python train.py \
  --train-labeled-json ./data/train_labeled.json \
  --train-unlabeled-json ./data/train_unlabeled.json \
  --valid-labeled-json ./data/valid.json \
  --model Echocare \
  --echo_care_ckpt ./pretrain/echocare_encoder.pth \
  --save_path ./checkpoints \
  --gpu 0 \
  --train_epochs 100 \
  --batch_size 4
```

### Lightweight Option (Lower Memory — UNet)

```bash
python train.py --model UNet --gpu 0
```

Training checkpoints:
```text
./checkpoints/best.pth
./checkpoints/latest.pth
```


## 🔍 5. Inference

Run inference on released validation images:

```bash
python inference.py \
  --val-dir ./data/val \
  --checkpoint ./checkpoints/best.pth \
  --encoder-pth ./pretrain/echocare_encoder.pth \
  --resize-target 256 \
  --gpu 0
```

Predictions are saved to:

```text
./data/val/preds/{case_name}_pred.h5
```

Each file contains:
- `long_mask`
- `trans_mask`
- `cls`


## 📦 6. Package Predictions for Submission

**⚠️ ATTENTION!**

This submission is for the validation phase. During this phase, the validation set images will be provided. 
Participants are required to save their valid results in H5 format (the same format as the training set labels) 
and package them into "preds.tar.gz" format for submission via the registration platform.

```bash
cd data/val
tar -czvf preds.tar.gz preds/
```
During the testing phase, participants are required to submit their method in Docker format.


## 🧠 7. Notes & Tips

- 🤖 Use `--model Echocare` for the best performance.
- 🤖 Use `--model UNet` for limited GPU memory.
- 🎯 Adjust `--batch-size` based on your GPU memory (8 for Echocare, 32 for UNet).
- 🔧 Use `--amp` for faster training with automatic mixed precision.
- 📊 Monitor training progress with TensorBoard: use the paths shown above for your chosen model

## 📚 Baseline Method Acknowledgement

This baseline is developed based on and inspired by the following works:

> **[1]** Yang L, Qi L, Feng L, et al.  
> *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation.*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

> **[2]** Zhang H, Wu Y, Zhao M, et al.  
> *A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications.*  
> arXiv preprint arXiv:2509.11752, 2025.

> **[3]** Hatamizadeh A, Nath V, Tang Y, et al.  
> *Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images.*  
> International MICCAI brainlesion workshop. Cham: Springer International Publishing, 2021: 272-284.

> **[4]** Hu E J, Shen Y, Wallis P, et al.  
> *Lora: Low-rank adaptation of large language models.*  
> ICLR, 2022, 1(2): 3.

The training method for the baseline is built upon the method in [1]. 
Echocare [2] is a self-supervised ultrasound foundation model trained on the Swin UNETR [3] architecture. 
We further fine-tune the encoder of Echocare with LoRA [4] to adapt it to our task.

---

## 🏁 Good Luck & Happy Research!

We look forward to your participation in the CSV 2026 Challenge 🚀
