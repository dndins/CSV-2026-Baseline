# 🧪 CSV 2026 Challenge Baseline — Quick Start

This repository provides the official baseline implementation for the **CSV Challenge**, including:
- Semi-supervised two-view training (UniMatch framework)
- Model inference and submission packaging
- Data splitting utilities and evaluation support

The baseline is designed to be **reproducible, extensible, and competition-ready**.

---

## 📁 1. Prepare Data

Place the downloaded training archive (provided by organizers) under the repository root and unzip it so the directory structure becomes:

```text
CSV2026_Baseline/
└─ data/
   └─ train/
      ├─ images/        # .h5 image files (long_img & trans_img)
      └─ labels/        # _label.h5 files (long_mask, trans_mask, cls)
```


## 🧰 2. Create Python Environment
 
 We recommend **Python 3.10** and **CUDA 12.1 (cu121)**.
 Minimum recommended PyTorch version: **>= 2.4.1**.

```bash
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
You can accquire the Echocare weight from: https://cashkisi-my.sharepoint.com/personal/cares-copilot_cair-cas_org_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcares%2Dcopilot%5Fcair%2Dcas%5Forg%5Fhk%2FDocuments%2Fpublic%5Fshares%2Fechocare%5Fencoder%2Epth&parent=%2Fpersonal%2Fcares%2Dcopilot%5Fcair%2Dcas%5Forg%5Fhk%2FDocuments%2Fpublic%5Fshares&ga=1

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

```bash
cd data/val
tar -czvf preds.tar.gz preds/
```


## 🧠 7. Notes & Tips

- 🏆 Use `--model Echocare` for the best performance.
- 🪶 Use `--model UNet` for limited GPU memory.

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

This baseline is built based on the methods in [1], applying a semi-supervised segmentation + classification approach to our task. Echocare[2] is a self-supervised ultrasound foundation model trained on the Swin UNETR[3] architecture. We further fine-tune the encoder of Swin UNETR with LoRA [4] to adapt it to the downstream task.

---

## 🏁 Good Luck & Happy Research!

We look forward to your participation in the CSV 2026 Challenge 🚀
