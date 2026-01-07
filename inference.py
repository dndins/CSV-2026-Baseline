import os
import glob
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model.Echocare import Echocare_UniMatch


class ValH5Dataset:
    """Simple dataset to iterate over val image .h5 files.
    Each file is expected to contain 'long_img' and 'trans_img' numpy arrays.
    """
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.paths = sorted(glob.glob(os.path.join(images_dir, "*.h5")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with h5py.File(p, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]
        # original shapes
        long_shape = long_img.shape
        trans_shape = trans_img.shape
        # to torch tensors: [1, H, W], float32, normalize to [0,1] if values not in [0,1]
        long_t = torch.from_numpy(long_img).unsqueeze(0).float()
        trans_t = torch.from_numpy(trans_img).unsqueeze(0).float()
        # attempt normalization if values seem 0-255
        if long_t.max() > 1.5:
            long_t = long_t / 255.0
        if trans_t.max() > 1.5:
            trans_t = trans_t / 255.0
        return p, long_t, trans_t, long_shape, trans_shape


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        # common pattern: {"model": state_dict, ...}
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model


def predict_and_save(model, device, file_path, long_t, trans_t, long_shape, trans_shape, resize_target, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    out_path = os.path.join(out_dir, f"{name_no_ext}_pred.h5")

    # batch dimension
    xL = long_t.unsqueeze(0).to(device)  # [1,1,H,W]
    xT = trans_t.unsqueeze(0).to(device)

    # resize to target (same as training) before forward
    xL_r = F.interpolate(xL, (resize_target, resize_target), mode="bilinear", align_corners=False)
    xT_r = F.interpolate(xT, (resize_target, resize_target), mode="bilinear", align_corners=False)

    with torch.no_grad():
        segL_logits, segT_logits, cls_logits = model(xL_r, xT_r)
        # upsample logits back to original sizes
        segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
        segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
        predL = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        predT = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # classification prediction (optional)
        cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
        cls_pred = (cls_prob >= 0.5).astype(np.uint8).reshape(-1)

    # save to h5 with same key names as label files: long_mask, trans_mask, cls
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("long_mask", data=predL, compression="gzip")
        hf.create_dataset("trans_mask", data=predT, compression="gzip")
        hf.create_dataset("cls", data=cls_pred)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Inference script for CSV challenge (Echocare)")
    parser.add_argument("--val-dir", type=str, required=True,
                        help="Path to validation folder that contains 'images' subfolder with .h5 files")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth", help="Path to checkpoint (best.pth)")
    parser.add_argument("--encoder-pth", type=str, default="./pretrain/echocare_encoder.pth", help="Pretrained encoder for Echocare")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory under val (default: val/output)")
    parser.add_argument("--resize-target", type=int, default=256, help="Resize input short side used during training")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = os.path.join(args.val_dir, "images")
    if args.output_dir is None:
        out_dir = os.path.join(args.val_dir, "preds")
    else:
        out_dir = args.output_dir

    ds = ValH5Dataset(images_dir)

    # build model (Echocare)
    model = Echocare_UniMatch(in_chns=1, seg_class_num=3, cls_class_num=1, encoder_pth=args.encoder_pth)
    model = model.to(device)

    # load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint, device)

    print(f"Running inference on {len(ds)} files, saving to {out_dir}")
    for idx in range(len(ds)):
        p, long_t, trans_t, long_shape, trans_shape = ds[idx]
        out_path = predict_and_save(model, device, p, long_t, trans_t, long_shape, trans_shape, args.resize_target, out_dir)
        print(f"[{idx+1}/{len(ds)}] Saved: {out_path}")


if __name__ == "__main__":
    main()


