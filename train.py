import argparse
import logging
import os
import sys
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.csv import CSVSemiDataset 
from util.utils import AverageMeter, count_params, DiceLoss, compute_nsd

from model.Echocare import Echocare_UniMatch
from model.unet import UNetTwoView


def main():
    parser = argparse.ArgumentParser("UniMatch Two-View Training")
    parser.add_argument("--train-labeled-json", type=str, default="./data/train_labeled.json")
    parser.add_argument("--train-unlabeled-json", type=str, default="./data/train_unlabeled.json")
    parser.add_argument("--valid-labeled-json", type=str, default="./data/valid.json")

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--conf_thresh", type=float, default=0.9)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)
    
    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument('--amp', type=bool, default=True, help='enable torch.cuda.amp')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=['fp16', 'bf16'])

    # model choice: Echocare (SwinUNETR-based) or UNet
    parser.add_argument("--model", type=str, default="Echocare", choices=["Echocare", "UNet"],
                        help="Model architecture to use: 'Echocare' or 'UNet'")

    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--gpu", type=str, default="3")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = build_logger(args.save_path)
    logger.info(str(args))

    cudnn.enabled = True
    cudnn.benchmark = True

    tb_logdir = os.path.join(args.save_path, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    logger.info(f"TensorBoard log dir: {tb_logdir}")

    model = get_model(args)
    
    logger.info("Total params: {:.1f}M".format(count_params(model)))
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.base_lr)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))
    
    db_train_u = CSVSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = CSVSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target, n_sample=len(db_train_u.case_list))
    db_valid_l = CSVSemiDataset(args.valid_labeled_json, "valid")

    train_loader_l = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True)
    train_loader_u = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True)
    train_loader_u_mix = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(db_valid_l, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, drop_last=False, pin_memory=True)

    total_iters = len(train_loader_u) * args.train_epochs

    # resume
    previous_best = 0.0
    start_epoch = 0
    previous_best_seg = 0.0
    previous_best_cls = 0.0
    latest_ckpt = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", 0.0)
        previous_best_seg = ckpt.get("previous_best_seg", 0.0)
        previous_best_cls = ckpt.get("previous_best_cls", 0.0)
        logger.info(f"************ Resume from {latest_ckpt}, epoch={start_epoch}, best={previous_best:.2f}")

    output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=0)


    for epoch in range(start_epoch, args.train_epochs):
        logger.info(f"===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}, Previous best: {previous_best:.2f}")
        
        stats = train_one_epoch(
            args=args,
            model=model,
            optimizer=optimizer,
            loader_l=train_loader_l,
            loader_u=train_loader_u,
            loader_u_mix=train_loader_u_mix,
            device=device,
            total_iters=total_iters,
            epoch=epoch,
            logger=logger,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler
        )

        writer.add_scalar("Train/Total_Loss", stats["loss"], epoch)
        writer.add_scalar("Train/Loss_x", stats["loss_x"], epoch)
        writer.add_scalar("Train/Loss_s", stats["loss_s"], epoch)
        writer.add_scalar("Train/Loss_fp", stats["loss_fp"], epoch)
        writer.add_scalar("Train/Loss_extra_cls", stats["loss_extra_cls"], epoch)

        output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=epoch)

        writer.add_scalar("Val/Dice/Long_Vessel", output_dict["dice_long_vessel"], epoch)
        writer.add_scalar("Val/Dice/Long_Plaque", output_dict["dice_long_plaque"], epoch)
        writer.add_scalar("Val/Dice/Trans_Vessel", output_dict["dice_trans_vessel"], epoch)
        writer.add_scalar("Val/Dice/Trans_Plaque", output_dict["dice_trans_plaque"], epoch)
        writer.add_scalar("Val/NSD/Long_Vessel", output_dict["nsd_long_vessel"], epoch)
        writer.add_scalar("Val/NSD/Long_Plaque", output_dict["nsd_long_plaque"], epoch)
        writer.add_scalar("Val/NSD/Trans_Vessel", output_dict["nsd_trans_vessel"], epoch)
        writer.add_scalar("Val/NSD/Trans_Plaque", output_dict["nsd_trans_plaque"], epoch)

        writer.add_scalar("Val/Cls/F1_Mean", output_dict["cls_score"], epoch)
        writer.add_scalar("Val/Total_Score", output_dict["total_score"], epoch)

        total_score = output_dict["total_score"]

        is_best = total_score > previous_best
        previous_best = max(previous_best, total_score)
        seg_score = output_dict.get("seg_score", 0.0)
        cls_score = output_dict.get("cls_score", 0.0)
        is_best_seg = seg_score > previous_best_seg
        is_best_cls = cls_score > previous_best_cls
        previous_best_seg = max(previous_best_seg, seg_score)
        previous_best_cls = max(previous_best_cls, cls_score)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
        }
        torch.save(ckpt, latest_ckpt)
        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, "best.pth"))
            logger.info(f"New best! total_score={total_score:.2f} saved to best.pth")
        if is_best_seg:
            torch.save(ckpt, os.path.join(args.save_path, "best_seg.pth"))
            logger.info(f"New best segmentation! seg_score={seg_score:.4f} saved to best_seg.pth")
        if is_best_cls:
            torch.save(ckpt, os.path.join(args.save_path, "best_cls.pth"))
            logger.info(f"New best classification! cls_score={cls_score:.4f} saved to best_cls.pth")
        # always save latest previous_best values into latest_ckpt for resume
        ckpt["previous_best_seg"] = previous_best_seg
        ckpt["previous_best_cls"] = previous_best_cls
    
    writer.close()
    logger.info("Training finished.")


@torch.no_grad()
def pseudo_from_logits(seg_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seg_logits: [B,K,H,W]
    returns:
      conf: [B,H,W] (max softmax prob)
      mask: [B,H,W] (argmax)
    """
    prob = torch.softmax(seg_logits, dim=1)
    conf, mask = prob.max(dim=1)
    return conf, mask



def cutmix_apply_image(img_s: torch.Tensor, img_mix: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    img_s/img_mix: [B,1,H,W], box: [B,H,W] 0/1
    """
    box_ = box.unsqueeze(1).expand_as(img_s)
    out = img_s.clone()
    out[box_ == 1] = img_mix[box_ == 1]
    return out


def cutmix_apply_pseudo(mask: torch.Tensor, conf: torch.Tensor,
                        mask_mix: torch.Tensor, conf_mix: torch.Tensor,
                        box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mask/conf: [B,H,W], box: [B,H,W]
    """
    mask_cm = mask.clone()
    conf_cm = conf.clone()
    mask_cm[box == 1] = mask_mix[box == 1]
    conf_cm[box == 1] = conf_mix[box == 1]
    return mask_cm, conf_cm


def ensure_cls_shape(y_cls: torch.Tensor) -> torch.Tensor:
    """
    Make y_cls -> [B,C] float for BCELoss.
    Handles [B], [B,1], [B,C].
    """
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(
    args,
    model,
    optimizer,
    loader_l,
    loader_u,
    loader_u_mix,
    device,
    total_iters,
    epoch,
    logger,
    use_amp,
    amp_dtype,
    scaler
):
    model.train()
    
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_cls_mse = nn.MSELoss()
    criterion_seg_ce = nn.CrossEntropyLoss()
    criterion_seg_dice = DiceLoss(n_classes=args.seg_num_classes)

    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_loss_fp = AverageMeter()
    total_loss_extra_cls = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(loader_l, loader_u, loader_u_mix)

    for i, (
        (x_long, x_trans, m_long, m_trans, y_cls),
        (uL_w, uL_s1, uL_s2, boxL1, boxL2, uT_w, uT_s1, uT_s2, boxT1, boxT2),
        (uL_wm, uL_s1m, uL_s2m, _, _, uT_wm, uT_s1m, uT_s2m, _, _),
    ) in enumerate(loader):

        # to device
        x_long = x_long.to(device); x_trans = x_trans.to(device)
        m_long = m_long.to(device); m_trans = m_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        uL_w = uL_w.to(device); uL_s1 = uL_s1.to(device); uL_s2 = uL_s2.to(device)
        uT_w = uT_w.to(device); uT_s1 = uT_s1.to(device); uT_s2 = uT_s2.to(device)

        boxL1 = boxL1.to(device); boxL2 = boxL2.to(device)
        boxT1 = boxT1.to(device); boxT2 = boxT2.to(device)

        uL_wm = uL_wm.to(device); uL_s1m = uL_s1m.to(device); uL_s2m = uL_s2m.to(device)
        uT_wm = uT_wm.to(device); uT_s1m = uT_s1m.to(device); uT_s2m = uT_s2m.to(device)

        # 1) pseudo-label from weak-mix
        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, cls_wm = model(uL_wm, uT_wm)
            segL_wm = segL_wm.detach()
            segT_wm = segT_wm.detach()
            cls_wm = cls_wm.detach()

            confL_wm, maskL_wm = pseudo_from_logits(segL_wm)
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm)

        # 2) CutMix on strong images (each view separately)
        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uL_s2 = cutmix_apply_image(uL_s2, uL_s2m, boxL2)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)
        uT_s2 = cutmix_apply_image(uT_s2, uT_s2m, boxT2)

        model.train()

        num_l_bs = x_long.size(0)
        num_u_bs = uL_w.size(0)

        # 3) forward labeled + weak unlabeled with need_fp=True (fp consistency)
        x_long_all = torch.cat([x_long, uL_w], dim=0)
        x_trans_all = torch.cat([x_trans, uT_w], dim=0)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            (segL_all, segL_fp_all), (segT_all, segT_fp_all), (cls_all, cls_fp_all) = model(
                x_long_all, x_trans_all, need_fp=True
            )

            # split back
            segL_x, segL_u_w = segL_all.split([num_l_bs, num_u_bs], dim=0)
            segT_x, segT_u_w = segT_all.split([num_l_bs, num_u_bs], dim=0)

            cls_x, cls_u_w = cls_all.split([num_l_bs, num_u_bs], dim=0)

            segL_u_w_fp = segL_fp_all[num_l_bs:]   # [B,K,H,W]
            segT_u_w_fp = segT_fp_all[num_l_bs:]   # [B,K,H,W]

            # 4) strong forward (s1 & s2 together)
            uL_s = torch.cat([uL_s1, uL_s2], dim=0)
            uT_s = torch.cat([uT_s1, uT_s2], dim=0)
            segL_s_out, segT_s_out, cls_s_out = model(uL_s, uT_s)

            segL_s1, segL_s2 = segL_s_out.chunk(2, dim=0)
            segT_s1, segT_s2 = segT_s_out.chunk(2, dim=0)

            # 5) pseudo-label from weak (non-mix)
            segL_u_w_detach = segL_u_w.detach()
            segT_u_w_detach = segT_u_w.detach()

            confL_w, maskL_w = pseudo_from_logits(segL_u_w_detach)
            confT_w, maskT_w = pseudo_from_logits(segT_u_w_detach)

            # CutMix pseudo-labels for strong s1/s2
            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskL_cm2, confL_cm2 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL2)

            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)
            maskT_cm2, confT_cm2 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT2)

            # 6) losses
            # labeled seg loss: long + trans average
            loss_x_long = (criterion_seg_ce(segL_x, m_long) +
                        criterion_seg_dice(segL_x, m_long, softmax=True, ignore=torch.zeros_like(m_long))) / 2.0
            loss_x_trans = (criterion_seg_ce(segT_x, m_trans) +
                            criterion_seg_dice(segT_x, m_trans, softmax=True, ignore=torch.zeros_like(m_trans))) / 2.0
            loss_x_seg = (loss_x_long + loss_x_trans) / 2.0

            # labeled cls loss (fused cls_x already)
            # ensure y_cls is [B,1] float and apply BCEWithLogitsLoss on logits
            loss_x_cls = criterion_cls(cls_x, y_cls.float())

            # unlabeled strong seg loss (Dice on pseudo) - average across (view,long/trans) and (s1/s2)
            ignL1 = (confL_cm1 < args.conf_thresh).float()
            ignL2 = (confL_cm2 < args.conf_thresh).float()
            ignT1 = (confT_cm1 < args.conf_thresh).float()
            ignT2 = (confT_cm2 < args.conf_thresh).float()

            loss_uL_s = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=ignL1) +
                        criterion_seg_dice(segL_s2, maskL_cm2, softmax=True, ignore=ignL2)) / 2.0
            loss_uT_s = (criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=ignT1) +
                        criterion_seg_dice(segT_s2, maskT_cm2, softmax=True, ignore=ignT2)) / 2.0
            loss_u_s_seg = (loss_uL_s + loss_uT_s) / 2.0

            # fp loss on weak (both views)
            ignLw = (confL_w < args.conf_thresh).float()
            ignTw = (confT_w < args.conf_thresh).float()
            loss_fp_L = criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=ignLw)
            loss_fp_T = criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=ignTw)
            loss_u_w_fp_seg = (loss_fp_L + loss_fp_T) / 2.0

            # extra cls constraint: strong-mix pairs should match weak-mix cls_wm
            _, _, cls_s1m = model(uL_s1m, uT_s1m)
            _, _, cls_s2m = model(uL_s2m, uT_s2m)
            # compare probabilities: sigmoid(logits)
            loss_u_s_cls = (criterion_cls_mse(torch.sigmoid(cls_s1m), torch.sigmoid(cls_wm))
                            + criterion_cls_mse(torch.sigmoid(cls_s2m), torch.sigmoid(cls_wm))) / 2.0

            # total loss (keep your original weights)
            loss = (
                loss_x_seg + loss_x_cls
                + loss_u_s_seg * 0.5            # = 0.25(s1)+0.25(s2) after averaging
                + loss_u_w_fp_seg * 0.5
                + loss_u_s_cls * 0.1
            )

            optimizer.zero_grad(set_to_none=True)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # poly lr (same as your old code)
        iters = epoch * len(loader_u) + i
        lr = args.base_lr * (1 - iters / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr

        # meters
        total_loss.update(loss.item())
        total_loss_x.update((loss_x_seg.item() + loss_x_cls.item()))
        total_loss_s.update(loss_u_s_seg.item())
        total_loss_fp.update(loss_u_w_fp_seg.item())
        total_loss_extra_cls.update(loss_u_s_cls.item())

        mask_ratio = (confL_w >= args.conf_thresh).sum() / confL_w.numel()
        total_mask_ratio.update(mask_ratio.item())

        if i % max(1, (len(loader_u) // 8)) == 0:
            logger.info(
                f"Iters: {i:4d} | "
                f"Total: {total_loss.avg:.3f} | "
                f"Loss_x: {total_loss_x.avg:.3f} | "
                f"Loss_s: {total_loss_s.avg:.3f} | "
                f"Loss_fp: {total_loss_fp.avg:.3f} | "
                f"Loss_extra_cls: {total_loss_extra_cls.avg:.3f} | "
                f"MaskRatio: {total_mask_ratio.avg:.3f}"
            )

    return {
        "loss": total_loss.avg,
        "loss_x": total_loss_x.avg,
        "loss_s": total_loss_s.avg,
        "loss_fp": total_loss_fp.avg,
        "loss_extra_cls": total_loss_extra_cls.avg,
        "mask_ratio": total_mask_ratio.avg,
    }

@torch.no_grad()
def validate(args, model, valid_loader, device, logger, writer=None, epoch=None):
    model.eval()

    # -------------------------
    # segmentation dice accumulators
    # -------------------------
    dice_long = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}

    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}

    # -------------------------
    # classification statistics
    # -------------------------
    cls_pred_list = []
    cls_gt_list = []

    num_batches = len(valid_loader)
    val_idx = 0

    for (x_long, x_trans, m_long, m_trans, y_cls) in valid_loader:
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)    # (B,H,W)
        m_trans = m_trans.to(device)
        y_cls = y_cls.to(device)      # (B,) or (B,1)

        # resize input
        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        x_long_r = F.interpolate(x_long, (args.resize_target, args.resize_target),mode="bilinear", align_corners=False)
        x_trans_r = F.interpolate(x_trans, (args.resize_target, args.resize_target),mode="bilinear", align_corners=False)

        # forward
        segL, segT, cls_out = model(x_long_r, x_trans_r)

        # -------------------------
        # classification (sigmoid -> threshold)
        # -------------------------
        cls_prob = torch.sigmoid(cls_out)               # (B,1)
        cls_pred = (cls_prob >= 0.5).long().view(-1)     # (B,)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(y_cls.view(-1).cpu().numpy().tolist())
        # -------------------------
        # segmentation Dice
        # -------------------------
        segL = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
        segT = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)

        predL = torch.argmax(segL, dim=1)  # (B,H,W)
        predT = torch.argmax(segT, dim=1)

        # -------------------------
        # Visualization to TensorBoard (per-sample) - placed after pred computed
        # -------------------------
        if writer is not None:
            try:
                # prepare numpy arrays: normalize image to [0,1]
                xL = x_long[0, 0].cpu().numpy()
                xT = x_trans[0, 0].cpu().numpy()
                def normalize_im(im):
                    mn = im.min()
                    mx = im.max()
                    if mx - mn < 1e-8:
                        return (im - mn)
                    return (im - mn) / (mx - mn)

                xL_n = normalize_im(xL)
                xT_n = normalize_im(xT)

                predL_np = predL[0].cpu().numpy()
                predT_np = predT[0].cpu().numpy()
                gtL_np = m_long[0].cpu().numpy()
                gtT_np = m_trans[0].cpu().numpy()

                # RGB base
                baseL = np.stack([xL_n, xL_n, xL_n], axis=0)  # C,H,W
                baseT = np.stack([xT_n, xT_n, xT_n], axis=0)

                def overlay(base, mask, color):
                    # base: C,H,W in [0,1], mask: H,W bool
                    over = base.copy()
                    alpha = 0.5
                    for c in range(3):
                        over[c][mask] = over[c][mask] * (1 - alpha) + color[c] * alpha
                    return over

                # colors: plaque (class 1) = red, vessel (class 2) = green
                red = [1.0, 0.0, 0.0]
                green = [0.0, 1.0, 0.0]

                predL_vis = baseL.copy()
                predL_vis = overlay(predL_vis, predL_np == 1, red)
                predL_vis = overlay(predL_vis, predL_np == 2, green)

                gtL_vis = baseL.copy()
                gtL_vis = overlay(gtL_vis, gtL_np == 1, red)
                gtL_vis = overlay(gtL_vis, gtL_np == 2, green)

                predT_vis = baseT.copy()
                predT_vis = overlay(predT_vis, predT_np == 1, red)
                predT_vis = overlay(predT_vis, predT_np == 2, green)

                gtT_vis = baseT.copy()
                gtT_vis = overlay(gtT_vis, gtT_np == 1, red)
                gtT_vis = overlay(gtT_vis, gtT_np == 2, green)

                # concatenate horizontally: [C, H, W*3]
                concatL = np.concatenate([baseL, predL_vis, gtL_vis], axis=2)
                concatT = np.concatenate([baseT, predT_vis, gtT_vis], axis=2)

                tagL = f"Val/vis/long/{val_idx}"
                tagT = f"Val/vis/trans/{val_idx}"
                step = epoch if epoch is not None and epoch >= 0 else 0
                writer.add_image(tagL, torch.from_numpy(concatL).float(), global_step=step)
                writer.add_image(tagT, torch.from_numpy(concatT).float(), global_step=step)
            except Exception as e:
                logger.warning(f"Failed to write val visualization: {e}")

        val_idx += 1
        for cls in [1, 2]:  # two foreground classes

            # ---------- Dice ----------
            interL = ((predL == cls) & (m_long == cls)).sum().item()
            unionL = (predL == cls).sum().item() + (m_long == cls).sum().item()
            diceL = 2.0 * interL / (unionL + 1e-8)
            dice_long[cls] += diceL

            # trans Dice
            interT = ((predT == cls) & (m_trans == cls)).sum().item()
            unionT = (predT == cls).sum().item() + (m_trans == cls).sum().item()
            diceT = 2.0 * interT / (unionT + 1e-8)
            dice_trans[cls] += diceT

            # ---------- NSD ----------
            predL_np = (predL[0] == cls).cpu().numpy()
            gtL_np = (m_long[0] == cls).cpu().numpy()
            nsd_long[cls] += compute_nsd(predL_np, gtL_np, tolerance=3.0)

            predT_np = (predT[0] == cls).cpu().numpy()
            gtT_np = (m_trans[0] == cls).cpu().numpy()
            nsd_trans[cls] += compute_nsd(predT_np, gtT_np, tolerance=3.0)

    # -------------------------
    # segmentation 结果
    # -------------------------
    idx_to_name = {1:"Plaque", 2:"Vessel"}
    for cls in [1, 2]:
        dice_long[cls] = dice_long[cls] / max(1, num_batches)
        dice_trans[cls] = dice_trans[cls] / max(1, num_batches)
        logger.info(f"[Dice] {idx_to_name[cls]} | Long Dice: {dice_long[cls]:.2f} | Trans Dice: {dice_trans[cls]:.2f}")

        nsd_long[cls] = nsd_long[cls] / max(1, num_batches)
        nsd_trans[cls] = nsd_trans[cls] / max(1, num_batches)
        logger.info(f"[NSD] {idx_to_name[cls]} | Long NSD: {nsd_long[cls]:.2f} | Trans NSD: {nsd_trans[cls]:.2f}")

    mean_dice = (dice_long[1] + dice_long[2] + dice_trans[1] + dice_trans[2]) / 4.0
    mean_NSD =  (nsd_long[1]  + nsd_long[2]  + nsd_trans[1]  + nsd_trans[2]) / 4.0

    logger.info(f"[Dice] Mean Foreground Dice: {mean_dice:.3f}")
    logger.info(f"[NSD] Mean Foreground NSD: {mean_NSD:.3f}")

    # -------------------------
    # classification F1
    # -------------------------
    cls_gt = np.array(cls_gt_list)
    cls_pred = np.array(cls_pred_list)

    f1 = f1_score(cls_gt, cls_pred)

    logger.info(f"[Cls] F1 Score: {f1:.4f}")

    # Format confusion matrix: counts + per-true-class percentages for readability
    cm = confusion_matrix(cls_gt, cls_pred)
    labels = list(range(cm.shape[0]))

    def _format_confusion_matrix(cm_array, lbls):
        """Return a human-readable multiline string for confusion matrix."""
        header = "\t" + "\t".join([f"Pred:{l}" for l in lbls])
        lines = [header]
        for i, l in enumerate(lbls):
            counts = "\t".join(str(int(x)) for x in cm_array[i])
            row_total = cm_array[i].sum()
            if row_total > 0:
                percents = "\t".join(f"{(cm_array[i, j] / row_total * 100):.1f}%" for j in range(cm_array.shape[1]))
            else:
                percents = "\t".join("0.0%" for _ in range(cm_array.shape[1]))
            lines.append(f"True:{l}\t{counts}\t| {percents}")
        return "\n".join(lines)

    logger.info("Confusion Matrix (rows=true labels, cols=pred labels):\n" + _format_confusion_matrix(cm, labels))

    return {
        "dice_long_vessel": dice_long[2],
        "dice_long_plaque": dice_long[1],
        "dice_trans_vessel": dice_trans[2],
        "dice_trans_plaque": dice_trans[1],
        
        "nsd_long_vessel": nsd_long[2],
        "nsd_long_plaque": nsd_long[1],
        "nsd_trans_vessel": nsd_trans[2],
        "nsd_trans_plaque": nsd_trans[1],
        
        "cls_score": f1,

        "seg_socre_long_vessel": (dice_long[2] + nsd_long[2])/2,
        "seg_socre_long_plaque": (dice_long[1] + nsd_long[1])/2,
        "seg_socre_trans_vessel": (dice_trans[2] + nsd_trans[2])/2,
        "seg_socre_trans_plaque": (dice_trans[1] + nsd_trans[1])/2,

        "seg_score":((dice_long[2] + nsd_long[2])/2 *0.4 +\
            (dice_long[1] + nsd_long[1])/2 *0.6 +\
            (dice_trans[2] + nsd_trans[2])/2 *0.4 +\
            (dice_trans[1] + nsd_trans[1])/2 *0.6)/2,
        
        # Note: total_score currently capped around 0.8 locally because the
        # time-efficiency component (20% of total) is not computed in local eval.
        "total_score": f1 *0.4 +\
            (dice_long[2] + nsd_long[2])/2 *0.4*0.2 +\
            (dice_long[1] + nsd_long[1])/2 *0.6*0.2 +\
            (dice_trans[2] + nsd_trans[2])/2 *0.4*0.2 +\
            (dice_trans[1] + nsd_trans[1])/2 *0.6*0.2,
    }

def build_logger(save_path: str):
    logger = logging.getLogger("UniMatch TwoView Training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(save_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_model(args):
    """
    Return model instance based on args.model.
      - 'Echocare' -> Echocare_UniMatch(...) (uses encoder checkpoint)
      - 'UNet'     -> UNetTwoView(...)
    """
    if args.model == "Echocare":
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.echo_care_ckpt,
        )
    elif args.model == "UNet":
        model = UNetTwoView(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    else:
        raise ValueError(f"Unknown model choice: {args.model}")
    return model


if __name__ == "__main__":
    main()
