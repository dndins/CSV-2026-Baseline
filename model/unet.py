from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNetTwoView(nn.Module):
    """
    Two-view UNet:
      - shared encoder (same weights)
      - two segmentation decoders (long/trans)
      - fused classification head (concat embeddings -> MLP -> sigmoid)
    """

    def __init__(self, in_chns: int, seg_class_num: int, cls_class_num: int):
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": seg_class_num,
            "bilinear": False,
            "acti_func": "relu",
        }

        self.encoder = Encoder(params)

        # two segmentation heads (can also share decoder weights, but separate is safer)
        self.seg_decoder_long = Decoder(params)
        self.seg_decoder_trans = Decoder(params)

        # classification fusion head
        bottleneck_dim = params["feature_chns"][-1]  # 256
        hidden_dim = params["feature_chns"][-2]      # 128

        # concat(long_embed, trans_embed) -> MLP
        self.cls_fuse = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            # single-logit output for binary classification (use BCEWithLogitsLoss)
            nn.Linear(hidden_dim, cls_class_num),
        )

    @staticmethod
    def _embed_from_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
        # bottleneck: [B, C, H, W] -> [B, C]
        return F.adaptive_avg_pool2d(bottleneck, 1).view(bottleneck.size(0), -1)

    def forward(self, x_long: torch.Tensor, x_trans: torch.Tensor, need_fp: bool = False):
        """
        Returns:
          if need_fp == False:
            seg_long: [B, K, H, W]
            seg_trans: [B, K, H, W]
            cls: [B, C_cls]  (sigmoid prob)
          if need_fp == True:
            (seg_long_1, seg_long_2), (seg_trans_1, seg_trans_2), (cls_1, cls_2)
            where *_1 is original, *_2 is dropout-perturbed branch (same as your old design)
        """
        feat_long = self.encoder(x_long)   # list of 5 feature maps
        feat_trans = self.encoder(x_trans)

        if need_fp:
            # mimic your old "feature duplication + Dropout2d" strategy :contentReference[oaicite:2]{index=2}
            def _perturb_feats(feats):
                return [torch.cat((f, nn.Dropout2d(0.5)(f)), dim=0) for f in feats]

            p_long = _perturb_feats(feat_long)
            p_trans = _perturb_feats(feat_trans)

            seg_long = self.seg_decoder_long(p_long)      # [2B, K, H, W]
            seg_trans = self.seg_decoder_trans(p_trans)   # [2B, K, H, W]

            emb_long = self._embed_from_bottleneck(p_long[-1])    # [2B, C]
            emb_trans = self._embed_from_bottleneck(p_trans[-1])  # [2B, C]
            emb_fuse = torch.cat([emb_long, emb_trans], dim=1)    # [2B, 2C]
            # return logits (no sigmoid) so training can use BCEWithLogitsLoss
            cls_logits = self.cls_fuse(emb_fuse)                 # [2B, 1]

            return seg_long.chunk(2, dim=0), seg_trans.chunk(2, dim=0), cls_logits.chunk(2, dim=0)

        # normal forward
        seg_long = self.seg_decoder_long(feat_long)
        seg_trans = self.seg_decoder_trans(feat_trans)

        emb_long = self._embed_from_bottleneck(feat_long[-1])
        emb_trans = self._embed_from_bottleneck(feat_trans[-1])
        emb_fuse = torch.cat([emb_long, emb_trans], dim=1)
        # return logits (no sigmoid) so training can use BCEWithLogitsLoss
        cls_logits = self.cls_fuse(emb_fuse)

        return seg_long, seg_trans, cls_logits
