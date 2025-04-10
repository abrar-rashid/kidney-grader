import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # resize in case of mismatches due to rounding
        if g1.shape != x1.shape:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, preds, targets, smooth=1.):
        preds = F.softmax(preds, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * one_hot_targets, dim=(2, 3))
        union = torch.sum(preds + one_hot_targets, dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_output = outputs[0]
        else:
            main_output = outputs

        ce_loss = self.ce(main_output, targets)
        dice_loss = self.dice_loss(main_output, targets)

        return self.alpha * dice_loss + self.beta * ce_loss


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class ImprovedUNet(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, features_only=True)
        encoder_channels = self.backbone.feature_info.channels()  # [24, 40, 80, 112, 1280]

        self.bottleneck = ResidualDoubleConv(encoder_channels[-1], 512, dropout_rate)

        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(512, 320, kernel_size=2, stride=2),
            nn.ConvTranspose2d(320, 112, kernel_size=2, stride=2),
            nn.ConvTranspose2d(112, 80, kernel_size=2, stride=2),
            nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2),
            nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2),
        ])

        skip_channels = encoder_channels[::-1][1:]  # [112, 80, 40, 24]
        decoder_channels = [320, 112, 80, 40, 24]

        self.attentions = nn.ModuleList([
            AttentionBlock(g, s, g // 2) for g, s in zip(decoder_channels[:-1], skip_channels)
        ])

        self.decoders = nn.ModuleList([
            ResidualDoubleConv(g + s, g, dropout_rate)
            for g, s in zip(decoder_channels[:-1], skip_channels)
        ])
        self.final_decoder = ResidualDoubleConv(decoder_channels[-1], 64, dropout_rate)

        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(ch, n_classes, kernel_size=1)
            for ch in decoder_channels[:-1] + [64]
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

        self.class_weights = nn.Parameter(torch.ones(n_classes), requires_grad=True)

    def forward(self, x):
        feats = self.backbone(x)  # list of [B, C, H, W] from 5 stages

        x = self.bottleneck(feats[-1])
        decoder_outputs = []

        for i in range(len(self.attentions)):
            x = self.upsamplers[i](x)
            skip = feats[-(i + 2)]
            att = self.attentions[i](x, skip)
            x = torch.cat([x, att], dim=1)
            x = self.decoders[i](x)
            decoder_outputs.append(x)

        x = self.upsamplers[-1](x)
        x = self.final_decoder(x)
        decoder_outputs.append(x)

        ds_outputs = []
        for i, dec_out in enumerate(decoder_outputs):
            out = self.deep_supervision[i](dec_out)
            if i != len(decoder_outputs) - 1:
                out = F.interpolate(out, size=decoder_outputs[-1].shape[2:], mode='bilinear', align_corners=False)
            ds_outputs.append(out)

        out = self.final_conv(decoder_outputs[-1])

        if self.training:
            return out, ds_outputs, self.class_weights
        else:
            return out
