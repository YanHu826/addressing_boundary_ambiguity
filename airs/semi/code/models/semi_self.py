# semi_self.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .modern_attention import CoordAttention
from utils.aug_function import SORDecoder


def cat(x1, x2, x3=None, dim=1):
    # center-pad and concatenate feature maps
    if x3 is None:
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim)
    else:
        # three-way concat if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim)
        diffY = x.size(2) - x3.size(2)
        diffX = x.size(3) - x3.size(3)
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x, x3], dim)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, transpose=False):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, in_ch // 4)
        self.conv2 = ConvBlock(in_ch // 4, out_ch)
        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)


class SideoutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, in_ch // 4)
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_ch // 4, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        return self.conv2(x)


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load("pretrain/backbone/resnet34.pth", weights_only=False))
        if in_ch == 3:
            self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        else:
            self.initial = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu
            )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        e1 = self.initial(x)
        e2 = self.layer1(self.maxpool(e1))
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)
        return e1, e2, e3, e4, e5


class MyModel(nn.Module):
    def __init__(self, args, num_classes=1, in_ch=3):
        super().__init__()
        self.no_ca = args.no_ca
        self.no_sor = args.no_sor
        self.no_scd = args.no_scd
        # Encoder
        self.encoder = Encoder(in_ch)
        # CoordAttention
        if self.no_ca:
            self.attention4 = nn.Identity()
            self.attention3 = nn.Identity()
            self.attention2 = nn.Identity()
            self.attention1 = nn.Identity()
        else:
            self.attention4 = CoordAttention(256)
            self.attention3 = CoordAttention(128)
            self.attention2 = CoordAttention(64)
            self.attention1 = CoordAttention(64)
        # Segmentation decoder
        self.seg5 = DecoderBlock(512, 512)
        self.seg4 = DecoderBlock(512 + 256, 256)
        self.seg3 = DecoderBlock(256 + 128, 128)
        self.seg2 = DecoderBlock(128 + 64, 64)
        self.seg1 = DecoderBlock(64 + 64, 64)
        self.seg_out = nn.Sequential(ConvBlock(64, 32), nn.Dropout2d(0.1), nn.Conv2d(32, num_classes, 1))
        # Inpainting decoder (unchanged)
        self.inp5 = DecoderBlock(512, 512, transpose=True)
        self.inp4 = DecoderBlock(512 + 256, 256, transpose=True)
        self.inp3 = DecoderBlock(256 + 128, 128, transpose=True)
        self.inp2 = DecoderBlock(128 + 64, 64, transpose=True)
        self.inp1 = DecoderBlock(64 + 64, 64, transpose=True)
        self.side5 = SideoutBlock(512, 1)
        self.side4 = SideoutBlock(256, 1)
        self.side3 = SideoutBlock(128, 1)
        self.side2 = SideoutBlock(64, 1)
        self.inp_out = nn.Sequential(ConvBlock(64, 32), nn.Dropout2d(0.1), nn.Conv2d(32, num_classes, 1))
        # Optional feature-drop decoder
        if args.no_ca:
            self.context_block = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        else:
            self.context_block = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                CoordAttention(512)
            )
        if self.no_sor:
            self.sor_decoder = nn.Identity()
        else:
            self.sor_decoder = SORDecoder(erase=0.4)

        self.boundary_out = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # ------------------------
        # 1. Encoding
        # ------------------------
        e1, e2, e3, e4, e5 = self.encoder(x)
        s5 = self.context_block(e5)

        # ------------------------
        # 2. Segmentation Decoder with Attention Gates
        # ------------------------
        d5 = self.seg5(s5)

        # Level 4
        f4_att = self.attention4(e4)
        d4 = self.seg4(cat(d5, f4_att))

        # Level 3
        f3_att = self.attention3(e3)
        d3 = self.seg3(cat(d4, f3_att))

        # Level 2
        f2_att = self.attention2(e2)
        d2 = self.seg2(cat(d3, f2_att))

        # Level 1
        f1_att = self.attention1(e1)
        d1 = self.seg1(cat(d2, f1_att))

        # d5 = self.seg5(e5)
        # d4 = self.seg4(cat(d5, e4))
        # d3 = self.seg3(cat(d4, e3))
        # d2 = self.seg2(cat(d3, e2))
        # d1 = self.seg1(cat(d2, e1))

        mask = torch.sigmoid(self.seg_out(d1))
        mask_binary = (mask > 0.5).float()
        boundary_pred = self.boundary_out(d1)

        # ------------------------
        # 3. Inpainting Decoder (unchanged)
        # ------------------------
        if self.no_sor:
            feature = e5
        else:
            feature = self.sor_decoder(e5, mask_binary)

        i5 = self.inp5(feature)
        out5 = torch.sigmoid(self.side5(i5))
        i4 = self.inp4(cat(i5, e4))
        out4 = torch.sigmoid(self.side4(i4))
        i3 = self.inp3(cat(i4, e3))
        out3 = torch.sigmoid(self.side3(i3))
        i2 = self.inp2(cat(i3, e2))
        out2 = torch.sigmoid(self.side2(i2))
        i1 = self.inp1(cat(i2, e1))
        preboud = torch.sigmoid(self.inp_out(i1))
        
        return mask, preboud, out2, out3, out4, out5, mask_binary, boundary_pred
