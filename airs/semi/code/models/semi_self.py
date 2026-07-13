"""
半监督分割模型主网络（SCRA框架）
论文：Addressing Boundary Ambiguity in Semi-Supervised Ultrasound Segmentation 
      via Structure-Consistent Representation Alignment

该模块实现了SCRA框架的核心架构，包括：
1. Coordinate Attention (CA) - 坐标注意力增强的编码器（第3.3节）
2. Structure-Oriented Regularization (SOR) - 结构导向正则化（第3.5节）
3. 双解码器架构：分割解码器和辅助解码器
"""
# semi_self.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .modern_attention import CoordAttention
from utils.aug_function import SORDecoder
from utils.path_utils import get_backbone_pretrain_path


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


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        resnet = models.resnet34(pretrained=False)
        backbone_weight_path = get_backbone_pretrain_path()
        if os.path.isfile(backbone_weight_path):
            resnet.load_state_dict(torch.load(backbone_weight_path, map_location='cpu'))
        else:
            print(f"[WARN] Backbone pretrained weights not found at {backbone_weight_path}; using random ResNet-34 initialization.")
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
    """
    SCRA主网络模型（论文第3.1节）
    
    架构组成：
    - 编码器：ResNet-34骨干网络，提取多尺度特征
    - CA模块：在skip connections中嵌入坐标注意力（第3.3节）
    - 分割解码器：主分割路径，输出最终分割结果
    - 辅助解码器：用于SOR的结构扰动路径（第3.5节）
    """
    def __init__(self, args, num_classes=1, in_ch=3):
        super().__init__()
        num_classes = int(getattr(args, 'nclasses', num_classes))
        self.no_ca = args.no_ca
        self.no_sor = args.no_sor
        self.no_scd = args.no_scd
        
        # ========== 编码器：ResNet-34骨干网络 ==========
        self.encoder = Encoder(in_ch)
        
        # ========== 坐标注意力模块（CA）- 论文第3.3节 ==========
        # CA模块嵌入在skip connections中，增强空间定位能力
        # 论文描述：CA模块在每个残差块后嵌入，实现多尺度边界增强
        if self.no_ca:
            self.attention4 = nn.Identity()
            self.attention3 = nn.Identity()
            self.attention2 = nn.Identity()
            self.attention1 = nn.Identity()
        else:
            # 不同层级的特征通道数：layer4(256), layer3(128), layer2(64), layer1(64)
            self.attention4 = CoordAttention(256)  # e4层特征增强
            self.attention3 = CoordAttention(128)   # e3层特征增强
            self.attention2 = CoordAttention(64)    # e2层特征增强
            self.attention1 = CoordAttention(64)     # e1层特征增强
        # ========== 分割解码器：主分割路径 ==========
        # U-Net风格的解码器，通过skip connections融合多尺度特征
        self.seg5 = DecoderBlock(512, 512)
        self.seg4 = DecoderBlock(512 + 256, 256)  # 融合e4层特征
        self.seg3 = DecoderBlock(256 + 128, 128)   # 融合e3层特征
        self.seg2 = DecoderBlock(128 + 64, 64)     # 融合e2层特征
        self.seg1 = DecoderBlock(64 + 64, 64)      # 融合e1层特征
        self.seg_out = nn.Sequential(ConvBlock(64, 32), nn.Dropout2d(0.1), nn.Conv2d(32, num_classes, 1))
        
        # ========== 辅助解码器：用于SOR的结构扰动路径（论文第3.5节） ==========
        # 该解码器用于生成结构扰动视图，实现结构一致性正则化
        self.inp5 = DecoderBlock(512, 512, transpose=True)
        self.inp4 = DecoderBlock(512 + 256, 256, transpose=True)
        self.inp3 = DecoderBlock(256 + 128, 128, transpose=True)
        self.inp2 = DecoderBlock(128 + 64, 64, transpose=True)
        self.inp1 = DecoderBlock(64 + 64, 64, transpose=True)
        self.inp_out = nn.Sequential(ConvBlock(64, 32), nn.Dropout2d(0.1), nn.Conv2d(32, num_classes, 1))
        
        # ========== 上下文块：e5层特征增强 ==========
        if args.no_ca:
            self.context_block = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        else:
            # 在e5层也应用CA增强（论文第3.3节）
            self.context_block = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                CoordAttention(512)
            )
        
        # ========== SOR解码器：结构导向正则化模块（论文第3.5节） ==========
        # SOR通过guided cutout对编码器特征进行结构扰动
        if self.no_sor:
            self.sor_decoder = nn.Identity()
        else:
            self.sor_decoder = SORDecoder(erase=args.sor_erase)

        self.boundary_out = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        前向传播
        
        返回:
            mask: 主分割解码器的输出（最终分割结果）
            preboud: 辅助解码器的输出（结构扰动视图）
            i2-i5: 辅助解码器的中间特征图（保留兼容性）
            mask_binary: 二值化掩码（仅保留兼容性）
            boundary_logits: 边界预测 logits
            e5: 编码器最深层的特征（用于SCD的特征F_u）
        """
        # ========== 1. 编码阶段：提取多尺度特征 ==========
        e1, e2, e3, e4, e5 = self.encoder(x)  # ResNet-34提取的特征
        s5 = self.context_block(e5)  # e5层特征增强（包含CA）

        # ========== 2. 分割解码器：主分割路径（论文第3.1节） ==========
        # 通过skip connections融合CA增强的特征
        d5 = self.seg5(s5)

        # Level 4: 融合CA增强的e4特征
        f4_att = self.attention4(e4)  # CA增强（论文第3.3节）
        d4 = self.seg4(cat(d5, f4_att))

        # Level 3: 融合CA增强的e3特征
        f3_att = self.attention3(e3)
        d3 = self.seg3(cat(d4, f3_att))

        # Level 2: 融合CA增强的e2特征
        f2_att = self.attention2(e2)
        d2 = self.seg2(cat(d3, f2_att))

        # Level 1: 融合CA增强的e1特征
        f1_att = self.attention1(e1)
        d1 = self.seg1(cat(d2, f1_att))

        # 最终分割输出
        mask = torch.sigmoid(self.seg_out(d1))
        boundary_logits = self.boundary_out(d1)

        # ========== 3. 辅助解码器：SOR结构扰动路径（论文第3.5节） ==========
        # SOR通过guided cutout对编码器特征进行结构扰动，生成扰动视图
        if self.no_sor:
            feature = e5  # 不使用SOR时，直接使用原始特征
        else:
            # Multi-slot predictions are merged before SOR so every dataset uses
            # one foreground structure cue while PSFHS can still keep PS/FH slots.
            sor_mask = mask
            if sor_mask.dim() == 4 and sor_mask.size(1) > 1:
                sor_mask = 1.0 - torch.prod(1.0 - sor_mask.clamp(0.0, 1.0), dim=1, keepdim=True)
            feature = self.sor_decoder(e5, sor_mask.detach())

        # 辅助解码器的前向传播（生成扰动视图的预测）
        i5 = self.inp5(feature)
        i4 = self.inp4(cat(i5, e4))
        i3 = self.inp3(cat(i4, e3))
        i2 = self.inp2(cat(i3, e2))
        i1 = self.inp1(cat(i2, e1))
        preboud = torch.sigmoid(self.inp_out(i1))  # 辅助解码器的主输出

        # forward 返回 4 元组：(主分割, 辅助分割, 边界 logits, 编码器顶层特征)
        # 历史上还返回 i2..i5 与 mask_binary，但外部从未消费，已移除以减少混淆。
        return mask, preboud, boundary_logits, e5
