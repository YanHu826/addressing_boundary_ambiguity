"""
坐标注意力模块（Coordinate Attention, CA）
论文第3.3节：Coordinate Attention-enhanced Encoder

该模块用于增强编码器的空间定位能力，通过建模方向依赖关系来改善边界精度。
"""
import torch
import torch.nn as nn


class CoordAttention(nn.Module):
    """
    坐标注意力机制（论文第3.3节，公式(10)-(12)）
    
    论文描述：CA将空间注意力分解为高度和宽度方向，从而编码位置和上下文信息。
    与传统的通道注意力不同，CA能够捕获长距离的空间依赖关系，这对于
    低对比度超声图像中的边界定位至关重要。
    
    公式(10): f^h(c, x) = (1/H) * sum_y F(c, y, x)
    公式(11): f^w(c, y) = (1/W) * sum_x F(c, y, x)
    公式(12): F' = F · σ(f^h) · σ(f^w)
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        # 方向池化：分别沿高度和宽度方向进行平均池化（论文公式(10)-(11)）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度方向池化

        # 中间维度：用于降维以减少计算量
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        # 生成方向特定的注意力图
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        """
        前向传播：计算坐标注意力并应用到特征图
        
        参数:
            x: [B, C, H, W] - 输入特征图
        返回:
            out: [B, C, H, W] - 增强后的特征图（论文公式(12)）
        """
        identity = x
        n, c, h, w = x.size()
        
        # 方向池化：分别沿高度和宽度方向聚合（论文公式(10)-(11)）
        x_h = self.pool_h(x)  # [B, C, H, 1] - 沿宽度方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]

        # 融合方向信息并变换
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离高度和宽度方向的注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, W, 1] -> [B, C, 1, W]

        # 生成方向特定的注意力权重（sigmoid激活）
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        
        # 应用注意力：F' = F · σ(f^h) · σ(f^w)（论文公式(12)）
        out = identity * a_h * a_w
        return out
