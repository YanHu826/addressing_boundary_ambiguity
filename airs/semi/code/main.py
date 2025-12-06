import shutil

import cv2
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from data.build_dataset import build_dataset
from models.build_model import build_model
from models.dc_gan import DCGAN_D
from utils.evaluate import evaluate
from opt import args
from utils.loss import BceDiceLoss, sigmoid_rampup, SemanticContrastiveLoss, dynamic_edge_loss, get_boundary_sobel
import math
import warnings
from utils.loss import edge_loss

warnings.filterwarnings("ignore", category=UserWarning)


def DeepSupSeg(pred, gt):
    criterion = BceDiceLoss()
    loss = criterion(pred, gt)
    return loss


def get_boundary_map(mask_batch):
    """
    Extract boundary map using Sobel operator as described in the paper.
    This replaces Canny edge detection to match the paper's methodology.
    """
    return get_boundary_sobel(mask_batch)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_lr_rate(argsimizer, iter, total_batch):
    lr = lr_poly(args.lr, iter, args.nEpoch * total_batch, args.power)
    argsimizer.param_groups[0]['lr'] = lr
    return lr


def train():
    """load data"""
    train_l_data, _, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_total_batch = math.ceil(len(valid_data) / args.batch_size)

    """load model"""
    model = build_model(args)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    print('\n---------------------------------')
    print('Start training')
    print('---------------------------------\n')

    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):
        model.train()

        print("Epoch: {}".format(epoch))
        total_batch = math.ceil(len(train_l_data) / args.batch_size)
        bar = tqdm(enumerate(train_l_dataloader), total=total_batch)
        for batch_id, data_l in bar:
            itr = total_batch * epoch + batch_id
            img, gt = data_l['image'], data_l['label']
            if args.GPUs:
                img = img.cuda()
                gt = gt.cuda()
            optim.zero_grad()
            mask = model(img)
            loss = DeepSupSeg(mask, gt)
            loss.backward()
            optim.step()
            adjust_lr_rate(optim, itr, total_batch)

        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
                ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice = evaluate(model, valid_dataloader, val_total_batch)

            print("Valid Result:")
            print(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            if dice > best:
                best = dice
            print("Best Dice:: ", best)

            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif (F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif (F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")

def train_semi():
    """
    SCRA框架的半监督训练函数
    
    训练流程概述（对应论文第3节）：
    1. 数据加载：标注数据D_l和无标注数据D_u
    2. 模型初始化：
       - 主分割网络（包含CA增强的编码器和双解码器）
       - SCD判别器（用于结构对比学习）
    3. 每个训练迭代：
       a) 标注数据：计算监督损失L_sup（BCE + Dice + 边界损失）
       b) 无标注数据：
          - 生成伪标签mask_boud
          - SCD模块：计算对抗损失L_adv和特征匹配损失L_FM（第3.2节）
          - SOR模块：计算结构一致性损失L_SOR（第3.5节）
          - 其他辅助损失（边界损失、CPS损失等）
       c) 总损失：L_total = L_sup + L_adv + L_FM + L_SOR + 其他（论文公式(19)）
       d) 反向传播和参数更新
    
    关键模块：
    - CA (Coordinate Attention): 第3.3节，增强空间定位
    - SCD (Structure-Contrast Discriminator): 第3.2节，结构对比学习
    - SOR (Structure-Oriented Regularization): 第3.5节，结构一致性正则化
    """
    # ========== 数据加载 ==========
    train_l_data, train_u_data, valid_data = build_dataset(args)
    train_l_dataloader = DataLoader(train_l_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_u_dataloader = DataLoader(train_u_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_sign = False
    if valid_data is not None:
        valid_sign = True
        valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_total_batch = math.ceil(len(valid_data) / args.batch_size)
    """load model"""
    model = build_model(args)
    scl_loss_fn = SemanticContrastiveLoss(temperature=0.1, momentum=0.9, num_classes=2)
    model_cps = build_model(args)  # Construct the CPS Branch Model
    model_cps.load_state_dict(model.state_dict())
    model_cps.eval()  # No second model is trained; it is only used to generate pseudo-labels.

    # ========== 初始化SCD判别器（论文第3.2节） ==========
    if not args.no_scd:
        """
        Structure-Contrast Discriminator (SCD) 初始化
        论文第3.2节：使用DCGAN风格的判别器架构
        
        输入设计：
        - 论文公式(8): Z = Concat(F_u, B)，其中F_u是512维特征，B是1维边界图
        - 理论上输入通道数应为 512 + 1 = 513
        - 为使用预训练权重，使用特征适配器将512维降维到3维
        - 最终输入：3（特征）+ 1（边界）= 4通道，兼容预训练判别器
        """
        # 初始化判别器：DCGAN架构，输入4通道，输出64x64特征图
        netD = DCGAN_D(isize=64, nz=100, nc=4, ndf=64, ngpu=1)
        netD.cuda()
        
        # 特征适配器：将512维编码器特征降维到3维
        # 用于将特征-边界拼接表示适配到预训练判别器的输入格式
        feature_adapter = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        ).cuda()
        
        # 加载预训练的判别器权重（论文提到判别器需要预训练以稳定训练）
        netD_weight = torch.load("models/pretrain/GAN/netD_epoch_10000.pth")
        new_state_dict = {}
        for k, v in netD_weight.items():
            # 适配输入通道数：从1通道改为4通道
            if k == "main.initial:1-64:conv.weight":
                print(f"Rename key: {k} -> main.initial:4-64:conv.weight")
                new_state_dict["main.initial:4-64:conv.weight"] = v
            else:
                new_state_dict[k] = v

        netD.load_state_dict(new_state_dict)
        netD.eval()  # 初始时设为评估模式
        
        # 特征适配器的优化器（需要单独优化）
        optim_adapter = torch.optim.Adam(feature_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        feature_adapter = None
        optim_adapter = None

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # train
    print('\n---------------------------------')
    print('Start training_semi')
    print('---------------------------------\n')
    F1_best, F1_second_best, F1_third_best = 0, 0, 0
    best = 0
    for epoch in range(args.nEpoch):
        model.train()
        print("Epoch: {}".format(epoch))
        loader = iter(zip(cycle(train_l_dataloader), train_u_dataloader))
        bar = tqdm(range(len(train_u_dataloader)))
        for batch_id in bar:
            data_l, data_u = next(loader)
            total_batch = len(train_u_dataloader)
            itr = total_batch * epoch + batch_id
            img_l, gt = data_l['image'], data_l['label']
            img_u = data_u
            if args.GPUs:
                img_l = img_l.cuda()
                gt = gt.cuda()
                img_u = img_u.cuda()
            # ========== 前向传播：标注数据 ==========
            optim.zero_grad()
            pred_l = model(img_l)
            # 模型输出：[mask, preboud, out2, out3, out4, out5, mask_binary, boundary_pred, e5]
            mask = pred_l[0]  # 主分割输出
            boundary_pred = pred_l[-2]  # 边界预测输出
            
            # 计算标注数据的监督损失（论文公式(20)）
            boundary_gt = get_boundary_map(gt)  # 使用Sobel算子提取真实边界
            loss_boundary = F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt)
            loss_l_seg = DeepSupSeg(mask, gt)  # BCE + Dice损失
            loss_l = loss_l_seg + 0.2 * loss_boundary  # 监督损失：分割损失 + 边界损失
            
            # ========== 前向传播：无标注数据 ==========
            pred_u = model(img_u)
            # 解包输出：predboud是辅助解码器输出，feat_u是编码器特征e5（用于SCD）
            _, predboud, sor_feat2, sor_feat3, sor_feat4, sor_feat5, mask_boud, _, feat_u = pred_u
            # feat_u = e5：编码器最深层的特征（512维），用于SCD的特征F_u（论文第3.2节）

            # ========== 无标注数据的分割损失 ==========
            # 使用伪标签（mask_boud）进行监督，但降低低置信度区域的权重
            with torch.no_grad():
                prob_map = torch.sigmoid(predboud)  # 预测概率图
                weights = torch.ones_like(prob_map)
                # 对低置信度区域（0.4-0.6）降低权重，避免噪声伪标签的影响
                weights[(prob_map >= 0.4) & (prob_map <= 0.6)] = 0.5

            # 加权分割损失：高置信度区域权重高，低置信度区域权重低
            loss_u_seg = (DeepSupSeg(predboud, mask_boud) * weights).mean()

            # ========== 训练判别器D（SCD模块 - 论文第3.2节） ==========
            if not args.no_scd:
                """
                Structure-Contrast Discriminator (SCD) - 结构对比判别器
                论文第3.2节：通过对抗学习区分真实和伪结构边界
                
                核心思想：
                1. 构建联合表示 Z = Concat(F_u, B)，其中：
                   - F_u: 编码器深层特征（e5，512维）
                   - B: 通过Sobel算子提取的边界图（论文公式(9)）
                2. 判别器学习区分真实边界（来自标注数据）和伪边界（来自无标注预测）
                3. 通过对抗训练，引导网络生成结构一致的预测
                
                论文公式(8): Z = Concat(F_u, B)
                论文公式(10)-(11): 对抗损失
                """
                
                # ========== 步骤1：提取边界图（论文公式(9)） ==========
                # 使用Sobel算子从掩码中提取边界图
                boundary_gt_l = get_boundary_sobel(gt)  # 真实边界：来自标注数据的ground truth
                boundary_pseudo_u = get_boundary_sobel(mask_boud)  # 伪边界：来自无标注数据的预测
                
                # ========== 步骤2：获取编码器特征F_u ==========
                # 对于标注数据：需要重新前向传播获取编码器特征
                with torch.no_grad():
                    _, _, _, _, _, _, _, _, feat_l = model(img_l)
                
                # 对于无标注数据：feat_u已经在前面获取（pred_u的最后一个输出）
                
                # ========== 步骤3：调整特征和边界图的空间尺寸 ==========
                # 将特征图调整到与边界图相同的空间尺寸
                feat_l_resized = F.interpolate(feat_l, size=boundary_gt_l.shape[2:], mode='bilinear', align_corners=False)
                feat_u_resized = F.interpolate(feat_u, size=boundary_pseudo_u.shape[2:], mode='bilinear', align_corners=False)
                
                # ========== 步骤4：特征适配（兼容预训练判别器） ==========
                # 将512维特征降维到3维，以便与边界图（1维）拼接成4通道输入
                # 这样可以使用预训练的判别器权重
                feat_l_adapted = feature_adapter(feat_l_resized.detach())
                feat_u_adapted = feature_adapter(feat_u_resized.detach())
                
                # ========== 步骤5：构建联合特征-边界表示（论文公式(8)） ==========
                # Z = Concat(F, B)
                real_feat_boundary = torch.cat([feat_l_adapted, boundary_gt_l], dim=1)  # 真实：Z_gt = (F_l, B(y))
                fake_feat_boundary = torch.cat([feat_u_adapted, boundary_pseudo_u], dim=1)  # 伪：Z_u = (F_u, B(P_u))
                
                # ========== 步骤6：计算特征匹配损失（论文公式(23)） ==========
                # L_FM用于稳定对抗训练，确保生成器特征与真实特征在判别器中间层相似
                # 论文描述：特征匹配损失使梯度传播更平滑
                _, real_features = netD(real_feat_boundary, return_features=True)
                _, fake_features = netD(fake_feat_boundary, return_features=True)

                fm_loss = 0
                # 计算判别器各层特征之间的L1距离
                for rf, ff in zip(real_features, fake_features):
                    fm_loss += F.l1_loss(ff, rf.detach())  # detach真实特征，避免反向传播到D
                fm_loss = fm_loss / len(real_features)  # 多层特征的平均损失
                # 论文权重：λ_FM = 1.0

                # ========== 步骤7：训练判别器（论文公式(21)-(22)） ==========
                criterion_GAN = nn.BCEWithLogitsLoss()
                
                # 判别器前向传播：区分真实和伪边界
                real_pred = netD(real_feat_boundary)  # 真实边界应该输出1
                fake_pred = netD(fake_feat_boundary)  # 伪边界应该输出0
                real_labels = torch.ones_like(real_pred)
                fake_labels = torch.zeros_like(fake_pred)

                # 计算判别器损失（论文公式(22)）
                errD_real = criterion_GAN(real_pred, real_labels)  # 真实边界损失
                errD_fake = criterion_GAN(fake_pred, fake_labels)  # 伪边界损失
                errD = (errD_real + errD_fake) * 0.5  # 总判别器损失

                # 更新判别器参数（每2个epoch更新一次，稳定训练）
                netD.train()
                optimizer_D = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
                if epoch % 2 == 0:
                    optimizer_D.zero_grad()
                    errD.backward()
                    optimizer_D.step()

            # ---------------- edge loss ----------------
            if not args.no_scd:
                loss_edge = edge_loss(predboud, mask_boud)
            else:
                loss_edge = torch.tensor(0.0, device=img_u.device)

            # ========== 生成器对抗损失（SCD - 论文公式(21)） ==========
            if not args.no_scd:
                """
                生成器损失：引导网络生成结构一致的预测
                论文公式(21): L_adv = E_{x_u}[-log(1 - D(Z_u))]
                目标：使伪边界被判别器误判为真实边界（输出接近1）
                """
                # 提取无标注数据的边界图
                boundary_pseudo_u = get_boundary_sobel(mask_boud)
                
                # 对于生成器：使用当前特征（不detach），允许梯度反向传播
                feat_u_resized_G = F.interpolate(feat_u, size=boundary_pseudo_u.shape[2:], mode='bilinear', align_corners=False)
                feat_u_adapted_G = feature_adapter(feat_u_resized_G)
                fake_feat_boundary_G = torch.cat([feat_u_adapted_G, boundary_pseudo_u], dim=1)
                
                # 生成器希望判别器将伪边界判断为真实（输出接近1）
                pred_fake_for_G = netD(fake_feat_boundary_G)
                errG_adv = criterion_GAN(pred_fake_for_G, real_labels)
                # 论文权重：λ_adv = 0.1
            else:
                errG_adv = torch.tensor(0.0, device=img_u.device)
                fm_loss = torch.tensor(0.0, device=img_u.device)

            # ========== SOR损失（结构导向正则化 - 论文第3.5节） ==========
            if not args.no_sor:
                """
                Structure-Oriented Regularization (SOR) - 结构导向正则化
                论文第3.5节：通过结构级一致性约束增强模型对边界模糊的鲁棒性
                
                核心思想：
                1. 对解码器输出施加dropout扰动，生成扰动视图
                2. 使用Sobel算子提取结构表示（边界图）
                3. 最小化干净预测和扰动预测的结构表示差异
                4. 确保模型在解码器扰动下仍能保持结构一致性
                
                论文公式(17): s = G(p), s_hat = G(p_hat)，其中G是Sobel算子
                论文公式(18): L_SOR = λ_SOR * ||s - s_hat||_1
                """
                
                # ========== 步骤1：对解码器输出施加扰动 ==========
                decoder_output_clean = predboud  # 干净预测（无扰动）
                decoder_output_perturbed = F.dropout2d(decoder_output_clean, p=0.1, training=True)  # 扰动预测
                
                # ========== 步骤2：提取结构表示（论文公式(17)） ==========
                # 使用Sobel算子从预测中提取结构边界表示
                # s = G(p)：干净预测的结构表示
                struct_clean = get_boundary_sobel(torch.sigmoid(decoder_output_clean))
                # s_hat = G(p_hat)：扰动预测的结构表示
                struct_perturbed = get_boundary_sobel(torch.sigmoid(decoder_output_perturbed))
                
                # ========== 步骤3：计算结构一致性损失（论文公式(18)） ==========
                # L_SOR = λ_SOR * ||s - s_hat||_1
                # 最小化干净和扰动预测的结构差异，增强结构稳定性
                loss_sor = 0.2 * F.l1_loss(struct_clean, struct_perturbed)
                # 论文权重：λ_SOR = 0.2
            else:
                loss_sor = torch.tensor(0.0, device=img_u.device)

            # ---------------- Additional losses (only when SCD is enabled) ----------------
            if not args.no_scd:
                # ========== Semantic Contrastive Loss ==========
                sor_feat4 = F.normalize(sor_feat4, p=2, dim=1)
                feat4 = sor_feat4
                label4 = F.interpolate(mask_boud, size=feat4.shape[2:], mode='nearest').squeeze(1).long()
                scl_weight = sigmoid_rampup(epoch, 20) * 0.1
                loss_scl = scl_weight * scl_loss_fn(feat4, label4)

                # ========== Prototype Matching Soft Pseudo Label ==========
                features = torch.cat([
                    F.interpolate(sor_feat3, size=sor_feat4.shape[2:], mode='bilinear', align_corners=False),
                    sor_feat4
                ], dim=1)
                logits = predboud.detach()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    conf_mask = ((probs > 0.3) & (probs < 0.7)).float()
                    probs_bin = (probs > 0.5).float() * (1 - conf_mask)  # Clear the blurred area
                    probs_bin_down = F.interpolate(probs_bin, size=(features.shape[2], features.shape[3]),
                                                   mode='bilinear', align_corners=False)

                    B, C, H, W = features.size()
                    features_flat = features.view(B, C, -1)
                    probs_flat = probs_bin_down.view(B, 1, -1)

                    eps = 1e-6
                    foreground_proto = (features_flat * probs_flat).sum(dim=2) / (probs_flat.sum(dim=2) + eps)
                    background_proto = (features_flat * (1 - probs_flat)).sum(dim=2) / (
                                (1 - probs_flat).sum(dim=2) + eps)

                    feat_norm = F.normalize(features_flat, dim=1)
                    fg_proto = F.normalize(foreground_proto.unsqueeze(2), dim=1)
                    bg_proto = F.normalize(background_proto.unsqueeze(2), dim=1)

                    fg_sim = torch.bmm(fg_proto.transpose(1, 2), feat_norm).squeeze(1)
                    bg_sim = torch.bmm(bg_proto.transpose(1, 2), feat_norm).squeeze(1)

                    sim_stack = torch.stack([bg_sim, fg_sim], dim=1)
                    soft_label = F.softmax(sim_stack, dim=1)[:, 1]
                    soft_label = soft_label.view(B, 1, H, W)

                soft_label_up = F.interpolate(soft_label, size=predboud.shape[2:], mode='bilinear', align_corners=False)
                loss_soft_pseudo = scl_weight * F.binary_cross_entropy_with_logits(predboud, soft_label_up)

            if args.no_scd:
                loss_edge = dynamic_edge_loss(predboud, mask_boud, epoch=epoch, total_epoch=args.nEpoch)
                loss = 2 * loss_l + 0.5 * loss_u_seg
                loss.mean().backward()
                optim.step()

            if not args.no_scd:
                # Obtain pseudo-labels using model_cps (without backpropagating gradients)
                with torch.no_grad():
                    pred_u_cps = model_cps(img_u)[0]
                    pseudo_u_cps = (torch.sigmoid(pred_u_cps) > 0.5).float()

                # ------------------- 🔹 NEW: Uncertainty-weighted consistency -------------------
                prob_u_main = torch.sigmoid(predboud).clamp(1e-6, 1 - 1e-6)
                entropy_u = - (prob_u_main * torch.log(prob_u_main) + (1 - prob_u_main) * torch.log(1 - prob_u_main))
                uncertainty_weight = 1 - entropy_u / math.log(2)  # 高不确定性区域权重低
                # -----------------------------------------------------------------------------

                # Align the predictions of the main model with the pseudo-labels to calculate the consistency loss.
                pred_u_main = predboud  # The output of your main model
                loss_cps = (F.binary_cross_entropy_with_logits(pred_u_main, pseudo_u_cps, reduction='none') * uncertainty_weight.detach()).mean()

                # ========== 总损失函数（论文第3.6节，公式(19)） ==========
                """
                论文公式(19): L_total = L_sup + λ_adv * L_adv + λ_FM * L_FM + λ_SOR * L_SOR
                
                损失组件：
                - L_sup: 监督损失（BCE + Dice），用于标注数据
                - L_adv: 对抗损失（SCD），权重λ_adv=0.1
                - L_FM: 特征匹配损失（SCD），权重λ_FM=1.0
                - L_SOR: 结构导向正则化损失，权重λ_SOR=0.2
                """
                cps_weight = sigmoid_rampup(epoch, rampup_length=10)
                
                # 无标注数据的损失组合
                loss_u = (
                        0.75 * loss_u_seg +        # 无标注分割损失
                        0.1 * errG_adv +          # λ_adv * L_adv（论文权重0.1）
                        1.0 * fm_loss +           # λ_FM * L_FM（论文权重1.0）
                        0.2 * loss_sor +         # λ_SOR * L_SOR（论文权重0.2）
                        0.05 * loss_edge +        # 边界损失（辅助）
                        cps_weight * loss_cps +   # CPS一致性损失（动态权重）
                        0.05 * loss_scl +         # 语义对比损失（辅助）
                        0.05 * loss_soft_pseudo   # 软伪标签损失（辅助）
                )

                # 总损失：L_total = L_sup + L_u
                # 其中L_sup包含监督损失和边界损失，L_u包含所有无标注损失组件
                loss = loss_l + loss_u
                loss.mean().backward()
                optim.step()
                if not args.no_scd and optim_adapter is not None:
                    optim_adapter.step()

            adjust_lr_rate(optim, itr, total_batch)
        model.eval()
        if valid_sign == True:
            recall, specificity, precision, F1, F2, \
                ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, *_ = evaluate(model, valid_dataloader, val_total_batch)
            save_output = False
            if dice > best:
                best = dice
                save_output = True
            print("Best Dice:: ", best)

            if save_output:
                result_dir = './result'
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                os.makedirs(result_dir)
                evaluate(model, valid_dataloader, val_total_batch, save_best=True)

            print("Valid Result:")
            print(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f' \
                % (recall, specificity, precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice))

            scheduler.step()
            if (F1 > F1_best):
                F1_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/best.pth")
            elif (F1 > F1_second_best):
                F1_second_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/second_best.pth")
            elif (F1 > F1_third_best):
                F1_third_best = F1
                torch.save(model.state_dict(), args.root + "/semi/checkpoint/" + args.ckpt_name + "/third_best.pth")


def test():
    print('loading data......')
    test_data = build_dataset(args)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    total_batch = math.ceil(len(test_data) / args.batch_size)
    model = build_model(args)
    if args.GPUs:
        model = model.cuda()
    # 优先加载 teacher 的 best；没有就回退 student 的 best；再没有就用随机初始化
    ckpt_dir_stu = os.path.join(args.root, "semi", "checkpoint", args.ckpt_name)
    ckpt_dir_tch = os.path.join(args.root, "semi", "checkpoint", args.ckpt_name + "_teacher")
    pth_tch = os.path.join(ckpt_dir_tch, "best.pth")
    pth_stu = os.path.join(ckpt_dir_stu, "best.pth")

    if os.path.exists(pth_tch):
        print(f"[Test] Loading EMA-Teacher checkpoint: {pth_tch}")
        model.load_state_dict(torch.load(pth_tch, map_location="cpu"))
    elif os.path.exists(pth_stu):
        print(f"[Test] Loading Student checkpoint: {pth_stu}")
        model.load_state_dict(torch.load(pth_stu, map_location="cpu"))
    else:
        print("[Test] WARNING: no checkpoint found; testing with randomly initialized weights.")

    model.eval()

    recall, specificity, precision, F1, F2, \
        ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice, _, _, table_metrics = \
        evaluate(model, test_dataloader, total_batch, spacing=(0.07, 0.07))

    if args.dataset.lower() == "hc18":
        # HC18 打印精简版
        print(
            'Valid Result: recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
            'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f, '
            'DSC_all: %.4f, Jacc_all: %.4f, HD95_all: %.4f, ASD_all: %.4f'
            % (recall, specificity, precision, F1, F2,
               ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,
               table_metrics['DSC'], table_metrics['Jaccard'],
               table_metrics['HD95'], table_metrics['ASD'])
        )
    else:
        # 其他数据集（例如 PSFH）打印全指标
        print(
            'Valid Result: recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
            'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, dice: %.4f, '
            'DSC_all: %.4f, Jacc_all: %.4f, HD95_all: %.4f, ASD_all: %.4f, '
            'DSC_PS: %.4f, Jacc_PS: %.4f, HD95_PS: %.4f, ASD_PS: %.4f, '
            'DSC_FH: %.4f, Jacc_FH: %.4f, HD95_FH: %.4f, ASD_FH: %.4f'
            % (recall, specificity, precision, F1, F2,
               ACC_overall, IoU_poly, IoU_bg, IoU_mean, dice,
               table_metrics['DSC'], table_metrics['Jaccard'], table_metrics['HD95'], table_metrics['ASD'],
               table_metrics['DSC_PS'], table_metrics['Jaccard_PS'], table_metrics['HD95_PS'], table_metrics['ASD_PS'],
               table_metrics['DSC_FH'], table_metrics['Jaccard_FH'], table_metrics['HD95_FH'], table_metrics['ASD_FH'])
        )


if __name__ == '__main__':

    checkpoint_name = os.path.join(args.root, 'semi/checkpoint/' + args.ckpt_name)
    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)
    else:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
    if args.manner == 'full':
        print('---{}-Seg Train---'.format(args.dataset))
        train()
    elif args.manner == 'semi':
        print('---{}-seg Semi-Train--'.format(args.dataset))
        train_semi()
    elif args.manner == 'test':
        print('---{}-Seg Test---'.format(args.dataset))
        test()
    print('Done')

