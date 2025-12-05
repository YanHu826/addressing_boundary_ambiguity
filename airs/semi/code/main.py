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
from utils.loss import BceDiceLoss, sigmoid_rampup, SemanticContrastiveLoss, dynamic_edge_loss
import math
import warnings
from utils.loss import edge_loss

warnings.filterwarnings("ignore", category=UserWarning)


def DeepSupSeg(pred, gt):
    criterion = BceDiceLoss()
    loss = criterion(pred, gt)
    return loss


def get_boundary_map(mask_batch):
    boundary_batch = []
    for mask in mask_batch:
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        edge = cv2.Canny(mask_np * 255, 100, 200) / 255.0
        boundary_batch.append(torch.from_numpy(edge).unsqueeze(0))
    return torch.stack(boundary_batch).float().to(mask_batch.device)


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
    """load data"""
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

    if not args.no_scd:
        netD = DCGAN_D(isize=64, nz=100, nc=4, ndf=64, ngpu=1)
        netD.cuda()
        netD_weight = torch.load("models/pretrain/GAN/netD_epoch_10000.pth")
        new_state_dict = {}
        for k, v in netD_weight.items():
            if k == "main.initial:1-64:conv.weight":
                print(f"Rename key: {k} -> main.initial:4-64:conv.weight")
                new_state_dict["main.initial:4-64:conv.weight"] = v
            else:
                new_state_dict[k] = v

        netD.load_state_dict(new_state_dict)
        netD.eval()

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
            optim.zero_grad()
            pred_l = model(img_l)
            mask = pred_l[0]
            boundary_pred = pred_l[-1]  # Obtain boundary predictions from the model output
            boundary_gt = get_boundary_map(gt)
            loss_boundary = F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt)
            loss_l_seg = DeepSupSeg(mask, gt)
            loss_l = loss_l_seg + 0.2 * loss_boundary
            pred_u = model(img_u)
            _, predboud, sor_feat2, sor_feat3, sor_feat4, sor_feat5, mask_boud, _ = pred_u  # 来自 SOR decoder

            if not args.no_scd:
                # print('未删除scd')
                shape_u_1 = F.interpolate(predboud, size=(64, 64), mode='bilinear', align_corners=False)
                shape_u_2 = F.interpolate(sor_feat2, size=(64, 64), mode='bilinear', align_corners=False)
                shape_u_3 = F.interpolate(sor_feat3, size=(64, 64), mode='bilinear', align_corners=False)
                shape_u_4 = F.interpolate(sor_feat4, size=(64, 64), mode='bilinear', align_corners=False)
                shape_u_5 = F.interpolate(sor_feat5, size=(64, 64), mode='bilinear', align_corners=False)
            # ---------------- Segmentation loss ----------------
            with torch.no_grad():
                prob_map = torch.sigmoid(predboud)  # soft prediction
                weights = torch.ones_like(prob_map)
                weights[(prob_map >= 0.4) & (prob_map <= 0.6)] = 0.5  # Downweight the regions with low confidence

            loss_u_seg = (DeepSupSeg(predboud, mask_boud) * weights).mean()

            # ---------------- Training of the discriminator D ----------------
            if not args.no_scd:
                # print('未删除scd')
                # Unify the spatial dimensions
                shape_u_1_up = F.interpolate(shape_u_1, size=img_u.shape[2:], mode='bilinear', align_corners=False)

                # Mosaic images and pseudo-label features
                real_feat = torch.cat([img_u, shape_u_1_up.detach()], dim=1)
                fake_feat = torch.cat([
                    img_u,
                    F.interpolate(sor_feat2.detach(), size=img_u.shape[2:], mode='bilinear', align_corners=False)
                ], dim=1)
                # Feature Matching Loss
                _, real_features = netD(real_feat, return_features=True)
                _, fake_features = netD(fake_feat, return_features=True)

                fm_loss = 0
                for rf, ff in zip(real_features, fake_features):
                    fm_loss += F.l1_loss(ff, rf.detach())  # detach real to avoid backprop to D
                fm_loss = fm_loss / len(real_features)  # The average loss of multiple feature layers

                criterion_GAN = nn.BCEWithLogitsLoss()
                # Upsample shape_u_1 to the same spatial size as img_u
                shape_u_1_up = F.interpolate(shape_u_1.detach(), size=img_u.shape[2:], mode='bilinear', align_corners=False)

                # The spliced images and pseudo-label features are then sent back to the discriminator
                real_pred = netD(torch.cat([img_u, shape_u_1_up], dim=1))
                fake_pred = netD(torch.cat([
                    img_u,
                    F.interpolate(sor_feat2.detach(), size=img_u.shape[2:], mode='bilinear', align_corners=False)
                ], dim=1))
                real_labels = torch.ones_like(real_pred)
                fake_labels = torch.zeros_like(fake_pred)

                errD_real = criterion_GAN(real_pred, real_labels)
                errD_fake = criterion_GAN(fake_pred, fake_labels)
                errD = (errD_real + errD_fake) * 0.5

                # Update D
                netD.train()
                optimizer_D = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
                if epoch % 2 == 0:
                    optimizer_D.zero_grad()
                    errD.backward()
                    optimizer_D.step()

            # ---------------- edge loss ----------------
            if not args.no_scd:
                loss_edge = edge_loss(predboud, mask_boud)

            # ---------------- Generator adversarial loss ----------------
            if not args.no_scd:
                pred_fake_for_G = netD(
                    torch.cat([
                        img_u,
                        F.interpolate(sor_feat2, size=img_u.shape[2:], mode='bilinear', align_corners=False)
                    ], dim=1)
                )

                errG_adv = criterion_GAN(pred_fake_for_G, real_labels)

            # ---------------- Summary of Final Losses ----------------
            if not args.no_scd:
                loss_u_shape = (
                                       netD(torch.cat([
                                           img_u,
                                           F.interpolate(shape_u_1, size=img_u.shape[2:], mode='bilinear',
                                                         align_corners=False)
                                       ], dim=1)) +
                                       netD(torch.cat([
                                           img_u,
                                           F.interpolate(shape_u_2, size=img_u.shape[2:], mode='bilinear',
                                                         align_corners=False)
                                       ], dim=1)) +
                                       netD(torch.cat([
                                           img_u,
                                           F.interpolate(shape_u_3, size=img_u.shape[2:], mode='bilinear',
                                                         align_corners=False)
                                       ], dim=1)) +
                                       netD(torch.cat([
                                           img_u,
                                           F.interpolate(shape_u_4, size=img_u.shape[2:], mode='bilinear',
                                                         align_corners=False)
                                       ], dim=1)) +
                                       netD(torch.cat([
                                           img_u,
                                           F.interpolate(shape_u_5, size=img_u.shape[2:], mode='bilinear',
                                                         align_corners=False)
                                       ], dim=1))
                               ) / 5

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

                # Dynamic weight ramp-up
                cps_weight = sigmoid_rampup(epoch, rampup_length=10)
                loss_u = (
                        0.75 * loss_u_seg +
                        0.1 * loss_u_shape +
                        0.05 * loss_edge +
                        0.1 * fm_loss +
                        0.05 * errG_adv +
                        cps_weight * loss_cps +
                        0.05 * loss_scl +
                        0.05 * loss_soft_pseudo
                )

                loss = 2 * loss_l + loss_u
                loss.mean().backward()
                optim.step()

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

