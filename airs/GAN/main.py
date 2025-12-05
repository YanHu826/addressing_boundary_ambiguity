from __future__ import print_function

import argparse
import json
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F

import models.dcgan as dcgan
import models.mlp as mlp
from data.BUSI import BUSIDataSet
from data.tn3k import tn3kDataSet
from data.HC18 import HC18Dataset
from data.PSFH import PSFHDataset

warnings.filterwarnings("ignore", category=UserWarning)
sobel_x = torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).unsqueeze(0)
sobel_y = torch.Tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).unsqueeze(0)


def compute_edges(x):
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=10001, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default='result', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--root', type=str, default='/root/Shape-Prior-Semi-Seg/airs/')
    parser.add_argument('--expID', type=int, default=1)
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset == 'tn3k':
        dataset = tn3kDataSet(opt.root, opt.expID, mode='train')
    elif opt.dataset == 'busi':
        dataset = BUSIDataSet(opt.root, opt.expID, mode='train')
    elif opt.dataset == 'hc18':
        dataset = HC18Dataset(opt.root, opt.expID, mode='train')
    elif opt.dataset == 'psfh':
        dataset = PSFHDataset(opt.root, opt.expID, mode='train')
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)  # number of gpu #1
    nz = int(opt.nz)  # size of the latent z vector #100
    ngf = int(opt.ngf)  # 64
    ndf = int(opt.ndf)  # 64
    nc = int(opt.nc)  # input images channels #3
    n_extra_layers = int(opt.n_extra_layers)  # Number of extra layers on gen and disc #0

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu,
                        "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu,
                        "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    netG.apply(weights_init)
    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
        print("using cuda ===================================== ")
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = next(data_iter)
                i += 1

                # train with real
                img_real = data['image']  # load image
                real_mask = data['label']  # load the real mask
                img_real = F.interpolate(img_real, size=(64, 64), mode='bilinear', align_corners=False)
                real_mask = F.interpolate(real_mask, size=(64, 64), mode='bilinear', align_corners=False)

                if opt.cuda:
                    img_real = img_real.cuda()
                    real_mask = real_mask.cuda()

                input_real = torch.cat([img_real, real_mask], dim=1)
                inputv = Variable(input_real)

                errD_real = netD(inputv)

                # Extract the intermediate features of the discriminator from the real image.
                features_real = netD.main[:-1](inputv).detach()  # Extract the output of the penultimate layer.
                errD_real.backward(one)
                # train with fake
                curr_batch_size = img_real.size(0)
                noise.resize_(curr_batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev)
                fake = fake[:img_real.size(0)]  # Keep the batch size consistent
                fake = F.interpolate(fake, size=img_real.shape[2:], mode='bilinear', align_corners=False)
                input_fake = torch.cat([img_real, fake], dim=1)  # joint images and pseudo masks
                inputv = input_fake
                errD_fake = netD(inputv)
                # Extract the intermediate features of the discriminator for fake images.
                features_fake = netD.main[:-1](inputv).detach()  # Extract the output of the penultimate layer.
                errD_fake.backward(mone)
                edges_real = compute_edges(real_mask)
                edges_fake = compute_edges(fake)
                edge_loss = F.l1_loss(edges_real, edges_fake)
                lambda_edge = 1.0
                errD = (errD_real - errD_fake) + lambda_edge * edge_loss
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            curr_batch_size = img_real.size(0)
            noise.resize_(curr_batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            fake = F.interpolate(fake, size=img_real.shape[2:], mode='bilinear', align_corners=False)
            input_fake = torch.cat([img_real, fake], dim=1)  # joint images and pseudo masks
            errG_main = netD(input_fake)
            feature_matching_loss = F.l1_loss(features_fake, features_real)

            lambda_fm = 10.0
            errG_total = (-1 * errG_main) + (lambda_fm * feature_matching_loss)

            errG_total.backward()
            optimizerG.step()

            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                     errD.data[0], errG_total.data[0], errD_real.data[0], errD_fake.data[0]))

        # do checkpointing
        if epoch % 500 == 0:
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
