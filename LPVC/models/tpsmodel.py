import torch
import torch.nn.functional as F
from PIL import ImageDraw

from util.tps import TPS
from . import networks
from .base_model import BaseModel
import util.util as util
from util.flowlib import save_flow_image
from util.metrics import util_of_lpips, util_of_dists, get_TensorMSSSIM, get_TensorPSNR
from compressai.entropy_models import EntropyBottleneck
from .flow_net import ME_Spynet, flow_warp, LiteFlownet, liteflownet
import numpy as np
import imageio
import os
import torchvision.transforms as transforms
from util.util import *
from torchvision import transforms
import time

# from models import EMA

class TPSModel(BaseModel):
    def name(self):
        return 'TPSModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        # if opt.isTrain:
        #     assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        # self.ema_updater = EMA(0.99)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.use_D = opt.isTrain and opt.lambda_GAN > 0.0
        self.bg_param = None

        self.loss_names = ['G_GAN', 'G_L1', 'Lp', 'Lf', 'equivariance', 'wrap']
        if self.use_D:
            self.loss_names += ['D', 'D_fake', 'D_real']

        if self.opt.bg:
            self.loss_names += ['bg']

        self.metric_names = ['psnr', 'ms_ssim', 'total_bpp', 'bpp_kp', 'value_lpips',
                             'value_dists']
        self.visual_names = ['fake_B', 'real_B', 'wrap', 'ref', 'add_bg']
        self.test_visual_names = ['raw', 'raw_bg', 'add_bg', 'lib_bg', 'recon_image', 'wrap'
            , 'flow_dense', 'flow_spy', 'raw_key_points',  'occlusion_map']


        # perceptual losses
        self.dists = util_of_dists(use_gpu=True)
        # self.lpips = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=self.opt.gpu_ids)
        self.lpips = util_of_lpips(net='alex', use_gpu=True)

        # models
        self.model_names = ['KP_Detector']
        self.netKP_Detector = networks.define_KPDetector(num_tps=opt.num_tps, gpu_ids=self.gpu_ids, mode=self.opt.mode,
                                                         dataset_name=self.opt.dataset_name)

        self.model_names += ['DenseMotionNet']
        self.netDenseMotionNet = networks.define_DenseMotionNet(num_tps=opt.num_tps, bg=self.opt.bg,
                                                                gpu_ids=self.gpu_ids)


        self.model_names += ['G']
        self.netG = networks.define_G(opt.use_LSTM, opt.use_BlurPool, gpu_ids=self.gpu_ids)

        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        self.gan_mode = opt.gan_mode
        use_sigmoid = self.gan_mode == 'dcgan'
        if self.use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          gpu_ids=self.gpu_ids)


        self.pyramid = networks.ImagePyramide(self.opt.scales, 3)  # self.netG.num_channels
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        if sum(self.opt.lambda_perceptual) != 0:
            self.vgg = networks.Vgg16()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        ##optimizers
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # self.criterionZ = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            if opt.no_TTUR:
                G_lr, D_lr = opt.lr, opt.lr / 2
            else:
                G_lr, D_lr = opt.lr / 2, opt.lr * 2

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=G_lr,
                                                betas=(opt.beta1, opt.beta2), weight_decay=1e-4)
            self.optimizers.append(self.optimizer_G)


            self.optimizer_KP_Detector = torch.optim.Adam(self.netKP_Detector.parameters(), lr=G_lr,
                                                          betas=(opt.beta1, opt.beta2), weight_decay=1e-4)
            self.optimizers.append(self.optimizer_KP_Detector)

            self.optimizer_DenseMotionNet = torch.optim.Adam(self.netDenseMotionNet.parameters(), lr=G_lr,
                                                             betas=(opt.beta1, opt.beta2), weight_decay=1e-4)
            self.optimizers.append(self.optimizer_DenseMotionNet)

            if self.use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=D_lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):

        self.real_B = self.raw = input['raw_image'].to(self.device)
        self.ref = input['ref_image'].to(self.device)

        self.raw_bg = input['raw_bg'].to(self.device)
        self.ref_bg = input['ref_bg'].to(self.device)

        self.lib_bg = input['lib_bg'].to(self.device)

        if input['add_bg'] is not None:
            self.add_bg = input['add_bg'].to(self.device)
            self.visual_names += ['add_bg']
        else:
            self.add_bg = None

        self.raw_fg = input['raw_fg'].to(self.device)
        self.ref_fg = input['ref_fg'].to(self.device)

        self.raw_seg = input['raw_seg'].to(self.device)
        self.ref_seg = input['ref_seg'].to(self.device)


        self.raw_mask = input['raw_mask'].to(self.device)
        self.ref_mask = input['ref_mask'].to(self.device)
        self.raw_mask_edge = input['raw_mask_edge'].to(self.device)

        self.im_shape = self.raw.size()  # [B,C,H,W]

    def forward(self, epoch):
        # print_info('model forward')
        start_time = time.time()

        kp_raw = self.netKP_Detector(self.raw, self.raw_seg)
        kp_ref = self.netKP_Detector(self.ref, self.ref_seg)

        self.raw_key_points = kp_raw
        entropy_bottleneck = EntropyBottleneck(channels=1).to(self.device)
        kp_raw_hat, feature_likelihoods_kp_raw = entropy_bottleneck(kp_raw['fg_kp'])
        kp_ref_hat, feature_likelihoods_kp_ref = entropy_bottleneck(kp_ref['fg_kp'])

        if self.add_bg is not None:
            add_bg_hat, feature_likelihoods_add_bg = entropy_bottleneck(self.add_bg)
        else:
            feature_likelihoods_add_bg = 0

        self.feature_likelihoods = {}
        self.feature_likelihoods['kp'] = feature_likelihoods_kp_raw + feature_likelihoods_kp_ref
        self.feature_likelihoods['add_bg'] = feature_likelihoods_add_bg
        if epoch >= self.opt.dropout_epoch:
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp
            dropout_flag = True
            dropout_p = min(epoch / self.opt.dropout_inc_epoch * self.opt.dropout_maxp
                            + self.opt.dropout_startp, self.opt.dropout_maxp)
        if not self.opt.isTrain and epoch == -1:  # test
            dropout_flag = False
            dropout_p = 0
        end_time = time.time()
        time_difference = end_time - start_time
        print(f"Encoder time: {time_difference} seconds")
        start_time = time.time()
        dense_motion = self.netDenseMotionNet(source_image=self.ref, kp_driving=kp_raw_hat,
                                              kp_source=kp_ref_hat, bg_param=self.bg_param,
                                              dropout_flag=dropout_flag, dropout_p=dropout_p)
        ##generate
        self.generated = self.netG(self.ref, dense_motion, self.lib_bg)
        self.real_A = self.generated['kp_driving']
        self.recon_image = self.fake_B = self.generated['prediction']  # G(A)
        self.flow_dense = dense_motion['deformation']
        self.wrap = self.netG.module.deform_input(self.ref, self.flow_dense)
        self.occlusion_map = dense_motion['occlusion_map']
        end_time = time.time()
        time_difference = end_time - start_time
        print(f"Decoder time: {time_difference} seconds")

        return self.fake_B, self.real_B


    def test(self, i):
        # print_info('model test')

        with torch.no_grad():
            self.forward(-1)

            # some metrics
            self.value_lpips = torch.mean(self.lpips.calc_lpips_value(self.fake_B, self.raw)).detach()
            self.value_dists = torch.mean(self.dists.calc_dists_value(self.fake_B, self.raw)).detach()

            self.psnr = get_TensorPSNR(self.raw, self.fake_B)
            self.ms_ssim = get_TensorMSSSIM(self.raw, self.fake_B)

            # bpp metric
            total_bits_feature = 0.0
            for v in self.feature_likelihoods.values():
                total_bits_feature += torch.sum(torch.log(v)) / (-np.log(2))
            self.total_bpp = total_bits_feature / (self.im_shape[0] * self.im_shape[2] * self.im_shape[3])

            bits_feature_kp = torch.sum(torch.log(self.feature_likelihoods['kp'])) / (-np.log(2))
            self.bpp_kp = bits_feature_kp / (self.im_shape[0] * self.im_shape[2] * self.im_shape[3])

            bits_feature_add_bg = torch.sum(torch.log(self.feature_likelihoods['add_bg'])) / (-np.log(2))
            self.bpp_add_bg = bits_feature_add_bg / (self.im_shape[0] * self.im_shape[2] * self.im_shape[3])

            self.bpp = {}
            self.bpp['total'] = self.total_bpp
            self.bpp['kp'] = self.bpp_kp
            self.bpp['add_bg'] = self.bpp_add_bg
            return self.fake_B, self.psnr, self.ms_ssim, self.value_lpips, self.value_dists, self.bpp


    def backward_D(self):
        # print_info('model backward_D')

        """Calculate GAN loss for the discriminator"""
        if self.opt.conditional_D:
            fake_data = torch.cat((self.real_A, self.fake_B),
                                  1)
            real_data = torch.cat((self.real_A, self.real_B), 1)
        else:
            fake_data = self.fake_B
            real_data = self.real_B

        pred_fake = self.netD(fake_data.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        # Real
        # real_AB = torch.cat((self.input_G, self.real_B), 1)
        pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_others(self):

        if self.opt.conditional_D:  # tedious conditoinal data
            fake_data = torch.cat((self.real_A, self.fake_B),
                                  1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        else:
            fake_data = self.fake_B

        if self.use_D:
            pred_fake = self.netD(fake_data)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0

        ## reconstruction loss

        ## L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # print_info('model loss_G_L1:', self.loss_G_L1.cpu().data.numpy())
        # warp loss
        if self.opt.lambda_warp != 0:
            # deformed_source = self.generated['deformed']
            self.loss_wrap = self.criterionL1(self.raw, self.wrap)  # * self.opt.lambda_warp

            occlusion_map = self.generated['occlusion_map']
            encode_map = self.netG.module.get_encode(self.raw, occlusion_map, self.raw_bg)
            decode_map = self.generated['warped_encoder_maps']  # 每层扭曲后的source image
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i] - decode_map[-i - 1]).mean()

            self.loss_wrap = (self.loss_wrap + value) * self.opt.lambda_warp

        ## Lp loss
        pyramide_real = self.pyramid(self.raw)
        pyramide_generated = self.pyramid(self.generated['prediction'])
        if sum(self.opt.lambda_Lp) != 0:
            value_total = 0
            for scale in self.opt.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.opt.lambda_perceptual):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.opt.lambda_perceptual[i] * value
            self.loss_Lp = value_total
        else:
            self.loss_Lp = 0

        ## Lf loss
        if self.opt.lambda_Lf != 0:
            self.loss_Lf = self.criterionL2((self.fake_B-self.real_B)*self.raw_seg) * self.opt.lambda_Lf
        else:
            self.loss_Lf = 0

        # equivariance loss
        if self.opt.lambda_equivariance != 0:
            transform_random = TPS(mode='random', bs=self.raw.shape[0])
            transform_grid = transform_random.transform_frame(self.raw)
            transformed_frame = F.grid_sample(self.raw, transform_grid, padding_mode="reflection",
                                              align_corners=True)
            transformed_kp = self.netKP_Detector(transformed_frame, self.raw_seg)

            self.generated['transformed_frame'] = transformed_frame
            self.generated['transformed_kp'] = transformed_kp

            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = self.raw_key_points['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            self.loss_equivariance = self.opt.lambda_equivariance * value

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_Lf + self.loss_Lf + self.loss_wrap \
                      + self.loss_equivariance

        self.loss_G.backward()


    def optimize_parameters(self, epoch):

        self.forward(epoch)

        # update D
        if self.use_D:
            self.set_requires_grad(self.netD, True)
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update others
        if self.use_D:
            self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_KP_Detector.zero_grad()
        self.optimizer_DenseMotionNet.zero_grad()
        self.backward_others()

        self.optimizer_G.step()
        self.optimizer_KP_Detector.step()
        self.optimizer_DenseMotionNet.step()
