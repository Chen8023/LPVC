import functools
import os
import math
from collections import namedtuple

import numpy as np

from models.triplet_attention import TripletAttention
from util.tps import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian, to_homogeneous, \
    from_homogeneous
from util.tps import UpBlock2d, TPS, ResBlock2d, SameBlock2d, DownBlock2d, DownBlock2d_blur, ResBlock2d_LSTM, \
    UpBlock2d_LSTM

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
import torch.utils.data
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def define_G(use_LSTM, use_BlurPool, gpu_ids):
    net = FgBgFusionNetwork(num_channels=6, block_expansion=64, max_features=512, num_down_blocks=3, multi_mask=True,
                            use_LSTM=use_LSTM, use_BlurPool=use_BlurPool)
    return init_net(net, init_type='normal', gpu_ids=gpu_ids)


def define_DenseMotionNet(num_tps, bg, gpu_ids):
    net = DenseMotionNetwork(block_expansion=64, num_blocks=5, max_features=1024, num_tps=num_tps,
                             num_channels=3, scale_factor=0.25, bg=bg, multi_mask=True, kp_variance=0.01)
    return init_net(net, init_type='normal', gpu_ids=gpu_ids)


def define_KPDetector(num_tps, gpu_ids, mode='half', dataset_name='fashion'):
    net = KPDetector(num_tps=num_tps, mode=mode, dataset_name=dataset_name)
    return init_net(net, init_type='normal', gpu_ids=gpu_ids)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu',
             use_sigmoid=False, init_type='xavier', gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                             use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_Feature_Net(requires_grad=False, net_type='vgg16', gpu_ids=[]):
    netFeature = None

    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())

    if net_type == 'vgg16':
        netFeature = Vgg16(requires_grad=requires_grad)
    else:
        raise NotImplementedError('Feature net name [%s] is not recognized' % net_type)

    if len(gpu_ids) > 0:
        netFeature.cuda(gpu_ids[0])

    return netFeature



# feature loss network
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super(Vgg16, self).__init__()
        os.environ['TORCH_HOME'] = './examples/VGGmodels/'
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Vgg19_(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=True):
        super(Vgg19_, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_tps, mode, dataset_name, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps
        self.mode = mode
        self.dataset_name = dataset_name
        self.inplanes = 128
        self.planes = 1024
        self.num_basicblocks = 3
        self.backbone = Vgg16()
        if torch.cuda.is_available():
            self.backbone = self.backbone.cuda()
        self.sa_relu2 = SpatialAttention()
        self.ca_relu2 = ChannelAttention(self.inplanes)
        self.sa_relu3 = SpatialAttention()
        self.ca_relu3 = ChannelAttention(self.inplanes * 2)
        self.sa_relu4 = SpatialAttention()
        self.ca_relu4 = ChannelAttention(self.inplanes * 4)

        down_blocks = []
        resblocks = []

        # self.basicblock =
        for i in range(self.num_basicblocks):
            if i == 0:
                resblocks.append(ResBlock2d(self.inplanes * (2 ** i), kernel_size=(3, 3), padding=(1, 1)))
                resblocks.append(ResBlock2d(self.inplanes * (2 ** i), kernel_size=(3, 3), padding=(1, 1)))
                down_blocks.append(
                    DownBlock2d(self.inplanes * (2 ** i), self.inplanes * (2 ** (i + 1)), kernel_size=(3, 3),
                                padding=(1, 1)))
            else:
                resblocks.append(ResBlock2d(self.inplanes * (2 ** (i + 1)), kernel_size=(3, 3), padding=(1, 1)))
                resblocks.append(ResBlock2d(self.inplanes * (2 ** (i + 1)), kernel_size=(3, 3), padding=(1, 1)))
                down_blocks.append(
                    DownBlock2d(self.inplanes * (2 ** (i + 1)), self.inplanes * (2 ** (i + 1)), kernel_size=(3, 3),
                                padding=(1, 1)))

        self.resblock = nn.ModuleList(resblocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes, num_tps * 5 * 2)

    def seg_weights_cal(self, seg):
        '''
        :param seg:BCHW
        :return: seg_mask,dit
        '''
        if self.dataset_name == 'fashion':
            seg_tensors = [torch.zeros_like(seg) for _ in range(18)]
            for class_idx in range(18):
                seg_tensors[class_idx] = torch.where(seg == class_idx / 255.0, torch.ones_like(seg),
                                                     torch.zeros_like(seg))

            out_dit = {}
            out_dit['bg'] = seg_tensors[0]
            out_dit['head'] = seg_tensors[1] + seg_tensors[2] + seg_tensors[3] + seg_tensors[11]
            out_dit['torso'] = seg_tensors[4] + seg_tensors[5] + seg_tensors[7] + seg_tensors[8] + seg_tensors[16] + \
                               seg_tensors[17]
            out_dit['limbs'] = seg_tensors[6] + seg_tensors[9] + seg_tensors[10] + seg_tensors[12] + seg_tensors[13] + \
                               seg_tensors[14] + seg_tensors[15]

        elif self.dataset_name == 'vox':
            seg_tensors = [torch.zeros_like(seg) for _ in range(7)]
            for class_idx in range(7):
                seg_tensors[class_idx] = torch.where(seg == class_idx / 255.0, torch.ones_like(seg),
                                                     torch.zeros_like(seg))

            out_dit = {}
            out_dit['bg'] = seg_tensors[0]
            out_dit['head'] = seg_tensors[1]
            out_dit['torso'] = seg_tensors[2]#out_dit['head']
            out_dit['limbs'] = seg_tensors[3] + seg_tensors[4] + seg_tensors[5] + seg_tensors[6]#out_dit['head']

        else:
            seg_tensors = [torch.zeros_like(seg) for _ in range(7)]
            for class_idx in range(7):
                seg_tensors[class_idx] = torch.where(seg == class_idx / 255.0, torch.ones_like(seg),
                                                     torch.zeros_like(seg))

            out_dit = {}
            out_dit['bg'] = seg_tensors[0]
            out_dit['head'] = seg_tensors[1]
            out_dit['torso'] = seg_tensors[2]
            out_dit['limbs'] = seg_tensors[3] + seg_tensors[4] + seg_tensors[5] + seg_tensors[6]

        return out_dit

    def forward(self, x, seg_mask):
        seg_mask_decomposed = self.seg_weights_cal(seg_mask)
        # print(seg_mask_decomposed['head'].shape)

        x_head = x * seg_mask_decomposed['head']
        x_torso = x * seg_mask_decomposed['torso']
        x_limbs = x * seg_mask_decomposed['limbs']
        # x_bg = x * seg_mask_decomposed['bg']  # x*(1-mask['head']-mask['torso']-mask['limbs'])

        if self.mode == 'head':

            x_relu4 = self.backbone(x_head)[3]
            x_relu3 = self.backbone(x_limbs)[2]
            x_relu2 = self.backbone(x_torso)[1]

        elif self.mode == 'half':
            x_relu4 = (self.backbone(x_head)[3] + self.backbone(x_limbs)[3]) / 2.0
            x_relu3 = (self.backbone(x_limbs)[2] + self.backbone(x_head)[2]) / 2.0
            x_relu2 = self.backbone(x_torso)[1]

        else:
            x_relu4 = self.backbone(x_limbs)[3]
            x_relu3 = self.backbone(x_head)[2]
            x_relu2 = self.backbone(x_torso)[1]

        x_relu2 = self.ca_relu2(x_relu2) * x_relu2
        x_relu2 = self.sa_relu2(x_relu2) * x_relu2
        x_relu3 = self.ca_relu3(x_relu3) * x_relu3
        x_relu3 = self.sa_relu3(x_relu3) * x_relu3
        x_relu4 = self.ca_relu4(x_relu4) * x_relu4
        x_relu4 = self.sa_relu4(x_relu4) * x_relu4

        out = self.resblock[0](x_relu2)
        out = self.resblock[1](out)
        out = self.down_blocks[0](out)

        out = self.resblock[2](torch.cat([x_relu3, out], dim=1))
        out = self.resblock[3](out)
        out = self.down_blocks[1](out)

        out = self.resblock[4](torch.cat([x_relu4, out], dim=1))
        out = self.resblock[5](out)
        out = self.down_blocks[2](out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        fg_kp = self.fc(out)

        bs, _, = fg_kp.shape  # [B, 50, 2]

        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1  # (0,1)→(0,2)→(-1，1)
        out = {'fg_kp': fg_kp.view(bs, self.num_tps * 5, -1)}

        return out


class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels,
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))

        if multi_mask:
            up = []
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4

            channel = [hourglass_output_size[-1] // (2 ** i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))
            occlusion = []

            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(
            heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)
        trans = TPS(mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        if not (bg_param is None):
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_tps + 1, -1, h, w))
        return deformed

    def dropout_softmax(self, X, P):

        drop = (torch.rand(X.shape[0], X.shape[1]) < (1 - P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2], X.shape[3], 1, 1).permute(2, 3, 0, 1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:, 1:, ...] /= (1 - P)
        mask_bool = (drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition

    def forward(self, source_image, kp_driving, kp_source, bg_param=None, dropout_flag=False, dropout_p=0):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)

        out_dict['transformations'] = transformations  # BHWC

        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict['deformed_source'] = deformed_source
        deformed_source = deformed_source.view(bs, -1, h, w)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input, mode=1)

        contribution_maps = self.maps(prediction[-1])
        if (dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps


        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)  # BCHW-BHWC

        out_dict['deformation'] = deformation  # Optical Flow

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion_map.append(
                    torch.sigmoid(self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))

        out_dict['occlusion_map'] = occlusion_map
        return out_dict








class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss


class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_loss, self).forward(img1, img2))


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor


    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).cuda()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(input, target_is_real).cuda()
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):

        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

