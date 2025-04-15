import numpy as np
import imageio
import os
#import util.flowlib as flowlib
#import util.metrics as metrics
import torch.nn as nn
import torch
import torch.nn.functional as F
from util.util import *

# from warp import tf_warp
import sys

# sys.path.append('/home/user321/tf_flownet2-master/FlowNet2_src/')
modelspath = './examples/flow_pretrain_np/'


# from flow_warp import flow_warp

def gather_nd(img, idx):
    """
    same as tf.gather_nd in pytorch
    """
    idx = idx.long()
    idx1, idx2, idx3 = idx.chunk(3, dim=3)
    output = img[idx1, idx2, idx3].squeeze(3)
    return output


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = x.size()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size).int().cuda()
    batch_idx = batch_idx.view(batch_size, 1, 1)
    b = batch_idx.repeat(1, height, width)

    indices = torch.stack([b, y, x], 3)
    # print(indices.size())

    return gather_nd(img, indices)


Backward_tensorGrid = {
    0: {},
    1: {},
    2: {},
    3: {},
    4: {},
    5: {},
    6: {},
    7: {},
    8: {},
    "cpu": {}
}


def torch_warp(tensorInput, tensorFlow):
    device_id = tensorInput.device.index
    if device_id == None:
        device_id = "cpu"
    if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical],
                                                                           1).cuda().to(device_id)
        # B, C, H, W = tensorInput.size()
        # xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        # yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        # Backward_tensorGrid[device_id][str(tensorFlow.size())] = Variable(torch.cat([xx, yy], 1).float().cuda()).to(device_id)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(
                Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border')


def log10(x):
    numerator = torch.log(x)
    denominator = torch.log(10)
    return numerator / denominator


def flow_warp(im, flow):
    warp = torch_warp(im, flow)

    return warp


def loadweightformnp(layername):
    index = layername.find('modelL')
    if index == -1:
        print('laod models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = modelspath + name + '-weight.npy'
        modelbias = modelspath + name + '-bias.npy'
        weightnp = np.load(modelweight)
        # weightnp = np.transpose(weightnp, [2, 3, 1, 0])
        # print(weightnp)
        biasnp = np.load(modelbias)

        # init_weight = lambda shape, dtype: weightnp
        # init_bias   = lambda shape, dtype: biasnp
        # print('Done!')

        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)
        # return init_weight, init_bias


class MEBasic(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv1.weight.data, self.conv1.bias.data = loadweightformnp(layername + '_F-1')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv2.weight.data, self.conv2.bias.data = loadweightformnp(layername + '_F-2')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv3.weight.data, self.conv3.bias.data = loadweightformnp(layername + '_F-3')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv4.weight.data, self.conv4.bias.data = loadweightformnp(layername + '_F-4')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        self.conv5.weight.data, self.conv5.bias.data = loadweightformnp(layername + '_F-5')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature


def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        # self.resblock = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2),
        #     nn.ReLU(),
        #     nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2),
        # )
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


class Warp_net(nn.Module):
    def __init__(self):
        super(Warp_net, self).__init__()
        channelnum = 64

        self.feature_ext = nn.Conv2d(6, channelnum, 3, padding=1)  # feature_ext
        self.f_relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)  # c0
        self.conv0_p = nn.AvgPool2d(2, 2)  # c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)  # c1
        self.conv1_p = nn.AvgPool2d(2, 2)  # c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)  # c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)  # c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)  # c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)  # c5
        self.conv6 = nn.Conv2d(channelnum, 3, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))
        c0 = self.conv0(feature_ext)
        c0_p = self.conv0_p(c0)
        c1 = self.conv1(c0_p)
        c1_p = self.conv1_p(c1)
        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling2(
            c3)  # torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling2(
            c4)  # torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
        c5 = self.conv5(c4_u)
        res = self.conv6(c5)
        return res


# def Preprocessing(im):
#     r, g, b = tf.split(im, 3, axis=3)
#     b = b - 0.406
#     g = g - 0.456
#     r = r - 0.485
#     b_v = b / 0.225
#     g_v = g / 0.224
#     r_v = r / 0.229

#     rgb = tf.concat(values=[r_v, g_v, b_v], axis=3)
#     return rgb

flowfiledsSamples = [{} for i in range(8)]


class ME_Spynet(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])
        # self.meBasic1 = MEBasic(layername + 'modelL1')
        # self.meBasic2 = MEBasic(layername + 'modelL2')
        # self.meBasic3 = MEBasic(layername + 'modelL3')
        # self.meBasic4 = MEBasic(layername + 'modelL4')
        # self.flow_warp = Resample2d()

        # self.meBasic = [self.meBasic1, self.meBasic2, self.meBasic3, self.meBasic4]

    # def Preprocessing(self, im):
    #     im[:, 0, :, :] -= 0.406
    #     im[:, 1, :, :] -= 0.456
    #     im[:, 2, :, :] -= 0.485
    #     im[:, 0, :, :] /= 0.225
    #     im[:, 1, :, :] /= 0.224
    #     im[:, 2, :, :] /= 0.229
    #     return im
    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat(
                [im1list[self.L - 1 - intLevel], flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample),
                 flowfiledsUpsample], 1))  # residualflow


        conf = (self.norm(im1 - flow_warp(im2, flowfileds)) < 0.02).float()

        return flowfileds,conf #flow_warp(im2,flowfileds)




def norm(t):
     return torch.sum(t*t, dim=1, keepdim=True)
def liteflownet(im1, im2,batch):
   # print('im1.size: ',im1.size())

    h = im1.size(2)
    w = im1.size(3)
    #print('liteflownet ',h,w)
    path_im1 = 'pytorch-liteflownet-master/images/one.png'
    path_im2 = 'pytorch-liteflownet-master/images/two.png'

    #batch = opt.batch_size
    flow = np.zeros((batch, 2, int(h), int(w)), dtype=np.float32)
    # if not opt.isTrain:
    #     im1 = torch.unsqueeze(im1,0)
    #     im2 = torch.unsqueeze(im2,0)


    for pic in range(batch):


        imageio.imsave(path_im1, tensor2im(torch.unsqueeze(im1[pic],0)))
        imageio.imsave(path_im2, tensor2im(torch.unsqueeze(im2[pic],0)))

        os.system('python ./pytorch-liteflownet-master/run.py --model default --one ./' + path_im1 +' --two ./'+ path_im2 +' --out ./pytorch-liteflownet-master/out.flo')

        f = open('./pytorch-liteflownet-master/out.flo', 'rb')
        x = np.fromfile(f, np.int32, count=1)  # not sure what this gives
        w = np.fromfile(f, np.int32, count=1)  # width
        h = np.fromfile(f, np.int32, count=1)  # height
        #print('x %d, w %d, h %d flo file' % (x, w, h))
        data = np.fromfile(f, np.float32)  # vector
        data_2D= np.reshape(data, newshape=(2,int(h), int(w)))#torch,caffe存储图片使用的是：B×C×H×W
        flow[pic] = data_2D
        f.close()
    flowfileds = torch.from_numpy(flow).cuda() #flow是numpy.ndarray类型
    conf = (norm(im1 - flow_warp(im2, flowfileds)) < 0.02).float()

    return flowfileds,conf

class LiteFlownet():
    def __init__(self, opt):
        self.path_im1 = 'pytorch-liteflownet-master/images/one.png'
        self.path_im2 = 'pytorch-liteflownet-master/images/two.png'
        self.opt = opt
        self.batch = opt.batch_size
        # print('liteflownet: batch = ', batch)

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)

    def forward(self, im1, im2):
        h = im1.shape(2)
        w = im1.shape(3)
        flow = np.zeros((self.batch, 2, int(h), int(w)), dtype=np.float32)
        # if not self.opt.isTrain:
        #     self.im1 = torch.unsqueeze(im1, 0)
        #     self.im2 = torch.unsqueeze(im2, 0)

        for pic in range(self.batch):
            imageio.imsave(self.path_im1, tensor2im(torch.unsqueeze(im1[pic], 0)))
            imageio.imsave(self.path_im2, tensor2im(torch.unsqueeze(im2[pic], 0)))

            os.system(
                'python ./pytorch-liteflownet-master/run.py --model default --one ./' + self.path_im1 + ' --two ./' + self.path_im2 + ' --out ./pytorch-liteflownet-master/out.flo')
            f = open('./pytorch-liteflownet-master/out.flo', 'rb')
            x = np.fromfile(f, np.int32, count=1)  # not sure what this gives
            w = np.fromfile(f, np.int32, count=1)  # width
            h = np.fromfile(f, np.int32, count=1)  # height
            data = np.fromfile(f, np.float32)  # vector

            data_2D = np.reshape(data, newshape=(2, int(h), int(w)))  # torch,caffe存储图片使用的是：B×C×H×W

            flow[pic] = data_2D

            f.close()
            flowfileds = torch.from_numpy(flow)  # flow是numpy.ndarray类型
            conf = (self.norm(im1 - flow_warp(im2, flowfileds)) < 0.02).float()

        return flowfileds, conf


# def ME_Ours(im1, im2):
#     h_d = im1.get_shape().as_list()[1]
#     w_d = im1.get_shape().as_list()[2]

#     im1_d = tf.contrib.layers.avg_pool2d(im1, kernel_size=[4, 4], stride=[4, 4])
#     im2_d = tf.contrib.layers.avg_pool2d(im2, kernel_size=[4, 4], stride=[4, 4])
#     flow_d, _ = ME_Spynet(im1_d, im2_d)
#     # feature extract...
#     with tf.variable_scope('FeatureExtract', reuse=tf.AUTO_REUSE) as scope:
#         c11 = conv2Block(layername='conv1a', inputfeature=im1, kernelNum=16, kernelSize=[3, 3], stride=[1, 1, 1, 1])
#         c21 = conv2Block(layername='conv1a', inputfeature=im2, kernelNum=16, kernelSize=[3, 3], stride=[1, 1, 1, 1])

#     with tf.variable_scope('GenerateFlow') as scope:
#         flow_up = tf.image.resize_images(flow_d, [h_d, w_d], align_corners=True)
#         c21_w = flow_warp(c21, flow_up)

#         c11 = conv2Block(
#             layername='conv1a',
#             inputfeature=tf.concat(values=[c11, c21_w], axis=3),
#             kernelNum=16,
#             kernelSize=[3, 3],
#             stride=[1, 1, 1, 1],
#             actFunc=leaky_relu)

#         flow_res = conv2Block(
#             layername='conv2a', inputfeature=c11, kernelNum=2, kernelSize=[3, 3], stride=[1, 1, 1, 1], actFunc=None)
#         flow = flow_res + flow_up

#         warpframe = flow_warp(im2, flow)

#     return flow, warpframe


def build_model():
    net = ME_Spynet().cuda()
    # inputImage1 = tf.placeholder(shape=[1,3, 448, 832], dtype=tf.float32, name="inputimage1")
    # inputImage2 = tf.placeholder(shape=[1,3, 448, 832], dtype=tf.float32, name="inputimage2")

    # # inputImage, inputRes, groudtruth
    # flowfiled, warpframe = ME_Spynet(inputImage1, inputImage2)

    # read images
    im1 = imageio.imread('input.png')
    im1 = im1 / 255.0
    im1 = np.expand_dims(im1, axis=0)
    im2 = imageio.imread('ref.png')
    im2 = im2 / 255.0
    im2 = np.expand_dims(im2, axis=0)

    # means = np.array([0.485, 0.456, 0.406])
    # stds = np.array([0.229, 0.224, 0.225])
    # input_image = (input_image - means) / stds
    # ref_image = (ref_image - means) / stds
    im1 = np.transpose(im1, [0, 3, 1, 2])
    im2 = np.transpose(im2, [0, 3, 1, 2])
    im1 = torch.from_numpy(im1).float().cuda()
    im2 = torch.from_numpy(im2).float().cuda()
    net.eval()
    flow, warp_frame = net(im1, im2)
    flow = flow.detach().cpu().numpy()
    #print(flow)
    warp_frame = warp_frame.cpu().detach().numpy()
    rgb_my = flowlib.flow_to_image(flow[0, :, :, :])
    print(rgb_my[0])
    #print(warp_frame)

    imageio.imwrite('flow.png', rgb_my[0])

    # # test spynet
    # flowspy = read_flow('result_hevc.flo')
    # flowspy_rgb = flow_to_image(flowspy)
    # imageio.imwrite('test2.png', flowspy_rgb)
    # psnr = CalcuPSNR(rgb_my / 255, flowspy_rgb / 255)
    # # print(rgb_my)
    # error = evaluate_flow(flow[0, :, :, :], flowspy)
    # # print(flow[0,:,:,:])
    # # print(flowspy)
    # print(error)

    # print(psnr)
    imageio.imwrite('warp2.png', warp_frame[0, :, :, :].transpose(1, 2, 0))
    # warp_frame_rgb = flow_warp(im2, flow)
    psnrwapr = metrics.CalcuPSNR(im1[0].cpu().numpy().transpose(1, 2, 0), warp_frame[0, :, :, :].transpose(1, 2, 0))
    print(psnrwapr)


if __name__ == '__main__':
    build_model()