"""This module contains metric functions """
import torch
import numpy as np
from PIL import Image
import os
import time
from termcolor import colored
import math
import torchvision.transforms as transforms

"""
skimage.measure.compare_ssim has been moved to '
         'skimage.metrics.structural_similarity. It will be removed from '
         'skimage.measure in version 0.18.
"""
from pytorch_msssim import ssim,ms_ssim

#Before version 0.18
from skimage.measure import compare_ssim, compare_psnr, compare_mse
#After version 0.18
from skimage.metrics import structural_similarity,peak_signal_noise_ratio

from skimage import io
import cv2
from DISTS_pytorch import DISTS
import lpips


###----------------------perceptual metrics--------------

class util_of_lpips():
    def __init__(self, net, use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips_value(self,tensor_img1,tensor_img2):
        return self.loss_fn.forward(tensor_img1, tensor_img2)
    def calc_lpips_loss(self,tensor_img1,tensor_img2):
        return self.loss_fn.forward(tensor_img1, tensor_img2)

    # def calc_lpips_value(self, img1_path, img2_path):
    #     '''
    #     Parameters
    #     ----------
    #     img1_path : str
    #         图像1的路径.
    #     img2_path : str
    #         图像2的路径.
    #     Returns
    #     -------
    #     dist01 : torch.Tensor
    #         学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).
    #
    #     References
    #     -------
    #     https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    #
    #     '''
    #     # Load images
    #     img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
    #     img1 = lpips.im2tensor(lpips.load_image(img2_path))
    #
    #     if self.use_gpu:
    #         img0 = img0.cuda()
    #         img1 = img1.cuda()
    #     dist01 = self.loss_fn.forward(img0, img1)
    #     return dist01

class util_of_dists():
    def __init__(self,use_gpu=False):
        self.use_gpu = use_gpu
        self.loss_fn = DISTS()
        if use_gpu:
            self.loss_fn.cuda()

    def calc_dists_value(self,tensor_img1,tensor_img2):
        return  self.loss_fn(tensor_img1,tensor_img2)

    def calc_dists_loss(self,tensor_img1,tensor_img2):
        # set 'require_grad=True, batch_average=True' to get a scalar value as loss.
        return self.loss_fn(tensor_img1,tensor_img2, require_grad=True, batch_average=True)


##------------------------calculate PSNR/SSIM/MS-SSIM---------------------------------#####

'''图像是tensor'''
#[b,c,h,w],使用现成的函数
def get_TensorSSIM(img1, img2):
    return ssim(img1, img2, data_range=1.0)

def get_TensorMSSSIM(img1, img2):
    return ms_ssim(img1, img2, data_range=1.0)

#[b,c,h,w]/[c,h,w],未经归一化,直接公式计算
def get_TensorPSNR(target, ref):
    mse_loss = torch.mean((target - ref).pow(2))
    psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
    return psnr#.cpu().detach().numpy()


'''图像是ndarray,未经归一化'''

#直接公式计算

def CalcuPSNR255(target, ref):
    mse = np.mean((target / 1.0 - ref / 1.0) ** 2)
    return 20*math.log10(255/math.sqrt(mse))

#使用现成的函数
def getSSIM(img1, img2):
    return compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
def getPSNR(img1, img2):
    return compare_psnr(img1, img2)
def getMSE(img1, img2):
    return compare_mse(img1, img2)


def getPSNR_new(img1, img2):
    return peak_signal_noise_ratio(img1, img2)
#这是均值结构相似度，不是多尺度ssim
def getSSIM_new(img1, img2):
    return structural_similarity(img1, img2,multichannel=True)# 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True

'''图像是ndarray,经过了归一化后的'''

#直接公式计算
# def CalcuPSNR(target, ref):
#     diff = ref - target
#     diff = diff.flatten('C')#按行展开成一维数组
#     rmse = math.sqrt(np.mean(diff**2.))
#     return 20 * math.log10(1.0 / (rmse))

def CalcuPSNR(target, ref,range = 1.0):
    diff = ref - target
    diff = diff.flatten('C')/range
    rmse = math.sqrt(np.mean(diff**2.))
    if rmse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / (rmse))

#加噪声
def make_salt_and_pepper_noise(img_path='test.png', noise_intensity=0.1):
    '''
    Parameters
    ----------
    img_path : str
        图像路径.
    noise_intensity : str
        椒盐噪声强度，0-1.

    Returns
    -------
    None.

    '''
    img = np.array(Image.open(img_path))
    noise = np.random.random(size=img.shape)
    img[noise < noise_intensity / 2] = 0
    img[1 - noise < noise_intensity / 2] = 255
    Image.fromarray(img).save('test-salt_and_pepper_noise-%.2f.png' % (noise_intensity))


'''图像是ndarray,带路径'''

def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    ssim_score = getSSIM_new(img1, img2)
    return ssim_score


def calc_psnr(img1_path, img2_path):

    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).

    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    psnr_score = getPSNR_new(img1, img2)
    return psnr_score

if __name__ == '__main__':
    for noise_intensity in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0]:
        make_salt_and_pepper_noise(noise_intensity=noise_intensity)

        print('SSIM', calc_ssim('test.png', 'test-salt_and_pepper_noise-%.2f.png' % (noise_intensity)), 'PSNR',
              calc_psnr('test.png', 'test-salt_and_pepper_noise-%.2f.png' % (noise_intensity)))




    im = Image.open('test.png').convert('RGB')
    im2 = Image.open('test-salt_and_pepper_noise-0.01.png').convert('RGB')
    # im = im.resize((512,512), Image.BICUBIC)
    # im2 = im2.resize((512,512), Image.BICUBIC)
    im_tensor = transforms.ToTensor()(im)
    im2_tensor = transforms.ToTensor()(im2)


    # 图像是np.array
    #直接计算
    psnr1 = CalcuPSNR255(np.array(im), np.array(im2))
    psnr2 = CalcuPSNR(np.array(im)/255.0, np.array(im2)/255.0)
    #现有函数计算
    psnr3 = getPSNR(np.array(im), np.array(im2))
    psnr4 = getPSNR_new(np.array(im), np.array(im2))

    ssim1 = getSSIM(np.array(im), np.array(im2))
    ssim2 = getSSIM_new(np.array(im), np.array(im2))

    #图像是tensor
    #psnr_tensor = get_TensorPSNR(im_tensor, im2_tensor)#get_TensorPSNR  输入是三通道四通道均可
    psnr_tensor = get_TensorPSNR(torch.unsqueeze(im_tensor,0), torch.unsqueeze(im2_tensor,0))
    # 输入必须是是四通道
    ssim_tensor = get_TensorSSIM(torch.unsqueeze(im_tensor,0), torch.unsqueeze(im2_tensor,0))
    #print("shape:", torch.unsqueeze(im_tensor,0).shape)

    msssim_tensor = get_TensorMSSSIM(torch.unsqueeze(im_tensor,0), torch.unsqueeze(im2_tensor,0))



    print("psnr1:", psnr1)
    print("psnr2:", psnr2)
    print("psnr3:", psnr3)
    print("psnr4:", psnr4)
    print("ssim1:", ssim1)
    print("ssim2:", ssim2)


    print("psnr_tensor:", psnr_tensor.cpu().detach().numpy())
    print("ssim_tensor:", ssim_tensor.cpu().detach().numpy())
    print("msssim_tensor:", msssim_tensor.cpu().detach().numpy())


    print(type(im)) #PIL.Image
    print(type(np.array(im))) # np.array
    print(type(im_tensor))  # torch.tensor

    #lpips
    img1_path ='./test_imgs/parrots.bmp'
    img2_path ='./test_imgs/parrots-salt_and_pepper_noise-0.01.bmp'

    lpips_model_vgg = util_of_lpips('vgg',False)
    lpips_model_alex = util_of_lpips('alex', False)

    lpips_dist_vgg = lpips_model_vgg.calc_lpips_value(img1_path,img2_path)
    lpips_dist_alex = lpips_model_alex.calc_lpips_value(img1_path,img2_path)

    lpips_dist_vgg = torch.squeeze(lpips_dist_vgg)
    lpips_dist_alex =torch.squeeze(lpips_dist_alex)

    print('vgg LPIPS score：',lpips_dist_vgg.cpu().detach().numpy())
    print('alex LPIPS score：',lpips_dist_alex.cpu().detach().numpy())

    #dists
    D = DISTS()
    im = Image.open('parrots.bmp').convert('RGB')
    im2 = Image.open('parrots_distorted.bmp').convert('RGB')

    im = im.resize((512, 512), Image.BICUBIC)
    im2 = im2.resize((512, 512), Image.BICUBIC)

    im_tensor = transforms.ToTensor()(im)
    im2_tensor = transforms.ToTensor()(im2)

    X = torch.unsqueeze(im_tensor, 0)
    Y = torch.unsqueeze(im2_tensor, 0)

    dists_value = D(X, Y)
    # set 'require_grad=True, batch_average=True' to get a scalar value as loss.
    dists_loss = D(X, Y, require_grad=True, batch_average=True)
    print(dists_loss.cpu().detach().numpy())
    # dists_loss.backward()