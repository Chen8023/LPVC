"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
from termcolor import colored
import math
import cv2
from pylab import *

# import random
# import ipdb
MSB2Num = [0,256,512,768] # for 10bit yuv, the Higher 2 bit could only be 00,01,10,11. The pixel_value = LSB_value + MSB2Num[MSB_value]
## randomly extract few frames from a sequence
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def get_binary_pic(ima):
    '''

    :param ima: PIL
    :return: 0-255
    '''

    ima = ima.convert('L')  # 转化为灰度图像,这一步是将原mask的3维降到1维
    ima = np.array(ima)  # 转化为二维数组
    for i in range(ima.shape[0]):  # 转化为二值矩阵
        for j in range(ima.shape[1]):
            if ima[i, j] >= 128:  #
                ima[i, j] = 255  #二值化
            else:
                ima[i, j] = 0
    return ima #[h,w]

def get_bg(ima,mask,white=True):
    '''
    :param ima: Image
    :param mask: Image
    :return: fg：Image
    '''

    ima = np.array(ima)
    mask = np.array(mask)
    ima_out = np.zeros_like(ima)
    mask_not = cv2.bitwise_not(mask)
    ima = cv2.bitwise_and(ima, ima, mask=mask_not)
    if white:
        #背景中人像部分用白色代替
        for i in range(3):
            ima_out[:, :, i] = ima[:, :, i] + mask
    else:
        ima_out = ima

    return Image.fromarray(ima_out)

def get_fg(ima,mask,white=True):
    '''
    :param ima: Image
    :param mask: Image
    :return: fg：Image
    '''
    ima = np.array(ima)
    mask = np.array(mask)

    ima_out = np.zeros_like(ima)

    mask_not = cv2.bitwise_not(mask)
    ima = cv2.bitwise_and(ima, ima, mask=mask)
    if white:
        for i in range(3):
            ima_out[:, :, i] = ima[:, :, i] + mask_not

    else:
        ima_out = ima

    return Image.fromarray(ima_out)




def get_binary_pic_3channel(ima):
    ima = np.array(ima)

    _,ima = cv2.threshold(ima, 128, 255, cv2.THRESH_BINARY)
    return ima

def wise(image):
    image = Image.fromarray(image)
    pixdata = image.load()
    #print(image.size)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if all(pixdata[x, y][i] < 2 for i in range(3)):
                pixdata[x, y] = 255, 255, 255
    return np.array(image)



def get_new_bg(curr_bg,raw,raw_mask,last_mask,white=True):

    raw_mask_1 = np.array(raw_mask)
    last_mask_1 = np.array(last_mask)
    raw = np.array(raw)
    curr_bg = np.array(curr_bg)

    raw_mask_3 = np.zeros_like(raw)
    last_mask_3 = np.zeros_like(raw)

    for i in range(3):
        raw_mask_3[:, :, i] = raw_mask_1
        last_mask_3[:, :, i] = last_mask_1

    #h,w = raw_mask_binary.shape[0],raw_mask_binary.shape[1]
    raw_mask_binary = get_binary_pic_3channel(raw_mask_3)
    last_mask_binary = get_binary_pic_3channel(last_mask_3)

    raw_mask_binary_not = cv2.bitwise_not(raw_mask_binary)
    variation_mask = cv2.bitwise_and(last_mask_binary, raw_mask_binary_not)
    variation_mask = cv2.cvtColor(variation_mask, cv2.COLOR_RGB2GRAY)
    # print(variation_mask.shape)  # (256, 256)
    variation_img = cv2.bitwise_and(raw, raw, mask=variation_mask)
    #h, w, c = variation_img.shape[0], variation_img.shape[1],variation_img.shape[2]
    add_bg = np.zeros_like(variation_img)
    #print("weight : %s, height : %s, channel : %s" % (h, w,c))
    h, w = variation_mask.shape[0], variation_mask.shape[1]
    for row in range(h):  #
        for col in range(w):

            condition_var = (variation_mask[row, col] == 255)


            if white:
                condition_cur = (curr_bg[row, col, 0] == 255) \
                                and (curr_bg[row, col, 1] == 255) \
                                and (curr_bg[row, col, 2] == 255)
            else:
                condition_cur = (curr_bg[row, col, 0] == 0) \
                                and (curr_bg[row, col, 1] == 0) \
                                and (curr_bg[row, col, 2] == 0)
            if condition_var and condition_cur:
                add_bg[row, col] = variation_img[row, col]
                #print(add_bg[row, col] )


    output = {}
    output['new_bg'] = Image.fromarray(uint8(curr_bg + add_bg))
    output['add_bg'] = Image.fromarray(uint8(add_bg))

    return output


def get_seg_weights(mode, seg, img):
    '''
    :param mode: 'head', 'half', 'full'
    :param seg: Image
    :return: Image
    '''
    img = np.array(img)
    x, y = seg.size  # (width, height)
    # print("w={},h={}".format(x, y))
    img_array = seg.load()

    head = Image.new("RGB", (x, y), "white")
    head_array = head.load()
    torso = Image.new("RGB", (x, y), "white")
    torso_array = torso.load()
    upperarms = Image.new("RGB", (x, y), "white")
    upperarms_array = upperarms.load()
    lowerarms = Image.new("RGB", (x, y), "white")
    lowerarms_array = lowerarms.load()
    upperlegs = Image.new("RGB", (x, y), "white")
    upperlegs_array = upperlegs.load()
    lowerlegs = Image.new("RGB", (x, y), "white")
    lowerlegs_array = lowerlegs.load()

    for i in range(x):
        for j in range(y):

            if img_array[i, j] == 1:
                head_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 2:
                torso_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 3:
                upperarms_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 4:
                lowerarms_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 5:
                upperlegs_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 6:
                lowerlegs_array[i, j] = img_array[i, j]
            elif img_array[i, j] == 0:
                continue
            else:
                raise NotImplementedError(
                    'Invalid semantic markers :[%s]' % img_array[i, j])

    head = head.convert('L')
    torso = torso.convert('L')
    upper_arms = upperarms.convert('L')
    lower_arms = lowerarms.convert('L')
    upper_legs = upperlegs.convert('L')
    lower_legs = lowerlegs.convert('L')

    headArray = np.array(head)
    torsoArray = np.array(torso)
    upper_armsArray = np.array(upper_arms)
    lower_armsArray = np.array(lower_arms)
    upper_legsArray = np.array(upper_legs)
    lower_legsArray = np.array(lower_legs)

    #
    _, headArray = cv2.threshold(headArray, 128, 255, cv2.THRESH_BINARY)
    _, torsoArray = cv2.threshold(torsoArray, 128, 255, cv2.THRESH_BINARY)
    _, upper_armsArray = cv2.threshold(upper_armsArray, 128, 255, cv2.THRESH_BINARY)
    _, lower_armsArray = cv2.threshold(lower_armsArray, 128, 255, cv2.THRESH_BINARY)
    _, upper_legsArray = cv2.threshold(upper_legsArray, 128, 255, cv2.THRESH_BINARY)
    _, lower_legsArray = cv2.threshold(lower_legsArray, 128, 255, cv2.THRESH_BINARY)

    #
    head_seg_binary = cv2.bitwise_not(headArray)
    torso_seg_binary = cv2.bitwise_not(torsoArray)
    upper_arms_seg_binary = cv2.bitwise_not(upper_armsArray)
    lower_arms_seg_binary = cv2.bitwise_not(lower_armsArray)
    upper_legs_seg_binary = cv2.bitwise_not(upper_legsArray)
    lower_legs_seg_binary = cv2.bitwise_not(lower_legsArray)

    # output
    seg = {}
    seg['head'] = head_seg_binary
    seg['torso'] = torso_seg_binary
    seg['upper_arms'] = upper_arms_seg_binary
    seg['lower_arms'] = lower_arms_seg_binary
    seg['upper_legs'] = upper_legs_seg_binary
    seg['lower_legs'] = lower_legs_seg_binary

    if mode == 'head':
        mask = seg['head']
        seg_img = cv2.bitwise_and(img, img, mask=mask)

    elif mode == 'half':
        mask = seg['head'] + seg['upper_arms'] + seg['lower_arms']
        seg_img = cv2.bitwise_and(img, img, mask=mask)

    else:
        mask = seg['upper_arms'] + seg['lower_arms'] + seg['upper_legs'] + seg['lower_legs']
        seg_img = cv2.bitwise_and(img, img, mask=mask)

    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    # input_seg = {}
    # input_seg['head'] = seg['head']
    # input_seg['torso'] = seg['torso']
    # input_seg['limbs'] = seg['upper_arms'] + seg['lower_arms'] + seg['upper_legs'] + seg['lower_legs']
    #
    # for k in input_seg.keys():
    #     input_seg[k] = Image.fromarray(input_seg[k])

    # for k in input_seg.keys():
    #     input_seg.update({k: Image.fromarray(input_seg[k])})    # frame_seg_weight = self.cu_transform(frame_seg_weight)
    head = seg['head']
    torso = seg['torso']
    limbs = seg['upper_arms'] + seg['lower_arms'] + seg['upper_legs'] + seg['lower_legs']
    return Image.fromarray(mask)#Image.fromarray(head),Image.fromarray(torso),Image.fromarray(limbs)#input_seg#Image.fromarray(mask) #Image.fromarray(seg_img)





#___________________test class_______________________________
#_______________flow ______________________________
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_image(flow, display=False, maxrad = None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # maxu = max(maxu, np.max(u))
    # minu = min(minu, np.min(u))
    #
    # maxv = max(maxv, np.max(v))
    # minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


# _________ text visualization ____________
def get_local_time():
    return time.strftime("%d %b %Y %Hh%Mm%Ss", time.localtime())

def print_info(info_string, quite=False):

    info = '[{0}][INFO]{1}'.format(get_local_time(), info_string)
    print(colored(info, 'green'))

def print_error(error_string):

    error = '[{0}][ERROR] {1}'.format(get_local_time(), error_string)
    print (colored(error, 'red'))

def print_warning(warning_string):

    warning = '[{0}][WARNING] {1}'.format(get_local_time(), warning_string)

    print (colored(warning, 'blue'))
# ___________ End text visualization


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def yuv_import_444(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    # fp=open(filename,'rb')

    blk_size = int(dims[0] * dims[1] * 3)
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    # print(dims[0])
    # print(dims[1])
    d00 = dims[0]
    d01 = dims[1]
    # print(d00)
    # print(d01)
    Yt = np.zeros((dims[0], dims[1]), np.int, 'C')
    Ut = np.zeros((d00, d01), np.int, 'C')
    Vt = np.zeros((d00, d01), np.int, 'C')
    print(dims[0])
    YUV = np.zeros((dims[0], dims[1], 3))

    for m in range(dims[0]):
        for n in range(dims[1]):
            Yt[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Ut[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Vt[m, n] = ord(fp.read(1))

    YUV[:, :, 0] = Yt
    YUV[:, :, 1] = Ut
    YUV[:, :, 2] = Vt
    fp.close()
    return YUV

def yuv420_import(filename,height,width,Org_frm_list,extract_frm,flag,isRef,isList0,deltaPOC,isLR):
    # print filename, height, width
    if flag:
        frm_size = int(float(height*width*3)/float(2*2)) ## for 10bit yuv, each pixel occupy 2 byte
    else:
        frm_size = int(float(height*width*3)/2)
    if isLR:
        frm_size=frm_size*4
        if flag:
            row_size = width * 2
        else:
            row_size = width
    Luma = []
    U=[]
    V=[]
    # Org_frm_list = range(1,numfrm)
    # random.shuffle(Org_frm_list)
    with open(filename,'rb') as fd:
        for extract_index in range(extract_frm):
            if isRef:
                if isList0:
                    current_frm = Org_frm_list[extract_index]-deltaPOC
                else:
                    current_frm = Org_frm_list[extract_index]+deltaPOC
            else:
                current_frm = Org_frm_list[extract_index]
            fd.seek(frm_size*current_frm,0)
            # ipdb.set_trace()
            if flag:
                Yt = np.zeros((height,width),np.uint16,'C')
                for m in range(height):
                    for n in range(width):
                        symbol = fd.read(2)
                        LSB = ord(symbol[0])
                        MSB = ord(symbol[1])
                        Pixel_Value = LSB+MSB2Num[MSB]
                        Yt[m,n]=Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                Luma.append(Yt)
                del Yt
            else:
                Yt = np.zeros((height,width),np.uint8,'C')
                for m in range(height):
                    for n in range(width):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Yt[m,n]=Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                Luma.append(Yt)
                del Yt
                Ut = np.zeros((height//2, width//2), np.uint8, 'C')
                for m in range(height//2):
                    for n in range(width//2):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Ut[m, n] = Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                U.append(Ut)
                del Ut
                Vt = np.zeros((height // 2, width // 2), np.uint8, 'C')
                for m in range(height // 2):
                    for n in range(width // 2):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Vt[m, n] = Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                V.append(Vt)
                del Vt
    return Luma,U,V

def YUV2RGB420_custom(Y_frames,U_frames,V_frames):
    shape = np.shape(Y_frames)
    n = shape[0]
    h = shape[1]
    w = shape[2]
    RGB_frames = np.zeros([n, h, w, 3], np.uint8)
    yuv_frame=np.zeros([h*3//2,w],np.uint8)

    for n_i in range(n):
        for f_i in range(1):
            Y = Y_frames[n_i, :, :, 0]
            U = U_frames[n_i, :, :, 0]
            V = V_frames[n_i, :, :, 0]
            yuv_frame[:h,:]=Y
            yuv_frame[h:5 * h // 4, :] = np.reshape(U, [h // 4, w])
            yuv_frame[5 * h // 4 :, :] = np.reshape(V, [h // 4, w])
            rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
            RGB_frames[n_i,:,:,:]=rgb
    return RGB_frames



def de_normalise(batch):
    # de normalise for regular normalisation
    batch = (batch + 1.0) / 2.0 * 255.0
    return batch


def normalise_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= mean
    batch = batch / std
    return batch

def print_test_metrics(i, tensors,test_metrics_log):

    message = '(iters: %d)' % (i)
    for k, v in tensors.items():
        message += '%s: %.3f ' % (k, v)
    with open(test_metrics_log, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message



def get_variation_encode(raw_mask_binary,last_mask_binary):


    h,w = raw_mask_binary.shape[0],raw_mask_binary.shape[1]
    variation =  np.zeros(shape=(h,w))#np.bitwise_xor(raw_mask_binary, last_mask_binary) * 255

    for i in range(h):
        for j in range(w):
            #print(raw_mask_binary[i, j], last_mask_binary[i, j])
            variation[i, j] = np.bitwise_xor(raw_mask_binary[i, j], last_mask_binary[i, j]) / 255.0
            #print(variation[i, j])
    return variation



def get_variation_decode(variation,image_mask):

    for i in range(image_mask.shape[0]):
        for j in range(image_mask.shape[1]):
            if variation[i, j] == 1.0:
                if image_mask[i, j] == 0.0:
                    image_mask[i, j] = 1.0
                else:
                    image_mask[i, j] = 0.0
    return image_mask


