import argparse
import os
from util import util
import torch


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""



        parser.add_argument("--mode", type=str, default='full',
                            choices=['head', 'half', 'full'],
                            help='mode name represents different scenarios')

        parser.add_argument("--dataset_name", type=str, default='fashion',
                            choices=['fashion', 'vox', 'taichi', 'ted'],
                            help='dataset name represents different scenarios')

        # set different mode for different dataset

        opt, _ = parser.parse_known_args()
        if opt.dataset_name == 'fashion' or opt.dataset_name == 'taichi':
            parser.set_defaults(mode='full')
        elif opt.dataset_name == 'ted':
            parser.set_defaults(mode='half')
        else:
            parser.set_defaults(mode='head')

        parser.add_argument("--bg", action='store_true', help='wether use bg')

        parser.add_argument('--name', type=str, default='TPS-VideoCompression',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # model parameters
        parser.add_argument('--input_nc', type=int, default=6,
                            help='# of input image channels: 4 for RGB concat grayscale,6 for 2 RGBs')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')

        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='batch',
                            help='instance normalization or batch normalization [instance | batch | none | GDN]')

        parser.add_argument('--use_LSTM', default=False, help='use LSTMcell')
        parser.add_argument('--use_BlurPool', default=False, help='use Down BlurPool')

        parser.add_argument('--use_EDe', default=False, help='whether to use the edge encoder/decoder')

        parser.add_argument('--input_channel_E', type=int, default=3,
                            help='#input channel of encoder')
        parser.add_argument('--out_channel_De', type=int, default=3,
                            help='#output channel of decoder')

        parser.add_argument('--out_channel_N', type=int, default=64, help='# of encoder filters in first conv layer')
        parser.add_argument('--out_channel_M', type=int, default=96, help='# output channels of encoder')

        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')

        parser.add_argument('--dropout_epoch', type=int, default=5, help=' use dropout for the generator')
        parser.add_argument('--dropout_maxp', type=float, default=0.7, help='weight for |B-G|,defaut 10,,40')
        parser.add_argument('--dropout_startp', type=float, default=0.0, help='weight for |B-G|,defaut 10,,40')
        parser.add_argument('--dropout_inc_epoch', type=int, default=10, help='weight for |B-G|,defaut 10,,40')

        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')


        # lambda parameters
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G|,defaut 10,,40')


        parser.add_argument('--lambda_equivariance', type=float, default=10,
                            help='Weights for value equivariance')

        parser.add_argument('--lambda_Lf', type=float, default=10.0)
        parser.add_argument('--lambda_Lp', type=float, default=10.0)

        parser.add_argument("--lambda_warp", type=float, default=10,
                            help='Weights for warp loss')

        parser.add_argument('--lambda_GAN', type=float, default=1,
                            help='weight on G(A), default 1,,2.0')

        # parser.add_argument('--lambda_feature_loss', type=float, default=0.3, help='weight for feature loss')
        # parser.add_argument('--lambda_dists_loss', type=float, default=0.3, help='weight for feature loss')
        # parser.add_argument('--lambda_lpips_loss', type=float, default=0.3, help='weight for feature loss')

        # models
        parser.add_argument("--num_tps", type=int, default=10,
                            help=' Number of TPS transformation')
        parser.add_argument('--scales', type=list, default=[1, 0.5, 0.25, 0.125],
                            help='Scales for perceptual pyramide loss. If scales = [1, 0.5, 0.25, 0.125] and image '
                                 'resolution is 256x256, '
                                 'than the loss will be computer on resolutions 256x256, 128x128, 64x64, 32x32.')

        # parser.add_argument('--num_Ds', type=int, default=1, help='number of Discrminators')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='dcgan|lsgan|wgan-gp|hinge')
        parser.add_argument('--netD', type=str, default='basic_256_multi',
                            help='selects model to use for netD, basic_256_multi for normal, basic_256_multi_class '
                                 'for adding class label')
        # parser.add_argument('--netD2', type=str, default='basic_256_multi', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='unet_256', help='selects model to use for netG')
        parser.add_argument('--netE', type=str, default='analysis_256', help='selects model to use for netE')
        parser.add_argument('--netDe', type=str, default='synthesis_256', help='selects model to use for netDe')
        parser.add_argument('--netF', type=str, default='resnet_256', help='selects model to use for netF')
        parser.add_argument('--conditional_D', default=False, help='if use conditional GAN for D')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        # parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        # parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='whether use TTUR training scheme')

        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        # if opt.gray_edge:
        #     parser.set_defaults(input_channel_E=1,out_channel_De=1,input_nc=4)

        parser.add_argument('--crop_height', type=int, default=256, help=' crop to this size when training')
        parser.add_argument('--crop_width', type=int, default=256, help=' crop to this size when training')


        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model，set to best to use '
                                 'best cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_['
                                 'load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
