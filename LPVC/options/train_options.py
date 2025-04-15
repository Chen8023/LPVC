from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #Compresser parameters
        parser.add_argument('--filefolderlist', type=str,
                            default='./fashion/fashion256/fashion_train.txt',
                            help='train dataset index')
        parser.add_argument('--rootdir', type=str,
                            default='./fashion/fashion256/train',
                            help='train dataset sequences dir')

        opt, _ = parser.parse_known_args()
        if opt.dataset_name == 'fashion':
            parser.set_defaults(rootdir='./fashion/fashion256/train')
            parser.set_defaults(
                filefolderlist='./fashion/fashion256/fashion_train.txt')

        elif opt.dataset_name == 'taichi':
            parser.set_defaults(rootdir='./taichi/taichi/train')
            parser.set_defaults(filefolderlist='./taichi/taichi/taichi_train.txt')

        elif opt.dataset_name == 'ted':
            parser.set_defaults(rootdir='./TED384-v2/TED384-v2/train')
            parser.set_defaults(filefolderlist='./TED384-v2/TED384-v2/ted_train.txt')

        # elif opt.dataset_name == 'meeting':
        #     parser.set_defaults(rootdir='./video_meeting_dataset/train')
        #     parser.set_defaults(filefolderlist='./video_meeting_dataset/meeting_train.txt')

        else:
            parser.set_defaults(rootdir='./vox/vox/train')
            parser.set_defaults(filefolderlist='./vox/vox/vox_train.txt')



        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

        parser.set_defaults(pool_size=0, gan_mode='vanilla')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=2, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', default=False, help='whether saves model by iteration')
        # parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        parser.add_argument('--niter', type=int, default=60, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=60,
                            help='# of iter to linearly decay learning rate to zero')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam,0.0002 by default')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
