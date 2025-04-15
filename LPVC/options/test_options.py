from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options


        parser.add_argument('--filefolderlist', type=str,
                            default='./fashion/fashion256/fashion_test.txt',
                            help='test dataset index')
        parser.add_argument('--rootdir', type=str,
                            default='./fashion/fashion256/test',
                            help='test dataset sequences dir')

        opt, _ = parser.parse_known_args()
        if opt.dataset_name == 'fashion':
            parser.set_defaults(rootdir='./ori_data/fashion/test')
            parser.set_defaults(filefolderlist='./ori_data/fashion/fashion_val.txt')

        elif opt.dataset_name == 'taichi':
            parser.set_defaults(rootdir='./ori_data/taichi/test')
            parser.set_defaults(filefolderlist='./ori_data/taichi/taichi_val.txt')

        elif opt.dataset_name == 'ted':
            parser.set_defaults(rootdir='./ori_data/ted256/test')
            parser.set_defaults(filefolderlist='./ori_data/ted256/ted_val.txt')
        # elif opt.dataset_name == 'meeting':
        #     parser.set_defaults(rootdir='./video_meeting_dataset/test')
        #     parser.set_defaults(filefolderlist='./video_meeting_dataset/meeting_test'
        #                                        '.txt')

        else:
            parser.set_defaults(rootdir='./ori_data/vox/test')
            parser.set_defaults(filefolderlist='./ori_data/vox/vox_test.txt')

        parser.add_argument('--results_dir', type=str, default='./exp_data/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        # visualize
        parser.add_argument('--visualize', default=True,
                            help='whether to visualize pics of each compressing stages')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run,每个序列取多少帧，最好是GOP的整数倍')
        parser.set_defaults(model='test')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
