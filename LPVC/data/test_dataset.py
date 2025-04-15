import torch.utils.data as data
from util.util import *
import torchvision.transforms as transforms
from util.metrics import get_TensorMSSSIM,get_TensorPSNR,util_of_lpips,util_of_dists


class TestDataSet(data.Dataset):
    def __init__(self, opt):

        self.opt = opt
        self.mode = self.opt.mode
        self.GOP = self.opt.GOP
        self.im_height = self.opt.crop_height
        self.im_width = self.opt.crop_width
        self.size = (self.im_width, self.im_height)

        self.key_frames_ref = []
        self.key_frames_raw = []
        self.input_img = []
        self.input_mask = []
        self.input_seg = []

        self.key_frames_bpp= []
        self.key_frames_psnr= []
        self.key_frames_msssim= []
        self.key_frames_lpips= []
        self.key_frames_dists= []

        self.lpips = util_of_lpips(net='alex', use_gpu=False)
        self.dists = util_of_dists(use_gpu=False)

        self.cu_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.len_dataset = self.get_frames(rootdir=self.opt.rootdir,
                           filefolderlist=self.opt.filefolderlist)

        print_info(" dataset images: " + str(self.len_dataset))
        print_info(" dataset GOPs: " + str(len(self.key_frames_raw)))


    def get_frames(self, rootdir="", filefolderlist=""):

        with open(filefolderlist) as f:
            self.data = f.readlines()
        path_results = os.path.join(str(self.opt.results_dir), str(self.opt.mode), 'keyframes')
        path_bin = os.path.join(path_results, 'bin')
        path_raw = os.path.join(path_results, 'raw')
        path_rec = os.path.join(path_results, 'recon_image')
        if not os.path.exists(path_bin):
            os.makedirs(path_bin, exist_ok=True)
        if not os.path.exists(path_raw):
            os.makedirs(path_raw, exist_ok=True)
        if not os.path.exists(path_rec):
            os.makedirs(path_rec, exist_ok=True)
        #########################################################
        for n, line in enumerate(self.data):
            y = os.path.join(rootdir, line.rstrip())

            if n % self.GOP == 0:
                f = n

                key_frame = Image.open(y)
                size = key_frame.size
                h = size[0] // 64 * 64
                w = size[1] // 64 * 64
                new_size = (h, w)
                #print(new_size)
                if size != self.size:
                    key_frame = key_frame.resize(new_size, Image.ANTIALIAS)
                key_frame_raw_path = path_raw + '/' + str(f + 1).zfill(7) + '.png'
                key_frame_rec_path = path_rec + '/' + str(f + 1).zfill(7) + '.png'

                key_frame.save(key_frame_raw_path)

                self.key_frames_ref.append(key_frame_rec_path)
                self.key_frames_raw.append(key_frame_raw_path)

                bits = os.path.getsize(path_bin + '/' + str(f + 1).zfill(7) + '.bin')
                bits = bits * 8
                bpp = bits / h / w * 1.0
                self.key_frames_bpp.append(bpp)

                #self.key_frames.append(y)

                img_path = []
                mask_path = []
                seg_path = []

                for p_frame in range(min(self.GOP,len(self.data)-f)):

                    img = y[0:-11] + str(f + 1).zfill(7) + '.png'

                    input_m = os.path.join(rootdir + '_mask', line.rstrip())
                    mask = input_m[0:-11] + str(f + 1).zfill(7) + '.png'

                    input_s = os.path.join(rootdir + '_seg', line.rstrip())
                    seg = input_s[0:-11] + str(f + 1).zfill(7) + '.png'

                    img_path.append(img)
                    mask_path.append(mask)
                    seg_path.append(seg)
                    f = f + 1

                self.input_img.append(img_path)
                self.input_mask.append(mask_path)
                self.input_seg.append(seg_path)

        return len(self.data)

    def __len__(self):
        return len(self.input_img)

    def __getitem__(self, index):

        output = {}

        frames = []
        frames_seg = []
        frames_fg = []
        frames_bg = []
        frames_mask = []
        frames_seg_weights = []
        frames_mask_edges = []

        frames_lib_bg = []
        frames_add_bg = []

        key_frame_psnr = None
        key_frame_msssim = None
        key_frame_lpips = None
        key_frame_dists = None

        raw_key_frame = None
        curr_bg = None
        last_mask = None

        for i in range(len(self.input_img[index])):

            if i == 0:
                frame = Image.open(self.key_frames_ref[index])
                raw_key_frame = Image.open(self.key_frames_raw[index])
            else:
                frame = Image.open(self.input_img[index][i])
            frame_mask = Image.open(self.input_mask[index][i])
            frame_mask = Image.fromarray(get_binary_pic(frame_mask)[:np.newaxis])#[h,w,3]→[h,w]→[h,w,1]
            frame_seg = Image.open(self.input_seg[index][i])

            size = frame.size
            h = size[0] // 64 * 64
            w = size[1] // 64 * 64
            new_size = (h,w)

            if size != self.size:
                frame = frame.resize(new_size, Image.ANTIALIAS)
                frame_mask = frame_mask.resize(new_size, Image.ANTIALIAS)
                frame_seg = frame_seg.resize(new_size, Image.ANTIALIAS)
            frame_fg = get_fg(frame, frame_mask)
            frame_bg = get_bg(frame, frame_mask)
            #frame_seg_weight = get_seg_weights(self.mode, frame_seg,frame)
            frame_mask_edge = get_mask_edge(frame_mask)
            if i == 0:
                ref_key_frame = self.cu_transform(frame)
                raw_key_frame = self.cu_transform(raw_key_frame)

                key_frame_psnr = get_TensorPSNR(raw_key_frame, ref_key_frame).data.cpu().numpy()
                key_frame_msssim = get_TensorMSSSIM(torch.unsqueeze(raw_key_frame, 0),
                                                    torch.unsqueeze(ref_key_frame, 0)).data.cpu().numpy()

                key_frame_lpips = torch.mean(self.lpips.calc_lpips_value(raw_key_frame, ref_key_frame)).detach().data.cpu().numpy()
                key_frame_dists = torch.mean(self.dists.calc_dists_value(torch.unsqueeze(raw_key_frame, 0),
                                                              torch.unsqueeze(ref_key_frame, 0))).detach().data.cpu().numpy()

                curr_bg = frame_bg
                last_mask = frame_mask
            else:
                bg = get_new_bg(curr_bg=curr_bg, raw=frame, raw_mask=frame_mask, last_mask=last_mask, white=True)
                lib_bg = bg['new_bg']
                add_bg = bg['add_bg']


                curr_bg = lib_bg
                last_mask = frame_mask

                lib_bg = self.cu_transform(lib_bg)
                add_bg = self.cu_transform(add_bg)
                frames_lib_bg.append(lib_bg)
                frames_add_bg.append(add_bg)

                ###################


            frame = self.cu_transform(frame)
            frame_mask = self.cu_transform(frame_mask)
            frame_fg = self.cu_transform(frame_fg)
            frame_bg = self.cu_transform(frame_bg)
            #frame_seg_weight = self.cu_transform(frame_seg_weight)
            frame_mask_edge = self.cu_transform(frame_mask_edge)
            frame_seg = self.cu_transform(frame_seg)

            frames.append(frame)
            frames_seg.append(frame_seg)
            frames_mask.append(frame_mask)
            frames_fg.append(frame_fg)
            frames_bg.append(frame_bg)
            #frames_seg_weights.append(frame_seg_weight)
            frames_mask_edges.append(frame_mask_edge)


        output['frames'] = frames
        output['frames_mask'] = frames_mask
        output['frames_fg'] = frames_fg
        output['frames_bg'] = frames_bg
        #output['frames_seg_weights'] = frames_seg_weights
        output['frames_mask_edges'] = frames_mask_edges

        output['frames_lib_bg'] = frames_lib_bg
        output['frames_add_bg'] = frames_add_bg

        output['frames_seg'] = frames_seg

        output['key_frame_bpp'] = self.key_frames_bpp[index]
        output['key_frame_psnr'] = key_frame_psnr
        output['key_frame_msssim'] = key_frame_msssim
        output['key_frame_lpips'] = key_frame_lpips
        output['key_frame_dists'] = key_frame_dists


        return output




