import torch.utils.data as data
from util.util import *
import torchvision.transforms as transforms


class Train2FDataSet(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.image_input_list, self.image_ref_list,self.input_mask_list,self.ref_mask_list,self.input_seg_list,self.ref_seg_list = \
            self.get_frames(rootdir=self.opt.rootdir, filefolderlist=self.opt.filefolderlist)

        self.mode = self.opt.mode
        self.im_height = self.opt.crop_height
        self.im_width = self.opt.crop_width

        self.size = (self.im_width, self.im_height)

        self.cu_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.CenterCrop(256),

        ])
        print_info(" dataset images: " + str(len(self.data)))
        print_info(" image_input_list: " + str(len(self.image_input_list)))


    def get_frames(self, rootdir="", filefolderlist=""):
        with open(filefolderlist) as f:
            self.data = f.readlines()
        fns_train_input = []
        fns_train_ref = []

        fns_input_mask = []
        fns_ref_mask = []

        fns_input_seg = []
        fns_ref_seg = []

        for n, line in enumerate(self.data, 1):
            if int(line.rstrip()[-11:-4]) == 1 or int(line.rstrip()[-11:-4]) == 0:
                continue
            y = os.path.join(rootdir, line.rstrip())

            fns_train_input += [y]
            refnumber = int(y[-11:-4]) - 1
            refname = y[0:-11] + str(refnumber).zfill(7) + '.png'
            fns_train_ref += [refname]
            input_mask = os.path.join(rootdir+'_mask', line.rstrip())
            fns_input_mask += [input_mask]
            ref_mask = input_mask[0:-11] + str(refnumber).zfill(7) + '.png'
            fns_ref_mask += [ref_mask]
            input_seg = os.path.join(rootdir+'_seg', line.rstrip())
            fns_input_seg += [input_seg]
            ref_seg = input_seg[0:-11] + str(refnumber).zfill(7) + '.png'
            fns_ref_seg += [ref_seg]


            #print(input_mask)

        return fns_train_input, fns_train_ref,fns_input_mask,fns_ref_mask,fns_input_seg,fns_ref_seg

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):

        #print('index = ',index)
        input_image = Image.open(self.image_input_list[index])
        ref_image = Image.open(self.image_ref_list[index])

        input_mask = Image.open(self.input_mask_list[index])
        ref_mask = Image.open(self.ref_mask_list[index])
        input_mask = Image.fromarray(get_binary_pic(input_mask)[:np.newaxis])
        ref_mask = Image.fromarray(get_binary_pic(ref_mask)[:np.newaxis])

        input_seg = Image.open(self.input_seg_list[index])
        ref_seg = Image.open(self.ref_seg_list[index])

        size = input_image.size

        if size != self.size:
            input_image = input_image.resize(self.size, Image.ANTIALIAS)
            ref_image = ref_image.resize(self.size, Image.ANTIALIAS)

            input_mask = input_mask.resize(self.size, Image.ANTIALIAS)
            ref_mask = ref_mask.resize(self.size, Image.ANTIALIAS)

            input_seg = input_seg.resize(self.size, Image.ANTIALIAS)
            ref_seg = ref_seg.resize(self.size, Image.ANTIALIAS)


        input_fg = get_fg(input_image, input_mask,True)
        input_bg = get_bg(input_image,input_mask,True)

        ref_fg = get_fg(ref_image,ref_mask,True)
        ref_bg = get_bg(ref_image, ref_mask,True)

        bg = get_new_bg(curr_bg=ref_bg, raw=input_image, raw_mask=input_mask, last_mask=ref_mask, white=True)
        lib_bg = bg['new_bg']
        add_bg = bg['add_bg']

        input_mask_edge = get_mask_edge(input_mask)


        input_image = self.cu_transform(input_image)
        ref_image = self.cu_transform(ref_image)

        input_mask = self.cu_transform(input_mask)
        ref_mask = self.cu_transform(ref_mask)
        input_seg = self.cu_transform(input_seg)
        ref_seg = self.cu_transform(ref_seg)

        input_bg = self.cu_transform(input_bg)
        ref_bg = self.cu_transform(ref_bg)

        lib_bg = self.cu_transform(lib_bg)
        add_bg = self.cu_transform(add_bg)

        input_fg = self.cu_transform(input_fg)
        ref_fg = self.cu_transform(ref_fg)
        # input_seg_weights = self.cu_transform(input_seg_weights)
        # ref_seg_weights = self.cu_transform(ref_seg_weights)

        input_mask_edge = self.cu_transform(input_mask_edge)

        output = {}

        output['raw_image'] = input_image
        output['ref_image'] = ref_image

        output['raw_mask'] = input_mask
        output['ref_mask'] = ref_mask

        output['raw_seg'] = input_seg
        output['ref_seg'] = ref_seg

        output['raw_bg'] = input_bg
        output['ref_bg'] = ref_bg

        output['lib_bg'] = lib_bg

        output['add_bg'] = add_bg

        output['raw_fg'] = input_fg
        output['ref_fg'] = ref_fg

        output['raw_mask_edge'] = input_mask_edge

        return output





