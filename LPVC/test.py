from options import test_options
from data import create_dataset
from models import create_model
from util.save_file import text_save
from util.util import *
import torch
import time
start_time = time.time()
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")


def Var(x):
    for k in x.keys():
        x[k] = Variable(x[k].cuda())
    return x

def totensor(x):
    return torch.tensor(x)

if __name__ == '__main__':
    
    torch.backends.cudnn.enabled = False

    opt = test_options.TestOptions().parse()  # get test options
    path = os.path.join(str(opt.results_dir), str(opt.mode))
    test_metrics_log = os.path.join(path, 'test_metrics_log.txt')

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # /results/256/ClassC/test_metrics_log.txt
    with open(test_metrics_log, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Test Metrics (%s) ================\n' % now)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()





    psnr_list = []
    msssim_list = []
    bpp_list = []
    lpips_list = []
    dists_list = []

    percentage_list = []

    ##################
    kp_bpp_list = []
    add_bg_bpp_list = []
    bg_bpp_list = []
    sum_kp_bpp = 0
    sum_add_bg_bpp = 0
    sum_bg_bpp = 0
    ###############



    sum_bpp = 0
    sum_psnr = 0
    sum_msssim = 0
    sum_dists = 0
    sum_lpips = 0

    lib_bg = None

    cnt = 0
    cntp = 0

    for i, data in enumerate(dataset):
        # if i % opt.GOP == 0:  # Gop
        #     print("testing : %d/%d" % (i, len(dataset)))
        print("testing : %d/%d" % (i+1, len(dataset)))

        frames = data['frames']
        frames_seg = data['frames_seg']

        frames_mask = data['frames_mask']
        frames_fg = data['frames_fg']
        frames_bg = data['frames_bg']
        #frames_seg_weights = data['frames_seg_weights']
        frames_mask_edges = data['frames_mask_edges']

        frames_lib_bg = data['frames_lib_bg']
        frames_add_bg = data['frames_add_bg']

        # #只有GOP-1个
        # frames_lib_bg = data['frames_lib_bg']
        # frames_add_bg = data['frames_add_bg']

        key_frame_bpp = data['key_frame_bpp']
        key_frame_psnr = data['key_frame_psnr']
        key_frame_msssim = data['key_frame_msssim']
        key_frame_lpips = data['key_frame_lpips']
        key_frame_dists = data['key_frame_dists']

        seqlen = len(frames)
        #print(len(frames))10

        sum_bpp += key_frame_bpp.data.cpu().numpy()
        sum_psnr += key_frame_psnr.data.cpu().numpy()
        sum_msssim += key_frame_msssim.data.cpu().numpy()
        sum_lpips += key_frame_lpips.data.cpu().numpy()
        sum_dists += key_frame_dists.data.cpu().numpy()

        bpp_list.append(key_frame_bpp.data.cpu().numpy())
        psnr_list.append(key_frame_psnr.data.cpu().numpy())
        msssim_list.append(key_frame_msssim.data.cpu().numpy())
        lpips_list.append(key_frame_lpips.data.cpu().numpy())
        dists_list.append(key_frame_dists.data.cpu().numpy())

        percentage_list.append(0)

        ##################
        kp_bpp_list.append(0)
        add_bg_bpp_list.append(0)
        bg_bpp_list.append(0)
        sum_kp_bpp += 0
        sum_add_bg_bpp += 0
        sum_bg_bpp += 0
        ###############################


        cnt += 1
        print_info('Frame' + str(i * opt.GOP + 1) + ' :' +
                   '   PSNR =' + str(psnr_list[i * opt.GOP]) +
                   '   BPP =' + str(bpp_list[i * opt.GOP]) +
                   '   MS-SSIM =' + str(msssim_list[i * opt.GOP]) +
                   '   LPIPS =' + str(lpips_list[i * opt.GOP]) +
                   '   DISTS =' + str(dists_list[i * opt.GOP])
                   )

        ref_seg = None
        ref_image = None
        ref_image_bg = None
        ref_image_fg = None
        #ref_image_seg = None
        ref_image_mask = None


        for j in range(seqlen): # 0...9

            #print("test.py : number of P input_images:" + str(j))

            p = i * opt.GOP + j + 1

            if j == 0:
                ref_image = frames[j]#[:, j, :, :, :]
                ref_seg = frames_seg[j]#[:, j, :, :, :]

                ref_image_mask = frames_mask[j]#[:, j, :, :, :]
                ref_image_bg = frames_bg[j]#[:, j, :, :, :]
                ref_image_fg = frames_fg[j]#[:, j, :, :, :]

                continue

            input = {}
            input['raw_image'] =raw_image = frames[j]#[:, j, :, :, :]
            input['raw_mask'] = raw_image_mask = frames_mask[j]#[:, j, :, :, :]
            input['raw_bg'] = raw_image_bg = frames_bg[j]#[:, j, :, :, :]
            input['raw_fg'] = raw_image_fg = frames_fg[j]#[:, j, :, :, :]
            #input['raw_seg_weights'] = raw_image_seg = frames_seg_weights[j]#[:, j, :, :, :]
            input['raw_mask_edge'] = frames_mask_edges[j]#[:, j, :, :, :]
            input['raw_seg'] = raw_seg = frames_seg[j]#[:, j, :, :, :]

            input['ref_image'] = ref_image
            input['ref_mask'] = ref_image_mask
            input['ref_bg'] = ref_image_bg
            input['ref_fg'] = ref_image_fg
            #input['ref_seg_weights'] = ref_image_seg
            input['ref_seg'] = ref_seg


            lib_bg = input['lib_bg'] = frames_lib_bg[j-1]#[:, j-1, :, :, :]
            add_bg = input['add_bg'] =frames_add_bg[j-1]#[:, j-1, :, :, :]

            input = Var(input)
            model.set_input(input)  # unpack data from data loader
            recon_image, psnr, ms_ssim, lpips, dists, bpp = model.test(p)  # run inference

            sum_bpp += bpp['total'].data.cpu().numpy()
            sum_psnr += psnr.cpu().data.numpy()
            sum_msssim += ms_ssim.cpu().data.numpy()
            sum_lpips += lpips.cpu().data.numpy()
            sum_dists += dists.cpu().data.numpy()

            bpp_list.append(bpp['total'].cpu().data.numpy())
            psnr_list.append(psnr.cpu().data.numpy())
            msssim_list.append(ms_ssim.cpu().data.numpy())
            lpips_list.append(lpips.cpu().data.numpy())
            dists_list.append(dists.cpu().data.numpy())


            ##################
            kp_bpp_list.append(bpp['kp'].cpu().data.numpy())
            add_bg_bpp_list.append(bpp['add_bg'].cpu().data.numpy())

            if opt.bg:
                bg_bpp_list.append(bpp['bg'].cpu().data.numpy())

            else:
                bg_bpp_list.append(bpp['bg'])

            sum_kp_bpp += bpp['kp'].data.cpu().numpy()
            sum_add_bg_bpp += bpp['add_bg'].data.cpu().numpy()
            if opt.bg:
                sum_bg_bpp += bpp['bg'].data.cpu().numpy()
            else:
                sum_bg_bpp += bpp['bg']
            ###############################
            #only p frame
            tensors = model.get_current_metrics()
            print_test_metrics(cnt, tensors, test_metrics_log)

            cnt += 1
            cntp += 1

            # ref_image = recon_image
            # ref_image_mask = raw_image_mask
            # ref_image_bg = raw_image_bg
            # ref_image_fg = raw_image_fg
            # ref_seg = raw_seg



            print_info('Frame' + str(p) + ' :'
                       + '   PSNR =' + str(psnr_list[p-1])
                       + '   BPP =' + str(bpp_list[p-1])
                       + '   MS-SSIM =' + str(msssim_list[p-1])
                       + '   LPIPS =' + str(lpips_list[p-1])
                       + '   DISTS =' + str(dists_list[p-1])

                       + '   kp_BPP =' + str(kp_bpp_list[p - 1])
                       + '   add_bg_BPP =' + str(add_bg_bpp_list[p - 1])
                       + '   bg_BPP =' + str(bg_bpp_list[p - 1])


                       )


    sum_bpp /= cnt
    sum_psnr /= cnt
    sum_msssim /= cnt
    sum_lpips /= cnt
    sum_dists /= cnt

    sum_kp_bpp /= cntp
    sum_add_bg_bpp /= cntp
    sum_bg_bpp /= cntp

    print_info("test.py : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf, "
               "average lpips : %.6lf, average dists : %.6lf, average kp_bpp : %.6lf, "
               "average add_bg_bpp : %.6lf, average bg_bpp : %.6lf\n"
               % (sum_bpp, sum_psnr, sum_msssim,sum_lpips,sum_dists,sum_kp_bpp,sum_add_bg_bpp,sum_bg_bpp))

    #conclude I/P frame metrics
    path_results = os.path.join(str(opt.results_dir), str(opt.mode))  # /results/256/ClassC/
    for name in ['PSNR','BPP','MS-SSIM','LPIPS','DISTS','kp_BPP','bg_BPP','add_bg_BPP','percentage']:
        filename = os.path.join(path_results + '/'+ name + '.txt')
        if name == 'PSNR':
            text_save(filename, psnr_list)
        if name == 'BPP':
            text_save(filename, bpp_list)
        if name == 'MS-SSIM':
            text_save(filename, msssim_list)
        if name == 'LPIPS':
            text_save(filename, lpips_list)
        if name == 'DISTS':
            text_save(filename, dists_list)
        ##########################
        if name == 'kp_BPP':
            text_save(filename, kp_bpp_list)
        if name == 'bg_BPP':
            text_save(filename, bg_bpp_list)
        if name == 'add_bg_BPP':
            text_save(filename, add_bg_bpp_list)

end_time = time.time()
time_difference = end_time - start_time
print(f"Elapsed time: {time_difference} seconds")

