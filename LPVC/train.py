
from options import train_options
from options import test_options

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import *
import warnings
from pytorch_msssim import ssim
from torch.autograd import Variable

warnings.filterwarnings("ignore")


def Var(x):
    for k in x.keys():
        x[k] = Variable(x[k].cuda())
    return x


def val(model, val_dataset):
    model.eval()
    cur_score = 0.0
    sum_bpp = 0.0
    sum_psnr = 0.0
    sum_msssim = 0.0
    sum_dists = 0.0
    sum_lpips = 0.0

    cnt = 0
    with torch.no_grad():

        for i, data in enumerate(val_dataset):
            print("val : %d/%d" % (i+1, len(val_dataset)))

            frames = data['frames']
            frames_seg = data['frames_seg']

            frames_mask = data['frames_mask']
            frames_fg = data['frames_fg']
            frames_bg = data['frames_bg']
            frames_mask_edges = data['frames_mask_edges']

            seqlen = len(frames)


            ref_seg = None

            ref_image = None
            ref_image_bg = None
            ref_image_fg = None
            ref_image_seg = None
            ref_image_mask = None
            for j in range(seqlen):

                p = i * opt.GOP + j + 1

                if j == 0:
                    ref_image = frames[j]  # [:, j, :, :, :]
                    ref_seg = frames_seg[j]  # [:, j, :, :, :]

                    ref_image_mask = frames_mask[j]  # [:, j, :, :, :]
                    ref_image_bg = frames_bg[j]  # [:, j, :, :, :]
                    ref_image_fg = frames_fg[j]  # [:, j, :, :, :]

                    continue

                input = {}
                input['raw_image'] = raw_image = frames[j]  # [:, j, :, :, :]
                input['raw_seg'] = raw_seg = frames_seg[j]  # [:, j, :, :, :]

                input['raw_mask'] = raw_image_mask = frames_mask[j]  # [:, j, :, :, :]
                input['raw_bg'] = raw_image_bg = frames_bg[j]  # [:, j, :, :, :]
                input['raw_fg'] = raw_image_fg = frames_fg[j]  # [:, j, :, :, :]
                input['raw_mask_edge'] = frames_mask_edges[j]  # [:, j, :, :, :]

                input['ref_image'] = ref_image
                input['ref_mask'] = ref_image_mask
                input['ref_bg'] = ref_image_bg
                input['ref_fg'] = ref_image_fg
                input['ref_seg'] = ref_seg



                input = Var(input)

                model.set_input(input)

                recon_image, psnr, ms_ssim, lpips, dists, bpp = model.test(p)

                sum_bpp += bpp['total'].data.cpu().numpy()
                sum_psnr += psnr.cpu().data.numpy()
                sum_msssim += ms_ssim.cpu().data.numpy()
                sum_lpips += lpips.cpu().data.numpy()
                sum_dists += dists.cpu().data.numpy()

                cnt += 1

        sum_bpp /= cnt
        sum_psnr /= cnt
        sum_msssim /= cnt
        sum_lpips /= cnt
        sum_dists /= cnt




    return sum_bpp, sum_psnr, sum_msssim, sum_lpips, sum_dists



if __name__ == '__main__':
    opt = train_options.TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # # ========== Validation ========

    val_opt = test_options.TestOptions().parse()

    if val_opt.dataset_name == 'fashion':
        val_opt.rootdir = './fashion/fashion256/val'
        val_opt.filefolderlist = './fashion/fashion256/fashion_val.txt'

    elif val_opt.dataset_name == 'taichi':
        val_opt.rootdir = './taichi/taichi/val'
        val_opt.filefolderlist = './taichi/taichi/taichi_val.txt'

    elif val_opt.dataset_name == 'ted':
        val_opt.rootdir = './TED384-v2/TED384-v2/val'
        val_opt.filefolderlist = './TED384-v2/TED384-v2/ted_val.txt'

    # elif val_opt.dataset_name == 'meeting':
    #     val_opt.rootdir = './video_meeting_dataset/val'
    #     val_opt.filefolderlist = './video_meeting_dataset/meeting_val.txt'

    else:
        val_opt.rootdir = './vox/vox/val'
        val_opt.filefolderlist = './vox/vox/vox_val.txt'





    val_dataset = create_dataset(val_opt)
    print_info('The number of val images = %d' % len(val_dataset))
    best_score = 0
    val_log_dir = os.path.join(opt.checkpoints_dir,opt.name,  "val_metrics.txt")

    # # ========== Train ========

    model = create_model(opt)
    print_info(" model created")
    model.setup(opt)
    print_info(" model setup")
    visualizer = Visualizer(opt)

    total_iters = 0

    sumloss = 0
    sumpsnr = 0
    sumbpp = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        t_data = 0
        visualizer.reset()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            input = {}
            input['raw_image'] = data['raw_image']
            input['raw_mask'] = data['raw_mask']
            input['raw_bg'] = data['raw_bg']
            input['raw_fg'] = data['raw_fg']
            input['raw_mask_edge'] = data['raw_mask_edge']

            input['raw_seg'] = data['raw_seg']
            input['ref_seg'] = data['ref_seg']


            input['ref_image'] = data['ref_image']
            input['ref_mask'] = data['ref_mask']
            input['ref_bg'] = data['ref_bg']
            input['ref_fg'] = data['ref_fg']
            input['ref_seg'] = data['ref_seg']


            input['lib_bg'] = data['lib_bg']
            input['add_bg'] = data['add_bg']



            input = Var(input)
            model.set_input(input)

            model.optimize_parameters(epoch)

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)


                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print_info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

                # # ========== Validation ========
                print('Validation with msssim metrics..')
                sum_bpp,sum_psnr, sum_msssim, sum_lpips, sum_dists = val(model, val_dataset)
                print_info("Val : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf, "
                           "average lpips : %.6lf, average dists : %.6lf\n" % (
                               sum_bpp, sum_psnr, sum_msssim, sum_lpips, sum_dists))

                with open(val_log_dir, 'a') as log_file:
                    log_file.write("sum_bpp, sum_psnr, sum_msssim, sum_lpips, sum_dists value of epoch %d is %.6f,%.6f,%.6f,%.6f,%.6f \n" % (epoch, sum_bpp, sum_psnr, sum_msssim, sum_lpips, sum_dists))
                if sum_msssim > best_score:
                    best_score = sum_msssim
                    model.save_networks('best')
                    print_info('saving the best model with the best ms-ssim score %f' % sum_msssim)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print_info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()
        print_info("model.update_learning_rate")
        print_info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
