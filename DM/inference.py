import argparse
import sys
sys.path.append('/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/')  # change this to your code directory

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import timeit
import math
from PIL import Image
from misc import Logger, grid2fig, conf2fig
from datasets_mug import MUG

import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion
from utils.dataset import CustomVideoDataset 
from torch.optim.lr_scheduler import MultiStepLR

start = timeit.default_timer()
BATCH_SIZE = 5
MAX_EPOCH = 1
epoch_milestones = [800, 1000]
root_dir = '/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/mug'
GPU = "1"
postfix = "-j-sl-vr-of-tr-rmm"
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model
if "random" in postfix:
    frame_sampling = "random"
elif "-vr" in postfix:
    frame_sampling = "very_random"
else:
    frame_sampling = "uniform"
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
if joint:
    null_cond_prob = 0.1
else:
    null_cond_prob = 0.0
split_train_test = "train" in postfix or "-tr" in postfix
use_residual_flow = "-rf" in postfix
config_pth = "/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/config/mug128.yaml"
# put your pretrained LFAE here
AE_RESTORE_FROM = "/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/weights/LFAE_MUG.pth"
# downloaded the pretrained DM model and put its path here
RESTORE_FROM = "/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/weights/DM_MUG.pth"
INPUT_SIZE = 128
N_FRAMES = 40
LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
NUM_EXAMPLES_PER_EPOCH = 465
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 4)
SAVE_VID_EVERY = 1
SAMPLE_VID_EVERY = 2
UPDATE_MODEL_EVERY = 3000

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
os.makedirs(VIDSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(root_dir)
print("update saved model every:", UPDATE_MODEL_EVERY)
print("save model every:", SAVE_MODEL_EVERY)
print("save video every:", SAVE_VID_EVERY)
print("sample video every:", SAMPLE_VID_EVERY)
print(postfix)
print("RESTORE_FROM", RESTORE_FROM)
print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
print("max epoch:", MAX_EPOCH)
print("image size, num frames:", INPUT_SIZE, N_FRAMES)
print("epoch milestones:", epoch_milestones)
print("split train test:", split_train_test)
print("frame sampling:", frame_sampling)
print("only use flow loss:", only_use_flow)
print("null_cond_prob:", null_cond_prob)
print("use residual flow:", use_residual_flow)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=100, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--n-frames", default=N_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, idx=0):
    rec_img = rec_img_batch[idx].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    """Create the model and start the training."""
    print("----------------------------------------------------------------------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(lr=LEARNING_RATE,
                          is_train=False,
                          img_size=INPUT_SIZE//4,
                          num_frames=N_FRAMES,
                          null_cond_prob=null_cond_prob,
                          sampling_timesteps=1000,
                          only_use_flow=only_use_flow,
                          use_residual_flow=use_residual_flow,
                          config_pth=config_pth,
                          pretrained_pth=AE_RESTORE_FROM)
    model.cuda()

    # Not set model to be train mode! Because pretrained flow autoenc need to be eval (BatchNorm)

    if args.fine_tune:
        pass
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(checkpoint['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")

    setup_seed(args.random_seed)
    dataset = CustomVideoDataset()
    dataloader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    print("Data Loaded!")
    data_time = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    epoch_cnt = start_epoch

    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(dataloader):
            print("iter:", i_iter, "actual_step:", actual_step, "epoch:", epoch_cnt)
            actual_step = int(args.start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            fake_vids, real_vids, ref_imgs = batch

            model.set_train_input(ref_img=ref_imgs, real_vid=real_vids, ref_text="")
            model.forward()
            
            if actual_step % args.save_vid_freq == 0 and cnt != 0:
                print("saving video...")
                num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_warp_img = sample_img(model.real_warped_vid[:, :, nf, :, :])
                    save_fake_img = sample_img(fake_vids[:, :, nf, :, :])
                    new_im = Image.new('RGB', (msk_size * 2, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_fake_img, 'RGB'), (msk_size, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + "_%d.gif" % 1
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            if actual_step >= args.final_step:
                break

            cnt += 1

    end = timeit.default_timer()
    print(end - start, 'seconds')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

