import argparse
import sys
sys.path.append('/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/')  # change this to your code directory

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import timeit
import math
from PIL import Image

import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion
from utils.dataset import DeepfakesDataset, FaceShifterDataset, FaceSwapDataset, NeuralTexturesDataset, Face2FaceDataset

start = timeit.default_timer()
BATCH_SIZE = 1
MAX_EPOCH = 5
epoch_milestones = [800, 1000]
root_dir = '/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/ATTN'
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
NUM_EXAMPLES_PER_EPOCH = 465
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 4)
SAMPLE_VID_EVERY = 2
UPDATE_MODEL_EVERY = 3000

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--type", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=100, type=int,
                        metavar='N', help='save image frequency')
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


class TransformerEncoderSA(nn.Module):
    def __init__(self,device , num_channels: int = 1, num_heads: int = 1):
        """A block of transformer encoder with multi-head self-attention from vision transformers paper,
        https://arxiv.org/pdf/2010.11929.pdf.
        """
        super(TransformerEncoderSA, self).__init__()
        self.num_channels = num_channels
        self.device = device
        # Initialize the multi-head attention module with identity weights
        self.mha = nn.MultiheadAttention(
            embed_dim=num_channels, num_heads=num_heads, batch_first=True, bias=False
        )
        self.mha.in_proj_weight.data.fill_(0.0)  # Set input projections to zero
        self.mha.out_proj.weight.data.fill_(0.0)  # Set output projection to zero

        # Initialize the layer normalization with scaling factor 1 and bias 0
        self.ln = nn.LayerNorm([num_channels], elementwise_affine=True)
        self.ln.weight.data.fill_(1.0)

        # Initialize the feed-forward network as an identity function
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels], elementwise_affine=True),
            nn.Linear(in_features=num_channels, out_features=num_channels, bias=False),
            nn.LayerNorm([num_channels], elementwise_affine=True),
            nn.Linear(in_features=num_channels, out_features=num_channels, bias=False),
        )
        self.ff_self[0].weight.data.fill_(0.0)
        self.ff_self[2].weight.data.fill_(0.0)
        self.ff_self[1].weight.data.fill_(0.0)
        self.ff_self[3].weight.data.fill_(0.0)

    def forward(self, x: torch.Tensor , y: torch.Tensor) -> torch.Tensor:
        """Self attention.

        Input feature map [4, 128, 32, 32], flattened to [4, 128, 32 x 32]. Which is reshaped to per pixel
        feature map order, [4, 1024, 128].

        Attention output is same shape as input feature map to multihead attention module which are added element wise.
        Before returning attention output is converted back input feature map x shape. Opposite of feature map to
        mha input is done which gives output [4, 128, 32, 32].
        """
        x = x.view(-1, self.num_channels).to(self.device)
        y = y.view(-1, self.num_channels).to(self.device)
        x_ln = self.ln(x)
        y_ln = self.ln(y)
        attention_value, _ = self.mha(query=x_ln, key=y_ln, value=y_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.view(-1)


def main(): 
    """Create the model and start the training."""
        
    root_dir = '/home/zeta/Workbenches/Diffusion/CVPR23_LFDM/ATTN_'+ str(args.type)
    print(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    device = "cuda:" + str(args.gpu)
    device_secondary = "cuda:" + str(args.gpu+1)
    print("max epoch:", MAX_EPOCH)
    print("image size, num frames:", INPUT_SIZE, N_FRAMES)
    print("----------------------------------------------------------------------------")
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

    lr=1e-4
    adam_betas=(0.9, 0.99)

    attention_real = TransformerEncoderSA(device)
    attention_real.to(device)
    attention_real.optim = torch.optim.Adam(attention_real.parameters(),
                                                   lr=lr, betas=adam_betas)
    
    attention_fake = TransformerEncoderSA(device_secondary)
    attention_fake.to(device_secondary)
    attention_fake.optim = torch.optim.Adam(attention_fake.parameters(),
                                                   lr=lr, betas=adam_betas)

    # Not set model to be train mode! Because pretrained flow autoenc need to be eval (BatchNorm)

    if args.restore_from:
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
    
    datasets = {
        0 : DeepfakesDataset,
        1 : FaceShifterDataset,
        2 : FaceSwapDataset,
        3 : NeuralTexturesDataset,
        4 : Face2FaceDataset
    }
    dataset = datasets[args.type]()
    dataloader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True)
    print("Data Loaded!")
    for epoch_cnt in range(MAX_EPOCH):
        epoch_loss = 0
        for i_iter, batch in enumerate(dataloader):
            print("Epoch:", epoch_cnt, "iter:", i_iter)

            fake_vids, real_vids, ref_imgs = batch
            fake_vids = fake_vids.cuda().requires_grad_()
            real_vids = real_vids.cuda().requires_grad_()
            ref_imgs = ref_imgs.cuda().requires_grad_()
            img_for_save = ref_imgs.clone()

            x, y, z =  ref_imgs, fake_vids[:,:,0,:,:], real_vids[:,:,0,:,:]
            x, y, z = x.flatten(1), y.flatten(1), z.flatten(1)
            x = x.to(device_secondary)
            x = y.to(device_secondary)
            x = attention_fake(x, y)
            x = x.to(device)
            z = z.to(device)
            x = attention_real(x, z)         

            ref_imgs = x.view(BATCH_SIZE,3,128,128)
            model.set_train_input(ref_imgs,fake_vids,"")
            model.forward()
            
            for params in model.parameters():
                params.requires_grad = True
            for params in attention_fake.parameters():
                params.requires_grad = True
            for params in attention_real.parameters():
                params.requires_grad = True

            fake_vids = fake_vids.cuda()
            model.real_out_vid = model.real_out_vid.cuda()
            loss = torch.nn.functional.smooth_l1_loss(model.real_warped_vid, fake_vids)
            attention_real.optim.zero_grad()
            attention_fake.optim.zero_grad()
            loss.backward()
            print("loss = ", loss.item())
            epoch_loss += loss.item()
            attention_real.optim.step()
            attention_fake.optim.step()
            
        print(f"Epoch-{epoch_cnt} Loss: {epoch_loss}")

    print("Training Done!")
    print("----------------------------------------------------------------------------")

    for i_iter, batch in enumerate(dataloader):
        print("Testing", "iter:", i_iter)

        fake_vids, real_vids, ref_imgs = batch
        img_for_save = ref_imgs.clone()

        x, y, z =  ref_imgs, fake_vids[:,:,0,:,:], real_vids[:,:,0,:,:]
        x, y, z = x.flatten(1), y.flatten(1), z.flatten(1)
        x = x.to(device_secondary)
        x = y.to(device_secondary)
        x = attention_fake(x, y)
        x = x.to(device)
        z = z.to(device)
        x = attention_real(x, z)             

        ref_imgs = x.view(BATCH_SIZE,3,128,128)
        model.set_train_input(ref_imgs,fake_vids,"")
        model.forward()
        
        print("saving video...")
        num_frames = real_vids.size(2)
        msk_size = ref_imgs.shape[-1]
        new_im_arr_list = []
        save_src_img = sample_img(img_for_save)
        for nf in range(num_frames):
            save_tar_img = sample_img(real_vids[:, :, nf, :, :])
            save_real_warp_img = sample_img(model.real_warped_vid[:, :, nf, :, :])
            save_real_out_img = sample_img(model.real_out_vid[:, :, nf, :, :])
            save_fake_img = sample_img(fake_vids[:, :, nf, :, :])
            new_im = Image.new('RGB', (msk_size * 5, msk_size))
            new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
            new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (msk_size,0))
            new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (2*msk_size, 0))
            new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (3*msk_size, 0))
            new_im.paste(Image.fromarray(save_fake_img, 'RGB'), (4*msk_size, 0))
            new_im_arr = np.array(new_im)
            new_im_arr_list.append(new_im_arr)
        new_vid_name = 'AS' + format(i_iter, "03d") + ".gif"
        new_vid_file = os.path.join(root_dir, new_vid_name)
        imageio.mimsave(new_vid_file, new_im_arr_list)

    print("Testing Done!")
    print("----------------------------------------------------------------------------")

    torch.save({'attention_real': attention_real.state_dict()}, os.path.join(root_dir, 'attention_real.pth'))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
