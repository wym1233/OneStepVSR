import os
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
os.environ['TORCH_HOME']='/data/wym123/VSRdataset/torchhome/'
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from pathlib import Path
import sys
sys.path.append('/code/')
from model.S3diff.de_net import DEResNet

from model.S3diff.my_utils.wavelet_color import wavelet_color_fix, adain_color_fix
from model.S3diff.my_utils.testing_utils import parse_args_paired_testing
def main(args):
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)




    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True



    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    offset = args.padding_offset


    from dataset.reds_dataset import REDSRecurrentDataset
    dataset_opts = OmegaConf.load('/code/dataset/config_reds.yaml')
    dataset_opts['dataset']['test']['num_frame'] = 4
    dataset_opts['dataset']['test']['crop'] = False
    dataset_test = REDSRecurrentDataset(dataset_opts['dataset']['test'])

    print('--------------')
    print(len(dataset_test))
    def process_indices():
        all_indices = []
        save_flags = []
        
        # Process each 100-frame chunk (0-99, 100-199, 200-299, 300-399)
        for base in range(0, 400, 100):
            # First chunk in each 100 (0,100,200,300): save first 3 frames
            all_indices.append(base)
            save_flags.append([True, True, True, False])
            
            # Middle chunks: save frames after the overlap
            for i in range(base + 2, base + 96, 2):
                all_indices.append(i)
                save_flags.append([False, True, True, False])
            
            # Last chunk in each 100: save remaining frames
            last_idx = (base + 96)
            all_indices.append(last_idx)
            save_flags.append([False, True, True, True])
        
        return all_indices, save_flags
    all_indices,save_flags=process_indices()

    from basicsr.archs.basicvsr_arch import BasicVSR
    precleanmodel = BasicVSR(num_feat=64, num_block=30)
    precleanmodel.load_state_dict(torch.load('/code/trains3diff/BasicVSR_REDS4-543c8261.pth')['params'], strict=True)
    precleanmodel.eval()
    precleanmodel = precleanmodel.to('cuda')

    
    progress_bar = tqdm(range(0, len(all_indices)), initial=0, desc="Steps",disable=not accelerator.is_local_main_process,)
    for idx, flag in zip(all_indices, save_flags):
        torch.cuda.empty_cache()
        batch_val=dataset_test.__getitem__(idx)

        im_lr = batch_val['lq'].squeeze().cuda()
        B,C,H,W=batch_val['gt'].squeeze().shape
        im_lr = im_lr.to(memory_format=torch.contiguous_format).float()


        with torch.no_grad():
            im_lr_resize = precleanmodel((im_lr.unsqueeze(0)/2+0.5)).squeeze(0)
            im_lr_resize = (im_lr_resize*2-1).clamp(-1, 1)

        B = im_lr_resize.size(0)
        with torch.no_grad():
            
            x_tgt_pred = im_lr_resize
            # x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).clamp(0,1).cpu().detach()
        lr_path = batch_val['key']
        for i in range(B):
            if not flag[i]:
                continue
            

            output_pil = transforms.ToPILImage()(out_img[i])

            if args.align_method == 'nofix':
                output_pil = output_pil
            else:
                im_lr_resizei = transforms.ToPILImage()((im_lr_resize[i]* 0.5 + 0.5).cpu().detach())
                if args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(output_pil, im_lr_resizei)
                elif args.align_method == 'adain':
                    output_pil = adain_color_fix(output_pil, im_lr_resizei)
            clip_name, frame_name = lr_path.split('/')
            frame_name=int(frame_name)+i
            frame_name=str(frame_name).zfill(8)
            
            pt=args.output_dir + clip_name + '/' + frame_name + '.png'
            # print(pt)
            output_pil.save(pt)
        progress_bar.update(1)

    # print_results = evaluate(args.output_dir, args.ref_path, None)
    # out_t = os.path.join(args.output_dir, 'results.txt')
    # with open(out_t, 'w', encoding='utf-8') as f:
    #     for item in print_results:
    #         f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    args = parse_args_paired_testing()
    args.base_config='/code/S3Diff-main/configs/sr_test.yaml'
    args.pretrained_path='/data/wym123/VSRdataset/S3diffLora/s3diff.pkl'
    args.sd_path='/data/wym123/VSRdataset/sd-turbo'
    args.gradient_accumulation_steps=1
    args.mixed_precision='no'
    args.report_to='tensorboard'
    args.seed=None
    
    args.lora_rank_unet=32
    args.lora_rank_vae=16
    args.de_net_path='/code/model/S3diff/assets/mm-realsr/de_net.pth'
    args.enable_xformers_memory_efficient_attention=True
    args.gradient_checkpointing=True
    args.allow_tf32=True
    args.padding_offset=32
    args.pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
    args.neg_prompt="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"
    args.align_method = 'wavelet'

    args.latent_tiled_size=96
    args.latent_tiled_overlap=32

    args.opt={
    "lr_path": "/data/wym123/VSRdataset/REDS/REDS4/train_sharp_bicubic/X4/",
    'meta_info_file':'/code/dataset/REDS4_eval_metadata.txt',
    "io_backend": {"type": "disk"}
    }
    args.output_dir='/data/wym123/VSRdataset/newinference_out/BasicVSR_overleap/'
    # args.batch_size=4
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'000', exist_ok=True)
    os.makedirs(args.output_dir+'011', exist_ok=True)
    os.makedirs(args.output_dir+'015', exist_ok=True)
    os.makedirs(args.output_dir+'020', exist_ok=True)
    main(args)

