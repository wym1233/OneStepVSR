import os
os.environ['TORCH_HOME'] = '/data/wym123/VSRdataset/torchhome/'
import gc
import tqdm
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import sys
sys.path.append('/code/')

from model.S3diff.my_utils.wavelet_color import wavelet_color_fix
from model.S3diff.onestepvsr import OneStepVSR
from model.S3diff.de_net import DEResNet
from dataset.test_dataset import SumTestDataset
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def main(args):
    split = True
    colorfix = True
    input_length = 4
    output_length = 4

    # Initialize dataset
    dataset_test = SumTestDataset(datasetname=args.dataset_name, frames=input_length)
    length_per_seq = dataset_test.length_per_seq
    
    print(f'length_per_seq: {length_per_seq}')
    print(f'dataset_test_len: {len(dataset_test)}')
    print(f'split: {split}')

    # Process indices for frame sequence handling
    def process_indices(input_len, outputlen):
        all_indices = []
        save_flags = []
        overleaplen = (input_len - outputlen) // 2
        
        for base in range(0, len(dataset_test), length_per_seq):
            # First chunk in each sequence
            all_indices.append(base)
            tmpflag = [True] * input_len
            for i in range(overleaplen):
                tmpflag[i] = True
            for i in range(overleaplen):
                tmpflag[-1-i] = False
            save_flags.append(tmpflag)
            
            # Middle chunks
            for i in range(base+outputlen, base+length_per_seq-input_len, outputlen):
                all_indices.append(i)
                tmpflag = [True] * input_len
                for i in range(overleaplen):
                    tmpflag[i] = False
                for i in range(overleaplen):
                    tmpflag[-1-i] = False
                save_flags.append(tmpflag)
            
            # Last chunk in each sequence
            last_idx = base + length_per_seq - input_len
            all_indices.append(last_idx)
            tmpflag = [True] * input_len
            for i in range(overleaplen):
                tmpflag[i] = False
            for i in range(overleaplen):
                tmpflag[-1-i] = True
            save_flags.append(tmpflag)
        
        return all_indices, save_flags
    
    all_indices, save_flags = process_indices(input_length, output_length)

    # Initialize models
    net_sr = OneStepVSR(
        lora_rank_unet=args.lora_rank_unet, 
        lora_rank_vae=args.lora_rank_vae, 
        pos_prompt=args.pos_prompt,
        neg_prompt=args.neg_prompt,
        sd_path=args.sd_path,
        parapth=args.pretrained_para_path,
    )
    net_sr.set_eval()
    net_sr.cuda()
    net_sr.unet.enable_gradient_checkpointing()
    
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    # Initialize optical flow model
    spynet = raft_large(weights=Raft_Large_Weights.DEFAULT)
    spynet.requires_grad_(False)
    spynet = spynet.to('cuda')
    
    def get_flow(raftmodel, x):
        x = x.unsqueeze(0)
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = raftmodel(x_1, x_2)[-1]
        flows_forward = raftmodel(x_2, x_1)[-1]
        return flows_forward.unsqueeze(0), flows_backward.unsqueeze(0)

    from basicsr.archs.basicvsr_arch import BasicVSR
    precleanmodel = BasicVSR(num_feat=64, num_block=30)
    if dataset_test.datasetname=='vimeo':
        precleanmodel.load_state_dict(torch.load(args.basicvsr_vimeo_path)['params'], strict=True)
    else:
        precleanmodel.load_state_dict(torch.load(args.basicvsr_vimeo_path)['params'], strict=True)
    precleanmodel.eval()
    precleanmodel = precleanmodel.to('cuda')

    # Inference loop
    progress_bar = tqdm(range(0, len(all_indices)), initial=0, desc="Steps")
    with torch.no_grad():
        for idx, flag in zip(all_indices, save_flags):
            gc.collect()
            torch.cuda.empty_cache()
            batch_val = dataset_test.__getitem__(idx)
            x_src = batch_val['lq'].squeeze().cuda()

            # Forward pass
            basicVSRimg=precleanmodel(x_src.unsqueeze(0).detach()*0.5+0.5).clamp(0,1).squeeze(0)
            basicVSRimg=(basicVSRimg*2-1).clamp(-1,1)
            ff, fb = get_flow(spynet, basicVSRimg)
            deg_score = net_de(basicVSRimg)

            if split:
                # Split processing for large images
                B, C, H, W = basicVSRimg.shape
                overlap_h = 64
                overlap_w = 64
                h_split = 1
                w_split = 2
                h_block = H // h_split
                w_block = W // w_split
                final_output = torch.zeros_like(basicVSRimg)
                blend_mask = torch.zeros_like(basicVSRimg)

                for i in range(h_split):
                    for j in range(w_split):
                        h_start = max(i*h_block - overlap_h, 0)
                        h_end = min((i+1)*h_block + overlap_h, H)
                        w_start = max(j*w_block - overlap_w, 0)
                        w_end = min((j+1)*w_block + overlap_w, W)
                        img_block = basicVSRimg[:,:,h_start:h_end,w_start:w_end]

                        try:
                            processed_block = net_sr.inference_forward(img_block, deg_score, 
                                                                      ff=ff[:,:,:,h_start:h_end,w_start:w_end], 
                                                                      fb=fb[:,:,:,h_start:h_end,w_start:w_end], 
                                                                      refinerimg=img_block)
                        except:
                            gc.collect()
                            torch.cuda.empty_cache()
                            processed_block = net_sr.inference_forward(img_block, deg_score, 
                                                                      ff=ff[:,:,:,h_start:h_end,w_start:w_end], 
                                                                      fb=fb[:,:,:,h_start:h_end,w_start:w_end], 
                                                                      refinerimg=img_block)
                        
                        # Blend overlapping regions
                        weight = torch.ones_like(processed_block)
                        if i > 0:
                            weight[:,:,:overlap_h,:] *= torch.linspace(0,1,overlap_h).view(1,1,-1,1).cuda()
                        if i < h_split-1:
                            weight[:,:,-overlap_h:,:] *= torch.linspace(1,0,overlap_h).view(1,1,-1,1).cuda()
                        if j > 0:
                            weight[:,:,:,:overlap_w] *= torch.linspace(0,1,overlap_w).view(1,1,1,-1).cuda()
                        if j < w_split-1:
                            weight[:,:,:,-overlap_w:] *= torch.linspace(1,0,overlap_w).view(1,1,1,-1).cuda()
                            
                        final_output[:,:,h_start:h_end,w_start:w_end] += processed_block * weight
                        blend_mask[:,:,h_start:h_end,w_start:w_end] += weight
                
                output_image = (final_output / (blend_mask + 1e-6))
                x_tgt_pred = output_image
                x_tgt_pred = (x_tgt_pred*0.5+0.5).clamp(0,1)
            else:
                x_tgt_pred = net_sr.inference_forward(basicVSRimg, deg_score, ff=ff, fb=fb, refinerimg=basicVSRimg)
                x_tgt_pred = (x_tgt_pred*0.5+0.5).clamp(0,1)

            # Save outputs
            lr_path_list = batch_val['key']
            for i in range(x_tgt_pred.shape[0]):
                if flag[i]:
                    output_pil = transforms.ToPILImage()(x_tgt_pred[i])
                    if colorfix:
                        im_lr_resizei = transforms.ToPILImage()((basicVSRimg[i]*0.5+0.5).cpu().detach())
                        output_pil = wavelet_color_fix(output_pil, im_lr_resizei)
                        
                    clip_name, frame_name = lr_path_list[i].split('/')
                    output_dir = os.path.join(args.inference_output_dir, args.lable)
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(os.path.join(output_dir, clip_name), exist_ok=True)
                    output_path = os.path.join(output_dir, clip_name, frame_name)
                    output_pil.save(output_path)
            
            progress_bar.update(1)

    progress_bar.close()
    
    # Generate videos from output frames
    from moviepy import ImageSequenceClip
    mp4dir = os.path.join(output_dir, 'mp4dir')
    os.makedirs(mp4dir, exist_ok=True)
    for seq in os.listdir(output_dir):
        if seq == 'mp4dir':
            continue
        ims = sorted(os.listdir(os.path.join(output_dir, seq)))
        ims = [os.path.join(output_dir, seq, im) for im in ims]
        clip = ImageSequenceClip(ims, fps=24)
        clip.write_videofile(os.path.join(mp4dir, seq + '.mp4'), codec='libx264')

    # Evaluate results
    from src.eval import evalREDS4rec
    evalREDS4rec(output_dir, dataset_test.dataset_dir)


def parse_inference_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description='OneStepVSR Inference')
    
    # Core inference parameters
    parser.add_argument("--dataset_name", type=str, default='REDS4', 
                       help="Name of the dataset to use for inference")
    parser.add_argument("--pretrained_para_path", type=str, 
                       default="/data/wym123/VSRdataset/newckptdir_s3diffinserted3/baseline_repetition/para_2_23600.pt",
                       help="Path to the pretrained model parameters")
    parser.add_argument("--inference_output_dir", type=str, 
                       default='/data/wym123/VSRdataset/inference_out_1',
                       help="Directory to save inference outputs")
    parser.add_argument("--lable", type=str, 
                       default='result1',
                       help="Directory to save inference outputs")


                       
    
    # Model architecture parameters
    parser.add_argument("--sd_path", type=str, default='/data/wym123/VSRdataset/sd-turbo',
                       help="Path to Stable Diffusion model")
    parser.add_argument("--de_net_path", type=str, default='/code/model/S3diff/assets/mm-realsr/de_net.pth',
                       help="Path to degradation network")
    parser.add_argument("--basicvsr_REDS_path", type=str, 
                       default="/code/src/BasicVSR_REDS4-543c8261.pth",
                       help="Path to BasicVSR weights") 
    parser.add_argument("--basicvsr_vimeo_path", type=str, 
                       default="/code/src/BasicVSR_Vimeo90K_BIx4-2a29695a.pth",
                       help="Path to BasicVSR weights") 


    parser.add_argument("--lora_rank_unet", type=int, default=32,
                       help="LoRA rank for UNet")
    parser.add_argument("--lora_rank_vae", type=int, default=16,
                       help="LoRA rank for VAE")
    
    # Prompt parameters
    parser.add_argument("--pos_prompt", type=str, 
                       default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
                       help="Positive prompt for generation")
    parser.add_argument("--neg_prompt", type=str, 
                       default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
                       help="Negative prompt for generation")
    
    # Performance parameters
    parser.add_argument("--enable_xformers", action="store_true", default=True,
                       help="Enable memory efficient attention")
    parser.add_argument("--allow_tf32", action="store_true", default=True,
                       help="Allow TF32 operations")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_inference_args()
    main(args)