import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['TORCH_HOME']='/data/wym123/VSRdataset/torchhome/'

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

import lpips
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.archs.arch_util import flow_warp
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# Custom imports
import sys
sys.path.append('/code/')
from model.S3diff.de_net import DEResNet
from model.S3diff.onestepvsr import OneStepVSR
from dataset.reds_dataset import REDSRecurrentDataset
import random

def main(args):
    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(project_dir='.', logging_dir=args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("train_onestepvsr")
    
    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda().eval()

    precleanmodel = BasicVSR(num_feat=64, num_block=30)
    precleanmodel.load_state_dict(torch.load(args.basicvsr_path)['params'], strict=True)
    precleanmodel = precleanmodel.cuda().eval()

    net_sr = OneStepVSR(
        lora_rank_unet=args.lora_rank_unet, 
        lora_rank_vae=args.lora_rank_vae, 
        pos_prompt=args.pos_prompt,
        neg_prompt=args.neg_prompt,
        sd_path=args.sd_path,
        parapth=args.pretrained_para_path,
    )
    net_sr.set_para_train()

    # Enable optimizations
    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        net_sr.unet.enable_xformers_memory_efficient_attention()
    
    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize loss networks
    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type='dino', 
            output_type='conv_multi_level', 
            loss_type=args.gan_loss_type, 
            device="cuda"
        )
        net_disc = net_disc.cuda()
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda().requires_grad_(False)
    spynet = raft_large(weights=Raft_Large_Weights.DEFAULT).cuda().requires_grad_(False)
    def get_flow(raftmodel, x):
        x=x.unsqueeze(0)
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = raftmodel(x_1, x_2)[-1]
        flows_forward = raftmodel(x_2, x_1)[-1]
        
        return flows_forward.unsqueeze(0),flows_backward.unsqueeze(0)

    # Load dataset
    dataset_opts = OmegaConf.load(args.dataset_config)
    dataset_train = REDSRecurrentDataset(dataset_opts['dataset']['train'])
    dl_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers
    )

    # Setup optimizers
    para_to_opt_list = net_sr.get_trainable_paralist(learning_rate=args.learning_rate)
    optimizer = torch.optim.AdamW(
        para_to_opt_list, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler, 
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, 
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler, 
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, 
        power=args.lr_power,
    )

    # Prepare for distributed training
    net_sr, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc, dl_train, net_de = accelerator.prepare(
        net_sr, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc, dl_train, net_de,
    )

    def warpMAEloss(videoframe,GT,ff,fb):
        T,C,H,W=videoframe.shape
        ff=ff.detach()
        fb=fb.detach()
        x_1 = videoframe[:-1, :, :, :].reshape(-1, C, H, W)
        x_2 = videoframe[1:, :, :, :].reshape(-1, C, H, W)         
        if random.random() < 0.5:
            # Compute forward loss
            forward_warped = flow_warp(x_1, ff.squeeze(0).permute(0,2,3,1).contiguous())
            loss = F.l1_loss(forward_warped, GT[1:].detach(), reduction="mean") 
        else:
            # Compute backward loss 
            backward_warped = flow_warp(x_2, fb.squeeze(0).permute(0,2,3,1).contiguous())
            loss = F.l1_loss(backward_warped, GT[:-1].detach(), reduction="mean") 
        return loss

    # Training loop
    global_step = 0
    for epoch in range(args.num_training_epochs):
        if accelerator.is_local_main_process:
            if epoch%2==0:
                os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch}_step_{global_step}.pt')
                accelerator.unwrap_model(net_sr).save_para(checkpoint_path)
        
        accelerator.wait_for_everyone()
        
        progress_bar = tqdm(range(len(dl_train)), disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(dl_train):
            torch.cuda.empty_cache()
            
            with accelerator.accumulate(net_sr, net_disc):
                x_src=batch['lq'].squeeze()
                x_tgt=batch['gt'].squeeze()
                T,C,H,W=x_tgt.shape

                with torch.no_grad():

                    basicVSRimg=precleanmodel(x_src.unsqueeze(0).detach()*0.5+0.5).clamp(0,1).squeeze(0)
                    basicVSRimg=(basicVSRimg*2-1).clamp(-1,1)
                    ff,fb=get_flow(spynet,basicVSRimg.detach())
                    deg_score = net_de(basicVSRimg.detach()).detach()
                
                pos_tag_prompt = [args.pos_prompt for _ in range(T)]                
                neg_tag_prompt = [args.neg_prompt for _ in range(T)]

                neg_probs = torch.rand(1).to(accelerator.device)
                if neg_probs < args.neg_prob:
                    mixed_tgt = basicVSRimg.detach()
                    mixed_tag_prompt = 'neg'
                else:
                    mixed_tgt = x_tgt.detach()
                    mixed_tag_prompt = 'pos'

    
                x_tgt_pred = net_sr(basicVSRimg.detach(), deg_score, mixed_tag_prompt, ff=ff.detach(),fb=fb.detach(),refinerimg=basicVSRimg.detach())

                
                loss_l2 = F.l1_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction="mean") 
                loss_lpips = net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() 
                loss_consis = warpMAEloss(x_tgt_pred.float(), mixed_tgt.detach().float(),ff.detach(),fb.detach())

                lossG = net_disc(x_tgt_pred.float(), for_G=True).mean()
                loss = loss_l2 * args.lambda_l2 + loss_lpips* args.lambda_lpips +lossG* args.lambda_gan+loss_l2 *loss_consis
    
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_sr.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean()
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean()
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake
                
                global_step += 1
                
                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "lossG": lossG.detach().item(),
                        "lossD": lossD.detach().item(),
                        "loss_consis": loss_consis.detach().item(),
                    }
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

def parse_args():
    parser = argparse.ArgumentParser(description="OnestepVSR Training Script")
    
    # Model paths
    parser.add_argument("--de_net_path", type=str, 
                       default="/code/model/S3diff/assets/mm-realsr/de_net.pth",
                       help="Path to DE Net weights")
    parser.add_argument("--basicvsr_path", type=str, 
                       default="/code/src/BasicVSR_REDS4-543c8261.pth",
                       help="Path to BasicVSR weights") 
    parser.add_argument("--pretrained_para_path", type=str, 
                       default="/data/wym123/VSRdataset/newckptdir_s3diffinserted3/baseline_repetition/para_2_23600.pt",
                       help="Path to S3Diff parameters")
    parser.add_argument("--sd_path", type=str, 
                       default="/data/wym123/VSRdataset/sd-turbo",
                       help="Path to Stable Diffusion model")
    parser.add_argument("--dataset_config", type=str, 
                       default="/code/dataset/config_reds.yaml",
                       help="Path to dataset config file")
    
    # Model architecture
    parser.add_argument("--lora_rank_unet", type=int, default=32,
                       help="LoRA rank for UNet")
    parser.add_argument("--lora_rank_vae", type=int, default=16,
                       help="LoRA rank for VAE")
    parser.add_argument("--gan_disc_type", default="vagan",
                       help="GAN discriminator type")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s",
                       help="GAN loss type")
    
    # Loss weights
    parser.add_argument("--lambda_gan", type=float, default=8.0,
                       help="Weight for GAN loss")
    parser.add_argument("--lambda_lpips", type=float, default=2.0,
                       help="Weight for LPIPS loss") 
    parser.add_argument("--lambda_l2", type=float, default=1.0,
                       help="Weight for L2 loss")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--num_training_epochs", type=int, default=10,
                       help="Total number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=50000,
                       help="Maximum number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2 parameter")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                       help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                       help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Learning rate scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="Number of warmup steps for LR scheduler")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                       help="Number of cycles for cosine scheduler")
    parser.add_argument("--lr_power", type=float, default=0.1,
                       help="Power factor for polynomial scheduler")
    
    # Prompts
    parser.add_argument("--pos_prompt", type=str, 
                       default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
                       help="Positive prompt for training")
    parser.add_argument("--neg_prompt", type=str, 
                       default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth",
                       help="Negative prompt for training")
    parser.add_argument("--neg_prob", type=float, default=0.05,
                       help="Probability of using negative prompt")
    
    # Technical settings
    parser.add_argument("--enable_xformers_memory_efficient_attention", 
                       action="store_true", default=True,
                       help="Enable xformers memory efficient attention")
    parser.add_argument("--gradient_checkpointing", 
                       action="store_true", default=False,
                       help="Enable gradient checkpointing")
    parser.add_argument("--allow_tf32", 
                       action="store_true", default=True,
                       help="Allow TF32 on Ampere GPUs")
    parser.add_argument("--mixed_precision", type=str, default="no", 
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    parser.add_argument("--set_grads_to_none", 
                       action="store_true", default=False,
                       help="Set gradients to None instead of zero")
    
    # Output directories
    parser.add_argument("--output_dir", type=str, 
                       default="/data/wym123/VSRdataset/ckptdir1",
                       help="Output directory for checkpoints")
    parser.add_argument("--logging_dir", type=str, 
                       default="/output/logs/",
                       help="Logging directory for tensorboard")
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    
    main(args)