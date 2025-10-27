import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKLTemporalDecoder,AutoencoderKL,UNet2DConditionModel
from peft import LoraConfig, get_peft_model,inject_adapter_in_model
from .modelutil import make_1step_sched, my_lora_fwd
from einops import rearrange
from basicsr.archs.arch_util import flow_warp
import torch.nn.functional as F

class OneStepVSR(torch.nn.Module):
    def __init__(self,sd_path,pos_prompt,neg_prompt,parapth=None,use3Dunet=True,useprop=True,residualconnection=True,lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64,propchannel=256):
        super().__init__()
        print('-----------------init model-----------------')

        self.use3Dunet=use3Dunet
        self.useprop=useprop
        self.residualconnection=residualconnection
        
        self.lora_rank_unet=lora_rank_unet
        self.lora_rank_vae=lora_rank_vae
        self.block_embedding_dim=block_embedding_dim
        num_embeddings = 64
        
        print('init SD2.1 tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        print('init SD2.1 text encoder')
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        if self.use3Dunet:
            caption_tokens_pos = self.tokenizer([pos_prompt for _ in range(4)], max_length=self.tokenizer.model_max_length,padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            self.caption_enc_pos = self.text_encoder(caption_tokens_pos)[0]
            caption_tokens_neg = self.tokenizer([neg_prompt for _ in range(4)], max_length=self.tokenizer.model_max_length,padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            self.caption_enc_neg = self.text_encoder(caption_tokens_neg)[0]
        else:
            caption_tokens_pos = self.tokenizer([pos_prompt for _ in range(1)], max_length=self.tokenizer.model_max_length,padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            self.caption_enc_pos = self.text_encoder(caption_tokens_pos)[0]
            caption_tokens_neg = self.tokenizer([neg_prompt for _ in range(1)], max_length=self.tokenizer.model_max_length,padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            self.caption_enc_neg = self.text_encoder(caption_tokens_neg)[0]
        del self.text_encoder
        del self.tokenizer

        print('init scheduler')
        self.sched = make_1step_sched(sd_path)

        print('init vae')
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae") 

        print('init vae mlp modules')
        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )
        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)

        print('init vae lora')
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        vae_lora_config = LoraConfig(r=self.lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
        vae=inject_adapter_in_model(vae_lora_config,vae,adapter_name="vae_skip")

        print('change vae lora forward')
        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name and 'decoder' not in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        if self.use3Dunet:
            print('init 3Dunet')
            from model.S3diff.animatediffmodels.unet import UNet3DConditionModel      
            unet_additional_kwargs = {
                'use_motion_module': True,
                'motion_module_resolutions': [1, 2, 4, 8],
                'unet_use_cross_frame_attention': False,
                'unet_use_temporal_attention': False,
                'motion_module_decoder_only': False,
                'use_inflated_groupnorm': True,
                'motion_module_mid_block': False,
                'motion_module_type': 'Vanilla',
                'motion_module_kwargs': {
                    'num_attention_heads': 8,
                    'num_transformer_block': 1,
                    'attention_block_types': ["Temporal_Self", "Temporal_Self"],
                    'temporal_position_encoding': True,
                    'temporal_position_encoding_max_len': 32,
                    'temporal_attention_dim_div': 1,
                    'zero_initialize': True
                }
            }
            unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet",unet_additional_kwargs=unet_additional_kwargs)

        else:
            print('init 2Dunet')
            unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet") 

        print('init unet lora')
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]        
        unet_lora_config = LoraConfig(r=self.lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        unet=inject_adapter_in_model(unet_lora_config,unet)

        print('change unet lora forward')        
        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name and 'motion_module' not in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        print('init unet mlp modules')
        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )
        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        print('init propagation module')
        from .propagation_module import Propagation
        self.propagator1 = Propagation(4, learnable=True,mid_channels=propchannel)
        self.propagator2 = Propagation(4, learnable=True,mid_channels=propchannel)
        
        
        print('init W')
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        unet.to("cuda")
        self.unet = unet
        vae.to("cuda")
        self.vae=vae

        self.timesteps = torch.tensor([999], device="cuda").long()
        
        if parapth is not None:
            self.load_para(parapth)


    def set_eval(self):
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.vae_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.vae_block_embeddings.requires_grad_(False)
        self.unet_de_mlp.eval()
        self.unet_block_mlp.eval()
        self.unet_fuse_mlp.eval()
        self.unet_block_embeddings.requires_grad_(False)

        
    def set_para_train(self):
        self.set_eval()

        if self.useprop:
            self.propagator1.requires_grad_(True)
            self.propagator2.requires_grad_(True)


        self.vae_de_mlp.train()
        self.vae_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.vae_block_embeddings.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
        self.unet_de_mlp.train()
        self.unet_block_mlp.train()
        self.unet_fuse_mlp.train()  
        self.unet_block_embeddings.requires_grad_(True)
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

    
    def forward(self, c_t, deg_score, prompt,ff=None,fb=None,refinerimg=None):

        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")

                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")

                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)
        
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        if self.useprop:
            encoded_control = self.propagator1(encoded_control,ff,fb)
        
        if prompt=='pos':
            caption_enc = self.caption_enc_pos.detach()
        elif prompt=='neg':
            caption_enc = self.caption_enc_neg.detach()
        else:
            print('prompt error')
            return

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
        x_denoised = self.sched.step(model_pred.unsqueeze(0), self.timesteps, encoded_control, return_dict=True).prev_sample
        
        if self.useprop:
            x_denoised = self.propagator2(x_denoised,ff,fb).squeeze(0)
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        if self.residualconnection:
            assert refinerimg is not None
            output_image=output_image+refinerimg
        return output_image    

    def inference_forward(self, c_t, deg_score,ff=None,fb=None,refinerimg=None,cfg=True):
        self.guidance_scale = 1.07

        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)
       
        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))
    
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")
                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        lq_latent = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        if self.useprop:
            lq_latent = self.propagator1(lq_latent,ff,fb)
        if cfg:
            pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=self.caption_enc_pos).sample
            neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=self.caption_enc_neg).sample
            model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=self.caption_enc_pos).sample
            model_pred = pos_model_pred
        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        if self.useprop:
            x_denoised = self.propagator2(x_denoised,ff,fb).squeeze(0)
       
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        if self.residualconnection:
            assert refinerimg is not None
            refinerimg=refinerimg.squeeze(0)
            output_image=output_image+refinerimg
            # output_image = self.post_filter(output_image,refinerimg)
        return output_image

    def save_para(self, outf):
        sd = {}
        total_params = 0
        print('-----------------save_para-----------------')
        if self.useprop:
            prop1_params = {k: v for k, v in self.propagator1.state_dict().items()}
            prop2_params = {k: v for k, v in self.propagator2.state_dict().items()}
            sd["propagator1"] = prop1_params
            sd["propagator2"] = prop2_params
            prop_total = sum(v.numel() for v in prop1_params.values()) + sum(v.numel() for v in prop2_params.values())
            print(f'Propagation parameters: {prop_total/1e6:.4f}M')
            total_params += prop_total

        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_unet_Motionmodules"] = {k: v for k, v in self.unet.state_dict().items() if "motion_modules" in k}

        unet_lora_total = sum(v.numel() for v in sd["state_dict_unet"].values()) 
        print(f'Unet lora parameters: {unet_lora_total/1e6:.4f}M')
        total_params+= unet_lora_total
        
        vae_lora_total = sum(v.numel() for v in sd["state_dict_vae"].values())
        print(f'VAE lora parameters: {vae_lora_total/1e6:.4f}M')
        total_params+= vae_lora_total


        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}

        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        
        sd["w"] = self.W

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }
        
    
        s3diff_total = sum(v.numel() for v in sd["state_dict_vae_de_mlp"].values()) + sum(v.numel() for v in sd["state_dict_unet_de_mlp"].values()) + \
            sum(v.numel() for v in sd["state_dict_vae_block_mlp"].values()) + sum(v.numel() for v in sd["state_dict_unet_block_mlp"].values()) + \
            sum(v.numel() for v in sd["state_dict_vae_fuse_mlp"].values()) + sum(v.numel() for v in sd["state_dict_unet_fuse_mlp"].values()) + \
            sum(v.numel() for v in sd["state_embeddings"]["state_dict_vae_block"].values()) + sum(v.numel() for v in sd["state_embeddings"]["state_dict_unet_block"].values())
    
        print(f'mlp parameters: {s3diff_total/1e6:.4f}M')
        total_params += s3diff_total
        

        print(f'Total trainable parameters: {total_params/1e6:.4f}M')
        print(f'Saving to {outf}')
        torch.save(sd, outf)

    def load_para(self, parapth):
        print('-----------------load model-----------------')
        print('load para from:',parapth)
        sd = torch.load(parapth, map_location="cpu")

        if sd.get("state_dict_vae") is not None:
            print('loading vae lora')
            m,u=self.vae.load_state_dict(sd["state_dict_vae"], strict=False)
            print(f"vae lora:\n### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
           
        if sd.get("state_dict_unet") is not None:
            print('loading unet lora')
            try:
                m,u=self.unet.load_state_dict(sd["state_dict_unet"], strict=False)
                print(f"unet lora:\n### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
                assert len(u)==0
            except:
                print('Failed to load unet lora, skipping...')
        if sd.get("state_dict_unet_Motionmodules") is not None:
            print('loading unet motion modules')
            m,u=self.unet.load_state_dict(sd["state_dict_unet_Motionmodules"], strict=False)
            print(f"unet motion modules:\n### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
            assert len(u)==0

           
        if sd.get("state_dict_unet_de_mlp") is not None:
            print('loading unet de mlp')
            m,u=self.unet_de_mlp.load_state_dict(sd["state_dict_unet_de_mlp"], strict=True)
        if sd.get("state_dict_unet_block_mlp") is not None:
            print('loading unet block mlp')
            m,u=self.unet_block_mlp.load_state_dict(sd["state_dict_unet_block_mlp"], strict=True)
        if sd.get("state_dict_unet_fuse_mlp") is not None:
            print('loading unet fuse mlp')
            m,u=self.unet_fuse_mlp.load_state_dict(sd["state_dict_unet_fuse_mlp"], strict=True)
        if sd.get("state_dict_vae_de_mlp") is not None:
            print('loading vae de mlp')
            m,u=self.vae_de_mlp.load_state_dict(sd["state_dict_vae_de_mlp"], strict=True)
        if sd.get("state_dict_vae_block_mlp") is not None:
            print('loading vae block mlp')
            m,u=self.vae_block_mlp.load_state_dict(sd["state_dict_vae_block_mlp"], strict=True)
        if sd.get("state_dict_vae_fuse_mlp") is not None:
            print('loading vae fuse mlp')
            m,u=self.vae_fuse_mlp.load_state_dict(sd["state_dict_vae_fuse_mlp"], strict=True)
        if sd.get("state_embeddings") is not None:
            print('loading vae block embedding')
            m,u=self.vae_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_vae_block"], strict=True)
            print('loading unet block embedding')
            m,u=self.unet_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_unet_block"], strict=True)
        if sd.get("w") is not None:
            print('loading w')
            self.W = sd["w"]
       
        if self.useprop:
            if sd.get("propagator1") is not None:
                print('loading propagator1')
                try:
                    m,u=self.propagator1.load_state_dict(sd["propagator1"], strict=True)
                except:
                    print('Failed to load propagator1, skipping...')
                    pass
            if sd.get("propagator2") is not None:
                print('loading propagator2')
                m,u=self.propagator2.load_state_dict(sd["propagator2"], strict=True)
        

    def get_trainable_paralist(self,learning_rate):
        print('---------------get_trainable_paralist----------------')
        para_to_opt_list=[]
    
        if self.useprop:
            layers_to_opt_normal=[]
            for k, _p in self.propagator1.named_parameters():
                assert _p.requires_grad
                layers_to_opt_normal.append(_p)
            for k, _p in self.propagator2.named_parameters():
                assert _p.requires_grad
                layers_to_opt_normal.append(_p)

            para_to_opt_list.append({'params': layers_to_opt_normal, 'lr':learning_rate*0.1})
            print('propagator1 and propagator2')

        layers_to_opt_normal1=[]
        layers_to_opt_normal1 = layers_to_opt_normal1 + list(self.vae_block_embeddings.parameters()) + list(self.unet_block_embeddings.parameters())
        layers_to_opt_normal1 = layers_to_opt_normal1 + list(self.vae_de_mlp.parameters()) + list(self.unet_de_mlp.parameters()) + \
                                list(self.vae_block_mlp.parameters()) + list(self.unet_block_mlp.parameters()) + \
                                list(self.vae_fuse_mlp.parameters()) + list(self.unet_fuse_mlp.parameters())
        
        print('vae and unet lora mlp')
        para_to_opt_list.append({'params': layers_to_opt_normal1, 'lr':learning_rate})

        layers_to_opt_normal2=[]
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                assert _p.requires_grad
                layers_to_opt_normal2.append(_p)
        layers_to_opt_normal2 += list(self.unet.conv_in.parameters())
        print('unet lora para')

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                assert _p.requires_grad
                layers_to_opt_normal2.append(_p)
        print('vae lora para')
        para_to_opt_list.append({'params': layers_to_opt_normal2, 'lr':learning_rate})


        return para_to_opt_list


        






