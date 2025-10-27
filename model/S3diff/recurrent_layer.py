import torch
from torch import nn
import numpy as np
import sys
# p = "/code/"
# sys.path.append(p)
from . import flow_utils as of
import torch.nn.functional as F

# class OffsetDiversity(nn.Module):
#     def __init__(self, in_channel,offset_num=2, group_num=16, max_residue_magnitude=5, inplace=False):
#         super().__init__()
#         self.in_channel = in_channel
#         self.offset_num = offset_num
#         self.group_num = group_num
#         self.max_residue_magnitude = max_residue_magnitude
#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel//2, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(in_channel//2, in_channel//2, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(in_channel//2, 3 * group_num * offset_num, 3, 1, 1),
#         )
#         self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

#     def forward(self, x, aux_feature, flow):
#         B, C, H, W = x.shape
#         out = self.conv_offset(aux_feature)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         mask = torch.sigmoid(mask)
#         # offset
#         offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))

#         offset = offset + flow.permute(0,3,1,2).repeat(1, self.group_num * self.offset_num, 1, 1)

#         # warp
#         offset = offset.view(B * self.group_num * self.offset_num, 2, H, W).permute(0,2,3,1)
#         mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
#         x = x.repeat(1, self.offset_num, 1, 1)
#         x = x.view(B * self.group_num * self.offset_num, C // self.group_num, H, W)
#         x = of.flow_warp(x,offset, interp_mode = 'bilinear')
#         x = x * mask
#         x = x.view(B, C * self.offset_num, H, W)
#         x = self.fusion(x)

#         return x




def flow_interpolate(flow,target_size):
    _,h,w,c=flow[0].shape
    assert c==2
    H,W=target_size

    resizeflow=[]

    for i in range(len(flow)):
        resizeflow.append(flow[i].permute(0,3,1,2))
        
        resizeflow[i] = F.interpolate(
                    resizeflow[i],
                    size=(H,W),
                    mode='bilinear',
                    align_corners=False # align_corners with this model causes the output to be shifted, presumably due to training without align_corners
                )

        resizeflow[i]=torch.cat((resizeflow[i][:,0:1,:,:]*H/h,resizeflow[i][:,1:2,:,:]*W/w),dim=1)

        resizeflow[i]=resizeflow[i].permute(0,2,3,1)
    return resizeflow

class recurrentlayer(nn.Module):
    def __init__(self, in_channel,direction):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(2*in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        # self.sflow=OffsetDiversity(in_channel=in_channel)
        self.direction=direction

    def forward(self,x,flow):
        B, C, H, W = x.shape
        resizeflow=flow_interpolate(flow,(H,W))
        x= [x[i:i+1] for i in range(B)]
        assert len(resizeflow)==B-1
        # flow= [flow[i:i+1] for i in range(B-1)]
        out=[]
        if self.direction=='backward':
            x.reverse()

        for i in range(B):
            if i==0:
                warpfeature=torch.zeros_like(x[0])
            else:
                warpfeature = of.flow_warp(out[-1],resizeflow[i-1], interp_mode = 'bilinear')
            out.append(x[i]+self.conv_layer(torch.cat((x[i],warpfeature),dim=1)))

        if self.direction=='backward':
            out.reverse()
        out=torch.cat(out,dim=0)
        return out

class recurrentlayer_baseline(nn.Module):
    def __init__(self, in_channel,direction):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        # self.sflow=OffsetDiversity(in_channel=in_channel)
        self.direction=direction

    def forward(self,x,flow):
        B, C, H, W = x.shape
        out=x+self.conv_layer(x)
        return out

class recurrentlayer_enh1(nn.Module):
    def __init__(self, in_channel,direction):
        super().__init__()
        self.conv_layer0 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3*in_channel+2, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        self.offsetdiver=nn.Sequential(
            nn.Conv2d(3*in_channel+2, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel, 3, 3, 1, 1),
        )
        # self.sflow=OffsetDiversity(in_channel=in_channel)
        self.direction=direction
        self.offsetV=10

    def forward(self,x,flow):
        B, C, H, W = x.shape
        resizeflow=flow_interpolate(flow,(H,W))
        x= [x[i:i+1] for i in range(B)]
        assert len(resizeflow)==B-1
        # flow= [flow[i:i+1] for i in range(B-1)]
        out=[]
        if self.direction=='backward':
            x.reverse()

        for i in range(B):
            if i==0:
                out.append(x[i]+self.conv_layer0(x[i]))
                print(i)
                print(torch.max(out[-1]))
                print(torch.min(out[-1]))
            else:
                auxfea0=torch.cat((x[i],x[i-1],out[-1],resizeflow[i-1].permute(0,3,1,2)),dim=1)
                print(i)
                print(torch.max(out[-1]))
                print(torch.min(out[-1]))
                offsetmask=self.offsetdiver(auxfea0)
                
                offset=offsetmask[:,0:2,:,:].permute(0,2,3,1)*self.offsetV
                mask=torch.sigmoid(offsetmask[:,2:3,:,:])
                
                warpfeature = of.flow_warp(out[-1],offset+resizeflow[i-1], interp_mode = 'bilinear')
                auxfea1=torch.cat((x[i],warpfeature,out[-1],resizeflow[i-1].permute(0,3,1,2)),dim=1)

                out.append(x[i]+self.conv_layer(auxfea1))

        if self.direction=='backward':
            out.reverse()
        out=torch.cat(out,dim=0)
        return out

# 训练batchsize过小
# randin=torch.rand(6, 320, 64,64).to('cuda')
# net=recurrentlayer_enh1(in_channel=320,direction='backward').to('cuda')
# flow=torch.rand(5,64,64,2).to('cuda')
# flow=[flow[i:i+1] for i in range(flow.shape[0])]
# out=net(randin,flow,)
# print(out.shape)


# from thop import profile

# flops, params = profile(net, (randin,flow,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
# FLOPs = 212.76229632G
# Params = 70.805856M


# import torch
# from PIL import Image
# import torchvision
# low_res_img0= Image.open("/data/wym123/VSRdataset/REDS/REDS4/train_sharp/000/00000000.png").convert("RGB")
# low_res_img1= Image.open("/data/wym123/VSRdataset/REDS/REDS4/train_sharp/000/00000010.png").convert("RGB")
# low_res_img0 = torchvision.transforms.ToTensor()(low_res_img0).unsqueeze(0).cuda()
# low_res_img1 = torchvision.transforms.ToTensor()(low_res_img1).unsqueeze(0).cuda()

# low_res_img0=(low_res_img0*2-1).clamp(-1,1)
# low_res_img1=(low_res_img1*2-1).clamp(-1,1)
# from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
# import pipeline.util.flow_utils as of
# import torch.nn.functional as F
# import math
# of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
# of_model.requires_grad_(False)
# of_model = of_model.to('cuda')

# def compute_flows(of_model, images, rescale_factor=1):
#     resize_h, resize_w = images.shape[2:]
#     pad_h = (math.ceil(resize_h / 8)) * 8 - resize_h
#     pad_w = (math.ceil(resize_w / 8)) * 8 - resize_w
#     images = F.pad(images, pad=(0, pad_w, 0, pad_h), mode='reflect')
#     forward_flows, backward_flows = [], []
#     for i in range(1, len(images)):
#         prev_image = images[i - 1:i]
#         cur_image = images[i:i+1]
#         fflow = of.get_flow(of_model, cur_image, prev_image, rescale_factor=rescale_factor)
#         fflow=fflow[:,:resize_h,:resize_w,:]
#         bflow = of.get_flow(of_model, prev_image, cur_image, rescale_factor=rescale_factor)
#         bflow=bflow[:,:resize_h,:resize_w,:]
#         forward_flows.append(fflow)
#         backward_flows.append(bflow)
#     backward_flows.reverse()
#     return {'forward':forward_flows,'backward':backward_flows}

# catimg=torch.cat((low_res_img0,low_res_img1),dim=0)
# flow=compute_flows(of_model,catimg)

# net = nn.ModuleList([
#     recurrentlayer_baseline(in_channel=3,direction='forward'),
#     recurrentlayer_baseline(in_channel=3,direction='backward'),
#     ]).to('cuda')
# catimg=net[0](catimg,flow['forward'])
# catimg=net[1](catimg,flow['backward'])
# catimg=(catimg/2+0.5).clamp(0,1)
# from torchvision import transforms
# output_pil = transforms.ToPILImage()(catimg[0])
# output_pil.save('/code/recurrentlayer_test0.png')
# output_pil = transforms.ToPILImage()(catimg[1])
# output_pil.save('/code/recurrentlayer_test1.png')