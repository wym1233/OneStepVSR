import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import flow_warp
def forward_backward_consistency_check(fwd_flow,bwd_flow,alpha=0.01,beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow.permute(0,2,3,1))  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow.permute(0,2,3,1))  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


#for eval
from model.S3diff.flow_utils import get_flow,flow_warp
def detect_occlusion_tensor(fw_flow_t, bw_flow_t):
    """
    Detect occlusions between two frames using optical flow
    Args:
        fw_flow_t: forward flow tensor (B,2,H,W)
        bw_flow_t: backward flow tensor (B,2,H,W)
    Returns:
        occlusion: tensor mask (B,1,H,W) with 1s indicating occluded regions
    """

    fw_flow_w = flow_warp(fw_flow_t, bw_flow_t.permute(0,2,3,1))

    # Calculate flow consistency
    fb_flow_sum = fw_flow_w + bw_flow_t
    
    # Calculate flow magnitudes
    fb_flow_mag = torch.sqrt(torch.sum(fb_flow_sum**2, dim=1, keepdim=True))
    fw_flow_w_mag = torch.sqrt(torch.sum(fw_flow_w**2, dim=1, keepdim=True)) 
    bw_flow_mag = torch.sqrt(torch.sum(bw_flow_t**2, dim=1, keepdim=True))

    # Mask 1: Flow consistency check
    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    # Calculate flow gradients for motion boundary detection
    dx = torch.tensor([[.0, 0, 0], [-1, 1, 0], [0, 0, 0]]).view(1,1,3,3).to(fw_flow_t.device)
    dy = torch.tensor([[.0, -1, 0], [0, 1, 0], [0, 0, 0]]).view(1,1,3,3).to(fw_flow_t.device)
    
    fx_du = torch.nn.functional.conv2d(bw_flow_t[:,0:1], dx, padding=1)
    fx_dv = torch.nn.functional.conv2d(bw_flow_t[:,1:2], dx, padding=1)
    fy_du = torch.nn.functional.conv2d(bw_flow_t[:,0:1], dy, padding=1) 
    fy_dv = torch.nn.functional.conv2d(bw_flow_t[:,1:2], dy, padding=1)

    # Calculate gradient magnitudes
    fx_mag = fx_du**2 + fx_dv**2
    fy_mag = fy_du**2 + fy_dv**2

    # Mask 2: Motion boundary check
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    # Combine masks
    bw_occlusion = torch.logical_or(mask1, mask2).float()

    return bw_occlusion

def compute_warp_loss(flows_backward, img1, img2, backward_occ_mask):
    """
    Compute warping loss in non-occluded regions
    Args:
        flow: optical flow tensor (B,H,W,2)
        img1: first image tensor (B,C,H,W)
        img2: second image tensor (B,C,H,W) 
        occ_mask: occlusion mask tensor (B,1,H,W)
    Returns:
        warp_loss: mean warping error in non-occluded regions
    """

    # Get non-occluded mask
    noc_mask = 1 - backward_occ_mask
    
    # Warp img2 to img1 using flow
    warped_img2 = flow_warp(img2, flows_backward.permute(0,2,3,1))
    
    # Compute warping error only in non-occluded regions
    diff = (warped_img2 - img1) * noc_mask
    
    # Average over non-occluded pixels
    N = torch.sum(noc_mask)
    warp_loss = torch.sum(diff**2) / N

    return warp_loss

def compute_Ewarp(img1,img2,of_model):
    flows_forward = get_flow(of_model, img2, img1)
    flows_backward = get_flow(of_model, img1, img2)
    bwocc_mask = detect_occlusion_tensor(flows_forward, flows_backward)
    bw_warp_loss = compute_warp_loss(flows_backward, img1, img2, bwocc_mask)
    fwocc_mask = detect_occlusion_tensor(flows_backward, flows_forward)
    fw_warp_loss = compute_warp_loss(flows_forward, img2, img1, fwocc_mask)
    return fw_warp_loss*1000,bw_warp_loss*1000