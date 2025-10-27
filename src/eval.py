import os
os.environ['TORCH_HOME']='/data/wym123/VSRdataset/torchhome/'

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pyiqa
from DISTS_pytorch import DISTS

from torchvision.models.optical_flow import raft_large as raft

import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop
import sys
sys.path.append('/code/')
from model.S3diff.flow_utils import get_flow,flow_warp
from model.S3diff.ewarp import compute_Ewarp
from DOVERmaster.evaluate_one_video import CalculateDOVER
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch.nn.functional as F
def evalREDS4rec(rec_path, gt_path='/data/wym123/VSRdataset/REDS/REDS4/train_sharp/'):
    print(f'----------------eval------------------')
    print(f'rec:{rec_path}')
    print(f'gt:{gt_path}')
    lable = rec_path.split('/')[-1]
    print(f'lable: {lable}')

    seqs = sorted(os.listdir(rec_path))

    device = torch.device('cuda')
    of_model = raft(pretrained=True).to(device)
    lpips = LPIPS(normalize=True).to(device)
    dists = DISTS().to(device)
    psnr = PSNR(data_range=1).to(device)
    ssim = SSIM(data_range=1).to(device)
    musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
    niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
    clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)#
    nrqm = pyiqa.create_metric('nrqm', device='cuda', as_loss=False)

    lpips_dict = {}
    psnr_dict = {}
    ssim_dict = {}
    dists_dict = {}
    musiq_dict = {}
    niqe_dict = {}
    clip_dict = {}
    nrqm_dict = {}
    tlpips_dict = {}
    tof_dict = {}
    fw_Ewarp_dict = {}
    bw_Ewarp_dict = {}
    tt = ToTensor()

    total = 0
    for root, dirs, files in os.walk(gt_path):
        total += len(files)

    pbar = tqdm(total=total, ncols=100)


    for seq in seqs:
        if seq == 'mp4dir':
            continue

        ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))
        ims_gt = sorted(os.listdir(os.path.join(gt_path, seq)))
        
        lpips_dict[seq] = []
        psnr_dict[seq] = []
        ssim_dict[seq] = []
        dists_dict[seq] = []
        musiq_dict[seq] = []
        niqe_dict[seq] = []
        clip_dict[seq] = []
        nrqm_dict[seq] = []
        tlpips_dict[seq] = []
        tof_dict[seq] = []
        fw_Ewarp_dict[seq] = []
        bw_Ewarp_dict[seq] = []

        

        for i, (im_rec, im_gt) in enumerate(zip(ims_rec, ims_gt)):
            with torch.no_grad():
                
                gt = Image.open(os.path.join(gt_path, seq, im_gt))
                rec = Image.open(os.path.join(rec_path, seq, im_rec))
                gt = tt(gt).unsqueeze(0).to(device)
                rec = tt(rec).unsqueeze(0).to(device)
                # rec = F.interpolate(
                #         rec,
                #         size=(720, 1272),  # 目标尺寸（高度, 宽度）
                #         mode='bicubic',     # 指定插值模式为bicubic
                #         align_corners=False # 推荐保持False（PyTorch官方建议非对齐模式）
                #     ).clamp(0,1)

                psnr_value = psnr(gt, rec)
                ssim_value = ssim(gt, rec)
                lpips_value = lpips(gt, rec)
                dists_value = dists(gt, rec)
                musiq_value = musiq(rec)
                niqe_value = niqe(rec)
                clip_value = clip(rec)
                nrqm_value = nrqm(rec)
                if i > 0:
                    

                    # tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
                    # if not i%4==0:
                    #     tlpips_dict[seq].append(tlpips_value.item())
                    # tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
                    # if not i%4==0:
                    #     tof_dict[seq].append(tof_value.item())
                    # fw_Ewarp, bw_Ewarp = compute_Ewarp(prev_rec, rec, of_model)
                    # if not i%4==0:
                    #     fw_Ewarp_dict[seq].append(fw_Ewarp.item())
                    #     bw_Ewarp_dict[seq].append(bw_Ewarp.item())

                    tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
                    tlpips_dict[seq].append(tlpips_value.item())
                    tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
                    tof_dict[seq].append(tof_value.item())
                    fw_Ewarp, bw_Ewarp = compute_Ewarp(prev_rec, rec, of_model)
                    fw_Ewarp_dict[seq].append(fw_Ewarp.item())
                    bw_Ewarp_dict[seq].append(bw_Ewarp.item())
                    # print(f'fw_Ewarp: {fw_Ewarp}, bw_Ewarp: {bw_Ewarp}')

            psnr_dict[seq].append(psnr_value.item())
            ssim_dict[seq].append(ssim_value.item())
            lpips_dict[seq].append(lpips_value.item())
            dists_dict[seq].append(dists_value.item())
            musiq_dict[seq].append(musiq_value.item())
            niqe_dict[seq].append(niqe_value.item())
            clip_dict[seq].append(clip_value.item())
            nrqm_dict[seq].append(nrqm_value.item())

            



            prev_rec = rec
            prev_gt = gt
            pbar.update()
            
    pbar.close()
    DOVER=[]
    mp4videodir = os.path.join(rec_path, 'mp4dir')
    if os.path.exists(mp4videodir):
        for mp4clip in os.listdir(mp4videodir):
            mp4clippth=os.path.join(mp4videodir,mp4clip)
            print(f'Processing {mp4clippth}')
            DOVER.append(CalculateDOVER(mp4clippth))
            print(f'DOVER: {DOVER}')
        aveDOVER = np.mean(DOVER)
        print(f'aveDOVER: {aveDOVER}')
        


    # Calculate means
    mean_lpips = np.round(np.mean([np.mean(lpips_dict[key]) for key in lpips_dict.keys()]), 3)
    mean_dists = np.round(np.mean([np.mean(dists_dict[key]) for key in dists_dict.keys()]), 3)
    mean_psnr = np.round(np.mean([np.mean(psnr_dict[key]) for key in psnr_dict.keys()]), 3)
    mean_ssim = np.round(np.mean([np.mean(ssim_dict[key]) for key in ssim_dict.keys()]), 3)
    mean_niqe = np.round(np.mean([np.mean(niqe_dict[key]) for key in niqe_dict.keys()]), 3)
    mean_clip = np.round(np.mean([np.mean(clip_dict[key]) for key in clip_dict.keys()]), 3)
    mean_nrqm = np.round(np.mean([np.mean(nrqm_dict[key]) for key in nrqm_dict.keys()]), 3)
    mean_musiq = np.round(np.mean([np.mean(musiq_dict[key]) for key in musiq_dict.keys()]), 3)
    mean_tlpips = np.round(np.mean([np.mean(tlpips_dict[key]) for key in tlpips_dict.keys()]) * 1e3, 3)
    mean_tof = np.round(np.mean([np.mean(tof_dict[key]) for key in tof_dict.keys()]) * 1e1, 3)
    mean_fw_Ewarp = np.round(np.mean([np.mean(fw_Ewarp_dict[key]) for key in fw_Ewarp_dict.keys()]) , 3)
    mean_bw_Ewarp = np.round(np.mean([np.mean(bw_Ewarp_dict[key]) for key in bw_Ewarp_dict.keys()]) , 3)
    mean_DOVER = np.round(np.mean(DOVER), 3)
    print('Group1:')

    print(f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}')
    print('|'+lable+f'|{mean_psnr}|{mean_ssim}|{mean_lpips}|{mean_dists}|')

    print('Group2:')

    print(f'CLIP: {mean_clip}, MUSIQ:{mean_musiq},NIQE: {mean_niqe} ,NRQM: {mean_nrqm},DOVER:{mean_DOVER}')
    print('|'+lable+f'|{mean_clip}|{mean_musiq}|{mean_niqe}|{mean_nrqm}|{mean_DOVER}|')

    print('Group3:')

    print(f'tLPIPS: {mean_tlpips}, tOF: {mean_tof}, fw_Ewarp: {mean_fw_Ewarp}, bw_Ewarp: {mean_bw_Ewarp}')
    print('|'+lable+f'|{mean_tlpips}|{mean_tof}|{mean_fw_Ewarp}|{mean_bw_Ewarp}|')


# evalREDS4rec('/data/wym123/VSRdataset/newinference_out3/stableVSR_UDM10','/data/wym123/VSRdataset/UDM10/data/UDM10/GT')
if __name__ == '__main__':
    evalREDS4rec('/data/wym123/VSRdataset/inference_out_1/result1','/data/wym123/VSRdataset/REDS/REDS4/train_sharp')
# evalREDS4rec('/data/wym123/VSRdataset/newinference_out3/prop_3DUnet_degreationLora_vimeo','/data/bitahub/vimeo_5/vimeo_test')




