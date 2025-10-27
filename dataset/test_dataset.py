import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torch
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow


import torchvision.transforms as transforms
import torch.nn.functional as F


class SumTestDataset(Dataset):
    def __init__(self,datasetname='REDS',frames=4):

        assert datasetname in ['REDS4','REDS30','UDM10','vimeo']
        self.datasetname=datasetname
        self.frames = frames

        if self.datasetname=='vimeo':
            self.dataset_dir = '/data/bitahub/vimeo_5/vimeo_test'
            self.LR_dataset_dir=None
            self.length_per_seq=7
            self.split=False
            self.dataset_dir_list=sorted(os.listdir(self.dataset_dir))
        elif self.datasetname=='REDS4':
            self.dataset_dir = '/data/wym123/VSRdataset/REDS/REDS4/train_sharp'
            self.LR_dataset_dir='/data/wym123/VSRdataset/REDS/REDS4/train_sharp_bicubic/X4'
            self.length_per_seq=100
            self.split=True
            self.dataset_dir_list=sorted(os.listdir(self.dataset_dir))
        elif self.datasetname=='REDS30':
            self.dataset_dir = '/data/wym123/VSRdataset/REDS/val/val_sharp/'
            self.LR_dataset_dir='/data/wym123/VSRdataset/REDS/val/val_sharp_bicubic/X4'
            self.length_per_seq=100
            self.split=True
            self.dataset_dir_list=sorted(os.listdir(self.dataset_dir))
        elif self.datasetname=='UDM10':
            self.dataset_dir = '/data/wym123/VSRdataset/UDM10/data/UDM10/GT'
            self.LR_dataset_dir='/data/wym123/VSRdataset/UDM10/data/UDM10/BIx4'
            self.length_per_seq=32
            self.split=True
            self.dataset_dir_list=sorted(os.listdir(self.dataset_dir))
        else:
            print('----------------dataet error---------------------')
        
        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
       
    def __len__(self):
        return len(self.length_per_seq*self.dataset_dir_list)

    def __getitem__(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        video_folder = self.dataset_dir_list[idx//self.length_per_seq]
        frameslist = []
        LR_frameslist=[]
        keylist=[]

        startidx=idx%self.length_per_seq
        ls=os.listdir(os.path.join(self.dataset_dir, video_folder))
        ls=sorted(ls)

        if self.datasetname=='vimeo':
            for i in range(startidx, startidx+self.frames):
                keylist.append(video_folder+'/'+ls[i])
                frameslist.append(os.path.join(self.dataset_dir, video_folder, ls[i]))
                
            img_gts = []
            for frampth in frameslist:
                img_bytes = self.file_client.get(frampth)
                img = imfrombytes(img_bytes, float32=True)
                img_gts.append(img)
          
            
            img_gts = img2tensor(img_gts)
            img_gts = torch.stack(img_gts, dim=0).clamp(0,1)
            
            T,C,H,W=img_gts.shape
            img_lqs = F.interpolate(
                        img_gts,
                        size=(H//4, W//4),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True  # 假设已手动完成抗混叠
                    ).clamp(0,1)
            img_gts = (img_gts * 2 - 1).clamp(-1, 1)
            img_lqs = (img_lqs * 2 - 1).clamp(-1, 1)
        else:
            for i in range(startidx, startidx+self.frames):
                keylist.append(video_folder+'/'+ls[i])
                frameslist.append(os.path.join(self.dataset_dir, video_folder, ls[i]))
                LR_frameslist.append(os.path.join(self.LR_dataset_dir, video_folder, ls[i]))

            img_gts = []
            for frampth in frameslist:
                img_bytes = self.file_client.get(frampth)
                img = imfrombytes(img_bytes, float32=True)
                img_gts.append(img)
            img_lqs = []
            for frampth in LR_frameslist:
                img_bytes = self.file_client.get(frampth)
                img = imfrombytes(img_bytes, float32=True)
                img_lqs.append(img)
            
            
            img_gts = img2tensor(img_gts)
            img_gts = torch.stack(img_gts, dim=0).clamp(0,1)
            img_gts = (img_gts * 2 - 1).clamp(-1, 1)
            img_lqs = img2tensor(img_lqs)
            img_lqs = torch.stack(img_lqs, dim=0).clamp(0,1)
            img_lqs = (img_lqs * 2 - 1).clamp(-1, 1)
        
        return {'lq': img_lqs, 'gt': img_gts,'key':keylist}



    