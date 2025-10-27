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
class VimeoSeptupletTrainDataset(Dataset):
    def __init__(self,frames=4):
       
    
        dataset_dir='/data/bitahub/vimeo_5/vimeo_train'
        self.frames = frames
        
        self.dataset_dir = dataset_dir
        self.dataset_dir_list=os.listdir(dataset_dir)

        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
       
    def __len__(self):
       
        return len(self.dataset_dir_list)

    def __getitem__(self, idx):
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        video_folder = self.dataset_dir_list[idx]
        frameslist = []

        assert self.frames < 8
        startidx=random.randint(0,7-self.frames)

        ls=os.listdir(os.path.join(self.dataset_dir, video_folder))
        ls=sorted(ls)
        for i in range(startidx, startidx+self.frames):
            frame_path = os.path.join(self.dataset_dir, video_folder, ls[i])
            frameslist.append(frame_path)

        img_gts = []
        for frampth in frameslist:
            img_bytes = self.file_client.get(frampth)
            img = imfrombytes(img_bytes, float32=True)
            img_gts.append(img)
        
        img_gts = img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0).clamp(0,1)

        transform = transforms.RandomCrop(256)
        img_gts =  transform(img_gts)

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

        return {'lq': img_lqs, 'gt': img_gts}




