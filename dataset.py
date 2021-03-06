import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.image_processing import normalize_input, compute_data_mean

class AnimeDataSet(Dataset):
    def __init__(self, args, transform=None):
        """   
        folder structure:
            - {data_dir}
                - train_photo #来自cycleGAN
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g shinkai hayao
                    smooth #通过script/edge_smooth.py实现
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """
        data_dir = args.data_dir
        dataset = args.dataset
        train_photo_path = args.train_photo_path

        #获取动漫dataset的路径
        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        #计算style中文件夹中所有图片的均值，用BGR格式
        self.mean = compute_data_mean(os.path.join(anime_dir, 'style'))
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.data_dir = data_dir
        self.image_files =  {}
        self.photo = f'{data_dir}/'+train_photo_path
        self.style = f'{anime_dir}/style'
        self.smooth =  f'{anime_dir}/smooth'
        self.dummy = torch.zeros(3, 256, 256)

        for opt in [self.photo, self.style, self.smooth]:
            folder = os.path.join(opt)
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return image, anime, anime_gray, smooth_gray

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)[:,:,::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)[:,:,::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return normalize_input(img)
