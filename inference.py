from cv2 import transform
import torch
import cv2
import os
import numpy as np
from modeling.anime_ganv2 import Generator
from utils.common import load_weight
from utils.image_processing import resize_image, normalize_input, denormalize_input,divisible
from utils import read_image
from tqdm import tqdm


cuda_available = torch.cuda.is_available()

VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

class Transformer:
    def __init__(self, weight='checkpoint/generate_shinkai.pth'):
        #weight 为对应的checkpoint存放位置
        self.G = Generator()

        if cuda_available:
            self.G = self.G.cuda()

        load_weight(self.G, weight)
        self.G.eval()

        print("Weight loaded, ready to predict")

    def transform(self, image):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        with torch.no_grad():
            fake = self.G(self.preprocess_images(image))
            fake = fake.detach().cpu().numpy()
            # Channel last
            fake = fake.transpose(0, 2, 3, 1)
            return fake

    def transform_file(self, file_path, save_path):
        if not save_path.endswith('png'):
            raise ValueError(f"{save_path} should be png format")

        image = read_image(file_path)

        if image is None:
            raise ValueError(f"Could not get image from {file_path}")

        anime_img = self.transform(resize_image(image))[0]
        anime_img = denormalize_input(anime_img, dtype=np.int16)
        cv2.imwrite(save_path, anime_img[..., ::-1])
        print(f"Anime image saved to {save_path}")

    def transform_in_dir(self, img_dir, dest_dir, max_images=0, img_size=(512, 512)):
        
        #Read all images from img_dir, transform and write the result to dest_dir

        
        os.makedirs(dest_dir, exist_ok=True)

        files = os.listdir(img_dir)
        files = [f for f in files if self.is_valid_file(f)]
        print(f'Found {len(files)} images in {img_dir}')

        if max_images:
            files = files[:max_images]

        for fname in tqdm(files):
            image = cv2.imread(os.path.join(img_dir, fname))[:,:,::-1]
            image = resize_image(image)
            anime_img = self.transform(image)[0]
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            anime_img = denormalize_input(anime_img, dtype=np.int16)
            cv2.imwrite(os.path.join(dest_dir, f'{fname}_anime.jpg'), anime_img[..., ::-1])

    def transform_video(self, input_path, output_path,fps_trans=5):
        
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        cap = cv2.VideoCapture(input_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        width, height = divisible((width, height))
        out = cv2.VideoWriter(output_path, fourcc, fps,(width, height))  # 写入视频
        i = 0
        while(True):
            ret, frame = cap.read()
            if(i%fps_trans==0):
                if ret == True:
                    frame = frame[:,:,::-1]
                    frame = resize_image(frame)
                    img = self.transform(frame)[0]
                    anime_img = denormalize_input(img, dtype=np.uint8)
                    out.write(anime_img[:,:,::-1])  # 写入帧
                else:
                    break
            i = i + 1
            
        cap.release()
        out.release()
        print(f'Animation video saved to {output_path}')


    def preprocess_images(self, images):
        '''
        Preprocess image for inference

        @Arguments:
            - images: np.ndarray

        @Returns
            - images: torch.tensor
        '''
        images = images.astype(np.float32)

        # Normalize to [-1, 1]
        images = normalize_input(images)
        images = torch.from_numpy(images)

        if cuda_available:
            images = images.cuda()

        # Add batch dim
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # channel first
        images = images.permute(0, 3, 1, 2)

        return images


    @staticmethod
    def is_valid_file(fname):
        ext = fname.split('.')[-1]
        return ext in VALID_FORMATS

        
