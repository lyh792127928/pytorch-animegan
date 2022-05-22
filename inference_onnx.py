from cv2 import transform
import torch
import cv2
import os
import numpy as np
from utils.image_processing import resize_image, normalize_input, denormalize_input,divisible
from tqdm import tqdm
from openvino.runtime import Core
import openvino.runtime as ov

VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

class Transformer_onnx:
    def __init__(self, weight='checkpoint/animegan.onnx'):
        self.core = Core()
        #weight 为对应的onnx存放位置
        self.model = self.core.compile_model(model=weight)
        self.weight = weight
        print("Weight loaded, ready to predict")

    def transform(self, image):
        '''
        Transform a image to animation

        @Arguments:
            - image: np.array, shape = (Batch, width, height, channels)

        @Returns:
            - anime version of image: np.array
        '''
        model_inference = self.core.compile_model(model=self.weight)
        input_tensor = self.preprocess_images(image)
        #方法一
        #results = model_inference.infer_new_request({0: input_tensor})
        #predictions = next(iter(results.values()))
        #pred = denormalize_input(predictions.transpose(0, 2, 3, 1)[0],dtype=np.uint8)

        #方法二
        infer_request = model_inference.create_infer_request()
        input_tensor = ov.Tensor(input_tensor)
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor()
        output_buffer = output.data
        pred = denormalize_input(output_buffer.transpose(0, 2, 3, 1)[0],dtype=np.uint8)
        return pred[:,:,::-1]

    def transform_file(self, file_path, save_path):
        if not save_path.endswith('png'):
            raise ValueError(f"{save_path} should be png format")

        image = cv2.imread(file_path)[:,:,::-1]

        if image is None:
            raise ValueError(f"Could not get image from {file_path}")

        anime_img = self.transform(resize_image(image))
        cv2.imwrite(save_path, anime_img)
        print(f"Anime image saved to {save_path}")

    def transform_in_dir(self, img_dir, dest_dir, max_images=0):
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
            anime_img = self.transform(image)
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            cv2.imwrite(os.path.join(dest_dir, f'{fname}_anime.jpg'), anime_img)

    def transform_video(self, input_path, output_path,fps_trans=5):
        
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'{input_path} does not exist')

        cap = cv2.VideoCapture(input_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        width, height = divisible((width, height))
        print(width)
        print(height)
        out = cv2.VideoWriter(output_path, fourcc, fps,(width, height))  # 写入视频
        i = 0
        while(True):
            ret, frame = cap.read()
            if(i%fps_trans==0):
                if ret == True:
                    frame = frame[:,:,::-1]
                    frame = resize_image(frame)
                    anime_img = self.transform(frame)
                    out.write(anime_img)  # 写入帧
                else:
                    break
            i = i + 1
            

        cap.release()
        out.release()
        print(f'Animation video saved to {output_path}')


    def preprocess_images(self,images):
        images = images.astype(np.float32)
        # Normalize to [-1, 1]
        images = normalize_input(images)
        # Add batch dim
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        # channel first
        images = images.transpose(0, 3, 1, 2)

        return images


    @staticmethod
    def is_valid_file(fname):
        ext = fname.split('.')[-1]
        return ext in VALID_FORMATS

        
