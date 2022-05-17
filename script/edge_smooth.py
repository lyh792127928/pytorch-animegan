# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth

import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm


def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='shinkai', help='dataset_name')
    parser.add_argument('--image_size', type=int, default=256, help='The size of image')

    return parser.parse_args()


def make_edge_smooth(data_dir,dataset_name, img_size) :
    #通过glob组件获取style（动漫文件）中所有图片
    file_list = glob((data_dir+'/{}/{}/*.*').format(dataset_name, 'style'))
    #设定保存路径
    save_dir = (data_dir + '/{}/smooth').format(dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)
        #读取彩色图片
        bgr_img = cv2.imread(f)
        #读取灰度图片，方便获取边缘
        gray_img = cv2.imread(f, 0)
        #将图片resize为规定大小
        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        #填充以确保计算边缘膨胀
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))
        #获取边缘
        edges = cv2.Canny(gray_img, 100, 200)
        #边缘膨胀
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        #进行高斯模糊
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
        #保存图片
        assert cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(args.data_dir,args.dataset, args.image_size)


if __name__ == '__main__':
    main()