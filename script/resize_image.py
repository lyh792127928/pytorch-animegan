import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm


def parse_args():
    desc = "resize image"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--dataset', type=str, default='anime_face', help='dataset_name')
    parser.add_argument('--image_size', type=int, default=64, help='The size of image')

    return parser.parse_args()


def resize_image(data_dir,dataset_name, img_size) :
    #通过glob组件获取所有图片
    file_list = glob((data_dir+'/{}/{}/*.*').format(dataset_name, 'origin'))
    #设定保存路径
    save_dir = (data_dir + '/{}/style').format(dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)
        #读取彩色图片
        bgr_img = cv2.imread(f)
        #将图片resize为规定大小
        bgr_img = cv2.resize(bgr_img, (img_size, img_size))

        #保存图片
        assert cv2.imwrite(os.path.join(save_dir, file_name), bgr_img)


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    resize_image(args.data_dir,args.dataset, args.image_size)


if __name__ == '__main__':
    main()