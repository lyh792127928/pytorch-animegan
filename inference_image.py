import os
import argparse
from inference import Transformer
from inference_openvino import Transformer_openvino


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint/generate_shinkai.pth')
    parser.add_argument('--src', type=str, default='dataset/train_photo/1.jpg', help='source dir, contain real images or source image')
    parser.add_argument('--dest', type=str, default='dataset/pred_photo/1.png', help='destination dir to save generated images')
    parser.add_argument('--type', type=str, default='openvino', help='use pytorch or openvino to inference')

    return parser.parse_args()


def main(args):
    if(args.type == 'pytorch'):
        transformer = Transformer(args.checkpoint)
    if(args.type=='openvino'):
        transformer = Transformer_openvino(args.checkpoint)
    else:
        print('pleaase set type as pytorch or openvino')
        return 

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        transformer.transform_in_dir(args.src, args.dest)
    else:
        transformer.transform_file(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)
