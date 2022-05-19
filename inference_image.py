import os
import argparse
from inference import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--src', type=str, default='dataset/train_photo/1.jpg', help='source dir, contain real images or source image')
    parser.add_argument('--dest', type=str, default='dataset/pred_photo/1.png', help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    transformer = Transformer(args.checkpoint)

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        transformer.transform_in_dir(args.src, args.dest)
    else:
        transformer.transform_file(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)
