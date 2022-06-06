from utils.common import load_weight
from modeling.anime_ganv2 import Generator
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_onnx', type=str, default='checkpoint/animegan.onnx')
    parser.add_argument('--output_ir', type=str, default='checkpoint/animegan_ir', help='destination dir to save generated ir model')

    return parser.parse_args()

def main(args):

    if os.path.exists(args.output_ir):
        print('onnx already exist')
    else:
        trans_ir_command = 'mo --input_model ' + args.input_onnx + ' --output_dir ' + args.output_ir
        os.system(trans_ir_command)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
