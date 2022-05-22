from utils.common import load_weight
from modeling.anime_ganv2 import Generator
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint/generate_shinkai.pth')
    parser.add_argument('--output_onnx', type=str, default='checkpoint/animegan.onnx', help='destination dir to save generated onnx model')

    return parser.parse_args()

def main(args):

    if os.path.exists(args.output_onnx):
        print('onnx already exist')
    else:
        torch_model = Generator()
        load_weight(torch_model,args.checkpoint)
        output_onnx = args.output_onnx
        print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
        input_names = ["input0"]
        output_names = ["output0"]
        # 转ONNX 动态输入转换代码
        #数字0，1等是指张量的维度，表示哪个维度需要动态输入,设置动态输入后可以设定为输入图片尺寸
        dynamic_axes= {'input0':[0, 2, 3], 'output0':[0,2,3]} 
        inputs = torch.randn(1, 3, 256, 256)
        torch.onnx.export(torch_model, inputs, output_onnx,input_names=input_names, output_names=output_names,opset_version=11, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    args = parse_args()
    main(args)
