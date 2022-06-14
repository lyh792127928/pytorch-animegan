import argparse
from inference import Transformer
from inference_openvino import Transformer_openvino


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint/generator_shinkai.pth')
    parser.add_argument('--src', type=str, default='dataset/video/city.gif', help='Path to input video')
    parser.add_argument('--dest', type=str, default='dataset/video/1.mp4', help='Path to save new video')
    parser.add_argument('--fps_trans', type=int, default=5, help='多少帧转化一次')
    parser.add_argument('--start', type=int, default=0, help='Start time of video (second)')
    parser.add_argument('--end', type=int, default=0, help='End time of video (second), 0 if not set')
    parser.add_argument('--type', type=str, default='onnx', help='use pytorch or openvino to inference')

    return parser.parse_args()


def main(args):
    if(args.type == 'pytorch'):
        transformer = Transformer(args.checkpoint)
    else :
        if(args.type=='openvino'):
            transformer = Transformer_openvino(args.checkpoint)
        else:
            print('pleaase set type as pytorch or openvino')
            return 

    transformer.transform_video(args.src, args.dest,args.fps_trans)

if __name__ == '__main__':
    args = parse_args()
    main(args)
