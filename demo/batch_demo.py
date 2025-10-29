import os
from argparse import ArgumentParser
from glob import glob

from mmdet.apis import inference_detector, init_detector
import mmrotate  # noqa: F401
from mmdet.apis import show_result_pyplot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Directory containing images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out_dir', help='Output directory for results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 构建模型
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 获取所有图像文件
    img_files = glob(os.path.join(args.img_dir, '*'))
    img_files = [f for f in img_files if os.path.isfile(f)]

    print(f"发现 {len(img_files)} 张图像")

    # 批量处理图像
    for img_path in img_files:
        # 推理
        result = inference_detector(model, img_path)

        # 获取输出文件名
        filename = os.path.basename(img_path)
        out_path = os.path.join(args.out_dir, filename)

        # 保存结果图像
        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=out_path
        )

        print(f"已保存结果到: {out_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
