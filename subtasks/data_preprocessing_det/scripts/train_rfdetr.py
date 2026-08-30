"""Fine-tune RF-DETR on the tape-measure detection dataset (COCO format).

RF-DETR (https://github.com/roboflow/rf-detr) ships its own self-contained
training loop (DINOv2 backbone + Hungarian-matching set loss) that doesn't
decompose into this repo's LitModel/AssignmentFactory/LossFactory framework, so
this experiment runs as an independent track via the official `rfdetr` package
and its pretrained checkpoints, on a dataset converted to COCO format by
convert_to_coco.py.

Usage:
    python -m subtasks.data_preprocessing_det.scripts.train_rfdetr \
        --dataset-dir data/internal_train/rfdetr_coco_front \
        --output-dir outputs/rfdetr_medium_front \
        --epochs 40 --batch-size 4 --grad-accum-steps 4
"""

import argparse
import json

from rfdetr import RFDETRMedium


def _class_names(dataset_dir):
    coco = json.load(open(f'{dataset_dir}/train/_annotations.coco.json', encoding='utf-8'))
    return [c['name'] for c in sorted(coco['categories'], key=lambda c: c['id'])]


def main(args):
    model = RFDETRMedium()
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_drop=args.lr_drop,
        resume=args.resume,
        class_names=_class_names(args.dataset_dir),
        run_test=True,
        tensorboard=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    # batch_size=4, grad_accum_steps=4 -> effective batch 16, sized for a 16 GB GPU
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-drop', type=int, default=100, help='epoch at which LR is decayed by 0.1x')
    parser.add_argument('--resume', default=None, help='path to a checkpoint.pth to resume training from')
    main(parser.parse_args())
