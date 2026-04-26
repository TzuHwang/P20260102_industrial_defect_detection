"""
Main entry point for training and inference.

Usage:
    python -m project_src.main --yml-config configs/yamls/train_config.yaml --task train
"""

from project_src.arguments import AccessArgs
from project_src.deploy import export_onnx
from project_src.inference import inference
from project_src.train import train


def main():
    """Main function to run training or inference."""
    # Parse arguments
    arg_parser = AccessArgs()
    args = arg_parser.get_args()

    # Run task
    if args.task == 'train':
        train(args)
    elif args.task == 'inference':
        inference(args)
    elif args.task == 'deploy':
        export_onnx(args)

    else:
        raise ValueError(f"Unknown task: {args.task}. Choose from ['train', 'inference', 'deploy']")


if __name__ == '__main__':
    main()
