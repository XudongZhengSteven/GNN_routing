import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Shortcut script for test-split evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--train-cfg", default="configs/train.yaml")
    parser.add_argument("--data-cfg", default="configs/data.yaml")
    parser.add_argument("--model-cfg", default="configs/model.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        os.path.join("scripts", "evaluate.py"),
        "--checkpoint",
        args.checkpoint,
        "--split",
        "test",
        "--train-cfg",
        args.train_cfg,
        "--data-cfg",
        args.data_cfg,
        "--model-cfg",
        args.model_cfg,
    ]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
