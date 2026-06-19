#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils.test_utils as utils
from utils.misc import viewVolume, make_dir


def run_case(case_dir, save_dir, ckp_path, model_cfg, gen_cfg, device):
    img_path = case_dir / "input.nii.gz"
    flip_path = case_dir / "input_flip_reg2orig.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing {img_path}")
    if not flip_path.exists():
        raise FileNotFoundError(f"Missing {flip_path}")

    _, img, _, aff = utils.prepare_image(
        str(img_path),
        win_size=[160, 160, 160],
        im_only=True,
        device=device,
    )

    _, img_flip_reg2orig, _, _ = utils.prepare_image(
        str(flip_path),
        win_size=[160, 160, 160],
        spacing=None,
        im_only=True,
        device=device,
    )

    outs = utils.evaluate_image(
        img,
        img_flip_reg2orig,
        ckp_path=str(ckp_path),
        device=device,
        gen_cfg=str(gen_cfg),
        model_cfg=str(model_cfg),
    )

    case_save_dir = make_dir(str(save_dir / case_dir.name), reset=False)

    for k, v in outs.items():
        if "feat" not in k and k != "segmentation" and isinstance(v, torch.Tensor):
            viewVolume(v, aff, names=["out_" + k], save_dir=case_save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ckp-path", type=Path, required=True)
    parser.add_argument("--model-cfg", type=str, default="test.yaml")
    parser.add_argument("--gen-cfg", type=str, default="test.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    cases = sorted([p for p in args.input_root.iterdir() if p.is_dir()])
    if args.limit is not None:
        cases = cases[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Cases: {len(cases)}")

    for i, case_dir in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case_dir.name}")
        run_case(
            case_dir=case_dir,
            save_dir=args.output_root,
            ckp_path=args.ckp_path,
            model_cfg=args.model_cfg,
            gen_cfg=args.gen_cfg,
            device=device,
        )

    print("DONE")


if __name__ == "__main__":
    main()
