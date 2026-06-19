import argparse
from pathlib import Path

from utils.demo_utils import prepare_image, get_feature
from utils.misc import viewVolume, make_dir
import torch

r"""
PYTHONPATH=. python3.10 scripts/infer.py \
  --input "/Users/kayra/Downloads/work/UMC/repo/brainage/lesion_agnostic/data/preprocessed/Andras/BraTS/BraTS-GLI-02093-103-t1n.nii.gz" \
  --checkpoint "/Users/kayra/Downloads/work/UMC/repo/brainage/lesion_agnostic/exp_1/brain_id/Brain-ID/assets/brain_id_pretrained.pth" \
  --out_dir "/Users/kayra/Downloads/work/UMC/repo/brainage/lesion_agnostic/exp_1/brain_id/Brain-ID/outputs/" \
  --device cpu
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input NIfTI image")
    parser.add_argument("--checkpoint", default="assets/brain_id_pretrained.pth")
    parser.add_argument("--out_dir", default="outs/reconstruction")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = make_dir(args.out_dir)

    print("Preparing image...")
    im, aff = prepare_image(str(input_path), device=args.device)
    print("Image prepared")

    print("Extracting features...")
    with torch.inference_mode():
        outputs = get_feature(
            im,
            args.checkpoint,
            feature_only=False,
            device=args.device,
        )
    print("Features extracted")

    recon = outputs["image"]
    print("Reconstructed.")

    out_name = input_path.name
    if out_name.endswith(".nii.gz"):
        out_name = out_name.replace(".nii.gz", "_brainid_recon")
    else:
        out_name = input_path.stem + "_brainid_recon"

    viewVolume(
        recon,
        aff,
        names=[out_name],
        save_dir=out_dir,
    )

    print(f"Saved Brain-ID reconstruction to: {out_dir}")


if __name__ == "__main__":
    main()