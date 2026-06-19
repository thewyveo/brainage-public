#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import random




def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")




def collect_nifti_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")


    return [p.resolve() for p in folder.iterdir() if p.is_file() and is_nifti(p)]




def write_txt(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")




def main():
    # reproducibility (change or remove if you want new randomness every run)
    random.seed(42)


    repo_root = Path(__file__).resolve().parents[4]
    print(repo_root)


    usb_preprocessed_dir = repo_root / "data" / "preprocessed" / "USB"
    healthy_dir = usb_preprocessed_dir / "healthy"
    masks_dir = usb_preprocessed_dir / "masks"


    usb_repo_dir = repo_root / "exp_0" / "synth_lesion_generator" / "USB" / "USB"
    assets_dir = usb_repo_dir / "assets"


    healthy_txt = assets_dir / "test_healthy.txt"
    mask_txt = assets_dir / "test_mask.txt"


    healthy_files = collect_nifti_files(healthy_dir)
    mask_files = collect_nifti_files(masks_dir)


    n_healthy = len(healthy_files)
    n_masks = len(mask_files)
    n = min(n_healthy, n_masks)


    if n == 0:
        raise RuntimeError(
            f"No files found.\nHealthy: {n_healthy}, Masks: {n_masks}"
        )


    # 🔥 RANDOM SHUFFLE
    random.shuffle(healthy_files)
    random.shuffle(mask_files)


    # take first n after shuffle
    selected_healthy = healthy_files[:n]
    selected_masks = mask_files[:n]


    # write txt files
    write_txt(selected_healthy, healthy_txt)
    write_txt(selected_masks, mask_txt)


    print("Done.")
    print(f"Healthy count: {n_healthy}")
    print(f"Mask count:    {n_masks}")
    print(f"Paired (random) n = {n}")


    print(f"\nSaved:\n  {healthy_txt}\n  {mask_txt}")


    print("\nExample pairs:")
    for i in range(min(5, n)):
        print(f"{i+1}")
        print(f"  H: {selected_healthy[i].name}")
        print(f"  M: {selected_masks[i].name}")




if __name__ == "__main__":
    main()
