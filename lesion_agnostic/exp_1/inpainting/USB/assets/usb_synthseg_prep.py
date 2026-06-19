#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import logging
import subprocess




def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")




def run_cmd(cmd):
    logging.info("CMD: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)




def find_images(input_dir: Path, name_filter: str | None = None):
    files = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".nii.gz") or name.endswith(".nii")):
            continue
        if name_filter and name_filter.lower() not in name:
            continue
        files.append(p)
    return sorted(files)




def main():
    setup_logging()


    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-filter", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docker-image", type=str, default="freesurfer/synthseg:latest")
    args = parser.parse_args()


    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    images = find_images(args.input_dir, args.name_filter)
    if args.limit is not None:
        images = images[:args.limit]


    if not images:
        raise RuntimeError(f"No NIfTI images found in {args.input_dir}")


    logging.info("Found %d images", len(images))


    for i, img in enumerate(images, start=1):
        rel = img.relative_to(args.input_dir)
        out = args.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)


        if out.exists() and not args.overwrite:
            logging.info("[%d/%d] Skipping existing: %s", i, len(images), out.name)
            continue


        logging.info("[%d/%d] SynthSeg: %s", i, len(images), img.name)


        cmd = [
            "docker", "run", "--rm",
            "-v", f"{args.input_dir}:/input",
            "-v", f"{args.output_dir}:/output",
            args.docker_image,
            "mri_synthseg",
            "--i", f"/input/{rel.as_posix()}",
            "--o", f"/output/{rel.as_posix()}",
            "--robust",
        ]

        run_cmd(cmd)


    logging.info("Done.")




if __name__ == "__main__":
    main()


