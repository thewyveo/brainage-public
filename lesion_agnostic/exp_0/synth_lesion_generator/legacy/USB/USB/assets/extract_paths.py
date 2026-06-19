import os


# === CHANGE THIS ===
INPUT_DIR = r"./results/h2p_edit"
OUTPUT_FILE = r"./assets/h2p_outputs.txt"


# ===================


def extract_paths(input_dir, output_file):
    paths = []


    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.abspath(os.path.join(root, file))
                paths.append(full_path)


    paths = sorted(paths)


    with open(output_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


    print(f"Saved {len(paths)} paths to: {output_file}")




if __name__ == "__main__":
    extract_paths(INPUT_DIR, OUTPUT_FILE)