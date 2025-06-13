import shutil
import os
from huggingface_hub import hf_hub_download

REPO_ID = "a-rashid/KidneyGrader"

CHECKPOINTS = {
    "checkpoints/detection/1949389_2.pt": "checkpoints/detection",
    "checkpoints/detection/1950672.pt": "checkpoints/detection",
    "checkpoints/detection/1952372.pt": "checkpoints/detection",
    "checkpoints/detection/instanseg_brightfield_monkey.pt": "checkpoints/detection",
    "checkpoints/detection/tiakong1.pth": "checkpoints/detection",
    "checkpoints/segmentation/kidney_grader_unet.pth": "checkpoints/segmentation",
}

def main():
    for remote_path, local_dir in CHECKPOINTS.items():
        filename = os.path.basename(remote_path)
        print(f"Downloading {filename} into {local_dir}...")
        os.makedirs(local_dir, exist_ok=True)

        # Download file (read-only in Hugging Face cache)
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path
        )

        target_path = os.path.join(local_dir, filename)

        # Copy instead of move to avoid cross-device errors
        shutil.copy2(downloaded_path, target_path)
        print(f"✔ Copied to {target_path}")

    print("All checkpoints downloaded and copied.")

if __name__ == "__main__":
    main()
