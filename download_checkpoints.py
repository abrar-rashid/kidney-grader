from huggingface_hub import hf_hub_download
import os
import shutil

REPO_ID = "ar2221/KidneyGrader"

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

        # Download to temp location
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
        )

        # Move file into correct local directory
        target_path = os.path.join(local_dir, filename)
        shutil.move(downloaded_path, target_path)
    print("All checkpoints downloaded.")

if __name__ == "__main__":
    main()
