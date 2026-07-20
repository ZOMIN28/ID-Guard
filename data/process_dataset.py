import os
import cv2
import numpy as np
from tqdm import tqdm

# Facial parts to be merged
TARGET_PARTS = {
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
}


def process_folder(folder_path, folder_idx, output_dir):
    """
    Merge masks in one annotation folder and save the results.

    Args:
        folder_path: Path of the current annotation folder.
        folder_idx: Folder index (0~14).
        output_dir: Directory to save merged masks.
    """

    print(f"Processing folder {folder_idx}...")

    # Store merged masks for the current folder only.
    merged_masks = {}

    # Traverse all mask files in the current folder.
    for filename in tqdm(os.listdir(folder_path)):

        if not filename.endswith(".png"):
            continue

        stem = filename[:-4]

        # Split only once because part names contain '_'
        try:
            img_id_str, part_name = stem.split("_", 1)
        except ValueError:
            continue

        if part_name not in TARGET_PARTS:
            continue

        img_id = int(img_id_str)

        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        binary_mask = mask > 0

        if img_id not in merged_masks:
            merged_masks[img_id] = binary_mask
        else:
            merged_masks[img_id] |= binary_mask

    # Save all images belonging to the current folder.
    start_id = folder_idx * 2000
    end_id = min(start_id + 2000, 30000)

    for img_id in range(start_id, end_id):

        if img_id in merged_masks:
            output_mask = merged_masks[img_id].astype(np.uint8) * 255
        else:
            # Create an empty mask if no target parts exist.
            output_mask = np.zeros((512, 512), dtype=np.uint8)

        save_path = os.path.join(output_dir, f"{img_id}.jpg")
        cv2.imwrite(save_path, output_mask)

    print(f"Folder {folder_idx} finished.")


def merge_masks(mask_root, output_dir):
    """
    Merge selected facial-part masks for the entire CelebAMask-HQ dataset.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Process folders in numerical order.
    folder_names = sorted(
        [f for f in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, f))],
        key=lambda x: int(x)
    )

    for folder_name in folder_names:

        folder_idx = int(folder_name)
        folder_path = os.path.join(mask_root, folder_name)

        process_folder(
            folder_path=folder_path,
            folder_idx=folder_idx,
            output_dir=output_dir,
        )

    print("Done.")


def main():

    mask_root = "dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    output_dir = "dataset/CelebAMask-HQ/mask_images"

    merge_masks(mask_root, output_dir)


if __name__ == "__main__":
    main()