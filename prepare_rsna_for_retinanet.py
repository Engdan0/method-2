import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm


def dicom_to_uint8(path):
    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image -= image.min()
    max_value = image.max()
    if max_value > 0:
        image /= max_value
    return (image * 255).astype(np.uint8)


def write_annotations(rows, patient_ids, image_dir, csv_path):
    patient_set = set(patient_ids)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for patient_id, group in rows.groupby("patientId", sort=False):
            if patient_id not in patient_set:
                continue

            image_path = (image_dir / f"{patient_id}.png").as_posix()
            positives = group[group["Target"] == 1]

            if positives.empty:
                writer.writerow([image_path, "", "", "", "", ""])
                continue

            for _, row in positives.iterrows():
                x1 = int(row["x"])
                y1 = int(row["y"])
                x2 = int(row["x"] + row["width"])
                y2 = int(row["y"] + row["height"])
                writer.writerow([image_path, x1, y1, x2, y2, "pneumonia"])


def main():
    parser = argparse.ArgumentParser(description="Prepare RSNA Pneumonia data for pytorch-retinanet CSV training.")
    parser.add_argument("--dicom_dir", required=True, help="Directory with RSNA train DICOM files.")
    parser.add_argument("--labels", required=True, help="stage_2_train_labels.csv path.")
    parser.add_argument("--output_dir", default="rsna_retinanet", help="Output directory.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation patient fraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=None, help="Optional limit for quick tests.")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_path)
    patient_ids = sorted(labels["patientId"].unique())

    if args.max_images is not None:
        patient_ids = patient_ids[: args.max_images]
        labels = labels[labels["patientId"].isin(patient_ids)]

    for patient_id in tqdm(patient_ids, desc="Converting DICOM to PNG"):
        src = dicom_dir / f"{patient_id}.dcm"
        dst = image_dir / f"{patient_id}.png"
        if dst.exists():
            continue
        Image.fromarray(dicom_to_uint8(src)).save(dst)

    rng = random.Random(args.seed)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_fraction))
    val_ids = set(shuffled[:val_count])
    train_ids = [patient_id for patient_id in shuffled if patient_id not in val_ids]

    with open(output_dir / "classes.csv", "w", newline="") as handle:
        csv.writer(handle).writerow(["pneumonia", 0])

    write_annotations(labels, train_ids, image_dir, output_dir / "train.csv")
    write_annotations(labels, val_ids, image_dir, output_dir / "val.csv")

    print(f"Images: {len(patient_ids)}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
