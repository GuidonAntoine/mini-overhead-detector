import random
import shutil
from pathlib import Path

from PIL import Image

CLASS_MAP = {
    1: "airplane",
    2: "ship",
    3: "storage tank",
    4: "baseball diamond",
    5: "tennis court",
    6: "basketball court",
    7: "ground track field",
    8: "harbor",
    9: "bridge",
    10: "vehicle",
}


def parse_annotation_line(line):
    """
    Parse a NWPU annotation line.

    Example:
        "(208,361),(272,418),1"
    """
    line = line.strip()
    line = line.replace("(", "").replace(")", "")
    x1, y1, x2, y2, class_id = map(int, line.split(","))
    return x1, y1, x2, y2, class_id


def parse_annotation_file(file_path):
    annotations = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            annotations.append(parse_annotation_line(line))

    return annotations


def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert absolute corner coordinates to normalized YOLO format:
    x_center, y_center, width, height.
    """
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    x_center = x1 + bbox_w / 2
    y_center = y1 + bbox_h / 2

    return (
        x_center / img_w,
        y_center / img_h,
        bbox_w / img_w,
        bbox_h / img_h,
    )


def split_image_ids(image_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    image_ids = list(image_ids)
    if not image_ids:
        raise ValueError("image_ids must not be empty")

    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n_images = len(image_ids)
    train_end = int(n_images * train_ratio)
    val_end = train_end + int(n_images * val_ratio)

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]

    return train_ids, val_ids, test_ids


def ensure_output_dirs(output_root):
    """
    Create the standard YOLO directory structure.
    """
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yolo_label_file(output_path, annotations, img_w, img_h):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for x1, y1, x2, y2, class_id in annotations:
        if class_id not in CLASS_MAP:
            raise ValueError(f"Unknown class_id {class_id} in annotation")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {(x1, y1, x2, y2)}")

        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1, y1, x2, y2, img_w, img_h
        )
        yolo_class_id = class_id - 1

        lines.append(
            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def write_dataset_yaml(output_root):
    class_names = [CLASS_MAP[class_id] for class_id in sorted(CLASS_MAP)]

    yaml_content = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            f"nc: {len(class_names)}",
            f"names: {class_names}",
            "",
        ]
    )

    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(yaml_content, encoding="utf-8")


def get_split_name(image_id, train_ids, val_ids, test_ids):
    if image_id in train_ids:
        return "train"
    if image_id in val_ids:
        return "val"
    if image_id in test_ids:
        return "test"
    raise ValueError(f"Image ID {image_id} was not assigned to any split")


def main():
    dataset_root = Path("data/interim/nwpu-vhr-10/dataset")
    positive_dir = dataset_root / "positive_image_set"
    ground_truth_dir = dataset_root / "ground_truth"
    output_root = Path("data/processed/nwpu_yolo")

    positive_images = sorted(positive_dir.glob("*.jpg"))
    image_ids = [image_path.stem for image_path in positive_images]

    train_ids, val_ids, test_ids = split_image_ids(image_ids)
    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    ensure_output_dirs(output_root)

    for image_path in positive_images:
        image_id = image_path.stem
        split_name = get_split_name(image_id, train_ids, val_ids, test_ids)

        annotation_path = ground_truth_dir / f"{image_id}.txt"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file for {image_id}")

        with Image.open(image_path) as image:
            img_w, img_h = image.size

        annotations = parse_annotation_file(annotation_path)

        output_image_path = output_root / "images" / split_name / image_path.name
        output_label_path = output_root / "labels" / split_name / f"{image_id}.txt"

        shutil.copy2(image_path, output_image_path)
        write_yolo_label_file(output_label_path, annotations, img_w, img_h)

    write_dataset_yaml(output_root)

    print("Conversion completed.")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
