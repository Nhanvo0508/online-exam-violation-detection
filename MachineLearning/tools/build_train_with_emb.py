import os
import json
from pathlib import Path

import numpy as np
from PIL import Image

from MachineLearning.schema import BASE_FEATURE_KEYS, EMBEDDING_DIM
from MachineLearning.mobilenetv2_embeddings import (
    load_mobilenetv2,
    embed_pil_image,
)


# ============================================================
# PATH
# ============================================================

# Project:
#
# source_code_nhom06/
# ├── app/
# ├── MachineLearning/
# ├── snapshots/
# │   ├── crops/      -> LABEL 1
# │   └── normal/     -> LABEL 0
# └── main.py

PROJECT_DIR = Path(__file__).resolve().parents[2]

SNAPSHOT_DIR = PROJECT_DIR / "snapshots"

CROP_DIR = SNAPSHOT_DIR / "crops"

NORMAL_DIR = SNAPSHOT_DIR / "normal"

OUT_JSON = (
    PROJECT_DIR
    / "MachineLearning"
    / "tools"
    / "train_with_emb.json"
)

DEVICE = os.getenv(
    "ML_DEVICE",
    "cpu"
)


# ============================================================
# FEATURE CHO ẢNH NORMAL / CROP
# ============================================================

def default_features():

    return {
        "dwell": 0.0,

        "score_mean": 0.0,
        "score_std": 0.0,
        "score_max": 0.0,

        "area_mean": 0.0,
        "area_std": 0.0,

        "aspect_mean": 1.0,
        "aspect_std": 0.0,

        "cx_mean": 0.5,
        "cy_mean": 0.5,

        "motion_mean": 0.0,
        "motion_peak": 0.0,

        "blur_mean": 0.0,

        "iou_stability": 0.0,

        "mean_speed": 0.0,
    }


# ============================================================
# TẠO EMBEDDING MOBILE NET V2
# ============================================================

def make_embedding(
    model,
    image_path
):

    try:

        with Image.open(
            image_path
        ) as img:

            img = img.convert(
                "RGB"
            )

            embedding = embed_pil_image(
                img,
                model,
                device=DEVICE
            )

    except Exception as e:

        raise RuntimeError(
            f"Không thể đọc ảnh "
            f"{image_path}: {e}"
        )

    embedding = np.asarray(
        embedding,
        dtype=float
    ).reshape(-1)

    # MobileNetV2 phải cho 1280 chiều
    if len(embedding) != EMBEDDING_DIM:

        raise ValueError(
            f"Embedding sai kích thước: "
            f"{len(embedding)}, "
            f"cần {EMBEDDING_DIM}"
        )

    return embedding.tolist()


# ============================================================
# LẤY DANH SÁCH ẢNH
# ============================================================

def get_images(folder):

    if not folder.exists():

        raise FileNotFoundError(
            f"Không tìm thấy thư mục:\n{folder}"
        )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    images = sorted(

        p

        for p in folder.iterdir()

        if p.is_file()
        and p.suffix.lower()
        in valid_extensions
    )

    return images


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" BUILD TRAIN DATASET")
    print("=" * 60)

    print(
        f"[INFO] Project : {PROJECT_DIR}"
    )

    print(
        f"[INFO] Crop    : {CROP_DIR}"
    )

    print(
        f"[INFO] Normal  : {NORMAL_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Lấy ảnh
    # --------------------------------------------------------

    crop_files = get_images(
        CROP_DIR
    )

    normal_files = get_images(
        NORMAL_DIR
    )

    print(
        f"[INFO] Crop images   : "
        f"{len(crop_files)}"
    )

    print(
        f"[INFO] Normal images : "
        f"{len(normal_files)}"
    )

    if len(crop_files) == 0:

        raise RuntimeError(
            "Thư mục crops không có ảnh."
        )

    if len(normal_files) == 0:

        raise RuntimeError(
            "Thư mục normal không có ảnh."
        )

    # --------------------------------------------------------
    # Load MobileNetV2
    # --------------------------------------------------------

    print()

    print(
        f"[INFO] Loading MobileNetV2 "
        f"on {DEVICE}..."
    )

    model = load_mobilenetv2(
        device=DEVICE
    )

    print(
        "[OK] MobileNetV2 loaded."
    )

    print()

    # Dataset
    data = []

    embedding_errors = 0

    # ========================================================
    # LABEL 1
    # VI PHẠM
    # ========================================================

    print("=" * 60)
    print("LABEL 1 - VIOLATION")
    print("=" * 60)

    for index, image_path in enumerate(
        crop_files,
        start=1
    ):

        print(
            f"[1] {index}/{len(crop_files)} "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Vì ảnh nằm trong crops nên mặc định label = 1
        # ----------------------------------------------------

        features = default_features()

        try:

            embedding = make_embedding(
                model,
                image_path
            )

        except Exception as e:

            embedding_errors += 1

            print(
                f"    [WARN] Embedding lỗi: "
                f"{e}"
            )

            continue

        features["emb"] = embedding

        data.append(
            {
                "features": features,
                "label": 1
            }
        )

    # ========================================================
    # LABEL 0
    # NORMAL
    # ========================================================

    print()
    print("=" * 60)
    print("LABEL 0 - NORMAL")
    print("=" * 60)

    for index, image_path in enumerate(
        normal_files,
        start=1
    ):

        print(
            f"[0] {index}/{len(normal_files)} "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Vì ảnh nằm trong normal nên label = 0
        # ----------------------------------------------------

        features = default_features()

        try:

            embedding = make_embedding(
                model,
                image_path
            )

        except Exception as e:

            embedding_errors += 1

            print(
                f"    [WARN] Embedding lỗi: "
                f"{e}"
            )

            continue

        features["emb"] = embedding

        data.append(
            {
                "features": features,
                "label": 0
            }
        )

    # ========================================================
    # THỐNG KÊ DATASET
    # ========================================================

    label_0_count = sum(
        1
        for item in data
        if item["label"] == 0
    )

    label_1_count = sum(
        1
        for item in data
        if item["label"] == 1
    )

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Crop images          : "
        f"{len(crop_files)}"
    )

    print(
        f"Normal images        : "
        f"{len(normal_files)}"
    )

    print(
        f"Embedding errors     : "
        f"{embedding_errors}"
    )

    print(
        "-" * 40
    )

    print(
        f"Total samples        : "
        f"{len(data)}"
    )

    print(
        f"Label 0              : "
        f"{label_0_count}"
    )

    print(
        f"Label 1              : "
        f"{label_1_count}"
    )

    print()

    # ========================================================
    # KIỂM TRA
    # ========================================================

    if label_0_count == 0:

        raise RuntimeError(
            "Không có label 0. "
            "Kiểm tra thư mục normal."
        )

    if label_1_count == 0:

        raise RuntimeError(
            "Không có label 1. "
            "Kiểm tra thư mục crops."
        )

    # ========================================================
    # TẠO THƯ MỤC OUTPUT
    # ========================================================

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # GHI train_with_emb.json
    # ========================================================

    with OUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # HOÀN TẤT
    # ========================================================

    print("=" * 60)

    print(
        "[OK] Đã tạo train_with_emb.json"
    )

    print(
        f"[OK] File: {OUT_JSON}"
    )

    print("=" * 60)

    print()

    print(
        "Bây giờ có thể train ML bằng:"
    )

    print()

    print(
        "py -3.13 -m MachineLearning.pipeline_train"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()