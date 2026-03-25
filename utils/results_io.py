from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from PIL import Image, ImageColor
from rasterio.merge import merge


RESULTS_PATH = Path("results")
NODATA_VALUE = 255


MUNICH_COLORS = [
    (1,  "Beet",       ImageColor.getcolor("#2B2B2B", "RGB")),  # dark neutral
    (2,  "Oat",        ImageColor.getcolor("#6E6A61", "RGB")),  # muted warm gray
    (3,  "Meadow",     ImageColor.getcolor("#1F78B4", "RGB")),  # clear blue
    (4,  "Rape",       ImageColor.getcolor("#FFD92F", "RGB")),  # bright yellow
    (5,  "Hop",        ImageColor.getcolor("#1B9E77", "RGB")),  # saturated green
    (6,  "Spelt",      ImageColor.getcolor("#E5D8BD", "RGB")),  # pale straw
    (7,  "Triticale",  ImageColor.getcolor("#A6761D", "RGB")),  # ochre brown
    (8,  "Beans",      ImageColor.getcolor("#D73027", "RGB")),  # strong red
    (9,  "Peas",       ImageColor.getcolor("#762A83", "RGB")),  # purple
    (10, "Potato",     ImageColor.getcolor("#8C510A", "RGB")),  # earthy brown
    (11, "Soybeans",   ImageColor.getcolor("#66A61E", "RGB")),  # olive green
    (12, "Asparagus",  ImageColor.getcolor("#66C2A5", "RGB")),  # fresh mint
    (13, "Wheat",      ImageColor.getcolor("#E6AB02", "RGB")),  # golden wheat
    (14, "W. Barley",  ImageColor.getcolor("#FC8D62", "RGB")),  # warm apricot
    (15, "Rye",        ImageColor.getcolor("#B2DF8A", "RGB")),  # light green
    (16, "S. Barley",  ImageColor.getcolor("#F46D43", "RGB")),  # orange-red
    (17, "Maize",      ImageColor.getcolor("#D95F02", "RGB")),  # vivid orange
]

LOMBARDIA_COLORS = [
    (1, "Cereals",   ImageColor.getcolor("#E6AB02", "RGB")),  # same as wheat
    (2, "Woods",     ImageColor.getcolor("#1B9E77", "RGB")),  # same as hop/forest
    (3, "Forage",    ImageColor.getcolor("#66C2A5", "RGB")),  # same as meadow
    (4, "Corn",      ImageColor.getcolor("#D95F02", "RGB")),  # same as maize family
    (5, "Rice",      ImageColor.getcolor("#1F78B4", "RGB")),  # water/blue
    (6, "Unk. Crop", ImageColor.getcolor("#BDBDBD", "RGB")),  # neutral
    (7, "No agric",  ImageColor.getcolor("#252525", "RGB")),  # dark neutral
]


def get_color_definitions(cmap: str):
    if cmap == "munich":
        return MUNICH_COLORS
    if cmap == "lombardia":
        return LOMBARDIA_COLORS
    raise ValueError(f"Unsupported cmap: {cmap}")


def normalize_filename(filename: str, dataset_key: str) -> Path:
    path = Path(filename)

    if path.parts and path.parts[0] == dataset_key:
        return Path(*path.parts[1:])

    return path


def apply_cmap(x: np.ndarray, cmap: str) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")

    colors = get_color_definitions(cmap)
    y = np.full((x.shape[0], x.shape[1], 3), fill_value=255, dtype=np.uint8)

    for class_id, _, rgb in colors:
        y[x == class_id] = rgb

    return y


def compute_difference_mask(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have same shape, got {pred.shape} and {target.shape}"
        )

    diff = pred != target
    diff[target == 0] = False
    return diff.astype(np.uint8)


def difference_to_rgb(diff: np.ndarray) -> np.ndarray:
    diff = np.asarray(diff)
    if diff.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {diff.shape}")

    rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    rgb[diff.astype(bool)] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def save_png(mask: np.ndarray, out_path: Path, cmap: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = apply_cmap(np.asarray(mask), cmap)
    Image.fromarray(rgb).save(out_path)


def save_difference_png(pred: np.ndarray, target: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff = compute_difference_mask(pred, target)
    diff_rgb = difference_to_rgb(diff)
    Image.fromarray(diff_rgb).save(out_path)


def get_patch_output_dir(arch: str, dataset_key: str, filename: str) -> Path:
    rel_path = normalize_filename(filename, dataset_key)
    return RESULTS_PATH / arch / dataset_key / rel_path


def save_tif(
    pred: np.ndarray,
    target: np.ndarray,
    filename: str,
    data_dir: Path,
    arch: str,
    dataset_key: str,
    cmap: str,
) -> None:
    patch_dir = get_patch_output_dir(arch, dataset_key, filename)
    patch_dir.mkdir(parents=True, exist_ok=True)

    label_filename = data_dir / filename / "y.tif"

    pred_out_path = patch_dir / "pred.tif"
    target_out_path = patch_dir / "truth.tif"
    pred_png_out_path = patch_dir / "pred.png"
    target_png_out_path = patch_dir / "truth.png"
    diff_png_out_path = patch_dir / "diff.png"

    with rasterio.open(label_filename) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=NODATA_VALUE, count=1)

    pred = np.asarray(pred, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)

    with rasterio.open(pred_out_path, "w", **profile) as dst:
        dst.write(pred, 1)

    with rasterio.open(target_out_path, "w", **profile) as dst:
        dst.write(target, 1)

    save_png(pred, pred_png_out_path, cmap)
    save_png(target, target_png_out_path, cmap)
    save_difference_png(pred, target, diff_png_out_path)


def iter_patch_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if "merged" in path.parts:
            continue
        if (path / "pred.tif").exists() and (path / "truth.tif").exists():
            yield path


def merge_tif_list(tif_files: list[Path], output_path: Path) -> None:
    if not tif_files:
        return

    src_files = [rasterio.open(fp) for fp in tif_files]
    try:
        mosaic, transform = merge(src_files, nodata=NODATA_VALUE)

        profile = src_files[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            nodata=NODATA_VALUE,
            compress="lzw",
            count=mosaic.shape[0],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in src_files:
            src.close()


def save_merged_png_from_tif(input_tif: Path, output_png: Path, cmap: str) -> None:
    with rasterio.open(input_tif) as src:
        arr = src.read(1)

    save_png(arr, output_png, cmap)


def save_merged_difference_png(
    pred_tif: Path,
    target_tif: Path,
    output_png: Path,
) -> None:
    with rasterio.open(pred_tif) as src:
        pred = src.read(1)

    with rasterio.open(target_tif) as src:
        target = src.read(1)

    save_difference_png(pred, target, output_png)


def build_group_rel_path(root: Path, patch_dir: Path) -> Path:
    return patch_dir.parent.relative_to(root)


def save_merged_patches(arch: str, dataset_key: str, cmap: str) -> None:
    root = RESULTS_PATH / arch / dataset_key
    merged_root = root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    grouped_preds: dict[Path, list[Path]] = {}
    grouped_truths: dict[Path, list[Path]] = {}

    for patch_dir in iter_patch_dirs(root):
        group_rel_path = build_group_rel_path(root, patch_dir)

        grouped_preds.setdefault(group_rel_path, []).append(patch_dir / "pred.tif")
        grouped_truths.setdefault(group_rel_path, []).append(patch_dir / "truth.tif")

    common_groups = sorted(set(grouped_preds) & set(grouped_truths))

    for group_rel_path in common_groups:
        merged_group_dir = merged_root / group_rel_path
        merged_group_dir.mkdir(parents=True, exist_ok=True)

        pred_out_tif = merged_group_dir / "pred.tif"
        truth_out_tif = merged_group_dir / "truth.tif"
        pred_out_png = merged_group_dir / "pred.png"
        truth_out_png = merged_group_dir / "truth.png"
        diff_out_png = merged_group_dir / "diff.png"

        merge_tif_list(sorted(grouped_preds[group_rel_path]), pred_out_tif)
        merge_tif_list(sorted(grouped_truths[group_rel_path]), truth_out_tif)

        if pred_out_tif.exists():
            save_merged_png_from_tif(pred_out_tif, pred_out_png, cmap)

        if truth_out_tif.exists():
            save_merged_png_from_tif(truth_out_tif, truth_out_png, cmap)

        if pred_out_tif.exists() and truth_out_tif.exists():
            save_merged_difference_png(pred_out_tif, truth_out_tif, diff_out_png)