import argparse
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from utils.results_io import apply_cmap, LOMBARDIA_COLORS, MUNICH_COLORS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["lombardia", "munich"],
    )

    parser.add_argument(
        "--lombardia_root",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets/sentinel2-crop-mapping"),
    )
    parser.add_argument(
        "--munich_root",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets/sentinel2-munich480/munich480/munich480"),
    )

    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exports/rgb_pngs"),
    )

    return parser.parse_args()


def get_patch_dirs(args):
    if args.dataset == "lombardia":
        root = args.lombardia_root
        dataset_roots = ["lombardia", "lombardia2", "lombardia3"]

        patch_dirs = []
        for ds_name in dataset_roots:
            ds_root = root / ds_name
            if not ds_root.exists():
                continue

            for year_dir in ds_root.glob("data*"):
                if not year_dir.is_dir():
                    continue

                for patch_dir in year_dir.iterdir():
                    if patch_dir.is_dir() and patch_dir.name.isdigit():
                        patch_dirs.append(patch_dir)

        return patch_dirs, root

    if args.dataset == "munich":
        root = args.munich_root

        patch_dirs = []
        for year_dir in root.glob("data*"):
            if not year_dir.is_dir():
                continue

            for patch_dir in year_dir.iterdir():
                if patch_dir.is_dir() and patch_dir.name.isdigit():
                    patch_dirs.append(patch_dir)

        return patch_dirs, root

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def has_enough_time_steps(patch_dir: Path, min_steps: int = 32) -> bool:
    tif_files = list(patch_dir.glob("*_10m.tif"))
    return len(tif_files) >= min_steps


def sample_dirs(dirs, percentage, seed):
    if not dirs:
        return []

    if not (0.0 < percentage <= 1.0):
        raise ValueError("--percentage must be in (0, 1].")

    rng = random.Random(seed)
    k = max(1, int(len(dirs) * percentage))
    return rng.sample(dirs, k)


def to_uint8(rgb):
    rgb = np.clip(rgb, 0, 10000) / 10000.0
    rgb = np.power(rgb, 0.5)
    rgb = rgb * 255.0
    return rgb.astype(np.uint8)


def convert_tif(tif_path, png_path, dataset):
    with rasterio.open(tif_path) as src:
        data = src.read()

    if tif_path.name == "y.tif":
        cmap_name = "munich" if dataset == "munich" else "lombardia"
        rgb = apply_cmap(data[0], cmap_name)
    else:
        rgb = data[[2, 1, 0]]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = to_uint8(rgb)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(png_path)


def save_legend_pdf(dataset, out_path):
    colors = MUNICH_COLORS if dataset == "munich" else LOMBARDIA_COLORS

    n_items = len(colors)
    n_cols = n_items

    cell_w = 1.2
    cell_h = 2.2
    square_size = 0.9

    fig_w = n_cols * cell_w
    fig_h = cell_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    for idx, (_, class_name, rgb) in enumerate(colors):
        x_center = idx * cell_w + cell_w / 2
        y_top = fig_h - 0.4

        square = plt.Rectangle(
            (x_center - square_size / 2, y_top - square_size),
            square_size,
            square_size,
            facecolor=tuple(c / 255.0 for c in rgb),
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(square)

        ax.text(
            x_center,
            y_top - square_size - 0.15,
            class_name,
            ha="center",
            va="top",
            fontsize=9,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def get_selected_time_files(patch_dir: Path):
    tif_files = sorted(patch_dir.glob("*_10m.tif"))

    wanted_indices = [0, 7, 15, 31]
    wanted_names = ["t1", "t8", "t16", "t32"]

    return [(tif_files[i], wanted_names[j]) for j, i in enumerate(wanted_indices)]


def main():
    args = parse_args()

    patch_dirs, root = get_patch_dirs(args)
    if not patch_dirs:
        raise RuntimeError("No patch folders found")

    valid_patch_dirs = [p for p in patch_dirs if has_enough_time_steps(p, min_steps=32)]
    skipped = len(patch_dirs) - len(valid_patch_dirs)

    if not valid_patch_dirs:
        raise RuntimeError("No valid patch folders found with at least 32 temporal files")

    selected = sample_dirs(valid_patch_dirs, args.percentage, args.seed)

    print(f"Dataset: {args.dataset}")
    print(f"Total patches: {len(patch_dirs)}")
    print(f"Valid patches (>=32 timestamps): {len(valid_patch_dirs)}")
    print(f"Skipped patches (<32 timestamps): {skipped}")
    print(f"Selected: {len(selected)}")

    dataset_out_dir = args.output_dir / args.dataset

    if dataset_out_dir.exists():
        shutil.rmtree(dataset_out_dir)
    
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    save_legend_pdf(args.dataset, dataset_out_dir / "legend.pdf")

    for patch_dir in tqdm(selected):
        rel = patch_dir.relative_to(root)
        out_dir = dataset_out_dir / rel

        for tif_path, out_name in get_selected_time_files(patch_dir):
            png_path = out_dir / f"{out_name}.png"
            convert_tif(tif_path, png_path, args.dataset)

        y_path = patch_dir / "y.tif"
        if y_path.exists():
            convert_tif(y_path, out_dir / "y.png", args.dataset)


if __name__ == "__main__":
    main()