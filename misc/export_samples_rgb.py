import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
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

    elif args.dataset == "munich":
        root = args.munich_root

        patch_dirs = []
        for year_dir in root.glob("data*"):
            if not year_dir.is_dir():
                continue

            for patch_dir in year_dir.iterdir():
                if patch_dir.is_dir() and patch_dir.name.isdigit():
                    patch_dirs.append(patch_dir)

        return patch_dirs, root

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


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
    rgb = np.power(rgb, 0.5)  # gamma correction
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

    handles = [
        Patch(
            facecolor=tuple(c / 255.0 for c in rgb),
            edgecolor="none",
            label=class_name,
        )
        for _, class_name, rgb in colors
    ]

    n_rows = int(np.ceil(len(handles) / 2))
    fig_height = max(2.0, 0.38 * n_rows + 0.6)

    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=2,
        fontsize=11,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.6,
        columnspacing=1.6,
        labelspacing=0.5,
        borderpad=0.2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    patch_dirs, root = get_patch_dirs(args)
    if not patch_dirs:
        raise RuntimeError("No patch folders found")

    selected = sample_dirs(patch_dirs, args.percentage, args.seed)

    print(f"Dataset: {args.dataset}")
    print(f"Total patches: {len(patch_dirs)}")
    print(f"Selected: {len(selected)}")

    dataset_out_dir = args.output_dir / args.dataset

    # create legend
    save_legend_pdf(args.dataset, dataset_out_dir / "legend.pdf")

    for patch_dir in tqdm(selected):
        tif_files = sorted(patch_dir.glob("*_10m.tif"))
        y_path = patch_dir / "y.tif"
        if y_path.exists():
            tif_files.append(y_path)

        if not tif_files:
            continue

        rel = patch_dir.relative_to(root)
        out_dir = dataset_out_dir / rel

        for tif_path in tif_files:
            png_path = out_dir / f"{tif_path.stem}.png"
            convert_tif(tif_path, png_path, args.dataset)


if __name__ == "__main__":
    main()