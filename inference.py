import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.lombardia_dataset import LombardiaDataset
from dataset.munich_dataset import MunichDataset
from utils.results_io import save_merged_patches, save_tif
from zoo.deeplabv3_3d import DeepLabV3_3D
from zoo.fpn_3d import FPN_3D
from zoo.swin_unetr import SwinUNETRTemporal
from zoo.unet_3d import UNet_3D


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights_path", type=Path, default=None, help="weights path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lombardia",
        choices=["lombardia", "munich"],
    )
    parser.add_argument("--test_id", type=str, default="A", choices=["A", "Y"])
    parser.add_argument(
        "--arch",
        type=str,
        default="swin_unetr",
        choices=["deeplabv3", "fpn", "swin_unetr", "unet"],
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/jovyan/shared/mgatti/datasets"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_tifs",
        action="store_true",
        help="save prediction and target GeoTIFFs",
    )
    parser.add_argument(
        "--merge_patches",
        action="store_true",
        help="merge saved patch GeoTIFFs after inference",
    )
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch, depth, in_channels, out_classes):
    if arch == "deeplabv3":
        return DeepLabV3_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    if arch == "fpn":
        return FPN_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    if arch == "swin_unetr":
        return SwinUNETRTemporal(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    if arch == "unet":
        return UNet_3D(
            depth=depth,
            in_channels=in_channels,
            out_classes=out_classes,
        )
    raise ValueError(f"Unsupported architecture: {arch}")


def build_inference_dataset(args):
    if args.dataset == "munich":
        data_dir = args.data_dir / "sentinel2-munich480" / "munich480"
        dataset = MunichDataset(
            data_dir,
            tileids=Path("tileids") / "eval.tileids",
            seqlength=32,
        )
        in_channels = 13
        out_classes = 18
        label_root_dir = dataset.root_dir

    elif args.dataset == "lombardia":
        data_dir = args.data_dir / "sentinel2-crop-mapping"

        if args.test_id == "A":
            dataset = LombardiaDataset(
                root_dirs=[data_dir / "lombardia3"],
                years=["data2019"],
                classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
                seqlength=32,
                tileids=Path("tileids") / "testA.tileids",
            )
            label_root_dir = data_dir

        elif args.test_id == "Y":
            dataset = LombardiaDataset(
                root_dirs=[data_dir / "lombardia", data_dir / "lombardia2"],
                years=["data2019"],
                classes_path=data_dir / "lombardia-classes" / "classes25pc.txt",
                seqlength=32,
                tileids=Path("tileids") / "testY2019.tileids",
            )
            label_root_dir = data_dir
        else:
            raise ValueError(f"Unsupported test_id: {args.test_id}")

        in_channels = 9
        out_classes = 8

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset, in_channels, out_classes, label_root_dir


def build_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )


def resolve_weights_path(args):
    if args.weights_path is not None:
        return args.weights_path

    return Path("exp") / args.arch / args.dataset / "train" / "weights" / "best.pt"


def load_weights(path, model, device):
    state_dict = torch.load(path, map_location=device)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dict in {path}, got {type(state_dict)}")

    if "model_state_dict" in state_dict:
        raise ValueError(
            f"{path} looks like a full checkpoint, but this script only accepts weights-only files."
        )

    model.load_state_dict(state_dict)
    return model


def build_dataset_key(args):
    if args.dataset == "lombardia":
        return f"lombardia{args.test_id}"
    return "munich"


@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
    arch,
    dataset_key,
    label_root_dir,
    cmap,
    save_tifs=False,
):
    model.eval()

    for x, y, filenames in tqdm(loader, desc="Inference", leave=False):
        x = x.to(device)
        y_hat = model(x)
        y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        if not save_tifs:
            continue

        for pred_i, target_i, filename_i in zip(y_pred, y_true, filenames):
            save_tif(
                pred=pred_i,
                target=target_i,
                filename=filename_i,
                data_dir=label_root_dir,
                arch=arch,
                dataset_key=dataset_key,
                cmap=cmap,
            )


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()

    dataset, in_channels, out_classes, label_root_dir = build_inference_dataset(args)
    loader = build_loader(dataset, args.batch_size)

    model = build_model(
        arch=args.arch,
        depth=32,
        in_channels=in_channels,
        out_classes=out_classes,
    )
    model = model.to(device)

    weights_path = resolve_weights_path(args)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = load_weights(weights_path, model, device)
    print(f"Loaded weights from {weights_path}")

    dataset_key = build_dataset_key(args)

    run_inference(
        model=model,
        loader=loader,
        device=device,
        arch=args.arch,
        dataset_key=dataset_key,
        label_root_dir=label_root_dir,
        cmap=args.dataset,
        save_tifs=args.save_tifs,
    )

    if args.merge_patches:
        save_merged_patches(args.arch, dataset_key, args.dataset)

    print("Inference completed.")


if __name__ == "__main__":
    main()