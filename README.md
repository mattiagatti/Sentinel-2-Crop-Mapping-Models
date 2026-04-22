# 🌾 Sentinel-2 Crop Mapping Models

Repository for the paper [Enhancing crop segmentation in satellite image
time-series with transformer
networks](https://arxiv.org/abs/2412.01944).

---

## ⚙️ Setup

``` shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
cd sentinel2-crop-mapping-models
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔑 Kaggle credentials

Create a Kaggle API token from: https://www.kaggle.com/settings

Then export:

``` shell
export KAGGLE_API_TOKEN=KGAT_...
```

---

## 📦 Datasets (auto-downloaded)

-   :de: Munich dataset\
    https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480

-   :it: Lombardia dataset\
    https://www.kaggle.com/datasets/ignazio/sentinel2-crop-mapping

---

## 🚀 Training

### :de: Munich

``` shell
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset munich
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset munich
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset munich
CUDA_VISIBLE_DEVICES=3 python train.py --arch unet --dataset munich
CUDA_VISIBLE_DEVICES=4 python train.py --arch vistaformer --dataset munich
```

### 🔁 Munich resume

``` shell
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset munich --ckpt_path exp/deeplabv3/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset munich --ckpt_path exp/fpn/munich/train/checkpoints/last.ckpt 
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset munich --ckpt_path exp/swin_unetr/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=3 python train.py --arch unet --dataset munich --ckpt_path exp/unet/munich/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=4 python train.py --arch vistaformer --dataset munich --ckpt_path exp/vistaformer/munich/train/checkpoints/last.ckpt
```

---

### :it: Lombardia

``` shell
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset lombardia
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset lombardia
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset lombardia
CUDA_VISIBLE_DEVICES=3 python train.py --arch unet --dataset lombardia
CUDA_VISIBLE_DEVICES=4 python train.py --arch vistaformer --dataset lombardia
```

### 🔁 Lombardia resume

``` shell
CUDA_VISIBLE_DEVICES=0 python train.py --arch deeplabv3 --dataset lombardia --ckpt_path exp/deeplabv3/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=1 python train.py --arch fpn --dataset lombardia --ckpt_path exp/fpn/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=2 python train.py --arch swin_unetr --dataset lombardia --ckpt_path exp/swin_unetr/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=3 python train.py --arch unet --dataset lombardia --ckpt_path exp/unet/lombardia/train/checkpoints/last.ckpt
CUDA_VISIBLE_DEVICES=3 python train.py --arch vistaformer --dataset lombardia --ckpt_path exp/vistaformer/lombardia/train/checkpoints/last.ckpt
```

---

## 🧪 Evaluation

### :de: Munich

``` shell
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset munich --weights_path exp/deeplabv3/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset munich --weights_path exp/fpn/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset munich --weights_path exp/swin_unetr/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch unet --dataset munich --weights_path exp/unet/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch vistaformer --dataset munich --weights_path exp/vistaformer/munich/train/weights/best.pt
```

### :it: Lombardia A

``` shell
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset lombardia --test_id A --weights_path exp/deeplabv3/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset lombardia --test_id A --weights_path exp/fpn/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset lombardia --test_id A --weights_path exp/swin_unetr/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch unet --dataset lombardia --test_id A --weights_path exp/unet/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch vistaformer --dataset lombardia --test_id A --weights_path exp/vistaformer/lombardia/train/weights/best.pt
```

### :it: Lombardia Y

``` shell
CUDA_VISIBLE_DEVICES=0 python test.py --arch deeplabv3 --dataset lombardia --test_id Y --weights_path exp/deeplabv3/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python test.py --arch fpn --dataset lombardia --test_id Y --weights_path exp/fpn/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=2 python test.py --arch swin_unetr --dataset lombardia --test_id Y --weights_path exp/swin_unetr/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=3 python test.py --arch unet --dataset lombardia --test_id Y --weights_path exp/unet/lombardia/train/weights/best.pt
CUDA_VISIBLE_DEVICES=4 python test.py --arch vistaformer --dataset lombardia --test_id Y --weights_path exp/vistaformer/lombardia/train/weights/best.pt
```

### Export sample RGB time series
``` shell
python -m misc.export_samples_rgb --dataset munich
```

### Run inference and merge patches into a full segmentation map
``` shell
CUDA_VISIBLE_DEVICES=0 python inference.py --arch unet --dataset munich --save_tifs --weights_path exp/unet/munich/train/weights/best.pt
CUDA_VISIBLE_DEVICES=1 python inference.py --arch unet --dataset lombardia --test_id A --save_tifs --merge_patches --weights_path exp/unet/lombardia/train/weights/best.pt
```

---

## 📝 Notes

-   📥 Datasets are downloaded automatically via Kaggle CLI
-   🔐 Make sure credentials are set before running
-   🎯 `CUDA_VISIBLE_DEVICES` controls GPU selection
-   📁 Logs and checkpoints are saved under
    `exp/<arch>/<dataset>/train/`