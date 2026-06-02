# RunPod запуск method 2

Цей репозиторій запускається як CSV training для PyTorch RetinaNet.

## 1. Pod

Рекомендований старт:

- Template: RunPod PyTorch template
- GPU: RTX 3090/4090 24GB, L4 24GB або T4 16GB
- Volume: мінімум 80 GB, краще 120 GB
- Робоча папка: `/workspace`

RunPod зберігає persistent disk у `/workspace`; container disk тимчасовий, тому код, датасет і чекпоїнти тримай у `/workspace`.

## 2. Код

```bash
cd /workspace
git clone <YOUR_METHOD_2_REPO_URL> method-2
cd method-2
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

## 3. Kaggle dataset

```bash
pip install kaggle
mkdir -p ~/.kaggle
# завантаж kaggle.json у ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p /workspace/rsna
cd /workspace/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip stage_2_train_images.zip -d stage_2_train_images
unzip stage_2_test_images.zip -d stage_2_test_images
cd /workspace/method-2
```

## 4. Підготовка CSV

```bash
python prepare_rsna_for_retinanet.py \
  --dicom_dir /workspace/rsna/stage_2_train_images \
  --labels /workspace/rsna/stage_2_train_labels.csv \
  --output_dir /workspace/rsna_retinanet \
  --val_fraction 0.1
```

Швидкий тест на 200 зображеннях:

```bash
python prepare_rsna_for_retinanet.py \
  --dicom_dir /workspace/rsna/stage_2_train_images \
  --labels /workspace/rsna/stage_2_train_labels.csv \
  --output_dir /workspace/rsna_retinanet_small \
  --max_images 200
```

## 5. Training

Почни обережно:

```bash
python train.py \
  --dataset csv \
  --csv_train /workspace/rsna_retinanet/train.csv \
  --csv_classes /workspace/rsna_retinanet/classes.csv \
  --csv_val /workspace/rsna_retinanet/val.csv \
  --depth 18 \
  --batch_size 2 \
  --workers 2 \
  --epochs 5
```

Для GPU з 16-24GB можна пробувати:

```bash
python train.py --dataset csv \
  --csv_train /workspace/rsna_retinanet/train.csv \
  --csv_classes /workspace/rsna_retinanet/classes.csv \
  --csv_val /workspace/rsna_retinanet/val.csv \
  --depth 50 --batch_size 2 --workers 4 --epochs 20
```

Чекпоїнти зберігаються як `csv_retinanet_<epoch>.pt`, фінальна модель як `model_final.pt`.
