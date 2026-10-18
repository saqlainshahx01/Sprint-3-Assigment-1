# Image Classification: Traditional ML → Deep Learning

An image-classification pipeline that compares six approaches on the same data split, from classical ML (SVM, KNN) up to transfer learning with ResNet50.

**Notebook:** `CNN_Assigment_01.ipynb` (written for Google Colab, TensorFlow/Keras)

---

## 1. Dataset

| Item | Value |
|---|---|
| Dataset used | [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (`puneet6060/intel-image-classification`) |
| Classes (6) | buildings, forest, glacier, mountain, sea, street |
| Images per class | 1,000, randomly sampled (`seed=42`) from `seg_train` |
| Total used | 6,000 images |
| Split | 70 / 15 / 15, stratified |

| Split | Images per class | Total |
|---|---|---|
| Train | 700 | 4,200 |
| Validation | 150 | 900 |
| Test | 150 | 900 |

The split is copied to `train/`, `val/`, `test/` folders so every model uses exactly the same images.

> **Note:** The assignment brief links the ISIC 2019 skin-lesion dataset. This notebook uses the Intel scene dataset instead. If your instructor requires ISIC 2019, only the dataset download and `CLASS_NAMES` cells need to change. The rest of the pipeline is dataset-agnostic.

---

## 2. Pipeline

### Step 1: Data preparation
- Random selection of 1,000 images per class, then a stratified 70/15/15 split.
- **Traditional ML / ANN:** resized to **64×64**, scaled to [0, 1], flattened to a 12,288-dim vector.
- **CNNs:** resized to **128×128** (custom CNNs) or **224×224** (ResNet50), loaded with `image_dataset_from_directory` and normalised inside the model.

### Step 2: Traditional ML
- **SVM:** linear kernel, `C=1.0`
- **KNN:** `k=5`
- Evaluated with accuracy and a confusion matrix (SVM).

### Step 3: Vanilla ANN
`Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(6, softmax)`
Adam, sparse categorical crossentropy, 88 epochs, batch size 32. 3.18M parameters.

### Step 4: Custom CNN (no augmentation)
`Rescaling → [Conv2D(32/64/128, 3×3, ReLU) + MaxPool2D] ×3 → Flatten → Dense(256) → Dropout(0.5) → Dense(6, softmax)`
Adam, sparse categorical crossentropy, 15 epochs. 8.48M parameters.

### Step 5: Custom CNN + Data Augmentation
Same architecture as Step 4, with `RandomFlip("horizontal")`, `RandomRotation(0.1)` and `RandomZoom(0.1)` at the input. 15 epochs.

### Step 6: Transfer learning (ResNet50)
ImageNet-pretrained **ResNet50** with the base frozen, plus `GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(6, softmax)`.
Adam (lr = 1e-4), 10 epochs. 526K trainable of 24.1M total parameters.

---

## 3. Results

### Final-epoch metrics

| Model | Train Acc | Train Loss | Val Acc | Val Loss | **Test Acc** | Test Loss |
|---|---|---|---|---|---|---|
| SVM (linear) | – | – | 44.89% | – | **49.78%** | – |
| KNN (k=5) | – | – | 42.56% | – | **42.89%** | – |
| Vanilla ANN | 31.31% | 1.5225 | 28.11% | 1.6208 | **31.67%** | – |
| Custom CNN | 97.86% | 0.0679 | 77.56% | 1.2688 | **78.89%** | 1.0628 |
| CNN + Augmentation | 78.90% | 0.5986 | 78.22% | 0.6042 | **81.11%** | 0.5743 |
| ResNet50 (frozen base) | 95.43% | 0.1299 | 93.22% | 0.1894 | **93.33%** | 0.1784 |

SVM and KNN have no epochs, so there are no training/validation loss curves for them. The notebook plots accuracy and loss curves for the ANN, the Custom CNN and ResNet50.

### Comparison

```
ResNet50 (transfer)   ██████████████████████████████████████████████  93.3%
CNN + Augmentation    ████████████████████████████████████████        81.1%
Custom CNN            ███████████████████████████████████████         78.9%
SVM                   ████████████████████████                        49.8%
KNN                   █████████████████████                           42.9%
Vanilla ANN           ███████████████                                 31.7%
```

### ResNet50 per-class report (test set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| buildings | 0.96 | 0.92 | 0.94 |
| forest | 0.99 | 0.99 | 0.99 |
| glacier | 0.87 | 0.89 | 0.88 |
| mountain | 0.90 | 0.88 | 0.89 |
| sea | 0.97 | 0.97 | 0.97 |
| street | 0.91 | 0.95 | 0.93 |

Overall: 93% accuracy, 0.93 macro F1. The weakest classes are **glacier** and **mountain**, which look alike.

---

## 4. Key observations

- **Pixel-based models fail on this data.** SVM, KNN and the ANN work on flattened raw pixels and lose all spatial structure, so they reach only 32–50%.
- **Convolutions help a lot.** The custom CNN gains about 29 points over SVM, the best of the non-CNN models.
- **The plain CNN overfits.** It reaches 97.9% train accuracy but only 77.6% validation, and its validation loss climbs (about 1.27). Augmentation closes that gap (78.9% train vs 78.2% val), lowers test loss from 1.06 to 0.57, and adds about 2 points of test accuracy.
- **Transfer learning is best.** ResNet50 with a frozen base reaches 93.3% test accuracy and generalises well: train 95.4% and val 93.2%.
- **The ANN barely learns.** It stays near 30% after 88 epochs, so the flattened 64×64 input and a small dense network are a poor fit for images.

---

## 5. How to run

1. Open `CNN_Assigment_01.ipynb` in Google Colab. Use a GPU runtime (`Runtime → Change runtime type → GPU`).
2. Upload your `kaggle.json` when the upload cell asks for it. Get it from Kaggle under *Settings → API → Create New Token*.
3. Run all cells from top to bottom. They download the dataset, build the split, train every model and produce the comparison table and chart.

**Requirements:** Python 3, TensorFlow/Keras, scikit-learn, NumPy, pandas, matplotlib, seaborn, Pillow, kaggle.

---

## 6. Known limitations and possible improvements

- **No fine-tuning stage.** The assignment asks to fine-tune after initial training. The notebook trains only the frozen-base classifier head. To add it, unfreeze the top layers of ResNet50 and retrain with a low learning rate (about 1e-5).
- **Only one pretrained model.** Only ResNet50 is used. VGG16, MobileNetV2, InceptionV3 or DenseNet121 can be added for comparison.
- **Confusion matrices only for SVM and ResNet50.** Adding them for KNN, ANN and both CNNs would complete the per-model report.
- **Loss functions.** The notebook uses `sparse_categorical_crossentropy` with integer labels. This is mathematically equivalent to categorical crossentropy with one-hot labels.
- **ANN training time.** 88 epochs with no early stopping gives no benefit, because validation accuracy plateaus early. Early stopping would save time.
- **Hyperparameters.** SVM and KNN use fixed hyperparameters with no tuning. A grid search (RBF kernel, different `k`) would likely improve them.
