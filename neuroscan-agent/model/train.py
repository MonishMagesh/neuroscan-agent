"""
NeuroScan Agent — Model Training Pipeline
==========================================
Reproduces the IEEE SCEECS 2026 paper methodology:
  InceptionV3 feature extraction → PCA → Neural Network classifier

Usage (Google Colab / Kaggle — GPU recommended):
    # 1. Download dataset
    kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
    unzip brain-tumor-mri-dataset.zip -d dataset/

    # 2. Run training
    python model/train.py --dataset dataset/Training --output model/weights/

    # 3. Set in .env:  ANALYSIS_MODE=local
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NeuroScan brain tumor classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to training dataset directory")
    parser.add_argument("--output", default="model/weights",
                        help="Directory to save model weights")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--kfolds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(
    dataset_dir: str, img_size: int, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image  # type: ignore
    images, labels = [], []
    dataset_path = Path(dataset_dir)
    for class_idx, cls in enumerate(classes):
        class_dir = dataset_path / cls
        if not class_dir.exists():
            print(f"  ⚠ Not found: {class_dir}")
            continue
        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"  Loading {cls}: {len(files)} images")
        for fpath in files:
            try:
                img = Image.open(fpath).convert("RGB").resize(
                    (img_size, img_size), Image.LANCZOS
                )
                arr = np.array(img, dtype=np.float32)
                arr = (arr / 127.5) - 1.0
                images.append(arr)
                labels.append(class_idx)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ Skipping {fpath.name}: {exc}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def extract_inception_features(
    images: np.ndarray, batch_size: int
) -> np.ndarray:
    import tensorflow as tf  # type: ignore
    print("\n[2/5] Building InceptionV3 feature extractor...")
    base = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet",
        input_shape=(224, 224, 3), pooling="avg",
    )
    base.trainable = False
    extractor = tf.keras.Model(inputs=base.input, outputs=base.output)
    print(f"      Extracting features from {len(images)} images...")
    t0 = time.time()
    features = extractor.predict(images, batch_size=batch_size, verbose=1)
    print(f"      Done in {time.time() - t0:.1f}s. Shape: {features.shape}")
    return features


def apply_pca(
    features: np.ndarray, n_components: int
) -> tuple[np.ndarray, object]:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    print(f"\n[3/5] Applying PCA (n_components={n_components})...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, random_state=42)
    features_reduced = pca.fit_transform(features_scaled)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"      Variance explained: {var:.1f}%  Shape: {features_reduced.shape}")
    return features_reduced, (scaler, pca)


def build_classifier(input_dim: int, num_classes: int) -> "object":
    import tensorflow as tf  # type: ignore
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="neuroscan_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_kfold_cv(
    features: np.ndarray, labels: np.ndarray,
    k: int, input_dim: int, num_classes: int,
    epochs: int, seed: int, classes: list[str],
) -> tuple[list[dict], object]:
    import tensorflow as tf  # type: ignore
    from sklearn.model_selection import StratifiedKFold  # type: ignore
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report  # type: ignore

    print(f"\n[4/5] Running {k}-fold cross-validation ({epochs} epochs each)...")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    best_model = None
    best_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
        print(f"\n  ── Fold {fold}/{k} ──")
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        model = build_classifier(input_dim, num_classes)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=10,
                restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=0,
            ),
        ]
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=64,
            callbacks=callbacks, verbose=0,
        )
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc = accuracy_score(y_val, y_pred)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
        try:
            auc = roc_auc_score(y_val_oh, y_pred_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        fold_results.append({"fold": fold, "accuracy": round(acc * 100, 2), "auc": round(auc, 4)})
        print(f"  Accuracy: {acc*100:.2f}%  AUC: {auc:.4f}")
        print(classification_report(y_val, y_pred, target_names=classes))
        if acc > best_acc:
            best_acc = acc
            best_model = model

    return fold_results, best_model


def save_model(model: object, output_dir: str, fold_results: list[dict]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = str(output_path / "neuroscan_model.keras")
    model.save(model_path)  # type: ignore[attr-defined]
    avg_acc = sum(f["accuracy"] for f in fold_results) / len(fold_results)
    avg_auc = sum(f["auc"] for f in fold_results) / len(fold_results)
    history = {
        "fold_results": fold_results,
        "average_accuracy": round(avg_acc, 2),
        "average_auc": round(avg_auc, 4),
    }
    (output_path / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"\n[5/5] Model saved → {model_path}")
    print(f"      Average Accuracy: {avg_acc:.2f}%")
    print(f"      Average AUC:      {avg_auc:.4f}")
    return model_path


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf  # type: ignore  # noqa: F401
        import sklearn  # type: ignore  # noqa: F401
    except ImportError:
        print("ERROR: pip install tensorflow scikit-learn")
        return

    np.random.seed(args.seed)

    CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    print("=" * 60)
    print("  NeuroScan Agent — Training Pipeline")
    print("  IEEE SCEECS 2026 | InceptionV3 + PCA + Neural Net")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    images, labels = load_dataset(args.dataset, args.img_size, CLASSES)
    if len(images) == 0:
        print("ERROR: No images loaded. Check --dataset path.")
        return
    print(f"      Total: {len(images)} images")

    features = extract_inception_features(images, args.batch_size)
    features_reduced, _ = apply_pca(features, args.pca_components)
    fold_results, best_model = run_kfold_cv(
        features_reduced, labels,
        k=args.kfolds, input_dim=features_reduced.shape[1],
        num_classes=len(CLASSES), epochs=args.epochs,
        seed=args.seed, classes=CLASSES,
    )
    save_model(best_model, args.output, fold_results)
    print("\n✓ Training complete. Set ANALYSIS_MODE=local in .env")


if __name__ == "__main__":
    main()
