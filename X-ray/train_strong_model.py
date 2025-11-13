"""High-performance training script for pneumonia classification.

This script introduces a stronger training pipeline than the baseline
MobileNetV2 fine-tuning approach. It relies on EfficientNetV2, richer
augmentation, class balancing, and threshold optimisation to reduce
false positives on the Normal class while preserving recall for the
Pneumonia class.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, precision_recall_curve,
                             roc_auc_score)
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import efficientnet_v2

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_SEED = 2024


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_dataset = project_root / "chest_xray"
    default_artifacts = project_root / "outputs" / "strong_model"
    default_model_path = project_root / "best_model_STRONG.h5"

    parser = argparse.ArgumentParser(
        description="Train a high-performing pneumonia classifier with EfficientNetV2"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset,
        help="Path to the chest_xray dataset root containing train/val/test folders",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=default_artifacts,
        help="Directory to store training artifacts (plots, reports, checkpoints)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path,
        help="Destination path for the final saved model",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=300,
        help="Input image size (height and width)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--initial-epochs",
        type=int,
        default=15,
        help="Number of frozen base training epochs",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=20,
        help="Additional epochs for fine-tuning",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=60,
        help="Number of EfficientNetV2 layers (from the end) to unfreeze during fine-tuning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def build_data_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1, 0.15),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def compute_class_weights(train_dir: Path) -> Dict[int, float]:
    class_dirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {train_dir}")

    counts = Counter()
    for class_dir in class_dirs:
        counts[class_dir.name] = sum(1 for _ in class_dir.glob("**/*.*"))
    if not counts:
        raise ValueError(f"No training images discovered under {train_dir}")

    total = sum(counts.values())
    class_indices = {name: idx for idx, name in enumerate(sorted(counts.keys()))}
    weights = {}
    for name, count in counts.items():
        weights[class_indices[name]] = total / (len(counts) * max(count, 1))
    return weights


def load_datasets(
    dataset_root: Path,
    img_size: int,
    batch_size: int,
    seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Tuple[str, ...]]:
    train_path = dataset_root / "train"
    val_path = dataset_root / "val"
    test_path = dataset_root / "test"

    if not all(path.exists() for path in (train_path, val_path, test_path)):
        raise FileNotFoundError(
            "Dataset structure must contain train/val/test folders under the provided root"
        )

    common_kwargs = {
        "image_size": (img_size, img_size),
        "batch_size": batch_size,
        "label_mode": "binary",
        "seed": seed,
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        shuffle=True,
        **common_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        shuffle=False,
        **common_kwargs,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        shuffle=False,
        **common_kwargs,
    )

    class_names = train_ds.class_names

    augment = build_data_augmentation()
    preprocess = efficientnet_v2.preprocess_input

    def prep(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        def _map(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            images = tf.cast(images, tf.float32)
            if training:
                images = augment(images, training=True)
            images = preprocess(images)
            return images, labels

        mapped = ds.map(_map, num_parallel_calls=AUTOTUNE)
        if not training:
            mapped = mapped.cache()
        return mapped.prefetch(AUTOTUNE)

    return (
        prep(train_ds, training=True),
        prep(val_ds, training=False),
        prep(test_ds, training=False),
        tuple(class_names),
    )


def build_model(img_size: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    inputs = layers.Input(shape=(img_size, img_size, 3))
    base_model = efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(
        320,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probabilities")(x)

    model = models.Model(inputs, outputs, name="efficientnetv2_pneumonia")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def train_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    artifact_dir: Path,
    initial_epochs: int,
    fine_tune_epochs: int,
    unfreeze_layers: int,
) -> Dict[str, list]:
    checkpoint_path = artifact_dir / "best_model_checkpoint.weights.h5"
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_pr_auc",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=3e-7,
            verbose=1,
        ),
    ]

    compile_model(model, learning_rate=3e-4)
    history = model.fit(
        train_ds,
        epochs=initial_epochs,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1,
    )

    base_model.trainable = True

    if 0 < unfreeze_layers < len(base_model.layers):
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    compile_model(model, learning_rate=5e-5)
    total_epochs = initial_epochs + fine_tune_epochs
    fine_history = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1,
    )

    model.load_weights(str(checkpoint_path))

    combined_history: Dict[str, list] = {}
    for key, values in history.history.items():
        combined_history[key] = values + fine_history.history.get(key, [])
    for key, values in fine_history.history.items():
        if key not in combined_history:
            combined_history[key] = fine_history.history[key]
    return combined_history


def find_optimal_threshold(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    artifact_dir: Path,
) -> float:
    val_probs = model.predict(val_ds, verbose=0).ravel()
    val_labels = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

    thresholds = np.linspace(0.3, 0.7, 81)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        preds = (val_probs >= threshold).astype(int)
        score = fbeta_score(
            val_labels,
            preds,
            beta=0.7,
            zero_division=0,
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    precision, recall, pr_thresholds = precision_recall_curve(val_labels, val_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    if pr_thresholds.size:
        idx = int(np.argmin(np.abs(pr_thresholds - best_threshold)))
        plt.scatter(
            recall[idx + 1],
            precision[idx + 1],
            label=f"Chosen threshold {best_threshold:.2f}",
            color="red",
        )
    plt.xlabel("Recall (Pneumonia)")
    plt.ylabel("Precision (Pneumonia)")
    plt.title("Validation Precision-Recall")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

    with (artifact_dir / "selected_threshold.json").open("w", encoding="utf-8") as fp:
        json.dump({"optimal_threshold": best_threshold, "fbeta_score": best_score}, fp, indent=2)

    return best_threshold


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: Tuple[str, ...],
    threshold: float,
    artifact_dir: Path,
) -> Dict[str, float]:
    probs = model.predict(test_ds, verbose=0).ravel()
    labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    preds = (probs >= threshold).astype(int)

    metrics_dict = {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "threshold": float(threshold),
    }

    report = classification_report(
        labels,
        preds,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    metrics_dict.update({"report": report})

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(artifact_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    with (artifact_dir / "classification_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return metrics_dict


def plot_training_curves(history: Dict[str, list], artifact_dir: Path) -> None:
    epochs_range = range(1, len(next(iter(history.values()), [])) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.get("accuracy", []), label="Train Acc")
    plt.plot(epochs_range, history.get("val_accuracy", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.get("loss", []), label="Train Loss")
    plt.plot(epochs_range, history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(artifact_dir / "training_curves.png", dpi=300)
    plt.close()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        dataset_root=args.dataset_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    class_weights = compute_class_weights(args.dataset_root / "train")
    model, base_model = build_model(args.img_size)
    history = train_model(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        artifact_dir=artifact_dir,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        unfreeze_layers=args.unfreeze_layers,
    )

    plot_training_curves(history, artifact_dir)
    best_threshold = find_optimal_threshold(model, val_ds, artifact_dir)
    metrics_dict = evaluate_model(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        threshold=best_threshold,
        artifact_dir=artifact_dir,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)

    with (artifact_dir / "test_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics_dict, fp, indent=2)

    print("Training complete.")
    print(f"Best model saved to: {args.model_path}")
    print(f"Artifacts stored in: {artifact_dir}")
    print(f"Optimal decision threshold: {best_threshold:.3f}")


if __name__ == "__main__":
    main()
