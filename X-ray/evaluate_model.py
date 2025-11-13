"""Evaluate a trained pneumonia classifier on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Keras model on the chest X-ray test set")
    parser.add_argument("--model-path", type=Path, default=Path("best_model_STRONG.h5"))
    parser.add_argument("--test-dir", type=Path, default=Path("chest_xray/test"))
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=("rescale", "efficientnet_v2"),
        default="efficientnet_v2",
        help="Image preprocessing pipeline to match the trained model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="best_model_STRONG")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    test_dir = args.test_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model = load_model(model_path)
    print("✅ النموذج تم تحميله بنجاح!")

    if args.preprocess == "rescale":
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    elif args.preprocess == "efficientnet_v2":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        raise ValueError(f"Unsupported preprocess option: {args.preprocess}")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=False,
    )

    y_prob = model.predict(test_generator).ravel()
    y_pred = (y_prob > args.threshold).astype(int)
    y_true = test_generator.classes

    class_names = [name for name, _ in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n📊 Classification Report:")
    print(report)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    with (output_dir / f"{args.output_prefix}_classification_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)

    cm = confusion_matrix(y_true, y_pred)
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
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.output_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.output_prefix}_roc_curve.png", dpi=300)
    plt.close()

    print("✅ Evaluation artifacts saved to:", output_dir)


if __name__ == "__main__":
    main()
