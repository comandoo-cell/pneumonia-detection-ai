import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2


PREFERRED_CONV_LAYERS = [
    "block6h_project_conv",
    "block6h_expand_conv",
    "block6g_project_conv",
    "block6f_project_conv",
    "block5f_project_conv",
    "block5e_project_conv",
    "top_conv",
]

FOCUS_LAYER_CANDIDATES = [
    "block6f_project_conv",
    "block5f_project_conv",
    "block5e_project_conv",
]


def build_spatial_weight(shape, top_shrink=0.12, center_sigma_y=0.24, center_sigma_x=0.32, min_weight=0.18):
    h, w = shape
    if h == 0 or w == 0:
        return 1.0

    y = np.linspace(0.0, 1.0, h)
    x = np.linspace(0.0, 1.0, w)

    y_weight = np.clip((y - top_shrink) / max(1e-6, 1.0 - top_shrink), 0.0, 1.0)
    x_weight = 1.0 - 0.65 * np.abs(x - 0.5) / 0.5
    x_weight = np.clip(x_weight, min_weight, 1.0)

    spatial_weight = np.outer(y_weight, x_weight)

    center_y = np.exp(-((y - 0.5) ** 2) / (2 * center_sigma_y ** 2))
    center_x = np.exp(-((x - 0.5) ** 2) / (2 * center_sigma_x ** 2))
    center_weight = np.outer(center_y, center_x)

    weight = spatial_weight * center_weight
    weight = np.clip(weight, min_weight, None)
    weight /= weight.max() if weight.max() > 0 else 1.0
    return weight


def find_last_conv_layer(model):
    """Return the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        layer_type = layer.__class__.__name__.lower()
        if "conv" in layer_type and len(getattr(layer, "output_shape", [])) == 4:
            return layer.name
        if layer.name == "top_conv":
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM")


def select_gradcam_layer(model, layer_name=None):
    if layer_name:
        try:
            model.get_layer(layer_name)
            return layer_name
        except ValueError as exc:
            raise ValueError(f"Requested Grad-CAM layer '{layer_name}' not found") from exc

    for candidate in PREFERRED_CONV_LAYERS:
        try:
            model.get_layer(candidate)
            return candidate
        except ValueError:
            continue
    return find_last_conv_layer(model)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    try:
        target_layer = select_gradcam_layer(model, last_conv_layer_name)
        conv_layer = model.get_layer(target_layer)
        grad_model = Model(inputs=model.input, outputs=[conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            index = pred_index if pred_index is not None else 0
            class_channel = predictions[:, index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if tf.less_equal(max_val, 0):
            return np.zeros_like(heatmap.numpy())
        heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        import traceback
        traceback.print_exc()
        raise


def postprocess_heatmap(
    heatmap,
    min_intensity=0.02,
    apply_threshold=True,
    apply_blur=True,
    blur_kernel=5,
):
    heatmap = np.maximum(np.array(heatmap, dtype=np.float32), 0.0)
    max_val = heatmap.max()
    if max_val <= 0:
        return np.zeros_like(heatmap)
    heatmap /= max_val

    spatial_weight = build_spatial_weight(heatmap.shape)
    heatmap = heatmap * spatial_weight
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    if apply_threshold and min_intensity is not None:
        mask = heatmap >= float(min_intensity)
        if np.count_nonzero(mask) > 0:
            heatmap = heatmap * mask

    if apply_blur:
        k = int(blur_kernel)
        k = k if k % 2 == 1 else k + 1
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
        peak = heatmap.max()
        if peak > 0:
            heatmap /= peak

    return heatmap


def create_superimposed_visualization(img_path, heatmap, alpha=0.35, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized, 0.0, 1.0) * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    mask = heatmap_uint8 > 0
    overlay[~mask] = img[~mask]
    return overlay


def compute_focus_score(heatmap, center_margin=0.30):
    heatmap = np.maximum(heatmap, 0.0)
    total = np.sum(heatmap)
    if total <= 1e-8:
        return 0.0

    h, w = heatmap.shape
    top = max(0, int(h * center_margin))
    bottom = min(h, int(h * (1.0 - center_margin)))
    left = max(0, int(w * center_margin))
    right = min(w, int(w * (1.0 - center_margin)))
    if bottom <= top or right <= left:
        return np.sum(heatmap) / total

    center_sum = np.sum(heatmap[top:bottom, left:right])
    edge_sum = total - center_sum
    center_ratio = center_sum / total
    edge_penalty = edge_sum / total
    score = center_ratio - 0.50 * edge_penalty
    return score


def choose_best_layer(model, img_array, candidate_layers):
    best_layer = None
    best_heatmap = None
    best_score = -np.inf

    for layer_name in candidate_layers:
        try:
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=layer_name)
        except ValueError:
            continue

        score = compute_focus_score(heatmap)
        if np.any(heatmap > 0) and score > best_score:
            best_score = score
            best_layer = layer_name
            best_heatmap = heatmap

    return best_layer, best_heatmap, best_score


def generate_gradcam(
    img_path,
    img_array,
    model,
    save_path,
    layer_name=None,
    candidate_layers=None,
    min_intensity=0.02,
    apply_threshold=True,
    apply_blur=True,
    blur_kernel=5,
    alpha=0.35,
):
    try:
        selected_layer = layer_name
        heatmap = None

        if selected_layer is None:
            layers_to_try = candidate_layers or FOCUS_LAYER_CANDIDATES
            best_layer, best_heatmap, best_score = choose_best_layer(model, img_array, layers_to_try)
            if best_layer is not None and best_heatmap is not None:
                selected_layer = best_layer
                heatmap = best_heatmap
                print(f"Grad-CAM layer selected: {selected_layer} (focus_score={best_score:.3f})")

        if heatmap is None:
            selected_layer = select_gradcam_layer(model, selected_layer)
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=selected_layer)

        heatmap = postprocess_heatmap(
            heatmap,
            min_intensity=min_intensity,
            apply_threshold=apply_threshold,
            apply_blur=apply_blur,
            blur_kernel=blur_kernel,
        )
        superimposed = create_superimposed_visualization(img_path, heatmap, alpha=alpha)
        cv2.imwrite(save_path, superimposed)
        return True
    except Exception as e:
        print(f"Error in generate_gradcam: {e}")
        import traceback
        traceback.print_exc()
        return False
