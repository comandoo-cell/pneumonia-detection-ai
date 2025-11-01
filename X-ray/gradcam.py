import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = None
        if hasattr(model, 'layers') and len(model.layers) > 0:
            base_model = model.layers[0]
            if hasattr(base_model, 'layers'):
                try:
                    conv_layer = base_model.get_layer(last_conv_layer_name)
                    grad_model = Model(
                        inputs=base_model.input,
                        outputs=[conv_layer.output, base_model.output]
                    )
                except Exception as e:
                    pass
        
        if grad_model is None:
            raise ValueError(f"Layer not found: {last_conv_layer_name}")
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = 0
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_superimposed_visualization(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


def get_last_conv_layer_name(model):
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'layers'):
            for layer in reversed(first_layer.layers):
                layer_name = layer.name
                if 'Conv' in type(layer).__name__ or 'block_16' in layer_name or 'out_relu' in layer_name:
                    return layer_name
    
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__
        if 'Conv' in layer_type:
            return layer.name
    
    return None


def generate_gradcam(img_path, img_array, model, save_path):
    try:
        possible_layers = ['out_relu', 'Conv_1_relu', 'block_16_project', 'block_16_expand_relu', 'Conv_1']
        last_conv_layer = None
        
        if hasattr(model, 'layers') and len(model.layers) > 0:
            base_model = model.layers[0]
            if hasattr(base_model, 'layers'):
                for layer_name in possible_layers:
                    try:
                        test_layer = base_model.get_layer(layer_name)
                        last_conv_layer = layer_name
                        break
                    except:
                        continue
        
        if last_conv_layer is None:
            last_conv_layer = get_last_conv_layer_name(model)
        
        if last_conv_layer is None:
            return False
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        superimposed = create_superimposed_visualization(img_path, heatmap)
        cv2.imwrite(save_path, superimposed)
        return True
    except Exception as e:
        print(f"Error in generate_gradcam: {e}")
        import traceback
        traceback.print_exc()
        return False
