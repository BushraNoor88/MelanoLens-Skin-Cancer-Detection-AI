"""MobileNetV2 transfer learning (lightweight, CPU friendly)."""
from .. import utils
from ..errors import ModelNotAvailable


def build_classifier(num_classes, img_size=224, alpha=1.0, weights="imagenet", dropout=0.3):
    """Input: RGB pixels 0-255. MobileNetV2 scaling (-1..1) is part of the model, so training and inference
    use exactly the same preprocessing. Returns (model, base_model)."""
    _, keras = utils.require_tf()
    layers = keras.layers
    w = None if weights in (None, "none") else weights
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_v2_scaling")(inputs)
    try:
        base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                              weights=w, alpha=alpha)
    except Exception as exc:
        if w:
            raise ModelNotAvailable("Could not download the ImageNet weights for MobileNetV2 "
                                    f"({type(exc).__name__}). Check your internet connection or train with "
                                    "--weights none (lower accuracy).", "model_invalid")
        raise
    base.trainable = w is None     # pretrained base starts frozen; a randomly initialised base must be trained
    x = base(x, training=False) if w else base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", name="diagnosis_probabilities")(x)
    return keras.Model(inputs, out, name="skin_lesion_classifier"), base


def compile_model(model, lr=1e-3):
    _, keras = utils.require_tf()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top(base, n_layers=30):
    """Fine-tuning: unfreeze the last n layers of the base (BatchNorm layers stay frozen)."""
    _, keras = utils.require_tf()
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
