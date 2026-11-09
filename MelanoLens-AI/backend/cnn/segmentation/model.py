"""A compact U-Net (encoder-decoder with skip connections). Small enough to train on a CPU."""
from .. import utils

_tf, _keras = None, None


def _libs():
    global _tf, _keras
    if _tf is None:
        _tf, _keras = utils.require_tf()
    return _tf, _keras


def dice_coef(y_true, y_pred, smooth=1.0):
    tf, _ = _libs()
    y_true, y_pred = tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (tf.reduce_sum(y_true, 1) + tf.reduce_sum(y_pred, 1) + smooth))


def iou_metric(y_true, y_pred, smooth=1.0):
    tf, _ = _libs()
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true, y_pred = tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, 1) + tf.reduce_sum(y_pred, 1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy + (1 - Dice): BCE gives stable gradients, Dice handles the small-lesion imbalance."""
    _, keras = _libs()
    bce = keras.ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coef(y_true, y_pred))


def custom_objects():
    return {"dice_coef": dice_coef, "iou_metric": iou_metric, "bce_dice_loss": bce_dice_loss}


def _block(x, filters, layers):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(img_size=128, base_filters=16, depth=4, dropout=0.1):
    _, keras = _libs()
    layers = keras.layers
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x, skips = inputs, []
    for d in range(depth):                                       # encoder
        x = _block(x, base_filters * 2 ** d, layers)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = _block(x, base_filters * 2 ** depth, layers)             # bottleneck
    x = layers.Dropout(dropout)(x)
    for d in reversed(range(depth)):                             # decoder
        x = layers.Conv2DTranspose(base_filters * 2 ** d, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = _block(x, base_filters * 2 ** d, layers)
    out = layers.Conv2D(1, 1, activation="sigmoid", name="lesion_probability")(x)
    return keras.Model(inputs, out, name="unet_skin_lesion")


def compile_model(model, lr=1e-3):
    _, keras = _libs()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=bce_dice_loss, metrics=[dice_coef, iou_metric])
    return model
