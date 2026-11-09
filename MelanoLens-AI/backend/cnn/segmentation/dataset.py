"""ISIC Task 1 dataset: image/mask matching by ID, consistent resizing, normalisation, augmentation."""
import numpy as np

from .. import utils


def preprocess_image(rgb_uint8, size):
    """THE preprocessing used for training AND inference: area-resize, scale to [0, 1]."""
    return utils.resize_image(rgb_uint8, size).astype(np.float32) / 255.0


def load_arrays(pairs, size):
    """pairs: [(id, image_path, mask_path)] -> uint8 images (N,S,S,3 RGB), uint8 masks (N,S,S,1 in {0,1}).
    Image and mask are both resized to SxS (images: INTER_AREA, masks: INTER_NEAREST) so they stay aligned."""
    imgs, masks, size_mismatch = [], [], 0
    for _id, ip, mp in pairs:
        rgb, m = utils.read_rgb(ip), utils.read_mask(mp)
        if rgb.shape[:2] != m.shape[:2]:
            size_mismatch += 1                                  # still consistent: both become SxS
        imgs.append(utils.resize_image(rgb, size))
        masks.append((utils.resize_mask(m, size) >= 128).astype(np.uint8)[..., None])
    return np.stack(imgs), np.stack(masks), size_mismatch


def make_tf_dataset(images, masks, batch_size, augment=False, seed=42):
    """tf.data pipeline. Augmentation (train only) transforms image AND mask together."""
    tf, _ = utils.require_tf()
    x = tf.constant(images); y = tf.constant(masks)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if augment:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)

    def prep(img, m):
        return tf.cast(img, tf.float32) / 255.0, tf.cast(m, tf.float32)

    def aug(img, m):
        both = tf.concat([img, m], axis=-1)                     # geometry applied to both at once
        both = tf.image.random_flip_left_right(both)
        both = tf.image.random_flip_up_down(both)
        both = tf.image.rot90(both, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img, m = both[..., :3], both[..., 3:]
        img = tf.image.random_brightness(img, 0.1)              # photometric: image only
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return tf.clip_by_value(img, 0.0, 1.0), m

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
