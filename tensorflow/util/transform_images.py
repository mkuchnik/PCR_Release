from functools import partial

import tensorflow as tf
from PIL import Image

from . import pcr_util

@tf.function
def transform_image_to_truncated_progressive(x, scan):
    if not tf.io.is_jpeg(x):
        x = tf.image.decode_image(x)
        tf.print("Converting image to JPEG")
        x = tf.image.encode_jpeg(x, progressive=True)
    encoding_fn = pcr_util.to_progressive_fn
    x = encoding_fn(x)
    x = pcr_util.to_scan_n_jpeg(x, scan)
    return x
