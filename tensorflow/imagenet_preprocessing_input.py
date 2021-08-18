import tensorflow.compat.v1 as tf

CROP_PADDING = 32
DEFAULT_IMAGE_SIZE = 224

def preprocess_image_fn(image_bytes, is_training=True, image_size=None,
                        dtype=tf.float32):
  """Preprocess the image. Inception style."""
  if image_size is None:
      image_size = DEFAULT_IMAGE_SIZE
  shape = tf.image.extract_jpeg_shape(image_bytes)
  if is_training:
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack(
        [offset_y, offset_x, target_height, target_width])
  else:
    crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)), tf.int32)
    offset_y, offset_x = [
        ((shape[i] - crop_size) + 1) // 2 for i in range(2)
    ]
    crop_window = tf.stack([offset_y, offset_x, crop_size, crop_size])

  image = tf.image.decode_and_crop_jpeg(
      image_bytes, crop_window, channels=3)
  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
  if is_training:
    image = tf.image.random_flip_left_right(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return tf.image.convert_image_dtype(image, dtype)