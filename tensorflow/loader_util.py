import argparse
import multiprocessing
import util.transform_images
import tf_util
tf_util.disable_tf_gpu()
tf_util.disable_tfds_gce_check()
tf_util.set_autograph_logging()

import tensorflow as tf
import imagenet_preprocessing_input
import tensorflow_datasets as tfds  # Load here to avoid slowdown
import pcr_util.PCR_helper as PCR_helper
import data_prefetcher
import pathlib
import numpy as np
import math


IMAGENET_IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGENET_IMAGE_STD = [0.229, 0.224, 0.225]

BENCH_MODES = ["decode", "load_data_only", "transcode"]
FILE_DATA_FORMATS = ["TFRecord", "files", "PCR"]
DEFAULT_BENCH_MODE = BENCH_MODES[0]
DEFAULT_DATA_FORMAT = FILE_DATA_FORMATS[0]
PARALLELISM_CORE_MULTIPLIER = 1
DEFAULT_PARALLELISM = int(multiprocessing.cpu_count() * PARALLELISM_CORE_MULTIPLIER)
DEFAULT_BUFFER_SIZE = 8
DEFAULT_PCR_BUFFER_SIZE = 10
PREPROCESSING_CHOICES = ["imagenet"]

def get_default_parallelism():
    return DEFAULT_PARALLELISM

def set_default_parallelism(parallelism):
    global DEFAULT_PARALLELISM
    DEFAULT_PARALLELISM = parallelism

def get_default_buffer_size():
    return DEFAULT_BUFFER_SIZE

def set_default_buffer_size(buffer_size):
    global DEFAULT_BUFFER_SIZE_
    DEFAULT_BUFFER_SIZE = buffer_size

def get_TFRecord_dataset(file_pattern: str, deterministic: bool=False, **kwargs):
    """
    Gets data from the dataset_path and returns a tf.data dataset with that
    path.
    Dataset is an iterator over (X, y) tuples of undecoded images.
    """
    if "*" in file_pattern:
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(file_pattern)
    tf.print("Dataset: {}".format(file_pattern))
    if kwargs["worker_index"] is not None:
        dataset = dataset.shard(kwargs["world_size"], kwargs["worker_index"])
    if not deterministic:
        dataset = dataset.shuffle(len(dataset))
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=DEFAULT_PARALLELISM,
                                 block_length=1,
                                 num_parallel_calls=DEFAULT_PARALLELISM,
                                 deterministic=bool(deterministic))
    dataset = dataset.map(unpack_image_record_fn,
                          num_parallel_calls=DEFAULT_PARALLELISM)
    return dataset

@tf.function
def wrapped_parse_pcr_fn(record):
    x, y = PCR_helper.parse_record_pcr(record)
    y = tf.cast(y, tf.int64)
    return x, y

def get_PCR_dataset(file_pattern: str, deterministic: bool=False, **kwargs):
    """
    Gets data from the dataset_path and returns a tf.data dataset with that
    path.
    Dataset is an iterator over (X, y) tuples of undecoded images.
    """
    if "*" in file_pattern:
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(file_pattern)
    index_filename = kwargs["index_filename"]
    scans = kwargs["scans"]
    tf.print("Dataset: {}".format(file_pattern))
    def PCR_Reader_fn(x):
        return PCR_helper.record_fn_imagenet_pcr(x, index_filename, scans)
    if kwargs["worker_index"] is not None:
        dataset = dataset.shard(kwargs["world_size"], kwargs["worker_index"])
    if not deterministic:
        dataset = dataset.shuffle(len(dataset))
    # NOTE(mkuchnik): Have to keep this low otherwise OOM is possible
    # NOTE(mkuchnik): We also expose interleave's prefetch via dataset_ops.py
    def patched_interleave(
            dataset,
            map_func,
            cycle_length=None,
            block_length=None,
            num_parallel_calls=None,
            deterministic=None,
            prefetch_input_elements=None):
        import tensorflow.python.data.ops.dataset_ops as data_ops
        if block_length is None:
          block_length = 1

        if cycle_length is None:
          cycle_length = data_ops.AUTOTUNE

        # NOTE(mkuchnik): Removed check for data_ops.DEBUG_MODE
        if num_parallel_calls is None:
          if deterministic is not None:
            warnings.warn("The `deterministic` argument has no effect unless the "
                          "`num_parallel_calls` argument is specified.")
          return data_ops.InterleaveDataset(dataset, map_func, cycle_length, block_length)
        else:
          return data_ops.ParallelInterleaveDataset(
              dataset,
              map_func,
              cycle_length,
              block_length,
              num_parallel_calls,
              deterministic=deterministic,
              prefetch_input_elements=prefetch_input_elements,
          )
    dataset = patched_interleave(dataset,
                                 PCR_Reader_fn,
                                 cycle_length=DEFAULT_PCR_BUFFER_SIZE,
                                 block_length=1,
                                 num_parallel_calls=DEFAULT_PCR_BUFFER_SIZE,
                                 deterministic=deterministic,
                                 prefetch_input_elements=0)
    dataset = dataset.prefetch(10)
    dataset = dataset.map(wrapped_parse_pcr_fn,
                          num_parallel_calls=DEFAULT_PARALLELISM)
    return dataset

def get_files_dataset(data_dir: str, deterministic: bool=False, directory:
                      str=None, **kwargs):
    """Same as above, but with folder structure"""
    builder = tfds.ImageFolder(data_dir)
    decoders = {
        "image": tfds.decode.SkipDecoding(),
    }
    if not directory:
        directory = "train"
    if kwargs["worker_index"] is not None:
        input_context = tf.distribute.InputContext(
            input_pipeline_id=kwargs["worker_index"],  # Worker id
            num_input_pipelines=kwargs["world_size"],  # Total number of workers
        )
        read_config = tfds.ReadConfig(
            input_context=input_context,
        )
        print("Sharding dataset for files", input_context)
    else:
        read_config = None

    # TODO(mkuchnik): for now, we are not using read_config, since sharding is
    # not controlled
    read_config = None

    # Enable shuffle_files to benchmark file IO
    dataset = builder.as_dataset(directory,
                                 shuffle_files=False,
                                 as_supervised=True,
                                 decoders=decoders,
                                 read_config=read_config)

    if kwargs["worker_index"] is not None and kwargs["world_size"] != -1:
        dataset = dataset.shard(kwargs["world_size"], kwargs["worker_index"])
    if not deterministic:
        dataset = dataset.shuffle(len(dataset))
    return dataset

def apply_dataset_options(dataset, deterministic: bool=False,
                          get_stats:bool=False):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    options.experimental_threading.max_intra_op_parallelism = 1
    num_cores = multiprocessing.cpu_count()
    options.experimental_threading.private_threadpool_size = num_cores
    if get_stats:
        options.experimental_optimization.autotune_stats_filename = "stats.pb"
    # Enable this if
    # dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.with_options(options)
    return dataset

@tf.function
def record_fn(record):
    """Extracts TFRecord content into dict"""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    return parsed

@tf.function
def unpack_image_record_fn(record):
    """Extracts TFRecord content into tuples"""
    parsed = record_fn(record)
    x, y =  parsed["image/encoded"], parsed["image/class/label"]
    labels = tf.cast(tf.reshape(y, []), tf.int64)
    # Subtract 1 to eliminate background class
    labels = labels - 1
    return x, labels

def get_raw_tfdata_loader(dataset_path: str, data_format: str,
                          deterministic: bool=False, directory: str=None,
                          worker_index=None, world_size=None,
                          **kwargs):
    """Reads files"""
    assert data_format in FILE_DATA_FORMATS
    if data_format == "TFRecord":
        ds = get_TFRecord_dataset(dataset_path, deterministic=deterministic,
                                  worker_index=worker_index,
                                  world_size=world_size,
                                  **kwargs)
    elif data_format == "PCR":
        ds = get_PCR_dataset(dataset_path, deterministic=deterministic,
                             worker_index=worker_index,
                             world_size=world_size,
                             **kwargs)
    else:
        assert data_format == "files"
        ds = get_files_dataset(dataset_path, deterministic=deterministic,
                               directory=directory,
                               worker_index=worker_index,
                               world_size=world_size,
                               **kwargs)
    return ds

def get_basic_tfdata_loader(dataset_path: str, data_format: str, scan: int,
                            deterministic: bool=False, directory: str=None,
                            worker_index: int=None, world_size: int=None,
                            finite: bool=True):
    """
    Gets the tf.data loader for the dataset_path 
    This only guarantees that the RAW data is returned.
    Performs scan emulation if desired.
    """
    if ((worker_index is not None or world_size is not None)
            and (not worker_index is not None or not world_size is not None)):
        raise ValueError("worker_index and world_size must both be set: {},"
                         " {}".format(worker_index, world_size))
    root_path = pathlib.Path(dataset_path).parent
    if data_format == "PCR":
        index_filename = (root_path / "PCR_index.pb").as_posix()
        scans = scan
        ds = get_raw_tfdata_loader(dataset_path, data_format, deterministic,
                                   directory, scans=scan,
                                   index_filename=index_filename,
                                   worker_index=worker_index,
                                   world_size=world_size)
    else:
        ds = get_raw_tfdata_loader(dataset_path, data_format, deterministic,
                                   directory,
                                   worker_index=worker_index,
                                   world_size=world_size)
    tf.print("Using scan: {}".format(scan))
    if scan and data_format != "PCR":
        raise RuntimeError("Requesting scan {} but not PCR format".format(scan))
    return ds

def get_dataset_length(dataset):
    for i, _ in enumerate(dataset):
        pass
    return i + 1

# Move channel to second slot
@tf.function
def normalize_image_imagenet(x):
    x /= 255.0     # Convert to [0, 1]
    x -= IMAGENET_IMAGE_MEAN  # Subtract mean
    x /= IMAGENET_IMAGE_STD   # Reduce std
    return x

def get_dataset_batch(ds):
    for x in ds:
        return x
    return None

def sample_dataset_images(ds, filename_prefix):
    batch = get_dataset_batch(ds)
    if isinstance(batch, tuple) and len(batch) and len(batch[0].shape) > 3:
        for i, x in enumerate(batch[0]):
            filename = str(filename_prefix) + str(i) + ".png"
            util.transform_images.save_image(x, filename)
    if isinstance(batch, tuple) and len(batch):
        i = 0
        x = batch[0]
        filename = str(filename_prefix) + str(i) + ".png"
        util.transform_images.save_image(x, filename)
    else:
        for i, x in enumerate(batch):
            filename = str(filename_prefix) + str(i) + ".png"
            util.transform_images.save_image(x, filename)

def sample_dataset_sizes(ds, dim_select=None, num_samples:int=None):
    if num_samples is None:
        num_samples = 100
    size_fn = lambda x: tf.strings.length(x, name="size")
    if dim_select is not None:
        dim_select = int(dim_select)
        ds = ds.map(lambda *x: size_fn(x[dim_select]), DEFAULT_PARALLELISM)
    else:
        ds = ds.map(lambda x: size_fn(x), DEFAULT_PARALLELISM)
    ds = ds.prefetch(10)
    ds = apply_dataset_options(ds, deterministic=True)
    samples = [x for x in ds.take(num_samples)]
    return samples

def get_imagenet_tfdata_sampler_loader(dataset_path: str, data_format: str,
                                       scan: int, decode: bool=True, directory:
                                       str=None):
    """Just for printing example images"""
    ds_base = get_basic_tfdata_loader(dataset_path, data_format, scan,
                                      deterministic=True, directory=directory)
    ds = ds_base.cache()
    if decode:
        ds = ds.map(lambda x, y:
                    (imagenet_preprocessing_input.preprocess_image_fn(x), y),
                    DEFAULT_PARALLELISM)
    ds = ds.prefetch(1)
    ds = apply_dataset_options(ds, deterministic=True)
    return ds

@tf.function
def apply_image_fused_transformations(x, image_size,
                                      processing_function,
                                      randomized_image_postprocessing:bool=True):
    """Normalize and crop.

    processing_function is something like
    imagenet_preprocessing_input.preprocess_image_fn
    """
    return normalize_image_imagenet(
        processing_function(x,
                            image_size=image_size,
                            is_training=randomized_image_postprocessing))

def apply_dataset_imagenet_transformations(
        ds, image_size: int, batch_size: int,
        permute: bool, cpu_permute: bool,
        image_normalization: bool=True,
        randomized_image_postprocessing: bool=True,
        preprocessing_type: str=None
):
    """Normalization and cropping, batching, and possibly transposing"""

    if preprocessing_type is None or preprocessing_type == "imagenet":
        processing_function = imagenet_preprocessing_input.preprocess_image_fn
    else:
        raise ValueError("Unknown preprocessing_type:"
                         "{} of {}".format(preprocessing_type,
                                           PREPROCESSING_CHOICES))
    if image_normalization:
        ds = ds.map(lambda x, y:
                    (apply_image_fused_transformations(
                        x, image_size=image_size,
                        processing_function=processing_function,
                        randomized_image_postprocessing=randomized_image_postprocessing), y),
                    DEFAULT_PARALLELISM)
    else:
        def preprocess_and_cast(x):
            x = processing_function(x,
                                    image_size=image_size,
                                    is_training=randomized_image_postprocessing)
            x = tf.cast(x, tf.float32)  # Cast to match normalization
            return x
        ds = ds.map(lambda x, y: (preprocess_and_cast(x), y),
                    DEFAULT_PARALLELISM)
    if batch_size:
        ds = ds.batch(batch_size)
    if permute and cpu_permute:
        transpose_array = [0, 3, 1, 2]
        ds = ds.map(
            lambda imgs, labels:
            (tf.transpose(imgs, transpose_array), labels),
            DEFAULT_PARALLELISM)
    return ds

def get_imagenet_tfdata_loader(dataset_path: str, data_format: str, scan: int,
                               batch_size: int, shuffle_size: int = None,
                               dataset_size: int = None,
                               cache_dataset: bool=True, permute: bool=True,
                               cpu_permute: bool=True, image_size: int=None,
                               randomized_image_postprocessing: bool=True,
                               snapshot_filename=None,
                               take_amount=None,
                               directory: str=None,
                               worker_index=None,
                               world_size=None,
                               deterministic=False,
                               preprocessing_type=None):
    """
    Directory is usually 'train', but can be others like 'val' or 'test',
    depending on the subdirectory under dataset_path.
    """
    ds_base = get_basic_tfdata_loader(dataset_path, data_format, scan,
                                      directory=directory,
                                      deterministic=deterministic,
                                      worker_index=worker_index,
                                      world_size=world_size,
                                      finite=bool(cache_dataset))

    if dataset_size is None and hasattr(ds_base, "_PCR_dataset_size"):
        pcr_size = ds_base._PCR_dataset_size
        tf.print("Found PCR dataset size: {}".format(pcr_size))
        if world_size is not None:
            dataset_size = math.ceil(pcr_size / batch_size / world_size)
        else:
            dataset_size = math.ceil(pcr_size / batch_size)
        shuffle_size = dataset_size
    if take_amount:
        tf.print("Taking subset of examples: {}".format(take_amount))
        ds_base = ds_base.take(take_amount)
    if snapshot_filename:
        tf.print("Adding snapshot to {}".format(snapshot_filename))
        ds_base = ds_base.apply(tf.data.experimental.snapshot(snapshot_filename))
    if cache_dataset:
        tf.print("Caching dataset")
        ds = ds_base.cache()
    else:
        ds = ds_base
    if shuffle_size is None:
        shuffle_size = get_dataset_length(ds)
        if dataset_size is not None:
            if dataset_size != shuffle_size:
                print("WARNING: shuffle_size and inferred size differ {} "
                      "{}".format(shuffle_size, dataset_size))
        else:
            dataset_size = shuffle_size
    if shuffle_size:
        ds = ds.shuffle(shuffle_size)
    ds_trans = apply_dataset_imagenet_transformations(ds, image_size,
                                                      batch_size, permute,
                                                      cpu_permute,
                                                      randomized_image_postprocessing=randomized_image_postprocessing,
                                                      preprocessing_type=preprocessing_type)
    ds = ds_trans  # Keep reference to finite variant
    ds = ds.repeat()  # Repeat at end for distinct epochs
    ds = ds.prefetch(DEFAULT_BUFFER_SIZE)
    if dataset_size is None:
        data_length = get_dataset_length(ds_trans)
        print("Using inferred dataset size: {}".format(data_length))
    else:
        data_length = dataset_size
        print("Using pre-computed dataset size: {}".format(data_length))
    ds = apply_dataset_options(ds, deterministic=False)
    ds = data_prefetcher.data_prefetcher(ds, length=data_length,
                                         permute_channel=permute and not cpu_permute)
    return ds

def get_synthetic_tfdata_loader(batch_size: int, image_size: int,
                                dataset_size: int,
                                permute: bool=True):
    # NOTE(mkuchnik): We don't use drop_remainder
    input_fn = get_synth_input_fn(height=image_size,
                                  width=image_size,
                                  num_channels=3,
                                  num_classes=1000,
                                  dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  flatten_labels=True,
                                  drop_remainder=False)
    is_training = True
    data_dir = None
    ds = input_fn(is_training, data_dir, batch_size)
    if permute:
        transpose_array = [0, 3, 1, 2]
        ds = ds.map(
            lambda imgs, labels:
            (tf.transpose(imgs, transpose_array), labels),
            DEFAULT_PARALLELISM)
    ds = apply_dataset_options(ds, deterministic=False)
    ds = data_prefetcher.data_prefetcher(ds, length=dataset_size,
                                         permute_channel=False)
    return ds


class ScanViewDataset:
    """Returns multiple views of the same base dataset.
    Base data is just a in-memory tuple of raw image_data and labels.
    """
    def __init__(self, data):
        assert isinstance(data, tuple) and len(data) == 2, \
            "Expected tuple of data: (image_data, labels)"
        assert len(data[0]) == len(data[1]), \
            "Expected data to have same length"
        self.data = data
        self.dataset_data = tf.data.Dataset.from_tensor_slices(data)

    def get_base_dataset(self):
        return self.dataset_data

    def apply_dataset_transformations(self, ds, image_size: int,
                                      batch_size: int, permute: bool,
                                      image_normalization: bool,
                                      randomized_image_postprocessing: bool):
        """Apply image processing to dataset"""
        return apply_dataset_imagenet_transformations(ds,
                                                      image_size=image_size,
                                                      batch_size=batch_size,
                                                      permute=permute,
                                                      cpu_permute=True,
                                                      image_normalization=image_normalization,
                                                      randomized_image_postprocessing=randomized_image_postprocessing)

    def get_scan_dataset(self, scan: int, image_size: int, batch_size: int,
                         permute: bool,
                         image_normalization: bool = True,
                         randomized_image_postprocessing: bool=True,
                         apply_prefetcher: bool=False):
        assert scan > 0
        ds = self.get_base_dataset()
        ds = ds.map(lambda x, y:
                    (util.transform_images.transform_image_to_truncated_progressive(
                        x, scan), y), DEFAULT_PARALLELISM)
        ds = self.apply_dataset_transformations(ds, image_size, batch_size,
                                                permute=permute,
                                                image_normalization=image_normalization,
                                                randomized_image_postprocessing=randomized_image_postprocessing)
        if apply_prefetcher:
            ds = data_prefetcher.data_prefetcher(ds, length=None,
                                                 permute_channel=False)
        return ds

    def get_baseline_dataset(self, image_size: int, batch_size: int,
                             permute: bool,
                             image_normalization: bool = True,
                             randomized_image_postprocessing: bool=True,
                             apply_prefetcher: bool=False):
        ds = self.get_base_dataset()
        ds = self.apply_dataset_transformations(ds, image_size, batch_size,
                                                permute=permute,
                                                image_normalization=image_normalization,
                                                randomized_image_postprocessing=randomized_image_postprocessing)
        if apply_prefetcher:
            ds = data_prefetcher.data_prefetcher(ds, length=None,
                                                 permute_channel=False)
        return ds

def get_synth_input_fn(height,
                       width,
                       num_channels,
                       num_classes,
                       dtype=tf.float32,
                       label_dtype=tf.float32,
                       flatten_labels=False,
                       drop_remainder=True):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tuning the full input pipeline.

  from Resnet official tensorflow example

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """

  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    inputs, labels = get_synth_data(
        height=height,
        width=width,
        num_channels=num_channels,
        num_classes=num_classes,
        dtype=dtype)
    if flatten_labels:
        labels = tf.reshape(labels, [])
    if label_dtype:
        # We can cast to float32 for Keras model.
        labels = tf.cast(labels, dtype=label_dtype)
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()

    # `drop_remainder` will make dataset produce outputs with known shapes.
    data = data.batch(batch_size, drop_remainder=drop_remainder)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn

def get_synth_data(height, width, num_channels, num_classes, dtype):
  """Creates a set of synthetic random data.

  from Resnet official tensorflow example

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    A tuple of tensors representing the inputs and labels.

  """
  # Synthetic input should be within [0, 255].
  inputs = tf.random.truncated_normal([height, width, num_channels],
                                      dtype=dtype,
                                      mean=127,
                                      stddev=60,
                                      name='synthetic_inputs')
  labels = tf.random.uniform([1],
                             minval=0,
                             maxval=num_classes - 1,
                             dtype=tf.int32,
                             name='synthetic_labels')
  return inputs, labels
