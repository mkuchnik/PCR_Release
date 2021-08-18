import os
import subprocess
import tensorflow.compat.v1 as tf

from pcr_util.PCR_importer import PCRDataset_fn

def get_file_pattern(PCR_dir):
    """
    Creates a glob pattern for PCRs
    """
    file_pattern = os.path.join(PCR_dir,
                                '{}'.format("PCR_*.pcr"))
    return file_pattern

def record_fn(x, scans):
    """
    A wrapper for backward-compatible interface to PCRs. Likely deprecated.
    """
    return tf.data.TFRecordDataset(x,
            compression_type=PCR_dir + "/PCR_index.pb",
            buffer_size=scans)

def parse_record_pcr(record):
  """Parse an ImageNet record from a serialized string Tensor."""
  label_data = tf.strings.substr(record, 0, 4) # Assumes little endian arch
  img_data = tf.strings.substr(record, 4, -1)
  label = tf.io.decode_raw(label_data, tf.int32)
  image_bytes = tf.reshape(img_data, shape=[])
  label = tf.reshape(label, shape=[])
  return image_bytes, label

def _pack_record_pcr(image_bytes, label):
  """Pack two tensors to one."""
  label_bytes = int(label.numpy()).to_bytes(4, byteorder="little")
  img_bytes = bytes(image_bytes.numpy())
  packed_bytes = label_bytes + img_bytes
  return packed_bytes

@tf.function
def pack_record_pcr(image_bytes, label):
  """Pack two tensors to one."""
  return tf.py_function(_pack_record_pcr, [image_bytes, label], tf.string)

@tf.function
def record_fn_imagenet_pcr(x, index_filename, scans):
    metadata_output_type = "labels_first"
    return PCRDataset_fn()(x, scan_groups=scans,
                           index_source_filename=index_filename,
                           metadata_output_type=metadata_output_type)
