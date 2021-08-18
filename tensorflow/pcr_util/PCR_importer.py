import tensorflow as tf
def PCRDataset_fn():
    """Imports from tfio is available, else assumes TensorFlow"""
    try:
        import tfio
        return tfio.experimental.pcr.ProgressiveCompressedRecordDataset
    except:
        return tf.data.ProgressiveCompressedRecordDataset

def parse_fn():
    """Imports from tfio is available, else assumes TensorFlow"""
    try:
        import tfio
        return tfio.experimental.pcr.parse_data_and_label_from_PCR_Dataset
    except:
        return tf.data.parse_data_and_label_from_PCR_Dataset
