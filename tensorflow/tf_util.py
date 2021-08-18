import os
import tensorflow as tf

def disable_tf_gpu():
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
        tf.get_logger().setLevel('ERROR')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

def disable_tfds_gce_check():
    os.environ["NO_GCE_CHECK"] = 'true'  # disable tfds checking

def set_autograph_logging():
    tf.autograph.set_verbosity(10)
