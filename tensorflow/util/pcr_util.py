import tensorflow as tf
import numpy as np
import subprocess
import io
import tempfile
from PIL import Image
import time

JPEGTRAN_NAME = "jpegtran"

def to_progressive_jpeg(image: str, scans_file: str = None, ignore_error: bool =
                        True, optimize=False, use_file=False):
    args = [JPEGTRAN_NAME, '-progressive']
    if optimize:
        args.append('-optimize')
    if scans_file:
        args.extend(["-scans", str(scans_file)])
    if use_file:
        f = tempfile.NamedTemporaryFile(mode="w+b")
        f.write(image)
        f.flush()
        args.append(f.name)
        p_input = None
        p_output = subprocess.PIPE
    else:
        p_input = image
        p_output = subprocess.PIPE
    if ignore_error:
        try:
            output = subprocess.run(args, stdout=p_output, input=p_input,
                                    check=True, bufsize=-1)
        except subprocess.CalledProcessError as ex:
            print(ex)
            try:
                image = to_jpeg(image)
                output = subprocess.run(args, stdout=p_output,
                                        input=p_input,
                                       check=True, bufsize=-1)
            except subprocess.CalledProcessError as ex:
                print(ex)
                return image
    if use_file:
        f.close()
    output = output.stdout
    return output

def get_jpeg_image_scans(image_filename: str):
    """
    Returns valid scans of image_filename
    """
    args = ["scan_only_jsk", image_filename]
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=-1)
    popen.wait()
    output = popen.stdout.read().splitlines()
    jsk_image_scans = list(map(lambda x: int(x), output))
    return jsk_image_scans

def _to_scan_n_jpeg(image, scan):
    image = image.numpy()
    scan = int(scan.numpy())
    with tempfile.NamedTemporaryFile(suffix="_image.jpeg") as image_temp:
        image_filename = image_temp.name
        with open(image_filename, "wb") as f:
            f.write(image)
        scans = get_jpeg_image_scans(image_filename)
        if not len(scans) == 10:
            print("Scan count is {}".format(len(scans)))
        scan_offset = scans[min(scan-1, len(scans)-1)]
        truncated_image_bytes = image[0:scan_offset]
        truncated_image_bytes += bytes.fromhex("FFD9")
        return truncated_image_bytes

@tf.function
def to_scan_n_jpeg(image, scan):
    return tf.py_function(_to_scan_n_jpeg, [image, scan], tf.string)

def _to_progressive_fn(x):
    x = x.numpy()
    return to_progressive_jpeg(x)

@tf.function
def to_progressive_fn(x):
    return tf.py_function(_to_progressive_fn, [x], tf.string)
