import subprocess
import pathlib
from PIL import Image

def jpeg_to_progressive(image_filename, out_filename=None, attempt_recovery=True):
    """
    Call JPEGTrans to get a bytestream of progressive image back
    Basically:
    cat img.0.jpg | jpegtran -optimize -progressive > img0.jpg
    Return bytes of progressive image
    """
    if out_filename is None:
        out_filename = image_filename
    jpeg_trans_args = ["-copy", "none", "-optimize", "-progressive", "-outfile",
    #jpeg_trans_args = ["-copy", "none", "-optimize", "-outfile",
                       out_filename]
    args = ["jpegtran", *jpeg_trans_args, image_filename]
    popen = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    popen.wait()
    out = popen.stdout.read()
    if len(out) > 0:
        if attempt_recovery:
            print("{}: jpegtran out: {}".format(image_filename, out))
            img = Image.open(image_filename).convert("RGB")
            img.save(image_filename, quality=100, optimize=True, format="JPEG")
            jpeg_to_progressive(image_filename, out_filename, False)
        else:
            raise RuntimeError("{}: jpegtran out: {}".format(image_filename, out))
    err = popen.stderr.read()
    if len(err) > 0:
        if attempt_recovery:
            print("{}: jpegtran err: {}".format(image_filename, err))
            img = Image.open(image_filename).convert("RGB")
            img.save(image_filename, quality=100, optimize=True, format="JPEG")
            jpeg_to_progressive(image_filename, out_filename, False)
        else:
            raise RuntimeError("{}: jpegtran err: {}".format(image_filename, err))
    return None

def jpeg_to_baseline(image_filename, out_filename=None, attempt_recovery=True):
    """
    Call JPEGTrans to get a bytestream of progressive image back
    Basically:
    cat img.0.jpg | jpegtran -optimize -progressive > img0.jpg
    Return bytes of progressive image
    """
    if out_filename is None:
        out_filename = image_filename
    jpeg_trans_args = ["-copy", "none", "-optimize", "-outfile",
                       out_filename]
    args = ["jpegtran", *jpeg_trans_args, image_filename]
    popen = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    popen.wait()
    out = popen.stdout.read()
    if len(out) > 0:
        if attempt_recovery:
            print("{}: jpegtran out: {}".format(image_filename, out))
            img = Image.open(image_filename).convert("RGB")
            img.save(image_filename, quality=100, optimize=True, format="JPEG")
            jpeg_to_baseline(image_filename, out_filename, False)
        else:
            raise RuntimeError("{}: jpegtran out: {}".format(image_filename, out))
    err = popen.stderr.read()
    if len(err) > 0:
        if attempt_recovery:
            print("{}: jpegtran err: {}".format(image_filename, err))
            img = Image.open(image_filename).convert("RGB")
            img.save(image_filename, quality=100, optimize=True, format="JPEG")
            jpeg_to_baseline(image_filename, out_filename, False)
        else:
            raise RuntimeError("{}: jpegtran err: {}".format(image_filename, err))
    return None

def jpeg_recompress(image_filename, quality=100):
    img = Image.open(image_filename).convert("RGB")
    img.save(image_filename, quality=quality, optimize=True, format="JPEG")


def is_JPEG(image_filename):
    try:
        img = Image.open(image_filename)
    except IOError:
        return False
    img_format = img.format
    return img_format == "JPEG"


def convert_dir_to_progressive_jpeg(root_dir):
    """
    Assumes directory is already cleaned (all files are JPEG).
    """
    root_image_path = pathlib.Path(root_dir)

    for image_path in root_image_path.rglob("*"):
        if not image_path.is_dir():
            image_filename = image_path.as_posix()
            if is_JPEG(image_filename):
                jpeg_to_progressive(image_filename)
            else:
                print("Bad file: {}".format(image_filename))
                raise RuntimeError(
                    "Failed conversion on: {}".format(image_filename)
                )


def get_jpeg_image_scans(image_filename):
    args = ["build/scan_only_jsk", image_filename]
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().splitlines()
    image_scans = list(map(lambda x: int(x), output))
    return image_scans

def get_jpeg_partial_images(image_filename):
    """
    Converts a progressive image at a quality setting into a series of scans,
    which are then taken appart via a partitioning function
    If this was done right, you should be able to run Unix tool "cat" to
    concatenate the files to get a valid JPEG.
    """
    image_scans = get_jpeg_image_scans(image_filename)
    with open(image_filename, "rb") as f:
        image_bytes = f.read()
    partial_images = []
    last_s = 0
    for i, s in enumerate(image_scans):
        partial_img_bytes = image_bytes[last_s:s]
        if i == 0:
            assert len(partial_img_bytes) == s
        partial_images.append(partial_img_bytes)
        last_s = s
    assert last_s == (len(image_bytes) - 2)
    return partial_images


if __name__ == "__main__":
    prog_img_bytes = jpeg_to_progressive("test.jpg", "test_prog.jpg")
    print(prog_img_bytes)

    partial_images = get_jpeg_partial_images("test_prog.jpg")
    print(partial_images)
    print(len(partial_images))

    convert_dir_to_progressive_jpeg("test_dir")
