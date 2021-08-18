import collections
import pathlib
import random
import io
from PIL import Image
import numpy as np
import multiprocessing
import tqdm
import pandas as pd
import shutil
import time
import itertools

import argparse

from sqlitedict import SqliteDict
import proto.LightMLRecords_pb2 as LightMLRecords_pb2
import proto.progressive_compressed_record_pb2 as PCR_pb2
import progressive_utils
import metric

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.compat.v1.disable_eager_execution()

def get_rocksdb_instance(name):
    import rocksdb
    options = rocksdb.Options()
    options.create_if_missing = True
    db = rocksdb.DB(name, options)
    return db

# https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    start_idx = 0
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            mb_inputs = [inputs[i] for i in excerpt]
            mb_targets = [targets[i] for i in excerpt]
            yield mb_inputs, mb_targets
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
    if start_idx < len(inputs):
        excerpt = slice(start_idx, len(inputs))
        yield inputs[excerpt], targets[excerpt]

def create_parser():
    """
    Creates a parser
    """
    parser = argparse.ArgumentParser(description='PyTorch PCR Training')
    # TODO(mkuchnik): Add tar wrapper option
    parser.add_argument('dataset_root', type=str,
                        help='The path to injest the dataset from')
    parser.add_argument('out_root', type=str,
                        help='The path to place the dataset in')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='the batch size to use for PCRs')
    parser.add_argument('--pool_size', default=None, type=int,
                        help='The number of threads to use for PCR creation')
    parser.add_argument('--convert_images', default=False,
                        action="store_true",
                        help='Convert the Images to Progressive format in place')
    parser.add_argument('--static_compression', default=100, type=int,
                        help='The level of compression to apply')
    parser.add_argument('--mssim_estimate_size', default=0, type=int,
                        help='The number of samples to use for MSSIM esimates')
    parser.add_argument('--force', action="store_true",
                        help='Override previous PCR dir')
    parser.add_argument('--no_shuffle', action="store_true",
                        help="don't shuffle the dataset")
    parser.add_argument('--raw_image_bytes', action="store_true",
                        help="don't pack images in protobuf")
    parser.add_argument('--backends', default=["protobuf"], nargs="+",
                        help="The converter backends to use (list)")
    parser.add_argument('--use_baseline_images', action="store_true",
                        help="Use baseline images for convert")
    parser.add_argument('--duplicate_dataset_factor', type=int, default=1,
                        help="Duplicate dataset n (int) times")
    parser.add_argument('--tf_records', action="store_true",
                        help="If to dump out tf_records rather than PCRs")
    return parser

def get_directory_filenames_and_labels(root_dir):
    root_filepath = pathlib.Path(root_dir).resolve()
    root_path_folders = []
    dir_to_filenames = collections.defaultdict(list)
    for x in root_filepath.iterdir():
        if x.is_dir():
            for f in x.iterdir():
                if f.is_dir():
                    print("Not file {}".format(f))
                else:
                    dir_to_filenames[x.as_posix()].append(f)
            root_path_folders.append(x)
        else:
            print("Not directory {}".format(x))

    # Assumption: The same amount of folders in train/test
    # PyTorch uses sorted order: https://github.com/pytorch/vision/issues/192
    root_path_folders = sorted(root_path_folders)
    dir_to_class = {x.as_posix(): i for i, x in enumerate(root_path_folders)}

    filename_class_tuples = []
    for folder in root_path_folders:
        folder_class = dir_to_class[folder.as_posix()]
        folder_filenames = dir_to_filenames[folder.as_posix()]
        file_classes = [(f_name, folder_class) for f_name in
                        folder_filenames]
        filename_class_tuples.extend(file_classes)

    print("file_class", filename_class_tuples)

    return filename_class_tuples

def prog_image_to_mssim(image_filename):
    image_partials = progressive_utils.get_jpeg_partial_images(image_filename)
    original_image = np.array(Image.open(image_filename).convert("RGB"),
                              dtype=np.float32)

    mssims = []
    for i in range(1, len(image_partials) + 1):
        partial_image = b"".join(image_partials[:i])
        if partial_image[-2:] != bytes.fromhex("FFD9"):
            partial_image += bytes.fromhex("FFD9")

        iobytes = io.BytesIO(partial_image)
        reconstructed_image = np.array(Image.open(iobytes).convert("RGB"),
                                       dtype=np.float32)
        mssim = metric.msssim(original_image, reconstructed_image)
        mssims.append(mssim)

    return mssims

def minibatch_to_PCR_records(image_filenames, image_classes, raw_bytes):
    """
    Converts a (X,Y) dataset into a tfrecord (progressive compression)
    optimized for partial (offset) reading.
    Returns a list of TFRecord representation, which can then be serialized
    The first record is metadata and labels, and all the following records are
    scan groups.
    raw bytes: If the images should be stored raw or with protobuf containers
    Returns a metadata record and
    the scan_records. If raw_bytes, the scan_records are just lists of the
    grouped partials; else, they're protobufs.
    The scan_index for can be recreated by the caller.
    """
    assert len(image_filenames) == len(image_classes)
    meta_record = LightMLRecords_pb2.MetadataRecord()
    all_partial_images = []
    most_scans = 0
    for (f, c) in zip(image_filenames, image_classes):
        print("F, c", f, c)
        partial_images = progressive_utils.get_jpeg_partial_images(f)
        most_scans = max(most_scans, len(partial_images))
        all_partial_images.append(partial_images)
        meta_record.labels.append(c)

    meta_record.progressive_levels = most_scans
    if raw_bytes:
        meta_record.version = 2
    else:
        meta_record.version = 1

    # We postprocess partials to add extra padding scans
    for image_scans in all_partial_images:
        while len(image_scans) < most_scans:
            # TODO we will ignore missed scans
            #image_scans.append(None)
            image_scans.append(b"")

    # We want to take all scans of index i and group them together
    all_grouped_partials = [[] for _ in range(most_scans)]
    for i in range(most_scans):
        grouped_partial = [image_scans[i]
                           for image_scans in all_partial_images]
        if b"" in grouped_partial:
            print("Found empty partial")
            print("on scan {} out of {}".format(i, most_scans))
            #raise RuntimeError("Variable scans not implemented: {}".format(image_filenames))
        all_grouped_partials[i].extend(grouped_partial)

    assert len(all_grouped_partials) == most_scans

    scan_records = []
    for grouped_partial in all_grouped_partials:
        assert len(grouped_partial) == len(image_classes)
        if raw_bytes:
            scan_records.append(grouped_partial)
        else:
            scan_record = LightMLRecords_pb2.ScanGroup()
            scan = scan_record.image_bytes.extend(grouped_partial)
            scan_records.append(scan_record)

    return [meta_record, *scan_records]

def minibatch_to_PCR_file(record_filename, image_filenames, image_classes,
                          raw_bytes=False):
    """
    Packs the records and writes them to disk. Returns indexing information
    (both a protobuf included scan_group index and a full scan_index).
    """
    PCR_records = minibatch_to_PCR_records(image_filenames, image_classes,
                                           raw_bytes)
    full_scan_index = [] # 2d (ragged) array of batch_size X scan_id
    prog_compressed_tf_records_str = []
    for i in range(len(PCR_records)):
        if i == 0:
            # metadata payload
            payload = PCR_records[i].SerializeToString()
        else:
            # TODO
            # each payload is a scan
            # len(payload) is batch_size
            # each element of payload is a partial image (bytes)
            payload = PCR_records[i]
            image_bytes = payload.image_bytes if not raw_bytes else payload
            scan_sizes = np.array([len(s) for s in image_bytes])
            full_scan_index.append(scan_sizes)
            if raw_bytes:
                payload = b"".join(payload)
            else:
                payload = payload.SerializeToString()
        prog_compressed_tf_records_str.append(payload)
    scan_group_index = list(
        map(len, prog_compressed_tf_records_str)
    )
    serialized_prog_records = b"".join(prog_compressed_tf_records_str)
    with open(record_filename, "wb") as record_file:
        record_file.write(serialized_prog_records)
    return (scan_group_index, full_scan_index)

# From Tensorflow tutorials
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    with tf.device("/CPU:0"):
        image_shape = tf.image.decode_jpeg(image_string).shape

        feature = {
            #'height': _int64_feature(image_shape[0]),
            #'width': _int64_feature(image_shape[1]),
            #'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

def minibatch_to_tfrecords_file(record_filename, image_filenames, image_classes,
                          raw_bytes=False):
    """
    Packs the records and writes them to disk. Returns indexing information
    (both a protobuf included scan_group index and a full scan_index).
    """
    # TODO remove raw bytes
    with tf.device("/CPU:0"):
        with tf.io.TFRecordWriter(record_filename) as record_file:
            for filename, label in zip(image_filenames, image_classes):
                image_string = open(filename, "rb").read()
                tf_example = image_example(image_string, label)
                record_file.write(tf_example.SerializeToString())
    scan_group_index = None
    full_scan_index = None
    return (scan_group_index, full_scan_index)


def minibatch_to_PCR_file_w_filename(record_filename, image_filenames,
                                     image_classes, raw_bytes):
    ret = minibatch_to_PCR_file(record_filename, image_filenames, image_classes,
                                raw_bytes)
    return (record_filename, ret)

def minibatch_to_tfrecords_file_w_filename(record_filename, image_filenames,
                                     image_classes, raw_bytes):
    ret = minibatch_to_tfrecords_file(record_filename, image_filenames, image_classes,
                                raw_bytes)
    return (record_filename, ret)

def minibatch_to_PCR_file_imap(args):
    record_filename, image_filenames, image_classes, raw_bytes = args
    return minibatch_to_PCR_file_w_filename(record_filename,
                                            image_filenames,
                                            image_classes,
                                            raw_bytes)

def minibatch_to_tfrecords_file_imap(args):
    record_filename, image_filenames, image_classes, raw_bytes = args
    return minibatch_to_tfrecords_file_w_filename(record_filename,
                                            image_filenames,
                                            image_classes,
                                            raw_bytes)

def jpeg_recompress_imap(args):
    f, quality = args
    return progressive_utils.jpeg_recompress(f, quality)

def pad_last_value(arr, arr_len):
    n_copies = arr_len - len(arr)
    assert n_copies >= 0
    if n_copies:
        print("Padding", n_copies)
    value = arr[-1]
    arr.extend([value for _ in range(n_copies)])
    return arr

def dataset_to_BCR(root_dir,
                   out_dir,
                   batch_size=512,
                   force=False,
                   pool_size=None,
                   mssim_estimate_size=256,
                   convert_images=False,
                   static_compression_quality=100,
                   shuffle=True,
                   backends=["protobuf"],
                   raw_image_bytes=False,
                   use_baseline_images=False,
                   duplicate_dataset_factor=1,
                   tf_records=False,
                   ):
    """
    Iterates over an epoch of a dataset and writes out the data to a directory
    """
    root_filepath = pathlib.Path(root_dir).resolve()
    out_filepath = pathlib.Path(out_dir).resolve()

    assert isinstance(duplicate_dataset_factor, int)

    f_c_list = get_directory_filenames_and_labels(root_dir)
    print("f_c_list", f_c_list)

    if pool_size is None:
        pool_size = 4 * multiprocessing.cpu_count()

    files, classes = list(zip(*f_c_list))
    print("Files", files)
    print("classes", classes)
    start_time = time.time()
    if static_compression_quality != 100:
        print("Downgrading quality of images")
        with multiprocessing.Pool(pool_size) as pool:
            inputs = [(f, static_compression_quality) for f in files]
            for _ in tqdm.tqdm(pool.imap_unordered(
                    jpeg_recompress_imap,
                    inputs,
                    chunksize=32), total=len(files)):
                pass
    if convert_images:
        print("Converting all images")
        if not use_baseline_images:
            print("Converting to progressive")
            # TODO add baseline images
            with multiprocessing.Pool(pool_size) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(
                        progressive_utils.jpeg_to_progressive,
                        files,
                        chunksize=32), total=len(files)):
                    pass
        else:
            print("Converting to baseline")
            with multiprocessing.Pool(pool_size) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(
                        progressive_utils.jpeg_to_baseline,
                        files,
                        chunksize=32), total=len(files)):
                    pass
    end_time = time.time()
    convert_time = end_time - start_time

    if shuffle:
        # Shuffle now
        random.shuffle(f_c_list)

    files, classes = list(zip(*f_c_list))
    print("Files", files)
    print("classes", classes)

    start_time = time.time()
    if mssim_estimate_size > 0:
        print("Calculating MSSIM")
        all_mssims = []
        with multiprocessing.Pool(pool_size) as pool:
            for mssims in tqdm.tqdm(pool.imap_unordered(
                    prog_image_to_mssim,
                    files[:mssim_estimate_size],
                    chunksize=4), total=mssim_estimate_size):
                all_mssims.append(mssims)
        all_mssims = list(map(lambda x: pad_last_value(x, 10), all_mssims))
        all_mssims = np.array(all_mssims)
        print("all mssims", all_mssims)
        print("all mssims", all_mssims.shape)
        df_mssims = pd.DataFrame.from_records(all_mssims)
        print("df mssims", df_mssims)
        df_mssims.to_csv("PCR_Conversion_MSSIM.csv")
        avg_mssims = np.mean(all_mssims, axis=0)
        print("avg mssims", avg_mssims)
    end_time = time.time()
    mssim_time = end_time - start_time

    # TODO maybe check that file is not overwritten
    if not out_filepath.exists():
        print("Creating PCR dir")
        out_filepath.mkdir(parents=True, exist_ok=True)

    def check_db_path_or_remove(db_path, force):
        if db_path.exists():
            print(db_path)
            if force:
                print("PCR db exists! Removing")
                shutil.rmtree(out_filepath.as_posix())
                out_filepath.mkdir(parents=True, exist_ok=True)
            else:
                print("PCR db exists! skipping")
                return

    if not tf_records:
        print("Connecting to DB")
        dbs = {}
        for backend in backends:
            if backend == "sql":
                db_path = out_filepath / "PCR.db"
                check_db_path_or_remove(db_path, force)
                db = SqliteDict(db_path.as_posix(),
                                autocommit=True)
                db["m_dataset_size"] = len(files)
                db["m_classes"] = np.unique(classes)
                db["m_batch_size"] = batch_size
            elif backend == "rocksdb":
                db_path = out_filepath / "PCR_rocksdb.db"
                check_db_path_or_remove(db_path, force)
                try:
                    db = get_rocksdb_instance(db_path.as_posix())
                except ImportError as ex:
                    raise ex
                    print(ex)
                    continue
                except AttributeError as ex:
                    raise ex
                    print(ex)
                    continue
                key_values = [("m_dataset_size", len(files)),
                              ("m_classes", ",".join(map(str, np.unique(classes)))),
                              ("m_batch_size", batch_size)]
                for k, v in key_values:
                    db.put(str.encode(k), str.encode(str(v)))
            elif backend == "protobuf":
                db_path = out_filepath / "PCR_index.pb"
                check_db_path_or_remove(db_path, force)
                db = PCR_pb2.DatasetOffsetsIndex()
            else:
                raise RuntimeError("Not supported backend: {}".format(backend))
            dbs[backend] = db
    else:
        print("Skipping DB!")

    print("top files", files[:10])
    print("top classes", classes[:10])
    mb_iter = iterate_minibatches(files,
                                  classes,
                                  batch_size,
                                  shuffle=False)
    PCR_args = []
    for i, (mb_f, mb_c) in enumerate(
        itertools.chain.from_iterable(itertools.repeat(list(mb_iter),
                                                       duplicate_dataset_factor))
    ):
        print("PCR_{}".format(i))
        assert len(mb_f) == len(mb_c)
        if tf_records:
            record_filepath = out_filepath / "tfrecord_{}.pcr".format(i)
        else:
            record_filepath = out_filepath / "PCR_{}.pcr".format(i)
        record_filename = record_filepath.as_posix()
        PCR_args.append((record_filename, mb_f, mb_c, raw_image_bytes))

    start_time = time.time()
    if not tf_records:
        print("Converting to PCR")
        all_PCR_tup = []
        with multiprocessing.Pool(pool_size) as pool:
            for PCR_tup in tqdm.tqdm(pool.imap_unordered(
                    minibatch_to_PCR_file_imap,
                    PCR_args,
                    chunksize=4), total=len(PCR_args)):
                print("PCR_tup", PCR_tup)
                record_name, (group_scan_index, full_scan_index) = PCR_tup
                print("record_name", record_name)
                for backend in backends:
                    k_clean = record_name
                    if len(record_name) and record_name[0] == "/":
                        # if a PCR filepath, normalize
                        k_file = pathlib.Path(record_name)
                        k_clean = "/" + k_file.name
                    try:
                        db = dbs[backend]
                    except:
                        print("Can't find backend: {}".format(backend))
                        continue
                    print("k_clean", k_clean)
                    if backend == "sql":
                        db[k_clean] = group_scan_index
                        for i, full_idx in enumerate(full_scan_index):
                            full_index_name = "m_PCR_full_index_{}_{}".format(k_clean[1:], i)
                            print("full_index_name", full_index_name)
                            db[full_index_name] = full_idx
                    elif backend == "rocksdb":
                        db.put(str.encode(k_clean),
                               str.encode(",".join(map(str, group_scan_index))))
                        for i, full_idx in enumerate(full_scan_index):
                            full_index_name = "m_PCR_full_index_{}_{}".format(k_clean[1:], i)
                            full_index_str = ",".join(map(str, full_idx))
                            print("full_index_name", full_index_name)
                            db.put(str.encode(full_index_name),
                                   str.encode(full_index_str))
                    elif backend == "protobuf":
                        for i, full_idx in enumerate(full_scan_index):
                            record_offsets_index = PCR_pb2.RecordOffsetsIndex()
                            record_offsets_index.name = k_clean[1:]
                            record_offsets_index.offsets.extend(group_scan_index)
                            db.records.append(record_offsets_index)
                    else:
                        raise RuntimeError("Not supported backend: {}".format(backend))
    else:
        print("Converting to PCR")
        all_PCR_tup = []
        with multiprocessing.Pool(pool_size) as pool:
            for PCR_tup in tqdm.tqdm(pool.imap_unordered(
                    minibatch_to_tfrecords_file_imap,
                    PCR_args,
                    chunksize=4), total=len(PCR_args)):
                print("PCR_tup", PCR_tup)
    for backend in backends:
        try:
            db = dbs[backend]
        except:
            print("Can't find backend: {}".format(backend))
            continue
        if backend == "protobuf":
            db_path = out_filepath / "PCR_index.pb"
            db_string = db.SerializeToString()
            with open(db_path, "wb") as record_file:
                record_file.write(db_string)

    end_time = time.time()
    record_create_time = end_time - start_time

    print("Convert time", convert_time)
    print("MSSIM time", mssim_time)
    print("Record create time", record_create_time)


def main():
    """
    Main (entrypoint) function
    """
    parser = create_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root
    out_root = args.out_root
    batch_size = args.batch_size
    mssim_estimate_size = args.mssim_estimate_size
    convert_images = args.convert_images
    force = args.force
    static_compression = args.static_compression
    shuffle = not args.no_shuffle
    raw_image_bytes = args.raw_image_bytes
    backends = args.backends
    use_baseline_images = args.use_baseline_images
    duplicate_dataset_factor = args.duplicate_dataset_factor
    tf_records = args.tf_records
    pool_size = args.pool_size

    start_time = time.time()
    print("backends: {}".format(backends))
    dataset_to_BCR(dataset_root,
                   out_root,
                   batch_size=batch_size,
                   mssim_estimate_size=mssim_estimate_size,
                   convert_images=convert_images,
                   force=force,
                   static_compression_quality=static_compression,
                   shuffle=shuffle,
                   backends=backends,
                   raw_image_bytes=raw_image_bytes,
                   use_baseline_images=use_baseline_images,
                   duplicate_dataset_factor = args.duplicate_dataset_factor,
                   tf_records=tf_records,
                   pool_size=pool_size,
                   )
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time", total_time)


if __name__ == "__main__":
    main()

