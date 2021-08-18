from sqlitedict import SqliteDict
import collections
import pathlib
import random
import io
from PIL import Image
import numpy as np
import multiprocessing
import tqdm
import pandas as pd
import proto.LightMLRecords_pb2 as LightMLRecords_pb2
import shutil
import argparse

import PCR_Iterator
import utils.metric as metric
import utils.progressive_utils as progressive_utils


def create_parser():
    """
    Creates a parser object
    """
    description = \
        """
        Utility to create Progressive Compressed Records (PCRs).
        PCRs are defined by a directory containing:
        1) a database of PCR metadata
        2) at least one, but probably many, .pcr files

        This PCR conversion script is meant to be used on a dataset directory
        (dataset root) and it outputs into a PCR directory (out_root).
        The dataset root is assumed to be a directory containing image data of
        organized by class label. This is the same format used by the
        PyTorch ImageFolder (See:
        https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).

        For example, *training_set* is a directory, which contains *cat* and
        *dog* directories. *cat* contains cat1.jpg, cat2.jpg, and cat3.jpg.
        *dog* contains dog1.jpg, dog2.jpg, and dog3.jpg.
        This utility can be run on *training_set* to generate a new directory
        that can be used for PCR training.

        To do so, you can run this script like:
        python3 PCR_Converter.py *training_set* *my_PCR_outputs*

        WARNING:
        This utility converts the images into progressive format ***IN PLACE***.
        If you do not want your images converted, make a copy of the dataset
        before using this utility.
        """
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('dataset_root', type=str,
                        help='The path to injest the dataset from')
    parser.add_argument('out_root', type=str,
                        help='The path to place the dataset in')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='the batch size to use for PCRs')
    parser.add_argument('--convert_images', default=True, type=bool,
                        help='Convert the Images to Progressive format in place')
    parser.add_argument('--mssim_estimate_size', default=0, type=int,
                        help='The number of samples to use for MSSIM esimates.'\
                        'This can take a long time.')
    parser.add_argument('--force', action="store_true",
                        help='Override previous PCR dir')
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

def minibatch_to_PCR_records(image_filenames, image_classes):
    """
    Converts a (X,Y) dataset into a PCR.
    Returns a list of PCR, which can then be serialized
    The first record is metadata and labels, and all the following records are
    scan groups.
    """
    assert len(image_filenames) == len(image_classes)
    meta_record = LightMLRecords_pb2.MetadataRecord()
    all_partial_images = []
    most_scans = 0
    for (f, c) in zip(image_filenames, image_classes):
        partial_images = progressive_utils.get_jpeg_partial_images(f)
        most_scans = max(most_scans, len(partial_images))
        all_partial_images.append(partial_images)
        meta_record.labels.append(c)

    meta_record.progressive_levels = most_scans

    # We postprocess partials to add extra padding scans
    for image_scans in all_partial_images:
        while len(image_scans) < most_scans:
            # We will ignore missed scans
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
    len_partials = np.array([len(partial) for partial in all_grouped_partials])
    assert np.all(len_partials == len(image_filenames))

    scan_records = []
    for grouped_partial in all_grouped_partials:
        scan_record = LightMLRecords_pb2.ScanGroup()
        assert len(grouped_partial) == len(image_classes)
        scan_record.image_bytes.extend(grouped_partial)
        scan_records.append(scan_record)

    return [meta_record, *scan_records]

def minibatch_to_PCR_file(record_filename, image_filenames, image_classes):
    PCR_records = minibatch_to_PCR_records(image_filenames, image_classes)
    prog_compressed_tf_records_str = [
        record.SerializeToString() for record in PCR_records
    ]
    prog_compressed_tf_records_len = list(
        map(len, prog_compressed_tf_records_str)
    )
    serialized_prog_records = b"".join(prog_compressed_tf_records_str)
    with open(record_filename, "wb") as record_file:
        record_file.write(serialized_prog_records)
    return prog_compressed_tf_records_len


def minibatch_to_PCR_file_w_filename(record_filename, image_filenames, image_classes):
    ret = minibatch_to_PCR_file(record_filename, image_filenames, image_classes)
    return (record_filename, ret)

def minibatch_to_PCR_file_imap(args):
    """Convenience for parallel implementations"""
    record_filename, image_filenames, image_classes = args
    return minibatch_to_PCR_file_w_filename(record_filename,
                                            image_filenames,
                                            image_classes)

def dataset_to_PCR(root_dir,
                   out_dir,
                   batch_size=512,
                   force=False,
                   pool_size=None,
                   mssim_estimate_size=256,
                   convert_images=True
                   ):
    """
    Iterates over an epoch of a dataset and writes out the data to a directory
    """
    root_filepath = pathlib.Path(root_dir).resolve()
    out_filepath = pathlib.Path(out_dir).resolve()

    f_c_list = get_directory_filenames_and_labels(root_dir)
    #print("files_list", f_c_list)

    if pool_size is None:
        pool_size = 4 * multiprocessing.cpu_count()

    files, classes = list(zip(*f_c_list))
    if convert_images:
        print("Converting all images")
        with multiprocessing.Pool(pool_size) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(
                    progressive_utils.jpeg_to_progressive,
                    files,
                    chunksize=32), total=len(files)):
                pass

    # Shuffle now
    random.shuffle(f_c_list)
    files, classes = list(zip(*f_c_list))

    if mssim_estimate_size > 0:
        print("Calculating MSSIM")
        all_mssims = []
        with multiprocessing.Pool(pool_size) as pool:
            for mssims in tqdm.tqdm(pool.imap_unordered(
                    prog_image_to_mssim,
                    files[:mssim_estimate_size],
                    chunksize=4), total=mssim_estimate_size):
                all_mssims.append(mssims)
        all_mssims = np.array(all_mssims)
        print("all mssims", all_mssims)
        df_mssims = pd.DataFrame.from_records(all_mssims)
        print("df mssims", df_mssims)
        df_mssims.to_csv("PCR_Conversion_MSSIM.csv")
        avg_mssims = np.mean(all_mssims, axis=0)
        print("avg mssims", avg_mssims)

    # TODO maybe check that file is not overwritten
    if not out_filepath.exists():
        print("Creating PCR dir")
        out_filepath.mkdir(parents=True, exist_ok=True)
    db_path = out_filepath / "PCR.db"
    if db_path.exists():
        print(db_path)
        if force:
            print("PCR db exists! Removing")
            shutil.rmtree(out_filepath.as_posix())
            out_filepath.mkdir(parents=True, exist_ok=True)
        else:
            print("PCR db exists! skipping")
            return

    print("Connecting to DB")
    db = SqliteDict(db_path.as_posix(),
                    autocommit=True)
    db["m_dataset_size"] = len(files)
    db["m_classes"] = np.unique(classes)
    db["m_batch_size"] = batch_size

    mb_iter = PCR_Iterator.iterate_minibatches(files,
                                               classes,
                                               batch_size,
                                               shuffle=False)
    PCR_args = []
    for i, (mb_f, mb_c) in enumerate(mb_iter):
        assert len(mb_f) == len(mb_c)
        record_filepath = out_filepath / "PCR_{}.pcr".format(i)
        record_filename = record_filepath.as_posix()
        PCR_args.append((record_filename, mb_f, mb_c))

    print("Converting to PCR")
    all_PCR_tup = []
    with multiprocessing.Pool(pool_size) as pool:
        for PCR_tup in tqdm.tqdm(pool.imap_unordered(
                minibatch_to_PCR_file_imap,
                PCR_args,
                chunksize=4), total=len(PCR_args)):
            record_name, record_info = PCR_tup
            # We convert to relative path
            pcr_filename = pathlib.Path(record_name).name
            db[pcr_filename] = record_info

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

    dataset_to_PCR(dataset_root,
                   out_root,
                   batch_size=batch_size,
                   mssim_estimate_size=mssim_estimate_size,
                   convert_images=convert_images,
                   force=force,
                   )


if __name__ == "__main__":
    main()

