"""
Converts a Cancer dataset to ImageFolder format
"""

import argparse

import pathlib
import shutil
import sklearn.model_selection
import tqdm
import multiprocessing
import scipy.io as sio

import pandas as pd

def create_parser():
    """
    Creates a parser
    """
    parser = argparse.ArgumentParser(description='TinyImagenet converter')
    parser.add_argument('--dataset_root', type=str,
                        required=True,
                        help='The path to place the dataset in')
    parser.add_argument('--output_root', type=str,
                        required=True,
                        help='The path to place the BCRs in')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="The percentage of data to use for test",
                        )
    parser.add_argument('--use_softlink',
                        action="store_true",
                        help='The path to place the BCRs in')
    return parser


def src_target_copy(src_path, output_path, use_softlink):
    target_path = output_path / src_path.name
    if use_softlink:
        target_path.symlink_to(src_path, target_is_directory=False)
    else:
        shutil.copyfile(str(src_path), str(target_path))

def src_target_copy_packed(input_tuple):
    src_path, output_path, use_softlink = input_tuple
    return src_target_copy(src_path, output_path, use_softlink)

def load_labels(meta_annotations):
    labels = [c for c in meta_annotations['class_names'][0]]
    labels = pd.DataFrame(labels, columns=['labels'])
    return labels


def load_train_df(train_annotations, train_path):
    frame = [[i.flat[0] for i in line] for line in train_annotations['annotations'][0]]
    columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    df_train = pd.DataFrame(frame, columns=columns)
    df_train['class'] = df_train['class']-1 # Python indexing starts on zero.
    df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path
    return df_train

def load_test_df(test_annotations, test_path):
    frame = [[i.flat[0] for i in line] for line in test_annotations['annotations'][0]]
    columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
    df_test = pd.DataFrame(frame, columns=columns)
    df_test['fname'] = [test_path/f for f in df_test['fname']] #  Appending Path
    return df_test

def main():
    """
    Main (entrypoint) function
    """
    parser = create_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_root = args.output_root
    use_softlink = args.use_softlink
    test_size = args.test_size
    pool_size = 4 * multiprocessing.cpu_count()

    print("dataset_root")

    dataset_path = pathlib.Path(dataset_root)
    dataset_train_path = dataset_path / "cars_train"
    dataset_val_path = dataset_path / "cars_test"
    output_path = pathlib.Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    output_train_path = output_path / "train"
    output_train_path.mkdir(parents=True, exist_ok=True)
    output_val_path = output_path / "val"
    output_val_path.mkdir(parents=True, exist_ok=True)


    annotations_meta = dataset_path / "car_devkit/devkit/cars_meta.mat"
    meta_annotations = sio.loadmat(str(annotations_meta))
    annotations_train_path = dataset_path / "car_devkit/devkit/cars_train_annos.mat"
    train_annotations = sio.loadmat(str(annotations_train_path))
    annotations_test_path = dataset_path / "car_devkit/devkit/cars_test_annos.mat"
    test_annotations = sio.loadmat(str(annotations_test_path))

    # https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up#DevKit
    labels = load_labels(meta_annotations)
    train_df = load_train_df(train_annotations, dataset_train_path)
    # Merge
    train_df = train_df.merge(labels, left_on='class', right_index=True)
    train_df = train_df.sort_index()
    # Somewhat confusing, but we use val and test interchangably
    #val_df = load_test_df(test_annotations, dataset_val_path)
    #val_df = val_df.merge(labels, left_on="class", right_index=True)
    #val_df = val_df.sort_index()

    train_df, val_df = sklearn.model_selection.train_test_split(
        train_df, test_size=test_size, random_state=42)
    print("train_df", train_df)
    print("val_df", val_df)

    classes = sorted(train_df["labels"].unique())
    print("classes", classes)

    for c in classes:
        class_name = c.replace(" ", "_")
        output_c_path = output_train_path / class_name
        output_c_path.mkdir(parents=True, exist_ok=True)
        relevant_df = train_df.query("labels == '{}'".format(c))
        source_paths = relevant_df["fname"]
        source_paths = list(map(pathlib.Path, source_paths))
        input_tuples = [(src, output_c_path, use_softlink) for src in
                        source_paths]
        with multiprocessing.Pool(pool_size) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(
                    src_target_copy_packed,
                    input_tuples,
                    chunksize=32), total=len(input_tuples)):
                pass

    for c in classes:
        class_name = c.replace(" ", "_")
        output_c_path = output_val_path / class_name
        output_c_path.mkdir(parents=True, exist_ok=True)
        relevant_df = val_df.query("labels == '{}'".format(c))
        source_paths = relevant_df["fname"]
        source_paths = list(map(pathlib.Path, source_paths))
        input_tuples = [(src, output_c_path, use_softlink) for src in
                        source_paths]
        with multiprocessing.Pool(pool_size) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(
                    src_target_copy_packed,
                    input_tuples,
                    chunksize=32), total=len(input_tuples)):
                pass


if __name__ == "__main__":
    main()
