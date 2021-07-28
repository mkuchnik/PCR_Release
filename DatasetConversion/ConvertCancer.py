"""
Converts a Cancer dataset to ImageFolder format
"""

import argparse

import pathlib
import shutil
import sklearn.model_selection
import tqdm
import multiprocessing

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
    output_path = pathlib.Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path_folders = [x for x in dataset_path.iterdir() if x.is_dir()]
    assert len(train_path_folders) == 2, "Need both HAM10000 image folders"
    output_train_path = output_path / "train"
    output_train_path.mkdir(parents=True, exist_ok=True)
    output_val_path = output_path / "val"
    output_val_path.mkdir(parents=True, exist_ok=True)


    annotations_path = dataset_path / "HAM10000_metadata.csv"
    annotations = pd.read_csv(str(annotations_path))

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    print(train_path_folders)

    imageid_path_dict = dict()
    for f in train_path_folders:
        for x in f.glob("*.jpg"):
            extensionless_name = x.with_suffix("").name
            imageid_path_dict[extensionless_name] = x

    print(imageid_path_dict)

    annotations["path"] = annotations["image_id"].map(imageid_path_dict)
    annotations["lession_long_name"] = annotations["dx"].map(lesion_type_dict)
    annotations["y"] = pd.Categorical(annotations["dx"])

    print("annotations", annotations)

    classes = annotations["y"].unique()
    print("classes", classes)

    train_df, val_df = sklearn.model_selection.train_test_split(
        annotations, test_size=test_size, random_state=42)

    for c in classes:
        output_c_path = output_train_path / c
        output_c_path.mkdir(parents=True, exist_ok=True)
        relevant_df = train_df.query("y == '{}'".format(c))
        source_paths = relevant_df["path"]
        input_tuples = [(src, output_c_path, use_softlink) for src in
                        source_paths]
        with multiprocessing.Pool(pool_size) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(
                    src_target_copy_packed,
                    input_tuples,
                    chunksize=32), total=len(input_tuples)):
                pass

    for c in classes:
        output_c_path = output_val_path / c
        output_c_path.mkdir(parents=True, exist_ok=True)
        relevant_df = val_df.query("y == '{}'".format(c))
        source_paths = relevant_df["path"]
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
