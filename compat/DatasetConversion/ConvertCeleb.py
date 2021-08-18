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

    output_train_path = output_path / "train"
    output_train_path.mkdir(parents=True, exist_ok=True)
    output_val_path = output_path / "val"
    output_val_path.mkdir(parents=True, exist_ok=True)


    annotations_path = dataset_path / "Anno/list_attr_celeba.txt"
    annotations = pd.read_csv(str(annotations_path), sep="\s+", skiprows=[0])

    image_list_path = dataset_path / "Anno/image_list.txt"
    image_list = pd.read_csv(str(image_list_path), sep="\s+")


    print(annotations)
    print(image_list)

    image_list.set_index("orig_file", inplace=True)
    print(image_list)

    joined_annot = image_list.join(annotations)
    joined_annot["new_image_name"] = ((joined_annot["idx"] + 1).
                                      map(lambda x: "{0:05d}.jpg".format(x))
                                     )
    #joined_annot.set_index("new_image_name", inplace=True)
    joined_annot.sort_index(inplace=True)
    print(joined_annot)

    smiling = joined_annot.loc[:, ["Smiling", "new_image_name"]]

    print(smiling)

    smiling.set_index("new_image_name", inplace=True)
    # Sorting is pointless, since the keys aren't proper ints (e.g., 10.jpg)
    #smiling.sort_index(inplace=True)

    print(smiling)


    smiling_dict = {
        1: "Smiling",
        -1: "Not_Smiling",
    }

    smiling["y"] = smiling["Smiling"].map(smiling_dict)
    smiling["path"] = smiling.index.astype("str").map(lambda x: dataset_path / x)

    print(smiling)

    classes = smiling["y"].unique()
    print("classes", classes)

    print(classes)

    train_df, val_df = sklearn.model_selection.train_test_split(
        smiling, test_size=test_size, random_state=42)

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
