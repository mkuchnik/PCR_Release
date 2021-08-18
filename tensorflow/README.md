# Progressive Compressed Records (tf.data release)
This is the Tensorflow tf.data implementation of Progressive Compressed Records (PCRs).
While the data pipeline is implemented in tf.data, it can be used with other
frameworks, such as PyTorch with appropriate tensor conversion operations.

## Dependencies
The Tensorflow implementation can be compiled using this
[fork](https://github.com/mkuchnik/TF_PCR.git).
This fork is compiled the same way Tensorflow is, which is described
[here](https://www.tensorflow.org/install/source).
Some utility functions in the `main.py` also assume that
[JPEGTran](https://jpegclub.org/jpegtran/) is on the
executable path and `scan_only_jsk` (compile `src/scan_only_jsk.c`) is also on the path.
Furthermore, the protobuf files must be compiled into Python files.
These last two steps are covered with the `init_repo.sh` script.

## Usage
Using PCRs involves two steps: conversion and loading.
A dataset has to be converted into PCR format, which can be done by first
transforming it into ImageFolder representation (described below).
Afterwards, the PCR loader can be used for training a model.

### Conversion
To utilize PCRs, we need to convert the dataset to PCR format.
To do so with the provided utility,
we need the format to be in ImageFolder representation e.g., PyTorch's
[ImageFolder representation](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
or Tensorflow dataset's [ImageFolder representation](https://www.tensorflow.org/datasets/api_docs/python/tfds/folder_dataset/ImageFolder).
For example, `training_set` is a directory, which contains `cat` and
`dog` directories. `cat` contains `cat1.jpg`, `cat2.jpg`, `cat3.jpg`, etc..
`dog` contains `dog1.jpg`, `dog2.jpg`, `dog3.jpg`, etc.

Many datasets, such as ImageNet, are already in the required form by default.
So the cats will be organized like this:
```
training_set/cat/cat1.jpg
training_set/cat/cat2.jpg
training_set/cat/cat3.jpg
...
```

and the dogs will be organized like this:
```
training_set/dog/dog1.jpg
training_set/dog/dog2.jpg
training_set/dog/dog3.jpg
...
```

We currently assume the dataset is full of JPEG files; if the dataset is not,
it's recommended to do that conversion with a seperate utility, as any
conversions the converter will do will use default, and therefore likely poor,
JPEG compression parameters.

The utility we use for converting datasets (specifically, training sets) into
PCR format is the `PCR_Converter.py` utility.

```bash
python3 PCR_Converter.py <input_dataset> <output_directory>
```

This utility can be run on `training_set` to generate a new directory
that can be used for PCR training.
```bash
python3 PCR_Converter.py training_set my_PCR_training_set
```

Running this script will take some time (5-10 minutes).
The MSSIM estimates are expensive to compute and unnecessary for running PCRs,
and they can be turned off.
***WARNING:*** The provided conversion utility modifies images **in place** to
convert them into progressive JPEG form.
Because of this, you will have to **pass flags** to allow the conversion and override the previous database.

```bash
python3 PCR_Converter.py training_set my_PCR_training_set --convert_images=True --force
```

This will create a directory as follows:
```
my_PCR_training_set/PCR_index.pb
my_PCR_training_set/PCR_0.db
my_PCR_training_set/PCR_1.db
my_PCR_training_set/PCR_2.db
my_PCR_training_set/PCR_3.db
...
```

In our included scripts (e.g., `main.py`), `PCR_index.pb` is assumed to be in
the same directory as the PCR files.

### Loading
The loader is currently easiest to use with PyTorch or TensorFlow, although it's a matter of
engineering to get it working with other frameworks.
To compare how code is written in Tensorflow's tf.data:

**TFRecords**
```python
# Get data out of files and into tuples of images and labels
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_record)

# Additional preprocessing such as crop
dataset = dataset.map(preprocessing)

# Training
for x, y in dataset:
  model.fit_step(x, y)
```

**ProgressiveCompressedRecord**
```python
# Get data out of files and into tuples of images and labels
# Note, data quality increases with scan number.
dataset = tf.data.ProgressiveCompressedRecordDataset(filenames, index_filename, scan)
dataset = dataset.map(parse_record)

# Additional preprocessing such as crop
dataset = dataset.map(preprocessing)

# Training
for x, y in dataset:
  model.fit_step(x, y)
```

For PyTorch, since the data is in Tensorflow's tensor format, we perform a
conversion operation on the materialized data to get it into PyTorch format.
This is currently done in the `data_prefetcher.py`.

### Running Training
We provide an ImageNet example in `main.py`.
You can run the code on a single or many machines with `run.sh`.
Please open the file and change the configurations if needed.
In particular, point the data directories to where your data is located.

`main.py` outputs files of the form `train_data_0.csv`, where 0 can be
some number representing a node rank.
This data can be used to plot accuracy over time.
