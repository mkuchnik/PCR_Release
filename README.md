# Progressive Compressed Records
Repository to along with the work:
**Progressive Compressed Records: Taking a Byte out of Deep Learning Data**

This repository contains PCR conversion scripts as well as code for running
examples of PCRs.

## Dependencies
PCRs currently use PyTorch and Nvidia Apex.
[PyTorch](https://pytorch.org/get-started/locally)
[NVIDIA Apex](https://github.com/NVIDIA/apex)
[NVIDIA DALI](https://github.com/NVIDIA/DALI) (at least v0.13.0)

For building native python code, you need python3-dev libraries, and you need
`eigen` and `pybind11` in the directory (we link against them).
```bash
git clone https://github.com/eigenteam/eigen-git-mirror.git
git clone https://github.com/pybind/pybind11.git
```
You also want to install CMake.
We clone `eigen` and `pybind11` in the `init_repo.sh` script (covered below).

In addition to these, [Protobuf 3.6.1](https://github.com/protocolbuffers/protobuf/releases/tag/v3.6.1) needs to be installed.
We use Python3.6 with Conda.
We provide a `requirements.txt` file which you can pip install (be careful to
use the conda pip):
```bash
python3 -m pip install -r requirements.txt
```

We provide a convenience script to install the python3 dependencies (DALI and
`requirements.txt`, which currently only has sqlitedict).
```bash
bash install_python_deps.sh
```

### Compiling code
Please run
```bash
bash init_repo.sh
```


### Docker
We provide a Dockerfile to facilitate using PCRs.
You will need to install
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to use it.
To build and run the container
```bash
bash docker_build.sh
bash docker_run.sh
```

Install python dependencies once you are in with:
```bash
bash install_python_deps.sh
```

If you did not compile code outside the container, you will have to do it
inside:
```bash
bash init_repo.sh
```

## Usage
Using PCRs involves two steps: conversion and loading.
A dataset has to be converted into PCR format, which can be done by first
transforming it into ImageFolder representation (described below).
Afterwards, the PCR loader can be used for training a model.

### Conversion
To utilize PCRs, we need to convert the dataset to PCR format.
To do so with the provided utility,
we need the format to be in PyTorch's
[ImageFolder representation](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).
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
my_PCR_training_set/PCR.db
my_PCR_training_set/PCR_0.db
my_PCR_training_set/PCR_1.db
my_PCR_training_set/PCR_2.db
my_PCR_training_set/PCR_3.db
...
```

We provide an advanced version of the conversion script, which provides some
additional features used in the paper (not necessary for running the
experiments).

### Loading
The loader is currently easiest to use with PyTorch, although it's a matter of
engineering to get it working with other frameworks.
`PCR_Iterator.py` provides an iterator over PCR datasets.
`PCR_Pipeline.py` provides some DALI pipelines to be used.
We show how to use these in `dali_main.py`


### Running Training
We provide an ImageNet ResNet18 example in `dali_main.py`.
You can run the code on a single machine with `dali_imagenet_bench_resnet18.sh`.
Please open the file and change the configurations if needed.
For instance, you will need to change `NUM_GPUS` if you want to use multiple
GPUs.
You will also want to point `pcr_traindir` to your PCR training directory.
Set `valdir` to a directory containing the validation data.
Note that the training data is in PCR form, but the validation data is in
regular imagefolder form.
We also provide ImageFolder conversion scripts for the other datasets in
`DatasetConversion`.

To run with 10 scans groups:
```bash
bash dali_imagenet_bench_resnet18.sh 10
```

To run with 1 scans groups:
```bash
bash dali_imagenet_bench_resnet18.sh 1
```

We also provide a distributed variant of the above script: `dali_imagenet_bench_resnet18_distributed_pcr.sh`.
You will have to set the IP address of the master node and port, in addition to
what you have to do for the non-distributed run.
You will also have to ensure that your environment plays nicely with distributed
pytorch, which as we show in the example script, may involve activating a
conda environment (e.g., py36env).
You will likely have to play with this script to get it to work.

`dali_main.py` outputs files of the form `train_data_0.csv`, where 0 can be
some number representing a node.
This file contains various metrics such as load times, training loss, and
test accuracies.
Feel free to plot these with tools such as Python Seaborn.

## Datasets
Information for obtaining datasets used with PCRs can be found at the respective
websites for the datasets:
[ImageNet](https://image-net.org/),
[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T),
[Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car\_dataset.html),
[CelebAHQ](https://github.com/tkarras/progressive\_growing\_of\_gans).
