# Progressive Compressed Records
Repository to along with the work:

**Progressive Compressed Records: Taking a Byte out of Deep Learning Data** (VLDB 21)

This repository contains Progressive Compressed Record (PCR) conversion scripts as well as code for running
examples of PCRs.
There are currently two implementations of PCRs: an external loader based [DALI
implementation](compat) and a [Tensorflow-native](tensorflow) tf.data implementation.
We recommend using the TensorFlow implementation as it is more efficient at an
implementation level, composes well with other tf.data operations, and works
well with both PyTorch and Tensorflow models.
The Tensorflow implementation can be compiled using this
[fork](https://github.com/mkuchnik/TF_PCR.git).

## A Quick Intro
Sometimes, deep learning workloads can become bottlenecked by data bandwidth
(e.g., storage).
The main idea of PCRs is to avoid performing large reads by truncating the reads
early.
On workloads where data bandwidth is a concern, this has the potential to save
bandwidth and yield training speedups.
We typically can't truncate data and get the same output back, so PCRs sacrifice
image quality for bytes read via JPEG's progressive compression.
For example, if training on sharks, we may view any of the following image
qualities (some of which are good enough and some of which are too poor quality
to train a machine learning model).

![The Effect of PCRs on Sharks](shark_PCR.gif)


## Datasets
Information for obtaining datasets used with PCRs can be found at the respective
websites for the datasets:
[ImageNet](https://image-net.org/),
[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T),
[Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car\_dataset.html),
[CelebAHQ](https://github.com/tkarras/progressive\_growing\_of\_gans).
