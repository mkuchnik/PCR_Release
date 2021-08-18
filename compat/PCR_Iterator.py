"""
Implements an iterator for PCR directories
"""
import collections
import numpy as np
import random
import pathlib
import build.python_bindings as native_bind
from sqlitedict import SqliteDict


def get_db_data(root_dir, metadata=False):
    """
    Gets DB data and possibly metadata
    """
    root_filepath = pathlib.Path(root_dir).resolve()
    db_path = root_filepath / "PCR.db"
    if not db_path.exists():
        print("No PCR DB found in {}".format(db_path))
        # TODO Backward compatibility
        db_path = root_filepath / "BCR.db"
        if not db_path.exists():
            raise ValueError("No PCR DB found in {}".format(db_path))
    db = SqliteDict(db_path.as_posix(), autocommit=True)
    db_data = {k: db[k] for k in db if not k.startswith("m_")}
    if metadata:
        db_metadata = {k: db[k] for k in db if k.startswith("m_")}
        return db_data, db_metadata
    else:
        return db_data

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Groups input, targets into batches
    From: https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            mb_inputs = [inputs[i] for i in excerpt]
            mb_targets = [targets[i] for i in excerpt]
            yield mb_inputs, mb_targets
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


class PCRIterator(object):
    def __init__(self, root_dir, batch_size, db_data, n_scans,
                 randomizer=None, random_buff_mult=4):
        """
        @param root_dir The PCR directory
        @param batch_size The minibatch sizes to yield
        @param db_data A dict of filename to record length
        @param n_scans The number of scans to use
        @param randomizer A random.Random object (for random seed)
        @param random_buff_mult An integer representing how many times
        the batch size should be fetched (improved randomization)
        """
        self.images_dir = root_dir
        self.root_dir = pathlib.Path(root_dir)
        self.files = [k for k in sorted(db_data.keys())]
        self.db_data = db_data
        self.n_scans = n_scans
        self.batch_size = batch_size
        self.randomizer = randomizer
        assert random_buff_mult >= 1 and isinstance(random_buff_mult, int)
        self.random_buff_mult = random_buff_mult
        self.shuffle_files()
        assert self.batch_size > 0

    def shuffle_files(self):
        if self.randomizer is None:
            random.shuffle(self.files)
        else:
            self.randomizer.shuffle(self.files)

    def shuffle(self, data):
        if self.randomizer is None:
            random.shuffle(data)
        else:
            self.randomizer.shuffle(data)
        return data

    def __iter__(self):
        self.i = 0
        self.j = 0
        self.n = len(self.files)
        self.minibatches = None
        return self

    def __next__(self):
        if self.minibatches is None:
            batch = []
            labels = []
            while len(batch) < self.batch_size * self.random_buff_mult:
                # We need to get at least 1 full batch of data
                pcr_filename = self.files[self.i]
                record_lens = self.db_data[pcr_filename]
                # We normalize path here
                pcr_filename = pathlib.Path(pcr_filename).name
                full_pcr_filename = (self.root_dir / pcr_filename).as_posix()
                all_image_bytes, Y = self.get_bytes_and_labels(
                    full_pcr_filename,
                    record_lens,
                    self.n_scans)
                for img_bytes, label in zip(all_image_bytes, Y):
                    np_img_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
                    batch.append(np_img_bytes)
                    np_label = np.array([label], dtype=np.int32)
                    labels.append(np_label)
                assert len(batch) > 0 and len(labels) == len(batch)
                if (self.i + 1) >= self.n:
                    self.shuffle_files()
                self.i = (self.i + 1) % self.n
            batch_labels = list(zip(batch, labels))
            batch_labels = self.shuffle(batch_labels)
            batch[:], labels[:] = zip(*batch_labels)
            self.minibatches = [x for x in
                                iterate_minibatches(batch,
                                                    labels,
                                                    self.batch_size)]
        try:
            minibatch = self.minibatches[self.j]
        except Exception as ex:
            raise ex
        self.j += 1
        if self.j >= len(self.minibatches):
            self.j = 0
            self.minibatches = None
        return minibatch

    def get_bytes_and_labels(self, filename, record_lens, n_scans):
        if n_scans is None:
            n_scans = len(record_lens)
        record_offsets = np.cumsum(
            record_lens
        )
        all_image_bytes, Y = native_bind.load_PCR(filename,
                                                  record_offsets,
                                                  n_scans
                                                  )
        return all_image_bytes, Y

    next = __next__
