"""
Based off PyTorch/NVIDIA AMP data preloader.
https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
"""

import torch


class data_prefetcher():
    """Prefetches data to GPU from a tf.loader"""
    def __init__(self, loader, length,
                 permute_channel: bool,
                 stream_prefetch: bool=True):
        """
        loader: A tf.data loader
        length: The length of the loader (None if infinite, but don't do that)
        permute_channel: If channel should be permuted on GPU

        """
        self.base_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.length = length
        self.permute_channel = permute_channel
        self.stream_prefetch = stream_prefetch
        self.idx = 0
        self.preload()

    def __len__(self):
        return self.length

    def query_idle_stream(self) -> bool:
        """
        Returns true if stream is not active with work (idle)
        """
        return self.stream.query()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
            self.next_input = self.next_input._numpy()
            self.next_target = self.next_target._numpy()
            self.next_input_py = torch.from_numpy(self.next_input).pin_memory()
            self.next_target_py = torch.from_numpy(self.next_target).pin_memory()
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.stream_prefetch:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input_py.cuda(non_blocking=True)
                self.next_target = self.next_target_py.cuda(non_blocking=True)
                if self.permute_channel:
                    self.next_input = self.next_input.permute(0, 3, 1, 2)
        else:
            self.next_input = self.next_input_py
            self.next_target = self.next_target_py
            if self.permute_channel:
                self.next_input = self.next_input.permute(0, 3, 1, 2)

    def reset(self):
        self.idx = 0
        self.loader = iter(self.base_loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def next(self):
        if self.stream_prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
        curr_input = self.next_input
        curr_target = self.next_target
        if self.stream_prefetch:
            if curr_input is not None:
                curr_input.record_stream(torch.cuda.current_stream())
            if curr_target is not None:
                curr_target.record_stream(torch.cuda.current_stream())
        self.preload()
        self.idx += 1
        return curr_input, curr_target

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.length is not None and self.idx >= self.length:
            raise StopIteration
        input, target = self.next()
        if input is None or target is None:
            raise StopIteration
        return input, target
