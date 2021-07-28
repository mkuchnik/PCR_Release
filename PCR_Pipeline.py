"""
Implements an DALI pipelines for PCR directories
"""

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

import PCR_Iterator

class PCRPipeline(Pipeline):
    """
    A Dali PCR pipeline
    """
    def __init__(self, batch_size, num_threads, device_id, pcr_iter):
        super(PCRPipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="cpu",
                                       output_type=types.RGB)
        self.cast = ops.Cast(device="cpu",
                             dtype=types.INT32)
        self.pcr_iter = pcr_iter

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        minibatch = self.pcr_iter.next()
        (images, labels) = minibatch
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)


class ImageNetPCRPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pcr_iter):
        super(ImageNetPCRPipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      )
        self.pcr_iter = pcr_iter
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 resize_x=256,
                                 resize_y=256,
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           output_dtype=types.FLOAT,
                                           crop=(224, 224),
                                           image_type=types.RGB,
                                           )
        self.cast = ops.Cast(device="gpu",
                             dtype=types.INT32)
        self.uniform = ops.Uniform(range = (0.0, 1.0))

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = images.gpu()
        images = self.resize(images)
        output = self.cmn(images,
                          crop_pos_x=self.uniform(),
                          crop_pos_y=self.uniform())
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        minibatch = self.pcr_iter.next()
        (images, labels) = minibatch
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)


class PCRHybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, crop,
                 pcr_iter, dali_cpu=False):
        super(PCRHybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.pcr_iter = pcr_iter
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

    def iter_setup(self):
        minibatch = self.pcr_iter.next()
        (images, labels) = minibatch
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

