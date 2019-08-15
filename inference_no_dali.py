#!/usr/bin/env python
import sys, os

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image
import numpy as np

import tensorflow as tf
# For preprocessing_factory and nets_factory, we need Slim to be available in the PYTHONPATH
# This can be cloned from https://github.com/tensorflow/models.git into the current directory
ROOT_DIR = os.path.dirname(__file__)
SLIM_DIR = os.path.join(ROOT_DIR, "models", "research", "slim")
sys.path.insert(1, SLIM_DIR)
from preprocessing import preprocessing_factory
from nets import nets_factory

TRT_LOGGER = trt.Logger()
IMAGE_PATH = os.path.join(ROOT_DIR, "lfw_cropped", "Zach_Parise", "Cropped1.bmp")

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = engine.get_binding_dtype(binding)
        # Allocate device buffers for each binding.
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(None, device_mem))
        else:
            # We only need to allocate host buffers for outputs.
            host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

def load_engine():
    with trt.Runtime(TRT_LOGGER) as runtime, open(os.path.join(ROOT_DIR, "resnet50.engine"), "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

# Build the TF preprocessing graph.
def build_preprocessing_graph():
    MODEL_NAME = "resnet_v1_50"
    image_size = nets_factory.get_network_fn(MODEL_NAME, 8, is_training=False).default_image_size
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(MODEL_NAME, is_training=False)

    graph = tf.Graph()
    with graph.as_default():
        inp_tensor = tf.placeholder(tf.uint8, shape=[None, None, 3])
        pre_image = image_preprocessing_fn(inp_tensor, image_size, image_size)
        out_tensor = tf.expand_dims(pre_image, 0)
    return graph, inp_tensor, out_tensor

def main():
    graph, inp_tensor, out_tensor = build_preprocessing_graph()
    # Using a ConfigProto, we can limit the amount of GPU memory TensorFlow uses.
    # By default, it will try to reserve ALL memory, which is clearly not needed here.
    config = tf.ConfigProto()
    # The two options are:
    # 1. Explicitly set the fraction of GPU memory available to TensorFlow
    # 2. Use allow_growth to allocate memory only as needed.
    # Option 1.
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    # Option 2.
    config.gpu_options.allow_growth = True
    # Finally, make sure you set up the session with the config.
    with tf.Session(graph=graph, config=config) as sess, load_engine() as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings = allocate_buffers(engine)
        stream = cuda.Stream()

        # Inference loop.
        img = np.array(Image.open(IMAGE_PATH))
        inputs[0].host = sess.run(out_tensor, feed_dict={inp_tensor: img}).ravel()
        [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

if __name__ == '__main__':
    main()
