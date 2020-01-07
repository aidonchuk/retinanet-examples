import os
import sys

import cv2
import numpy as np
import tensorrt as trt
from common.pre_process_utils import normalized, DTYPE

remove_after_predict = False
camera = None
cam_id = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except NameError:
    FileNotFoundError = IOError

sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

DTYPE = trt.float32

def normalized(img, buff, input_shape):
    def normalize_image(image):
        image_arr = np.asarray(cv2.resize(image, (input_shape[1], input_shape[2])).transpose(
            [2, 0, 1]).astype(
            trt.nptype(DTYPE)).ravel())
        return (image_arr / 255.0 - 0.456) / 0.225

    np.copyto(buff, normalize_image(img))
    return img


def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()


def load_engine_trt(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as trt_runtime:
        engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine


def infer(context, images, h_input, d_input, h_output, d_output, stream, input_shape, output_shape):
    r = []
    for img in images:
        normalized(img, h_input, input_shape)
        # start_inf = time.time()
        do_inference(context, h_input, d_input, h_output, d_output, stream)
        r.append(np.reshape(h_output.copy(), output_shape))
        # print("TensorRT inference time: {} ms".format(int(round((time.time() - start_inf) * 1000))))
    return r


imgs = None  # np array

INPUT_SHAPE = (3, 320, 320)
OUTPUT_SHAPE = (4, 320, 320)

with load_engine_trt('trt_engine_path_pads') as trt_engine:
    h_input, d_input, h_output, d_output, stream = allocate_buffers(trt_engine)
    with trt_engine.create_execution_context() as context:
        output = infer(context, imgs, h_input, d_input, h_output, d_output, stream, INPUT_SHAPE, OUTPUT_SHAPE)
        del context
    del trt_engine
    del h_input, d_input, h_output, d_output, stream
