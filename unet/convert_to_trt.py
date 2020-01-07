import os
import shutil
import sys

import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except NameError:
    FileNotFoundError = IOError

sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def GiB(val):
    return val * 1 << 30


def onnx_engine_trt(engine_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        builder.fp16_mode = False
        with open(engine_path, 'rb') as model:
            parser.parse(model.read())
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        engine = builder.build_cuda_engine(network)
        with open(engine_path[:-5] + '.trt', 'wb') as f:
            f.write(engine.serialize())


print('Export...')
onnx_engine_trt('../models/onnx/resnet.onnx')
shutil.move('../models/onnx/resnet.engine', '../models/resnet.engine')
