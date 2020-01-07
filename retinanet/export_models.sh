#!/bin/sh

/opt/conda/bin/python -W ignore main.py export ../models/onnx/retinanet_rn18fpn_1_stage.pth ../models/retinanet_rn18fpn_1_stage.engine --full-precision --size 1280 1024 --batch 1
/opt/conda/bin/python -W ignore main.py export ../models/onnx/retinanet_rn34fpn_2_stage.pth ../models/retinanet_rn34fpn_2_stage.engine --full-precision --size 256 512 --batch 1
