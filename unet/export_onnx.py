import torch

from unet.common.models_common import ResNet34, ResNet18


def get_model(model_path, model_type='ResNet34'):
    num_classes = 4

    if model_type == 'ResNet18':
        model = ResNet18(num_classes=num_classes)
    if model_type == 'ResNet34':
        model = ResNet34(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    model.eval()
    if torch.cuda.is_available():
        return model.cuda()

    return model


model = get_model('39_model_0.pt', model_type='ResNet34')

p = torch.randn(1, 3, 320, 320, device='cuda')

'''
import torch.onnx.symbolic_opset9 as onnx_symbolic
def upsample_nearest2d(g, input, output_size):
    # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
    # needed to support dynamic upsampling ONNX forumlation
    # Here we hardcode scale=2 as a temporary workaround
    scales = g.op("Constant", value_t=torch.tensor([1., 1., 2., 2.]))
    return g.op("Upsample", input, scales, mode_s="nearest")


onnx_symbolic.upsample_nearest2d = upsample_nearest2d

print('Exporting to ONNX...')
onnx_bytes = io.BytesIO()
extra_args = {'verbose': True}
torch.onnx.export(model, p, onnx_bytes, *extra_args)
'''

torch.onnx.export(model, p, 'resnet34.onnx', verbose=True)

import onnx

# Load the ONNX model
model = onnx.load('resnet34.onnx')

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
