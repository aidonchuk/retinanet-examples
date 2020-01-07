import torch
from PIL.Image import Image
import torch.nn.functional as F
from retinanet.main import get_model_trt

model = get_model_trt('trt_engine_path_retina')

CUDA_VISIBLE_DEVICES = 0


def infer(model, imgs):
    r = []
    resize = 1024
    max_size = 1280
    stride = model.stride

    for k, i in enumerate(imgs):
        data, ration = resize_normalize_pad(i, stride, resize, max_size)
        scores, boxes, classes = model(data)
        score, box, clazz = scores[0][0].cpu().numpy(), boxes[0][0].cpu().numpy(), classes[0][0].cpu().numpy()
        if score > 0.8:
            print(score, box, clazz)

    return r


def resize_normalize_pad(img, stride, resize, max_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img.copy()[..., ::-1]
    im = Image.fromarray(img)
    ratio = resize / min(im.size)
    if ratio * max(im.size) > max_size:
        ratio = max_size / max(im.size)
    im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
    data = data.float().div(255).view(*im.size[::-1], len(im.mode))
    data = data.permute(2, 0, 1)

    for t, mean, std in zip(data, mean, std):
        t.sub_(mean).div_(std)

    pw, ph = ((stride - d % stride) % stride for d in im.size)
    data = F.pad(data, (0, pw, 0, ph)).to(CUDA_VISIBLE_DEVICES)
    data = data.unsqueeze_(0)
    return data, ratio
