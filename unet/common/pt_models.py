from pytorch_tools.segmentation_models import Unet


def resnet34_blur(num_classes):
    model = Unet(decoder_use_batchnorm=False, classes=num_classes, antialias=False)
    return model
