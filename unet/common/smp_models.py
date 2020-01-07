import segmentation_models_pytorch as smp


def se_resnet50(num_classes):
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=num_classes,
        activation=None,
    )

    return model


def resnet34_fpn(num_classes):
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'

    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=num_classes,
        activation=None,
    )

    return model


def resnet34_psp(num_classes):
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'

    model = smp.PSPNet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=num_classes,
        activation=None,
    )

    return model
