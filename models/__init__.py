from models.AttDepth import AttDepth
from models.DORN.DORN_kitti import DORN


def build_model(model_name, args=None):
    if model_name == "DORN":
        return DORN(args.crop_size, pretrained=args.use_pretrain)
    elif model_name == "AttDepth":
        return AttDepth(args)
    else:
        raise NotImplementedError
