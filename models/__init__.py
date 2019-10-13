from models.AttDepth import AttDepth


def build_model(model_name, args=None):
    if model_name == "AttDepth":
        return AttDepth(args)
    else:
        raise NotImplementedError
