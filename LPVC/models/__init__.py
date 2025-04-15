
from models.tpsmodel import TPSModel


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """

    instance = TPSModel(opt)
    print("model [%s] was created" % (instance.name()))

    return instance
