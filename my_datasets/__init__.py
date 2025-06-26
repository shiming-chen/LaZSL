from enum import Enum
from .cub import CUBDataset
from .dtd import DTD
from .food101 import Food101
from .imagenetv2 import ImageNetV2Dataset
from .oxford_pets import OxfordIIITPet
from .place365 import Places365
from .imagenet import ImageNet
from .imagenet2 import ImageNet2


class MyDataset(str, Enum):
    ImageNet = "imagenet"
    ImageNetV2 = "imagenetv2"
    ImageNetR = "imagenet-r"
    ImageNetS = "imagenet-s"
    ImageNetA = "imagenet-a"
    CUB = "cub"
    DTD = "dtd"
    Food101 = "food101"
    OxfordIIITPet = "oxford_pet"
    Place365 = "place365"

    def __str__(self) -> str:
        return self.value
