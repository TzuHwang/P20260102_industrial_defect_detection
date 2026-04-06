import albumentations as A

from .__template__ import AlbumentationsAPI


class BrightnessContrast(AlbumentationsAPI):
    def __init__(self, args=None):
        super().__init__()
        self.transform = A.RandomBrightnessContrast(
            brightness_limit=getattr(args, 'brightness', 0.2),
            contrast_limit=getattr(args, 'contrast', 0.2),
            p=getattr(args, 'p', 0.5),
        )


class ColorJitter(AlbumentationsAPI):
    def __init__(self, args=None):
        super().__init__()
        self.transform = A.ColorJitter(
            brightness=getattr(args, 'brightness', 0.2),
            contrast=getattr(args, 'contrast', 0.2),
            saturation=getattr(args, 'saturation', 0.2),
            hue=getattr(args, 'hue', 0.1),
            p=getattr(args, 'p', 0.5),
        )
