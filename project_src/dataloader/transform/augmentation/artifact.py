import albumentations as A

from .__template__ import AlbumentationsAPI


class BrightnessContrast(AlbumentationsAPI):
    def __init__(self, args):
        super().__init__()
        self.transform = A.RandomBrightnessContrast(
            brightness_limit=args.brightness,
            contrast_limit=args.contrast,
            p=args.p
        )
