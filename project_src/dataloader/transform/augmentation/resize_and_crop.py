import cv2
import albumentations as A

from .__template__ import AlbumentationsAPI


class Resize(AlbumentationsAPI):
    supported_interpolations = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, args):
        super().__init__()
        input_size = args.input_size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        interpolation = self.supported_interpolations.get(
            args.interpolation, cv2.INTER_LINEAR)
        mask_interpolation = self.supported_interpolations.get(
            args.mask_interpolation, cv2.INTER_NEAREST)

        self.transform = A.Resize(
            height=input_size[0],
            width=input_size[1],
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=args.p
        )
