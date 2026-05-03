import albumentations as A
import cv2

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
        target_size = args.target_size
        if isinstance(target_size, int):
            input_size = (target_size, target_size)

        interpolation = self.supported_interpolations.get(
            args.interpolation, cv2.INTER_LINEAR)
        mask_interpolation = self.supported_interpolations.get(
            getattr(args, 'mask_interpolation', 'NEAREST'), cv2.INTER_NEAREST)

        self.transform = A.Resize(
            height=input_size[0],
            width=input_size[1],
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=args.p
        )
