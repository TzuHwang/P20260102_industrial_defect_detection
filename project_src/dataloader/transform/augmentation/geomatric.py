import albumentations as A

from .__template__ import AlbumentationsAPI


class Rotate(AlbumentationsAPI):
    def __init__(self, args):
        """
        args.degree: max rotation angle (int)
        args.p: probability to apply rotation (float)
        """
        super().__init__()
        self.transform = A.Rotate(
            limit=args.degree,
            p=args.p,
            border_mode=0  # 0 = constant padding
        )


class HorizontalFlip(AlbumentationsAPI):
    def __init__(self, args):
        """
        args.p: probability to apply rotation (float)
        """
        super().__init__()
        self.transform = A.HorizontalFlip(
            p=args.p,
        )


class Rotate90(AlbumentationsAPI):
    def __init__(self, args):
        """
        args.p: probability to apply rotation (float)
        """
        super().__init__()
        self.transform = A.RandomRotate90(
            p=args.p,
        )
