"""Overlay detections on a frame. Uses PIL because the class names are Chinese
and cv2.putText cannot render CJK glyphs."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# First existing font wins; covers the Docker dev image and Windows deployment.
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",   # Microsoft YaHei
    "C:/Windows/Fonts/msjh.ttc",   # Microsoft JhengHei
    "C:/Windows/Fonts/simhei.ttf",
]
BOX_COLOR = (220, 50, 50)   # RGB
_font_cache: dict = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_detections(frame_bgr: np.ndarray, detections, font_size: int = 16) -> np.ndarray:
    """Return a copy of the BGR frame with boxes + `class conf` labels drawn."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=2)
        label = f"{det.class_name} {det.score:.2f}"
        tb = draw.textbbox((x1, y1), label, font=font)
        label_y = y1 - (tb[3] - tb[1]) - 4
        if label_y < 0:
            label_y = y1 + 2
        bg = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(bg, fill=BOX_COLOR)
        draw.text((x1, label_y), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
