"""
Detection-task preprocessing: extract all bounding box annotations per image.

Unlike the classification task (which picks the largest defect per image),
this script collects every valid annotation from each XML as a list of
{"class": int, "bbox": [x, y, w, h]} records. Images with no valid XML
annotations are kept as negatives (empty annotation list).
"""

import json
import os
import re
from pathlib import Path

from project_src.arguments import AccessArgs
from subtasks.data_preprocessing.scripts.label_match import label_match_dict
from subtasks.data_preprocessing_det.scripts.draw_specimens import draw_all_specimens


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

_BACK_RE = re.compile(r'back|背面', re.IGNORECASE)
_FRONT_RE = re.compile(r'front|正面', re.IGNORECASE)


def detect_side(path):
    if _BACK_RE.search(path):
        return 'back'
    if _FRONT_RE.search(path):
        return 'front'
    return 'unknown'


def is_excluded(path, data_dir, exclude_subdirs, exclude_keywords):
    for subdir in exclude_subdirs:
        excluded_abs = os.path.join(data_dir, subdir)
        if os.path.abspath(path).startswith(os.path.abspath(excluded_abs)):
            return True
    for kw in exclude_keywords:
        if kw.lower() in path.lower():
            return True
    return False


def find_images(data_dir, exclude_subdirs, exclude_keywords):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if is_excluded(dirpath, data_dir, exclude_subdirs, exclude_keywords):
            dirnames.clear()
            continue
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                yield os.path.join(dirpath, fname)


def get_annotations_from_xml(xml_path):
    """Parse XML (JSON format) and return list of detection annotations.

    Each annotation is {"class": int, "bbox": [x, y, w, h]}.
    Returns empty list if no valid/known labels are found or file is unreadable.
    """
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        faces = data.get('faces', [])
    except (json.JSONDecodeError, IOError):
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            raw_types = re.findall(r'"type":\s*"([^"]+)"', content)
            faces = [{'type': t} for t in raw_types]
        except IOError:
            return []

    annotations = []
    for face in faces:
        tag = face.get('type', '')
        if tag not in label_match_dict:
            continue
        annotations.append({
            "class": label_match_dict[tag],
            "bbox": [face.get('x', 0), face.get('y', 0), face.get('w', 0), face.get('h', 0)],
        })
    return annotations


def build_data_list(data_dir, exclude_subdirs, exclude_keywords):
    """Build list of detection records.

    Each record has keys: filename, file_dir, side, annotations.
    annotations is a list of {"class": int, "bbox": [x, y, w, h]}.
    Images without a valid XML get an empty annotations list (negative sample).
    Side defaults to 'front' when path keywords are ambiguous.
    """
    records = []
    for img_path in find_images(data_dir, exclude_subdirs, exclude_keywords):
        xml_path = img_path + '.xml'
        side = detect_side(img_path)
        annotations = get_annotations_from_xml(xml_path) if os.path.exists(xml_path) else []
        records.append({
            "filename": os.path.basename(img_path),
            "file_dir": os.path.dirname(img_path),
            "side": 'front' if side == 'unknown' else side,
            "annotations": annotations,
        })
    return records


def main(args):
    data_dir = getattr(args, 'data_dir', '/root/data/tap_measure')
    exclude_subdirs = getattr(args, 'exclude_subdirs', [])
    exclude_keywords = getattr(args, 'exclude_keywords', [])
    output = getattr(args, 'output', 'data/internal_train/data_list_det')

    if isinstance(exclude_subdirs, str):
        exclude_subdirs = [exclude_subdirs]
    if isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]

    print(f"Scanning: {data_dir}")
    records = build_data_list(data_dir, exclude_subdirs, exclude_keywords)
    print(f"Total images: {len(records)}")

    out_path = Path(output).with_suffix('.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    front = sum(1 for r in records if r['side'] == 'front')
    back = sum(1 for r in records if r['side'] == 'back')
    has_ann = sum(1 for r in records if r['annotations'])
    print(f'  front: {front};  back: {back}')
    print(f'  with annotations: {has_ann};  negatives: {len(records) - has_ann}')
    print(f'Saved to: {out_path}')

    print('Drawing specimen grids ...')
    draw_all_specimens(records, out_path.parent / 'specimens')


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
