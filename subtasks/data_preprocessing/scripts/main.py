"""
That is not a clean dataset, there are some labels with numeric prefix, and some labels with non-numeric prefix.
We need to match them to the same label. For example, "1表面脏污" and "表面脏污" should be matched to the same
label "表面脏污". We can use the label_match_dict to match the labels. And for meeting the time constraint, we
have to split the dataset without data replication and source checking to prevent data leakage.

For the information about the label matching, please refer to the label_match_dict in label_match.py.
"""

import csv
import json
import os
import re
from pathlib import Path

from project_src.arguments import AccessArgs
from subtasks.data_preprocessing.scripts.label_match import label_match_dict


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

_BACK_RE = re.compile(r'back|背面', re.IGNORECASE)
_FRONT_RE = re.compile(r'front|正面', re.IGNORECASE)


def detect_side(path):
    """Return 'front', 'back', or 'unknown' based on directory path keywords."""
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
    """Walk data_dir and yield image file paths, skipping excluded subdirs and keywords."""
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if is_excluded(dirpath, data_dir, exclude_subdirs, exclude_keywords):
            dirnames.clear()
            continue
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                yield os.path.join(dirpath, fname)


def get_class_from_xml(xml_path):
    """Parse XML (JSON format) and return the mapped class label.

    Picks the defect with the largest bounding box area.
    Returns None if no valid/known labels are found.
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
            faces = [{'type': t, 'h': 1, 'w': 1} for t in raw_types]
        except IOError:
            return None

    best_class = None
    best_area = -1
    for face in faces:
        tag = face.get('type', '')
        if tag not in label_match_dict:
            continue
        area = face.get('h', 1) * face.get('w', 1)
        if area > best_area:
            best_area = area
            best_class = label_match_dict[tag]

    return best_class


def build_data_list(data_dir, exclude_subdirs, exclude_keywords):
    """Build list of (filename, class, file_dir, side, side_preferred) tuples.

    Images without a valid XML label are assigned class 0 (negative).
    Side is detected from directory path keywords ('front'/'正面', 'back'/'背面').
    Undetected paths default to 'front'; side_preferred=True marks these cases.
    """
    records = []
    for img_path in find_images(data_dir, exclude_subdirs, exclude_keywords):
        xml_path = img_path + '.xml'
        filename = os.path.basename(img_path)
        file_dir = os.path.dirname(img_path)
        side = detect_side(img_path)

        if os.path.exists(xml_path):
            cls = get_class_from_xml(xml_path)
            if cls is None:
                cls = 0  # XML exists but no known label → treat as negative
        else:
            cls = 0  # no annotation → negative

        resolved_side = 'front' if side == 'unknown' else side
        records.append((filename, cls, file_dir, resolved_side))
    return records


def main(args):
    data_dir = getattr(args, 'data_dir', '/root/data/tap_measure')
    exclude_subdirs = getattr(args, 'exclude_subdirs', [])
    exclude_keywords = getattr(args, 'exclude_keywords', [])
    output = getattr(args, 'output', 'data/internal_train/data_list')
    output_format = getattr(args, 'output_format', 'csv')

    if isinstance(exclude_subdirs, str):
        exclude_subdirs = [exclude_subdirs]
    if isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]

    print(f"Scanning: {data_dir}")
    records = build_data_list(data_dir, exclude_subdirs, exclude_keywords)
    print(f"Total images: {len(records)}")

    out_path = Path(output).with_suffix('.csv' if output_format == 'csv' else '')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class', 'file_dir', 'side'])
        writer.writerows(records)

    front = sum(1 for r in records if r[3] == 'front')
    back = sum(1 for r in records if r[3] == 'back')
    print(f'  front: {front};  back: {back}')
    print(f'Saved to: {out_path}')


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
