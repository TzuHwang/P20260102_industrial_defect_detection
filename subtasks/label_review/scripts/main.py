import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from project_src.arguments import AccessArgs


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def find_image_xml_pairs(root_dir):
    """Find all image files that have corresponding XML annotations.

    XML files can be named as:
    - <image_name>.xml (e.g., image123.xml -> image123.jpg)
    - <image_name>.<ext>.xml (e.g., image123.jpg.xml -> image123.jpg)

    Args:
        root_dir: Root directory to search for image-xml pairs.

    Returns:
        List of tuples (image_path, xml_path) for valid pairs.
    """
    pairs = []

    for dirpath, _, filenames in os.walk(root_dir):
        xml_files = [f for f in filenames if f.endswith('.xml')]

        for xml_file in xml_files:
            xml_path = os.path.join(dirpath, xml_file)
            base_name = xml_file[:-4]  # remove .xml

            # Case 1: XML is named <image_name>.<ext>.xml (e.g., image.jpg.xml)
            # Check if base_name ends with an image extension
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                if base_name.endswith(ext):
                    # The image path is the base_name itself
                    candidate_image = os.path.join(dirpath, base_name)
                    if os.path.exists(candidate_image):
                        image_path = candidate_image
                        break

            # Case 2: XML is named <image_name>.xml (e.g., image.xml -> image.jpg)
            if image_path is None:
                for ext in IMAGE_EXTENSIONS:
                    candidate_image = os.path.join(dirpath, base_name + ext)
                    if os.path.exists(candidate_image):
                        image_path = candidate_image
                        break

            if image_path:
                pairs.append((image_path, xml_path))

    return pairs


def extract_tags_from_xml(xml_path):
    """Extract all tag types from an XML annotation file.

    The XML files are actually JSON format with 'faces' containing defect annotations.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        Set of tag types found in the file.
    """
    tags = set()
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        faces = data.get('faces', [])
        for face in faces:
            tag_type = face.get('type')
            if tag_type:
                tags.add(tag_type)
    except (json.JSONDecodeError, IOError):
        # Fallback: parse as text if JSON parsing fails
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r'"type":\s*"([^"]+)"', content)
        tags.update(matches)

    return tags


def create_composite(image_paths, tag):
    """Create a composite image with up to 5 images arranged horizontally on a white canvas,
    with the tag label on the top left.

    Args:
        image_paths: List of image file paths.
        tag: The label/tag name.

    Returns:
        Composite image as numpy array.
    """
    resized_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        resized_images.append(img)

    if not resized_images:
        # Empty canvas if no images
        canvas = np.ones((250, 1000, 3), dtype=np.uint8) * 255
    else:
        # Arrange images horizontally
        canvas_height = 250  # 50 for text + 200 for images
        canvas_width = 200 * len(resized_images)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        for i, img in enumerate(resized_images):
            y_start = 50
            x_start = i * 200
            canvas[y_start: y_start + 200, x_start: x_start + 200] = img

    # Put text on top left, ensuring no overlap
    cv2.putText(canvas, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return canvas


def main(args):
    """Main function for label_wash subtask.

    Creates composite images for each label, showing up to 5 images per label on a white canvas
    with the label text on the top left.

    Args:
        args: Arguments namespace from AccessArgs.
    """
    data_dir = getattr(args, 'data_dir', '/root/data/tap_measure')

    print(f"Scanning directory: {data_dir}")
    print("-" * 60)

    # Find all image-xml pairs
    pairs = find_image_xml_pairs(data_dir)
    print(f"Found {len(pairs)} image-xml pairs")

    # Collect tag to images mapping
    tag_to_images = {}
    for img_path, xml_path in pairs:
        tags = extract_tags_from_xml(xml_path)
        for tag in tags:
            if tag not in tag_to_images:
                tag_to_images[tag] = []
            tag_to_images[tag].append(img_path)

    # Output directory
    output_dir = getattr(args, 'output', None)
    if output_dir is None:
        output_dir = 'outputs/label_wash'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating composites in: {output_path}")
    print("-" * 60)

    for tag, images in tag_to_images.items():
        if not images:
            continue
        selected = images[:5]  # Select up to 5 images
        composite = create_composite(selected, tag)
        composite_file = output_path / f"{tag}.png"
        cv2.imwrite(str(composite_file), composite)
        print(f"Saved composite for '{tag}' with {len(selected)} images to {composite_file}")

    print(f"\nCompleted creating composites for {len(tag_to_images)} labels.")


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
