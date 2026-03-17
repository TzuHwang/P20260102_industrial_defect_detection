import json
import os
import re
from pathlib import Path

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


def collect_all_tags(pairs):
    """Collect all unique tag types from XML annotations.

    Args:
        pairs: List of (image_path, xml_path) tuples.

    Returns:
        Sorted list of unique tag types.
    """
    all_tags = set()
    for _, xml_path in pairs:
        tags = extract_tags_from_xml(xml_path)
        all_tags.update(tags)

    return sorted(all_tags)


def main(args):
    """Main function for label_wash subtask.

    Iterates through /root/data/tap_measure to find all image files with
    corresponding XML annotations, and lists all possible tag types.

    Args:
        args: Arguments namespace from AccessArgs.
    """
    data_dir = getattr(args, 'data_dir', '/root/data/tap_measure')

    print(f"Scanning directory: {data_dir}")
    print("-" * 60)

    # Find all image-xml pairs
    pairs = find_image_xml_pairs(data_dir)
    print(f"Found {len(pairs)} image-xml pairs")

    # Collect all unique tags
    all_tags = collect_all_tags(pairs)

    print(f"\nAll possible tag types ({len(all_tags)}):")
    print("-" * 60)
    for tag in all_tags:
        print(f"  - {tag}")

    # Optional: save results to output file if configured
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Image-XML Pairs:\n")
            for img, xml in pairs:
                f.write(f"{img} <-> {xml}\n")
            f.write("\nAll Tags:\n")
            for tag in all_tags:
                f.write(f"{tag}\n")
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    args = AccessArgs().get_args()
    main(args)
