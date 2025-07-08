"""
script to debug the interface detection algorithm in `interface_detection.vial_contour.get_interfaces`

This script does NOT perform vial detection. You need to use `-xy` to specify the bounding box of the vial in the image.
This script focuses on the interface detection part only. If you need to debug the entire vial detection process, use `debug.py` instead.
"""

import argparse
import re
import os

import cv2

from interface_detection.utils import increase_image_contrast, get_gray_image
from interface_detection.vial_contour import get_interfaces
from interface_detection.vial_contour import (
    get_vessel_mask,
    get_cap_range,
    compensate_for_background,
    find_label_mask,
)  # other functions that might be useful for debugging


parser = argparse.ArgumentParser(description="Find the contour of a vial in an image")
parser.add_argument(
    "-i",
    "--image",
    type=str,
    default=f"tests/img/example_1.jpg",
    help="Path to the image or directory to images (use --pattern for filtering filenames)",
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    default="",
    help="regex pattern for image filenames (should match pattern in filename)",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Debug mode, show images with cv2.imshow()",
)
parser.add_argument(
    "-xy",
    "--xyxy",
    type=int,
    nargs=4,
    default=(466, 537, 614, 962),
    help="xyxy bbox for the vial",
)
args = parser.parse_args()

image_path = args.image
IMG_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
if os.path.isdir(image_path):
    image_paths = [
        os.path.join(image_path, f)
        for f in os.listdir(image_path)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]
    if args.pattern:
        image_paths = [
            image_path
            for image_path in image_paths
            if re.match(args.pattern, os.path.basename(image_path))
        ]
elif not os.path.isfile(image_path):
    image_paths = [os.path.join(os.path.dirname(__file__), "vial", image_path)]
else:
    image_paths = [image_path]
assert all(os.path.isfile(f) for f in image_paths), "Invalid image path or directory"

for image_path in image_paths:
    img = cv2.imread(image_path)
    if args.xyxy is not None:
        img = img[args.xyxy[1] : args.xyxy[3] + 1, args.xyxy[0] : args.xyxy[2] + 1]

    median = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    over_exposure_threshold = 200
    img_enhanced = increase_image_contrast(image=img, thresh=over_exposure_threshold)
    gray_enhanced = get_gray_image(img_enhanced)

    test = get_interfaces(
        img,
        debug=args.debug,
        title=os.path.splitext(os.path.basename(image_path))[0],
    )
    print(test)
