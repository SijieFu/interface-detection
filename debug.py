"""
script to debug the entire interface detection algorithm in `interface_detection.vial_detection.detect_interfaces`

You can supply the raw image to this script. It performs vial detection first, and then interface detection on
the cropped vial(s) in the image.

Note that the `interface_detection` module depends on an external module for vial detection.
Check `README.md` for more details.
"""

import os
import argparse
import re

from Flask_utils import visualize_analysis
from interface_detection.vial_detection import detect_interfaces

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--img",
    type=str,
    default="tests/img/example_1.jpg",
    help="Path to the image or directory to images (use --pattern for filtering filenames)",
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    default="",
    help="pattern for image names (should contain pattern in file name)",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Debug mode, show images with cv2.imshow() in `visualize_analysis`",
)
parser.add_argument(
    "-dd",
    "--debug-details",
    action="store_true",
    help="Show detailed debug information from `detect_interfaces` (and `visualize_analysis`)",
)
parser.add_argument(
    "-a",
    "--use-autohsp-config",
    action="store_true",
    help="Use autoHSP config for analyzing the image, otherwise use default config",
)
parser.add_argument(
    "-v", "--vial-detection", type=str, default="http://localhost:5002/",
    help="URL for the vial detection service (default: http://localhost:5002/)",
)
args = parser.parse_args()

IMG_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
if os.path.isdir(args.img):
    imgs = [
        os.path.join(args.img, img)
        for img in os.listdir(args.img)
        if os.path.splitext(img)[-1].lower() in IMG_EXTENSIONS
    ]
    if args.pattern:
        imgs = [img for img in imgs if re.match(args.pattern, os.path.basename(img))]
elif os.path.isfile(args.img):
    imgs = [args.img]
else:
    raise ValueError("Invalid input")

if len(imgs) == 0:
    raise ValueError("No image found in the specified directory")

weights = "yolov8n.pt"
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

for img in imgs:
    if args.use_autohsp_config:
        result = detect_interfaces(
            img,
            weights=weights,
            name="vial",
            # interface_detection.vial_detection.detect_interfaces
            conf=0.7,
            iou=0.5,
            max_det=100,
            width_expansion=0.025,
            alignment_cutoff=0.5,
            config=None,
            vial_detection_server= args.vial_detection,
            # interface_detection.vial_contour.get_interfaces
            over_exposure_threshold=200,
            bg_correction=True,
            bg_tolerance=30,
            bg_resolution=0.02,
            bg_sobel_ratio=0.8,
            cap_ratio=(0.1, 0.25),
            cap_target=-1,
            cap_tolerance=40,
            vessel_sobel_threshold=31,
            sobel_threshold=24,
            sobel_xy_ratio=1.0,
            dilation=True,
            phase_resolution=0.08,
            boundary_resolution=(0.4, 0.1),
            label_correction=True,
            label_low=150,
            label_high=220,
            label_check_gray=False,
            interface_signal=0.55,
            debug=args.debug_details,
        )
    else:
        result = detect_interfaces(img, weights=weights, debug=args.debug_details)
    print(f"Working on {img}")
    for vial in result["vials"]:
        print(f"{vial}: {' '.join([str(x) for x in result[vial]['xyxy']])}")
    _ = visualize_analysis(
        img,
        result,
        zoomin=True,
        force=True,
        save_path=os.path.join(out_dir, f"{os.path.basename(img)}"),
        debug=args.debug or args.debug_details,
        title=img.split(os.path.sep)[-1].split(".")[0],
    )
