"""
Utilities for Flask
"""

import os
import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np
import cv2
from PIL import Image

from info import info
from interface_detection.vial_detection import detect_interfaces


def get_md5_hash(file: str | bytes, chunk_size: int = 4096) -> str:
    """
    Get the MD5 hash of a file (designed for images)

    :param str | bytes file: the filepath or the file content
    :param int chunk_size: chunk size in kb
    :return str: the MD5 hash of the file
    """
    md5 = hashlib.md5()
    if isinstance(file, str):
        with open(file, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
    elif isinstance(file, bytes):
        md5.update(file)
    return md5.hexdigest()


def get_image_md5_hash(image: str | bytes) -> str:
    """
    Get the MD5 hash of an image, regardless of the file format or EXIF data.

    :param str | bytes image: the image path or the image content.
        Note: `image` should be a valid argument for `PIL.Image.open()`.
    :return str: the MD5 hash of the image
    """
    img = Image.open(image)
    return hashlib.md5(img.tobytes()).hexdigest()


def _parse_bool(value: str | bool) -> bool:
    """
    Parse a string or boolean value to a boolean value. This is designed to handle URL query parameters.

    :param str | bool value: the value to parse. Can be a literal string or a boolean.

        > Acceptable True strings (not case sensitive): "true", "yes", "1", "y", "t"
    :return bool: the parsed boolean value
    """
    if isinstance(value, str):
        return True if value.lower() in ["true", "yes", "1", "y", "t"] else False
    return bool(value)


def analyze_image(
    file_md5: str,
    file_path: str | None = None,
    force_rerun: str | bool = "false",
    **kwargs: Any,
) -> tuple[str, bool]:
    """
    Analyze an image and return the result

    :param str file_md5:
        the MD5 hash of the file/image. This is used to name the result file(s).
    :param str | None file_path:
        the path to the saved file/image for analysis
    :param str | bool force_rerun:
        whether to force rerun the analysis even if the result file exists (from cache).
    :param Any kwargs:
        additional keyword arguments for `interface_detection.vial_detection.detect_interfaces`

    :return str:
        the path to the result json file
    :return bool:
        whether the analysis was from cache or not. A True value indicates that:

        - the result file already exists and `force` is False
        - the parameters are the same as the previous analysis
    """
    force_rerun = _parse_bool(force_rerun)
    current_parameters = detect_interfaces(
        image=np.ndarray([]), return_parameters=True, **kwargs
    )

    result_dir = (
        os.path.join(os.path.dirname(file_path), info.IMG_RESULT_DIRNAME)
        if file_path
        else info.IMG_RESULT_DIR
    )
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{file_md5}.json")
    previous_result = defaultdict(dict)
    try:  # preserve the human intervention record; MD5 hash collision should not be of concern here
        with open(result_path, "r") as f:
            previous_result.update(json.load(f))
        if current_parameters != previous_result.get("parameters", {}):
            force_rerun = True
    except:
        force_rerun = True

    if force_rerun or not os.path.isfile(result_path):
        # Load a model
        img = cv2.imread(file_path)
        result = detect_interfaces(
            image=img, return_parameters=False, force_rerun=force_rerun, **kwargs
        )
        result["md5_hash"] = file_md5
        result["human_intervention"].update(
            previous_result.get("human_intervention", {})
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
    return result_path, force_rerun


def _get_adaptive_font_scale(
    text: str, font: int, thickness: int, max_width: int, max_height: int
) -> float:
    """
    Get the adaptive font scale for a text

    :param str text: the text to display
    :param int font: the font face
    :param int thickness: the thickness of the text
    :param int max_width: the maximum width of the text
    :param int max_height: the maximum height of the text
    :return float: the adaptive font scale
    """
    scale = 0.05
    while True:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        if w > max_width or h > max_height:
            break
        scale += 0.05
    return scale


def visualize_analysis(
    img: str | np.ndarray,
    result: str | dict,
    zoomin: bool = False,
    save_path: str | None = None,
    force_rerun: str | bool = "false",
    vial_detection_only: bool = False,
    annotated_img_only: bool = False,
    debug: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Visualize the analysis result by annotating the image with the detected vials and their properties.

    :param str | np.ndarray img:
        the image to visualize, either the path to the image or the numpy array of the image
    :param str | dict result:
        the analysis result from the function `analyze_image`, either the path to the result json file or the dictionary
    :param bool zoomin:
        whether to zoom in the image (zoom in the detected vial instead of the whole image)
    :param str | None save_path: the path to save the result
    :param str | bool force_rerun:
        in case of found result file, whether to force rerun the visualization even if the result file exists
    :param bool vial_detection_only:
        whether to only visualize the vial detection result (without the interface detection result)
    :param bool annotated_img_only:
        whether to only visualize the annotated image (without the original image)
    :param bool debug:
        whether to show the debug information. If True, the image will be displayed in a window on the screen,
        blocking the program until the user closes the window.
    :param str | None title:
        the title of the visualization image
    :param Any kwargs:
        additional keyword arguments for visualization. For compatibility with other functions and not used in this function.
    :return np.ndarray: the annotated image
    """
    force_rerun = _parse_bool(force_rerun)

    if save_path and not force_rerun and os.path.isfile(save_path):
        if zoomin == ("zoomin" in save_path):
            return cv2.imread(save_path)

    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(result, str):
        with open(result, "r") as f:
            result = json.load(f)

    # check the img width and height, resize to at least 720px height if necessary
    if img.shape[0] < 20 or img.shape[1] < 20:
        return img  # return the original image if it is too small

    factor: float = 1.0
    new_height = int(np.clip(img.shape[0], 480, 2160))
    factor = new_height / img.shape[0]
    if not np.isclose(factor, 1.0):
        img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))

    img_original = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = img.shape[:2]
    if result["nvials"] == 0:  # if no vial is detected
        boxwidth = np.clip(1920, width // 3, width)
        boxheight = np.clip(height // 10, min(30, height // 3), height // 2)
        boxheight = min(boxheight, boxwidth // 3)
        cv2.rectangle(img, (0, 0), (boxwidth, boxheight), (0, 0, 255), -1)
        text, text_thickness = ("No vial detected", np.clip(int(boxheight / 15), 1, 4))
        cv2.putText(
            img,
            text,
            (int(boxwidth * 0.05), int(boxheight * 0.95)),
            font,
            _get_adaptive_font_scale(
                text, font, text_thickness, int(boxwidth * 0.9), int(boxheight * 0.9)
            ),
            (255, 255, 255),
            text_thickness,
        )
        if save_path:
            cv2.imwrite(save_path, img)

        if debug:
            cv2.imshow(title or "result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    height_original, width_original = result["image_dimension_height_width"]
    # check if the aspect ratio is maintained
    if not np.isclose(width_original / height_original, width / height, rtol=0.05):
        raise ValueError(
            f"Aspect ratio mismatch: [original]{width_original / height_original} vs. [current]{width / height}"
        )
    factor: float = height / height_original

    max_boxheight = 0
    for vial in result["vials"]:
        x1, y1, x2, y2 = (np.array(result[vial]["xyxy"]) * factor).astype(int)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        conf = result[vial]["confidence"]
        cap_top, cap_bottom = (np.array(result[vial]["cap_abs"]) * factor).astype(int)
        interfaces = (np.array(result[vial]["interfaces_abs"]) * factor).astype(int)
        polylines = [
            (np.array(polyline) * factor).astype(int)
            for polyline in result[vial]["label_mask_abs"]
        ]

        thickness = max(1, (w + h) // 150)
        if not vial_detection_only:
            # draw the label mask in white
            for poly in polylines:
                cv2.polylines(
                    img,
                    [np.array(poly, dtype=int).reshape(-1, 1, 2)],
                    True,
                    (255, 255, 255),
                    thickness,
                )
            # draw the cap top and bottom in blue
            cv2.line(img, (x1, cap_top), (x2, cap_top), (255, 0, 0), thickness)
            cv2.line(img, (x1, cap_bottom), (x2, cap_bottom), (255, 0, 0), thickness)
            # draw the interfaces in blue
            for interface in interfaces:
                cv2.line(
                    img, (x1, interface), (x2, interface), (255, 255, 0), thickness
                )

        # draw the bounding box in red or green
        _is_miscible = bool(result[vial]["is_miscible"])
        _human_intervention = result["human_intervention"].get(vial, -1)
        _needs_intervention = result[vial]["needs_intervention"]
        if vial_detection_only:
            color = (255, 0, 0)  # blue if only vial detection
        elif _human_intervention == 1 or (_human_intervention < 0 and _is_miscible):
            color = (0, 160, 0)  # green if miscible
        else:
            color = (0, 0, 255)  # red if not miscible
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # draw vial name and confidence within a red filled rectangle with transparent text
        text = f"{vial} {conf:.2f}"
        if vial_detection_only:
            pass
        elif _needs_intervention or _human_intervention >= 0:
            if _human_intervention < 0:
                text = f"?{text}"  # human assessment is needed but not provided
                color = tuple(int(_x * 0.5) for _x in color)  # darken the color
            elif bool(_human_intervention) != _is_miscible:
                text = f"!{text}"  # human assessment is inconsistent with automatic assessment
            else:
                text = f"${text}"  # human assessment is consistent with automatic assessment

        if y1 < 30:  # too few pixels to display the label above the bounding box
            pt1 = (int(x1 * 0.85 + x2 * 0.15), 0)
            pt2 = (int(x1 * 0.15 + x2 * 0.85), min(h // 3, max(30, h // 10)))
        else:  # display the label above the bounding box
            pt1 = (x1, max(0, y1 - max(30, h // 10)))
            pt2 = (x2, y1)

        boxwidth, boxheight = pt2[0] - pt1[0] + 1, pt2[1] - pt1[1] + 1
        if boxheight > boxwidth // 3:
            boxheight = boxwidth // 3
            pt1 = (pt1[0], pt2[1] - boxheight + 1)
        text_thickness = np.clip(int(boxheight / 15), 1, 4)
        cv2.rectangle(img, pt1, pt2, color, -1)
        cv2.putText(
            img,
            text,
            (int(pt1[0] + boxwidth * 0.05), int(pt2[1] - boxheight * 0.05)),
            font,
            _get_adaptive_font_scale(
                text,
                font,
                text_thickness,
                max_width=int(boxwidth * 0.9),
                max_height=int(boxheight * 0.9),
            ),
            (255, 255, 255),
            text_thickness,
        )

        max_boxheight = max(max_boxheight, boxheight * 1.2)

    if zoomin:
        vial_height, vial_width = (
            np.array(result["vial_dimension_height_width"]) * factor
        ).astype(int)
        xx1, yy1, xx2, yy2 = (np.array(result["xyxy"]) * factor).astype(int)
        xx1, yy1, xx2, yy2 = (
            max(0, xx1 - vial_width // 4),
            max(0, yy1 - int(max_boxheight)),
            min(img.shape[1], xx2 + vial_width // 4),
            min(img.shape[0], yy2 + vial_height // 16),
        )
        img = img[yy1 : yy2 + 1, xx1 : xx2 + 1]
        img_original = img_original[yy1 : yy2 + 1, xx1 : xx2 + 1]

    # concatenate the original image and the result image together along thr longer axis
    if annotated_img_only:
        pass
    elif img.shape[0] >= img.shape[1] * 0.8:
        # vertically concatenate & add padding betteen the two images
        img = np.concatenate(
            (
                img_original,
                np.zeros((img.shape[0], 10, *[3] * (img.ndim - 2)), dtype=img.dtype),
                img,
            ),
            axis=1,
        )
    else:
        # horizontally concatenate & add padding betteen the two images
        img = np.concatenate(
            (
                img_original,
                np.zeros((10, img.shape[1], *[3] * (img.ndim - 2)), dtype=img.dtype),
                img,
            ),
            axis=0,
        )

    if save_path:
        cv2.imwrite(save_path, img)

    if debug:
        cv2.imshow(title or "result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img
