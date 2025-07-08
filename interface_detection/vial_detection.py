import os
import json
import warnings
from typing import Any, Literal
from dataclasses import dataclass
import requests

import numpy as np
import cv2

from .utils import number_to_excel_column
from .vial_contour import get_interfaces


@dataclass
class VialDetectionResult:
    """
    A class to represent the result of vial detection.

    Attributes:
        xyxy (list[int, int, int, int]): The bounding box of the detected vial. Should contain
            the coordinates of the bounding box in the format [x1, y1, x2, y2].
        confidence (float): The confidence score of the detection.
        name (str): The name of the detected object.
    """

    xyxy: list[int, int, int, int]
    confidence: float
    name: str


def detect_vials(
    image: np.ndarray | str,
    vial_detection_server: str = "http://localhost:5002",
    weights: str = "yolov8n.pt",
    name: str = "vial",
    conf: float = 0.7,
    iou: float = 0.5,
    device: str = "cpu",
    max_det: int = 100,
    force_rerun: bool | str = False,
) -> list[VialDetectionResult]:
    """
    detect vials in an image

    :param np.ndarray | str image:
        input image as a numpy array or path to the image
    :param str vial_detection_server:
        the URL of the vial detection service to use for detecting vials. A POST request will be sent to this URL and
        a successful response should return a JSON object.

        For the POST request, the following information will be included in the request body:
        `requests.post(vial_detection_server, files={"file": (filename, image_bytes, "image/jpeg")}, data=params, params=url_params, timeout=60)`,
        where `params` is a dictionary containing the following:

        - "weights": the weight filename
        - "name": the name of the class to detect (e.g., "vial")
        - "conf": confidence threshold for the detection
        - "iou": IoU threshold for the detection
        - "device": device to use for inference (e.g., "cpu" or "cuda")
        - "max_det": maximum number of detections to return

        and the `url_params` will contain the following:

        - "force": a boolean value indicating whether to force re-run the detection, ignoring any cached results.
        - "saveflag": a boolean value indicating whether to permanently save the detection results (default is False).

        The expected response should be a JSON object with the following structure:

        - "vials": a list of detected vials, each object is a dictionary with:

            - "xyxy": a list of four integers representing the bounding box of the detected vial in the format [x1, y1, x2, y2]
            - "confidence": a float representing the confidence score of the detection
            - "name": a string representing the name of the detected object (e.g., "vial")
        - "dimensions": a list of two integers representing the height and width of the image
    :param str weights:
        the weight filename. By default, weights should be under the `runs` directory.
    :param str name:
        name of the class to detect. This is the name of the class when training or tuning the object detection model.
    :param float conf:
        confidence threshold for the detection.
    :param float iou:
        IoU threshold for the detection. Should be between 0 and 1. The IoU threshold is used to filter out overlapping detections.
        The higher the IoU threshold, the more strict the filtering is.
    :param str device:
        device to use for inference. By default, it is set to "cpu". If you have a GPU, you can set it to "cuda" to use the GPU for inference.
    :param int max_det:
        maximum number of detections to return. This is used to limit the number of detections returned by the model.
    :param bool | str force_rerun:
        if True, request the vial detection service at `vial_detection_server` to ignore any cached results
        and re-run the detection. A string starting with `t`, `y`, or `1` will be treated as True.

    :return list[VialDetectionResult]:
        a list of VialDetectionResult objects, each containing the bounding box ("xyxy": list[int, int, int, int]),
        confidence ("confidence": float), and name ("name": str) of the detected vial.

    :raises FileNotFoundError: if the image file is not found
    :raises ValueError: if the input arguments are not valid
    """
    # Check input arguments
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"image file not found: {image}")
        image = cv2.imread(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("`image` must be a numpy array or a path to an image")

    if not isinstance(weights, str) or not weights:
        raise ValueError("`weights` must be a non-empty string")
    if not 0 <= conf <= 1:
        raise ValueError("confidence threshold `conf` must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise ValueError("`iou` threshold must be between 0 and 1")
    if device not in ["cpu", "cuda"]:
        raise ValueError("`device` must be 'cpu' or 'cuda'")
    if not isinstance(max_det, int) or max_det <= 0:
        raise ValueError("`max_det` must be a positive integer")

    height, width = image.shape[:2]
    params = {
        "weights": weights,
        "name": name,
        "conf": conf,
        "iou": iou,
        "device": device,
        "max_det": max_det,
    }
    try:
        response = requests.post(
            vial_detection_server,
            files={
                "file": (
                    "image.jpg",
                    cv2.imencode(".jpg", image)[1].tobytes(),
                    "image/jpeg",
                )
            },
            data=params,
            params={"force": force_rerun, "saveflag": False},
            timeout=60,
        )
        response.raise_for_status()

        response_data = response.json()
        height_new, width_new = response_data["dimensions"]
        if not np.isclose(height, height_new) or not np.isclose(width, width_new):
            raise ValueError(
                "The object detection service API returned an image with different dimensions than the input image."
            )

        vials = response_data["vials"]
        if not isinstance(vials, list):
            raise ValueError(
                "The object detection service API did not return a valid list of objects."
            )

        vials = [
            VialDetectionResult(
                xyxy=v["xyxy"], confidence=v["confidence"], name=v["name"]
            )
            for v in vials
            if v["name"] == name
        ]
        return vials
    except requests.RequestException as e:
        raise ValueError(f"Failed to connect to the vial detection service: {e}")
    except json.JSONDecodeError:
        raise ValueError(
            "Failed to decode the response from the vial detection service."
        )
    except Exception as e:
        raise ValueError(
            f"An error occurred while processing the response from the object detection service. "
            f"Are you sure the service is hosting the correct API? Error: {e} - {response_data}"
        )


def detect_interfaces(
    image: np.ndarray | str,
    weights: str = "yolov8n.pt",
    name: str = "vial",
    conf: float = 0.7,
    iou: float = 0.5,
    device: str = "cpu",
    max_det: int = 100,
    width_expansion: float = 0.0,
    alignment_cutoff: float = 0.5,
    config: str | list[int] | None = None,
    return_parameters: bool = False,
    vial_detection_server: str = "http://localhost:5002",
    force_rerun: bool | str = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Detect interfaces in vials in an image

    :param np.ndarray | str image:
        input image as a numpy array or path to the image
    :param str weights:
        the weight filename. By default, weights should be under the `runs` directory.
    :param str name:
        name of the class to detect. This is the name of the class when training or tuning the object detection model.
    :param float conf:
        confidence threshold for the detection.
    :param float iou:
        IoU threshold for the detection. Should be between 0 and 1. The IoU threshold is used to filter out overlapping detections.
        The higher the IoU threshold, the more strict the filtering is.
    :param str device:
        device to use for inference. By default, it is set to "cpu". If you have a GPU, you can set it to "cuda" to use the GPU for inference.
    :param int max_det:
        maximum number of detections to return. This is used to limit the number of detections returned by the model.
    :param float width_expansion:
        width expansion for the vial. This is used to expand the vial width by a certain percentage to both left and right.
        Should be between 0 and 0.25. If 0, no expansion is applied.
    :param float alignment_cutoff:
        cutoff for alignment ratio, should be between 0 and 1 (recommended >= 0.5).
        For the detected vials, how to group them into rows. If the vertical alignment ratio of the vials is greater than this value, they are considered to be in the same row.
        See `align_detected_vials` for more details
    :param str | list[int] | None config:
        how many vials in each row; if not provided, the function will try to infer. See `align_detected_vials` for more details
    :param bool return_parameters:
        if True, return the parameters used for detection. This is useful to compile the parameters for comparison with previous runs,
        without having to go through the actual detection process.
    :param str vial_detection_server:
        the URL of the vial detection service to use for detecting vials. The detected vials will be passed to
        crop the vials out and analyze the interfaces in each vial, respectively.

        A POST request will be sent to this URL. Check `.detect_vials` for the expected API format.
    :param bool | str force_rerun:
        if True, request the vial detection service at `vial_detection_server` to ignore any cached results
        and re-run the detection. A string starting with `t`, `y`, or `1` will be treated as True.
    :param Any **kwargs:
        additional keyword arguments for `vial_contour.get_interfaces`
    :return dict[str, Any]:
        If `return_parameters` is True, return a dictionary containing the parameters used for detection.

        If `return_parameters` is False, return a dictionary containing the detected vials and their interfaces.

        Specifically, for the key `human_intervention`, the value will be a dictionary with:
            - key as the string index for a vial
            - value as if human intervention was performed on the vial
                -1: not assessed by human,
                0: not miscible by human intervention,
                1: miscible by human intervention.
    """
    # Check input arguments
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"image file not found: {image}")
        image = cv2.imread(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("`image` must be a numpy array or a path to an image")

    if not isinstance(weights, str) or not weights:
        raise ValueError("`weights` must be a non-empty string")
    if not 0 <= conf <= 1:
        raise ValueError("confidence threshold `conf` must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise ValueError("`iou` threshold must be between 0 and 1")
    if device not in ["cpu", "cuda"]:
        raise ValueError("`device` must be 'cpu' or 'cuda'")
    if not isinstance(max_det, int) or max_det <= 0:
        raise ValueError("`max_det` must be a positive integer")
    if not isinstance(width_expansion, float) or not 0 <= width_expansion <= 0.25:
        raise ValueError("`width_expansion` must be a float between 0 and 0.25.")
    if not 0 <= alignment_cutoff <= 1:
        raise ValueError("alignment_cutoff must be between 0 and 1, recommended >= 0.5")
    if isinstance(config, str) and config != "":
        config = list(map(int, config.split(",")))

    max_det = min(max_det, 300)
    if isinstance(config, list):
        max_det = sum(config)

    parameters = {
        "vial_detection": {
            "weights": weights,
            "name": name,
            "conf": conf,
            "iou": iou,
            "device": device,
            "max_det": max_det,
            "alignment_cutoff": alignment_cutoff,
            "width_expansion": width_expansion,
            "config": config,
        },
        "interface_detection": kwargs,
    }
    if return_parameters:
        return parameters

    # detect vials in the image
    vials = detect_vials(
        image=image,
        vial_detection_server=vial_detection_server,
        weights=weights,
        name=name,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        force_rerun=force_rerun,
    )
    vial_positions = np.array([vial.xyxy for vial in vials]).astype(int)

    sorted_vial_indices = align_detected_vials(
        vial_positions, format="xyxy", cutoff=alignment_cutoff, config=config
    )

    # expand the vial width by 2.5% to both left and right
    if width_expansion > 0:
        vial_widths = vial_positions[:, 2] - vial_positions[:, 0]
        vial_widths = np.ceil(vial_widths * width_expansion).astype(int)
        vial_positions[:, 0] = np.maximum(0, vial_positions[:, 0] - vial_widths)
        vial_positions[:, 2] = np.minimum(
            image.shape[1] - 1, vial_positions[:, 2] + vial_widths
        )
    result_detected_vials = {
        "vials": [],
        "nvials": len(vials),
        "nrows": len(sorted_vial_indices),
        "rows": list(
            map(number_to_excel_column, range(1, len(sorted_vial_indices) + 1))
        ),
        "nvials_per_row": list(map(len, sorted_vial_indices)),
        "xyxy": (
            np.concatenate(
                (
                    np.min(vial_positions[:, 0:2], axis=0),
                    np.max(vial_positions[:, 2:4], axis=0),
                )
            )
            .astype(int)
            .tolist()
            if 0 not in vial_positions.shape
            else []
        ),
        "vial_dimension_height_width": (
            np.flip(np.max(vial_positions[:, 2:4] - vial_positions[:, 0:2], axis=0))
            .astype(int)
            .tolist()
            if not vial_positions.size == 0
            else []
        ),
        "image_dimension_height_width": [image.shape[0], image.shape[1]],
        "human_intervention": {},  # of `is_miscible`, -1: not assessed, 0: not miscible, 1: miscible
        "parameters": parameters,
    }

    for irow, row in enumerate(
        result_detected_vials["rows"]
    ):  # row is a letter, e.g., 'A', 'B', 'C', ...
        for ivial, vial_index in enumerate(
            sorted_vial_indices[irow], start=1
        ):  # ivial is the index of the vial in the row to get 'A1', 'A2', 'A3', ...
            key = f"{row}{ivial}"
            result_detected_vials["vials"].append(key)

            vial_summary = {}
            vial_summary["xyxy"] = vial_positions[vial_index].astype(int).tolist()
            vial_summary["confidence"] = round(vials[vial_index].confidence, 3)

            x1, y1, x2, y2 = vial_summary["xyxy"]
            (
                interfaces,
                (cap_top, cap_bottom, cap_color),
                is_miscible,
                needs_intervention,
                label_mask_polylines,
            ) = get_interfaces(
                image[y1 : y2 + 1, x1 : x2 + 1].copy(), title=key, **kwargs
            )
            interfaces = list(map(int, interfaces))
            # relative position as to the vial y1
            vial_summary["interfaces"] = interfaces
            # absolute position
            vial_summary["interfaces_abs"] = [y1 + _x for _x in interfaces]
            # relative position as to the vial y1
            vial_summary["cap"] = [int(cap_top), int(cap_bottom)]
            # absolute position
            vial_summary["cap_abs"] = [int(y1 + cap_top), int(y1 + cap_bottom)]
            vial_summary["cap_color"] = cap_color
            vial_summary["label_mask"] = label_mask_polylines
            vial_summary["label_mask_abs"] = [
                (np.array(polylines) + np.array([[x1, y1]])).astype(int).tolist()
                for polylines in label_mask_polylines
            ]
            vial_summary["is_miscible"] = is_miscible
            vial_summary["needs_intervention"] = needs_intervention

            result_detected_vials[key] = vial_summary

    # of `is_miscible`;
    # -1: not assessed by human,
    # 0: not miscible by human intervention
    # 1: miscible by human intervention
    result_detected_vials["human_intervention"] = {
        k: -1 for k in result_detected_vials["vials"]
    }
    return result_detected_vials


def align_detected_vials(
    positions: np.ndarray | list[list],
    format: Literal["xyxy", "xywh"] = "xyxy",
    cutoff: float = 0.5,
    config: str | list[int] | None = None,
) -> list[list[int]]:
    """
    Align the detected vials to a grid

    :param np.ndarray | list[list] positions:
        list of vial positions or a numpy array; each entry is (x1, y1, x2, y2) by default
    :param Literal["xyxy", "xywh"] format:
        format of the positions, either 'xyxy' or 'xywh'
    :param float cutoff:
        cutoff for alignment ratio, should be between 0 and 1 (recommended >= 0.5).
        For the detected vials, how to group them into rows.
        If the vertical alignment ratio of the vials is greater than this value, they are considered to be in the same row.
    :param str | list[int] | None config:
        how many vials in each row; if not provided, the function will try to infer.

        - if list[int], it represents the number of vials in each row from top to bottom
        - if str, the string must be in the format of "n1,n2,n3,...,nk" where ni is the number of vials in the i-th row
    :return list[list[int]]:
        a list of lists, where each inner list contains the indices of the vials in the same row (0-indexed).
        For example, [[0, 1], [2, 3]] means that vials 0 and 1 are in the first row, and vials 2 and 3
        are in the second row. The indices are the indices of the vials in the original list of vials.

    """
    positions = np.array(positions).reshape(-1, 4)  # x1, y1, x2, y2 if format is 'xyxy'
    if format.lower() == "xyxy":
        pass
    elif format.lower() == "xywh":
        positions[:, 2] += positions[:, 0]
        positions[:, 3] += positions[:, 1]
    else:
        raise ValueError("format must be `xyxy` or `xywh`")

    # first, cluster the vials into rows (no sorting of columns)
    sorted_rows = []
    if config:
        if isinstance(config, str):
            config = list(map(int, config.split(",")))
        elif isinstance(config, list):
            config = list(map(int, config))
        else:
            raise ValueError(
                "config must be a list of integers or a string with comma-separated integers"
            )
        assert sum(config) == len(
            positions
        ), "the total number of vials must match the sum of the config"
        sorted_indices = np.argsort(positions[:, 1])  # sort by y1
        start = 0
        for c in config:
            sorted_rows.append(sorted_indices[start : start + c].tolist())
            start += c
        # perform a validity check
        for row in sorted_rows:
            if len(row) > 1:
                alignment = np.abs(
                    positions[row[1:], 1] - positions[row[0], 1]
                ) / np.minimum(positions[row[1:], 3], positions[row[0], 3])
                if not np.all(alignment > cutoff):
                    warnings.warn(
                        f"The vials in the same row are not aligned properly given the "
                        f"configuration {config}. Manual inspection is recommended."
                    )
    else:  # try to inference rows
        # sort by y1
        sorted_indices = np.argsort(positions[:, 1])
        while len(sorted_indices) > 0:  # while there are still rows to be sorted
            row_anchor = sorted_indices[0]
            sorted_indices = sorted_indices[1:]
            if len(sorted_indices) == 0:  # if there are no more rows to be sorted
                sorted_rows.append([row_anchor])
                break

            alignment = _calculate_alignment_ratio(
                positions[row_anchor, 1],
                positions[row_anchor, 3],
                positions[sorted_indices, 1],
                positions[sorted_indices, 3],
            )
            aligned_ = alignment > cutoff
            in_same_row = sorted_indices[aligned_].tolist()
            sorted_rows.append([row_anchor] + in_same_row)
            sorted_indices = sorted_indices[np.logical_not(aligned_)]

    # then, sort the vials in each row according to x1
    for i, row in enumerate(sorted_rows):
        sorted_rows[i] = sorted(row, key=lambda x: positions[x, 0])

    # finally, check if the vials are overlapping
    for row in sorted_rows:
        if len(row) == 1:
            continue
        alignment = _calculate_alignment_ratio(
            positions[row[0], 0],
            positions[row[0], 2],
            positions[row[1:], 0],
            positions[row[1:], 2],
        )
        if np.any(alignment > 0.25):
            warnings.warn(
                f"The vials in the same row are overlapping. Manual inspection is recommended."
            )

    return sorted_rows


def _calculate_alignment_ratio(
    cood1_tl: int | float | np.ndarray,
    cood1_br: int | float | np.ndarray,
    cood2_tl: int | float | np.ndarray,
    cood2_br: int | float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the alignment ratio between two sets of y-coordinates or x-coordinates

    :param int | float | np.ndarray cood1_tl:
        top-or-left y-coordinate or x-coordinate of the first set
    :param int | float | np.ndarray cood1_br:
        bottom-or-right y-coordinate or x-coordinate of the first set
    :param int | float | np.ndarray cood2_tl:
        top-or-left y-coordinate or x-coordinate of the second set
    :param int | float | np.ndarray cood2_br:
        bottom-or-right y-coordinate or x-coordinate of the second set
    :return float | np.ndarray:
        alignment ratio between the two sets of coordinates
    """
    overlapping = np.maximum(
        0, np.minimum(cood1_br, cood2_br) - np.maximum(cood1_tl, cood2_tl)
    )
    return overlapping / np.minimum(cood1_br - cood1_tl, cood2_br - cood2_tl)
