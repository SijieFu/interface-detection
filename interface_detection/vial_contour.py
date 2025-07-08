import warnings
import math
from copy import deepcopy
from typing import Iterable

import numpy as np
from scipy.stats import linregress
import cv2

from .color_inference import infer_cap_color, get_cap_mask_by_color
from .mask_to_contours import mask_to_polylines, polylines_to_mask, get_scaled_polylines
from .mask_to_contours import contours_to_mask, get_trimmed_mask_bbox
from .utils import display_image, no_modification_to_args
from .utils import get_gray_image, increase_image_contrast
from .utils import union_of_domains, find_longest_in_sequence

_BRACKETS = r"()[]{}"


@no_modification_to_args
def get_vessel_contour(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    where: Iterable[str] | str | None = None,
    sobel_threshold: int = 63,
    percentiles: tuple[int] | str | None = None,
    debug: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """
    Find the contour of a vessel in an image

    :param np.ndarray image:
        the image to find the contour in, grayscale or color (BGR)
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param Iterable[str] | str | None where:
        where to find the vessel contour (top, bottom, left, right)
    :param int sobel_threshold:
        threshold for Sobel gradient markers
    :param tuple[int] | str | None percentiles:
        percentile range for making the contour. The percentiles to use for the transformation. For example, for a top contour, the y-coordinates of the contour will be
        identified. But some noise may be present in the mask. The percentiles will be used to filter the outlier coordinates outside the
        defined percentiles. The y-coordinates within the percentiles will be used to find the slope and intercept of the line.
    :param bool debug:
        whether to show debug images in a window
    :return np.ndarray | list[np.ndarray]:
        the contour of the vessel

        - If `where` is str or only has one element, the contour will be a single array of shape (n, 2), where n is the number of points in the contour.
        - If `where` is a list of strings, the contour will be a list of arrays, each of shape (n, 2), where n is the number of points in the contour.
    """
    if where is None:
        locations = ["top", "bottom", "left", "right"]
    elif isinstance(where, str):
        locations = [where]
    else:
        locations = where
        assert all(
            loc in ["top", "bottom", "left", "right"] for loc in locations
        ), "Invalid value for 'where'"

    if isinstance(percentiles, str):  # compatibility with the API string input
        percentiles = tuple(int(x) for x in percentiles.strip(_BRACKETS).split(","))
        if len(percentiles) == 2:
            percentiles = [percentiles] * len(locations)
        else:
            assert (
                len(percentiles) == len(locations) * 2
            ), "Invalid value for `percentiles`"
            percentiles = (
                percentiles[i : i + 2] for i in range(0, len(percentiles), 2)
            )

    # Convert to grayscale if necessary
    if enhanced_image is None:
        enhanced_image = image.copy()
    gray_original = get_gray_image(enhanced_image)
    gray_original = cv2.medianBlur(gray_original, 5)

    _top, _left = 40, 50
    percentile_mapping = {
        "top": (_top, 100 - _top // 2),
        "bottom": (_top, 100 - _top // 2),
        "left": (_left, 100 - _left // 2),
        "right": (_left, 100 - _left // 2),
    }
    if not percentiles:
        percentiles = [percentile_mapping[loc] for loc in locations]
    elif (
        isinstance(percentiles, (tuple, list))
        and isinstance(percentiles[0], int)
        and len(percentiles) == 2
    ):
        percentiles = [percentiles] * len(locations)
    else:
        assert len(percentiles) == len(locations), "Invalid value for `percentiles`"

    contours = []
    for loc, percentile in zip(locations, percentiles):
        require_transpose = True if loc.lower() in ["top", "bottom"] else False
        require_flip = True if loc.lower() in ["bottom", "right"] else False

        gray = gray_original.copy()
        gray = gray.T if require_transpose else gray
        gray = np.flip(gray, axis=1) if require_flip else gray

        # mark vertical edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.minimum(np.absolute(sobelx), 255).astype(np.uint8)

        xdim, ydim = gray.shape
        xmid, ymid = xdim // 2, ydim // 2
        vertical_markers = np.where(sobelx > sobel_threshold, 255, 0).astype(np.uint8)

        # the following code is to find the indices of the contour points
        # contour_left is of shape (xdim, 2), where the first column is the x index and the second column is the y index
        contour_left = np.hstack(
            [
                np.arange(xdim).reshape(-1, 1),
                np.argmax(vertical_markers, axis=1).reshape(-1, 1),
            ]
        )
        contour_left = contour_left[
            np.bitwise_and(contour_left[:, 1] >= 1, contour_left[:, 1] <= ymid), :
        ]
        x1, x2 = np.percentile(contour_left[:, 1], percentile)
        x1, x2 = math.floor(x1), math.ceil(x2)
        contour_left = contour_left[
            np.bitwise_and(contour_left[:, 1] >= x1, contour_left[:, 1] <= x2), :
        ]

        if len(contour_left) == 0:
            raise ValueError(
                f"No vessel contour found for {loc} location in your image. Are you sure the vessel is present?"
            )
        elif len(contour_left) == 1:
            slope, intercept = 0, contour_left[0, 1]
        else:
            slope, intercept, _, _, _ = linregress(*contour_left.T)

        contour_left = np.hstack(
            [
                np.arange(xdim).reshape(-1, 1),
                slope * np.arange(xdim).reshape(-1, 1) + intercept,
            ]
        )
        contour_left[:, 1] = np.maximum(contour_left[:, 1], 0)
        contour_left[:, 1] = np.minimum(contour_left[:, 1], ydim - 1)

        # return the original indices
        contour_left[:, 1] = (
            ydim - 1 - contour_left[:, 1] if require_flip else contour_left[:, 1]
        )
        contour_left = (
            np.flip(contour_left, axis=1) if require_transpose else contour_left
        )
        contours.append(contour_left.astype(np.int32))

    if debug:
        # Draw the contour on the original image
        image_with_contour = gray_original.copy()
        for contour in contours:
            image_with_contour[contour[:, 0], contour[:, 1]] = 255
        display_image(
            {
                "Grayscale": gray_original,
                "With contour": image_with_contour,
            },
            title="Vessel contour",
            axis=1,
        )

    return contours[0] if isinstance(where, str) else contours


@no_modification_to_args
def get_vessel_mask(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    sobel_threshold: int = 63,
    top: np.ndarray | None = None,
    bottom: np.ndarray | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Get the mask of the vessel in the image.

    :param np.ndarray image:
        the image, grayscale or color (BGR)
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param int sobel_threshold:
        threshold for Sobel gradient markers. See `get_vessel_contour` for more details.
    :param np.ndarray | None top:
        the contour of the top of the vessel. If None, then will assume the top of the image is the top of the vessel.
    :param np.ndarray | None bottom:
        the contour of the bottom of the vessel. If None, then will assume the bottom of the image is the bottom of the vessel.
    :param np.ndarray | None left:
        the contour of the left of the vessel. If None, then will assume the left of the image is the left of the vessel.
    :param np.ndarray | None right:
        the contour of the right of the vessel. If None, then will assume the right of the image is the right of the vessel.

        > `top`, `bottom`, `left`, and `right` are the contours of the vessel in the image.
        > They can be obtained using the `get_vessel_contour` function.
        > If all four contours are None, they will be calculated using the `get_vessel_contour` function.
    :param bool debug:
        whether to show debug images in a window
    :return np.ndarray:
        the 2D mask of the vessel in the image (pixels inside the vessel are True)
    """
    if all(contour is None for contour in [top, bottom, left, right]):
        top, bottom, left, right = get_vessel_contour(
            image,
            enhanced_image=enhanced_image,
            where=["top", "bottom", "left", "right"],
            sobel_threshold=sobel_threshold,
        )

    crop_mask = contours_to_mask(
        top, bottom, left, right, mask=np.ones_like(image, dtype=bool)
    )

    if debug:
        display_image(
            {
                "Original": image,
                "Cropped": np.where(crop_mask, (1, 1, *image.shape[2:]), image, 0),
            },
            title="Cropped vessel",
            axis=1,
        )

    return crop_mask


@no_modification_to_args
def get_cap_range(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    force_gray: bool = False,
    mask: np.ndarray | None = None,
    sobel_threshold: int = 63,
    cap_ratio: float | tuple[float] | str = (0.1, 0.3),
    target: int | tuple[int, int, int] | str = -1,
    tolerance: int | tuple[int, int, int] | str = 40,
    debug: bool = False,
) -> tuple[int, int, tuple[int] | int, np.ndarray]:
    """
    Find the contour of a cap in an image

    :param np.ndarray image:
        the image, grayscale or color (BGR)
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param bool force_gray:
        whether to force the image to be grayscale if it is colored. Typically not recommended.
        If `enhanced_image` is provided and `force_gray` is True, the `enhanced_image` will be used for the grayscale conversion.
    :param np.ndarray | None mask:
        the mask of the vessel in the image to refine the cap detection.
        If None, it will be calculated using the `get_vessel_mask` function.
    :param int sobel_threshold:
        threshold for Sobel gradient markers. The vertical gradient (sobely) should be larger than this threshold
        for a pixel to be considered as a part of a horizontal edge.
    :param float | tuple[float] | str cap_ratio:
        the min/max ratio of the cap height to the vessel height. The vessel height is estimated with `get_vessel_mask` function.
        The upper cap bound cannot be upper than the first cap bound. The top of the image is 0 and the bottom of the image is 1.
        The lower cap bound cannot be lower than the second cap bound.

        - If `cap_ratio` is a float, it will be used as the lower cap bound. And the upper cap bound will be `cap_ratio * 0.5`.
        - If `cap_ratio` is a tuple of two floats, it will be used as the lower and upper cap bounds.
        - If `cap_ratio` is a string, it will be converted to a float or a tuple of two floats. This is for
            compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str target:
        the target color of the cap, either a single value or a tuple of values for each channel.
        If target is -1 or is not valid, it will be inferred using the `.color_inference.infer_cap_color` function.

        - If `target` is an int, it will be used as the target color for all channels.
        - If `target` is a tuple of three ints, it will be used as the target color for each channel (BGR).
        - If `target` is a string, it will be converted to an int or a tuple of three ints. This is for
            compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str tolerance:
        the tolerance for the target color, either a single value or a tuple of values for each channel.
        Check the function `.color_inference.infer_cap_color` for more details.
    :param bool debug:
        whether to show debug images in a window
    :return int:
        the index of the first row of the cap (0-indexed from the top of the image)
    :return int:
        the index of the last row of the cap (0-indexed from the top of the image)
    :return tuple[int] | int:
        the color of the cap, either a single value or a tuple of values for each channel (BGR).
        If the image is grayscale, it will be a single value.
    :return np.ndarray:
        the mask of the cap in the image, a 2D binary mask.
    """
    if isinstance(cap_ratio, str):  # compatibility with the API string input
        cap_ratio = tuple(float(x) for x in cap_ratio.strip(_BRACKETS).split(","))
        cap_ratio = cap_ratio[0] if len(cap_ratio) == 1 else cap_ratio
    if isinstance(target, str):
        target = tuple(int(x) for x in target.strip(_BRACKETS).split(","))
        target = target[0] if len(target) == 1 else target
    if isinstance(tolerance, str):
        tolerance = tuple(int(x) for x in tolerance.strip(_BRACKETS).split(","))
        tolerance = tolerance[0] if len(tolerance) == 1 else tolerance

    if enhanced_image is None:
        enhanced_image = image

    gray = get_gray_image(enhanced_image)
    if force_gray:
        image = gray.copy()
        enhanced_image = gray.copy()
    ndim = image.ndim

    if mask is None:
        mask = get_vessel_mask(image, enhanced_image=enhanced_image)
    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    vessel_top = int(np.percentile(np.argmax(mask_2d, axis=0), 50))

    # get the average of values between 30 and 70 percentiles
    heights = np.sum(mask_2d, axis=0)
    p1, p2 = np.percentile(heights, [30, 70])
    p1, p2 = int(p1), int(p2)
    # estimate height of the vessel
    H = np.mean(heights[np.bitwise_and(heights >= p1, heights <= p2)])
    if not isinstance(cap_ratio, Iterable):
        cap_ratio = (cap_ratio / 2.0, cap_ratio)
    else:
        cap_ratio = (min(cap_ratio), max(cap_ratio))
    # the estimated min and max height of the cap
    Hcap_min, Hcap_max = int(H * cap_ratio[0]), int(H * cap_ratio[1])

    if np.any(np.array(target) < 0) or np.any(np.array(target) > 255):
        cap_color, _upper, _lower, _mask = infer_cap_color(
            image,
            enhanced_image=enhanced_image,
            mask=mask_2d,
            sobel_threshold=sobel_threshold,
            color_tolerance=tolerance,
            cap_ratio=cap_ratio,
            debug=False,
        )
    else:
        cap_color = target

    if ndim == 3:
        if not isinstance(cap_color, Iterable):
            cap_color = np.repeat(cap_color, 3)
        cap_color = np.array(cap_color).astype(int).flatten()
        cap_color = np.clip(cap_color, 0, 255)
        cap_color = tuple(cap_color.tolist())
    elif ndim == 2:
        if isinstance(cap_color, Iterable):
            raise ValueError(
                "`cap_color` must be a single value for grayscale images, but got a tuple"
            )
        cap_color = int(cap_color)
    else:
        raise ValueError("Invalid image dimension")

    cap_mask = get_cap_mask_by_color(
        image=image,
        color=cap_color,
        enhanced_image=enhanced_image,
        mask=mask_2d,
        sobel_threshold=sobel_threshold,
        cap_ratio=cap_ratio,
        color_tolerance=tolerance,
    )
    # relax the condition to find the cap since sobely will be used to refine the search
    cap_ratio_per_row: np.ndarray = np.mean(cap_mask, axis=1)
    row_is_cap_color = cap_ratio_per_row >= 0.5

    kernel = np.ones(3, np.uint8)
    row_is_cap_color = cv2.erode(
        row_is_cap_color.astype(np.uint8), kernel, iterations=2
    )
    row_is_cap_color = cv2.dilate(row_is_cap_color, kernel, iterations=2).astype(bool)

    _cap_top_idx, _cap_bottom_idx = find_longest_in_sequence(
        row_is_cap_color[vessel_top : vessel_top + Hcap_max + 1], value=1
    )
    if _cap_top_idx == -1:
        cap_top_low_idx = vessel_top + Hcap_min
    else:
        cap_top_low_idx = vessel_top + _cap_top_idx

    # get sobel y
    sobely = cv2.Sobel(gray[: cap_top_low_idx + 1], cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely) >= sobel_threshold
    sobely = np.mean(sobely, axis=1) >= 0.4
    _cap_top_idx, _ = find_longest_in_sequence(
        sobely[vessel_top : cap_top_low_idx + 1], value=True
    )
    if _cap_top_idx == -1:
        cap_top_idx = vessel_top
    else:
        cap_top_idx = vessel_top + _cap_top_idx
    cap_top = np.clip(cap_top_idx, vessel_top, vessel_top + Hcap_min)
    cap_top = int(cap_top)

    # update vessel height based on the cap top
    H2 = H - (cap_top - vessel_top)
    Hcap_min2, Hcap_max2 = int(H2 * cap_ratio[0]), int(H2 * cap_ratio[1])

    _cap_top_idx, _cap_bottom_idx = find_longest_in_sequence(
        row_is_cap_color[cap_top : cap_top + Hcap_max2 + 1], value=1
    )
    if _cap_bottom_idx == -1:
        cap_bottom_idx = cap_top + min((Hcap_min2 + Hcap_max2) // 2, Hcap_min2 * 2)
    else:
        cap_bottom_idx = cap_top + _cap_bottom_idx
    cap_bottom = np.clip(cap_bottom_idx, cap_top + Hcap_min2, cap_top + Hcap_max2)
    cap_bottom = int(cap_bottom)

    # refine cap color based on `cap_top` and `cap_bottom`
    shift = int(image.shape[1] * 0.25)
    cap_color = np.percentile(
        image[cap_top : cap_bottom + 1, shift:-shift], 50, axis=(0, 1)
    )
    if ndim == 3:
        cap_color = tuple(cap_color.astype(int).tolist())
    else:
        cap_color = int(cap_color)

    if debug:
        print(
            f"Cap top: {cap_top} ({cap_top_idx}, {vessel_top} - {vessel_top + Hcap_min})"
        )
        print(
            f"Cap bottom: {cap_bottom} ({cap_bottom_idx}, {cap_top + Hcap_min2} - {cap_top + Hcap_max2})"
        )
        print(f"Cap color: {cap_color}")
        bg_color = 0 if np.mean(cap_color) >= 128 else 255
        if ndim == 3:
            cap_by_pixel = np.where(cap_mask[:, :, np.newaxis], image, bg_color)
            cap_rows = np.zeros_like(image, dtype=np.uint8)
            cap_rows[np.where(row_is_cap_color)[0], :, :] = np.array(
                [255, 0, 0]
            ).reshape(1, 1, 3)
            marked_cap = image.copy()
            marked_cap[[cap_top, cap_bottom], :, :] = np.array([255, 0, 0]).reshape(
                1, 1, 3
            )
        else:
            cap_by_pixel = np.where(cap_mask, image, bg_color)
            cap_rows = np.zeros_like(image, dtype=np.uint8)
            cap_rows[np.where(row_is_cap_color)[0], :] = 255
            marked_cap = image.copy()
            marked_cap[[cap_top, cap_bottom], :] = bg_color
        display_image(
            {
                "Original": image,
                "Enhanced": enhanced_image,
                "Cap mask": cap_mask.astype(np.uint8) * 255,
                "Cap by pixel": cap_by_pixel,
                "Cap rows": cap_rows,
                "Marked cap": marked_cap,
            },
            title="Cap contour",
            axis=1,
        )

    return cap_top, cap_bottom, cap_color, cap_mask


@no_modification_to_args
def compensate_for_background(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    contours: list[np.ndarray] | None = None,
    vessel_mask: np.ndarray | None = None,
    sobel_threshold: int = 63,
    sobel_xy_ratio: float = 0.75,
    dilation: np.ndarray | bool = True,
    bg_resolution: int | float = 0.02,
    bg_sobel_ratio: float = 0.55,
    color_tolerance: int | tuple[int, int, int] | str = 30,
    debug: bool = False,
    title: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compensate for the background in an image and get the mask for horizontal edges from the background (instead of the liquid interface).

    :param np.ndarray image:
        the image, grayscale or color (BGR)
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param list[np.ndarray] | None contours:
        the contours of the vessel in the order of (top, bottom, left, right).
        If None, the contours will be calculated using the `get_vessel_contour` function.
    :param np.ndarray | None vessel_mask:
        the mask of the vessel in the image. If None, it will be calculated using the `get_vessel_mask` function.
    :param int sobel_threshold:
        threshold for Sobel gradient markers. The vertical gradient (sobely) should be larger than this threshold
        for a pixel to be considered as a part of a horizontal interface.
    :param float sobel_xy_ratio:
        horizontal gradient (sobelx) should be less than this ratio of the vertical gradient (sobely)
        for a pixel to be considered as a part of a horizontal interface. Otherwise, it is considered as a part of a vertical interface or noise.
    :param np.ndarray | bool dilation:
        whether to dilate the horizontal markers or not. In other words, an identified horizontal marker will be dilated to include the surrounding pixels.
    :param int | float bg_resolution:
        the resolution of the background interfaces. This is used to determine when the background outside the vessel is creating horizontal markers,
        how far above or below should we check within the vessel to find the background interface. This is because the vessel and the liquid within
        the vessel will cause displacements and distortions in the background.

        - If `bg_resolution` is an integer, it will be used as the number of pixels to check above and below the horizontal marker.
        - If `bg_resolution` is a float, it will be converted to an integer by multiplying it with the height of the vessel.
    :param float bg_sobel_ratio:
        for background correction, if the vertical Sobel gradient (sobely) is smaller than this ratio of the vertical Sobel gradient (sobely)
        in the background, then it is assumed this is a background-induced gradient.

        This parameter should be between 0.5 and 1.0, and is proportional to the background correction intensity. Typically, if the image is
        in a noisy background, this value should be lower so that true interfaces are not removed.
        Under a relatively clean background, this value can be higher (e.g., 0.8) for more aggressive background correction.
    :param int | tuple[int, int, int] | str color_tolerance:
        the tolerance for the color of the background to the color within the vessel (inferred to the HSV space).
        When checking if a horizontal marker is a background interface, the color of the pixel should also match the color of the background.
        And the match is checked in the HSV space within the `color_tolerance` range.

        - If `color_tolerance` is an int, the HSV color tolerance will be a half of the value for the H and S channels, and the full value for the V channel.
        - If `color_tolerance` is a tuple of three ints, it will be used as the color tolerance for each channel (H, S, V).
        - If `color_tolerance` is a string, it will be converted to an int or a tuple of three ints. This is for
            compatibility with URL query parameters.
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        title for the debug images in the debug window
    :return np.ndarray:
        The compensated/corrected vessel horizontal markers
        (horizontal markers for the vessel only, with the background correction within vessel interfaces).
    :return np.ndarray:
        the mask of the background pixels that give interfaces within the vessel
    """
    if isinstance(color_tolerance, str):  # compatibility with the API string input
        color_tolerance = tuple(
            int(x) for x in color_tolerance.strip(_BRACKETS).split(",")
        )
        color_tolerance = (
            color_tolerance[0] if len(color_tolerance) == 1 else color_tolerance
        )
    color_tolerance: np.ndarray = np.array(color_tolerance).flatten()
    color_tolerance = np.clip(color_tolerance, 0, 255).astype(int)
    if len(color_tolerance) == 1:
        _tol = color_tolerance[0]
        color_tolerance = np.maximum(10, np.array([_tol // 2, _tol // 2, _tol]))
    elif not len(color_tolerance) == 3:
        raise ValueError(
            "Invalid value for `color_tolerance` for background compensation"
        )

    gray = get_gray_image(image)
    if enhanced_image is None:
        enhanced_image = image.copy()
    enhanced_gray = get_gray_image(enhanced_image)

    if image.ndim == 3:
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        HSV = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    if contours is None:
        contours = get_vessel_contour(gray, where=["top", "bottom", "left", "right"])
    top, bottom, left, right = contours
    Hvessel = int(np.percentile(bottom[:, 0] - top[:, 0], 50))

    if vessel_mask is None:
        vessel_mask = get_vessel_mask(
            gray,
            enhanced_image=enhanced_gray,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )
    bg_mask = np.bitwise_not(vessel_mask)
    bg_width = np.maximum(np.sum(bg_mask, axis=1), 10)

    # vertical gradient, horizontal edge
    sobely = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.minimum(np.absolute(sobely), 255).astype(np.uint8)
    # horizontal gradient, vertical edge
    sobelx = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.minimum(np.absolute(sobelx), 255).astype(np.uint8)
    # a horizontal edge should have a larger vertical gradient than horizontal gradient
    possible_interface_mask = sobelx <= (sobely * sobel_xy_ratio)
    hmarkers = np.bitwise_and(sobely >= sobel_threshold, possible_interface_mask)

    # WARNING: dilation also enables noise to be considered as interfaces
    if (isinstance(dilation, np.ndarray) and dilation.size > 0) or np.all(dilation):
        if isinstance(dilation, bool):
            dilation_kernel = np.ones((1, 3), np.uint8)
        else:
            dilation_kernel = dilation.astype(np.uint8)
        niter = 2
        hmarkers = hmarkers.astype(np.uint8)
        hmarkers = cv2.erode(hmarkers, dilation_kernel, iterations=niter)
        hmarkers = cv2.dilate(hmarkers, dilation_kernel, iterations=niter).astype(bool)
        dilation = True
    else:
        dilation = False

    vessel_hmarkers = np.bitwise_and(hmarkers, vessel_mask)
    bg_hmarkers = np.bitwise_and(hmarkers, bg_mask)

    bg_row_is_interface = (np.sum(bg_hmarkers, axis=1) / bg_width) >= 0.3
    bg_row_is_interface = np.bitwise_and(
        bg_row_is_interface, np.sum(bg_hmarkers, axis=1) >= 10
    )
    bg_interface_rows = np.where(bg_row_is_interface)[0]

    if isinstance(bg_resolution, float):
        bg_resolution = math.ceil(Hvessel * bg_resolution)
    nearby = max(2, bg_resolution)

    # the upper tolerance for the V channel should be limited (the glass should deplete background brightness)
    _ctol = color_tolerance.reshape(1, 3)
    bg_sobel_percentile = int(np.clip(bg_sobel_ratio, 0.2, 1.0) * 100)

    def _consistent_bg_vessel_color_mask(_rows: np.ndarray | list) -> np.ndarray:
        # performance warning: this function is slow because it uses a for loop
        # TODO: vectorize this function
        _compensation_mask = np.zeros_like(gray, dtype=bool)
        if len(_rows) == 0:
            return _rows, np.zeros((0, gray.shape[1]), dtype=bool)

        _bg_colors_low, _bg_colors_high = np.array(
            [
                np.percentile(HSV[_row, bg_hmarkers[_row]], (30, 70), axis=0)
                for _row in _rows
            ]
        ).transpose(1, 0, 2)
        _bg_colors_low = np.maximum(0, _bg_colors_low - _ctol).astype(np.uint8)
        _bg_colors_high = np.minimum(255, _bg_colors_high + _ctol).astype(np.uint8)

        _bg_sobely_cutoff = (
            np.array(
                [
                    np.percentile(sobely[_row, bg_hmarkers[_row]], bg_sobel_percentile)
                    for _row in _rows
                ]
            )
            .astype(int)
            .reshape(-1, 1)
        )

        for _irow, _row in enumerate(_rows):
            _row_min, _row_max = max(0, _row - nearby), min(
                gray.shape[0], _row + nearby + 1
            )
            _slice = slice(_row_min, _row_max)
            _comp_mask = cv2.inRange(
                HSV[_slice], _bg_colors_low[_irow], _bg_colors_high[_irow]
            ).astype(bool)
            _comp_mask_edge = sobely[_slice] <= _bg_sobely_cutoff[_irow]
            _comp_mask = np.bitwise_and(_comp_mask, _comp_mask_edge)
            _compensation_mask[_slice] = np.bitwise_or(
                _compensation_mask[_slice], _comp_mask
            )

        _rows = np.where(np.any(_compensation_mask, axis=1))[0]
        _compensation_mask = _compensation_mask[_rows]
        return _rows, _compensation_mask

    if len(bg_interface_rows) > 0:
        rows_compensated, compensation_mask = _consistent_bg_vessel_color_mask(
            bg_interface_rows
        )
        if len(rows_compensated) > 0:
            if dilation:
                compensation_mask = cv2.erode(
                    compensation_mask.astype(np.uint8),
                    dilation_kernel,
                    iterations=niter,
                )
            vessel_hmarkers_compensated = vessel_hmarkers.copy()
            vessel_hmarkers_compensated[rows_compensated] = np.bitwise_and(
                vessel_hmarkers[rows_compensated], np.bitwise_not(compensation_mask)
            )
            vessel_hmarkers_bg_mask = np.zeros_like(vessel_hmarkers, dtype=bool)
            vessel_hmarkers_bg_mask[rows_compensated] = np.bitwise_and(
                compensation_mask, vessel_hmarkers[rows_compensated]
            )

    if len(bg_interface_rows) == 0 or len(rows_compensated) == 0:
        vessel_hmarkers_compensated = vessel_hmarkers.copy()
        vessel_hmarkers_bg_mask = np.zeros_like(vessel_hmarkers, dtype=bool)

    if debug:
        display_image(
            {
                "Gray": gray.copy(),
                "Sobel y": np.where(sobely >= sobel_threshold, 255, 0).astype(np.uint8),
                "Bg markers": np.where(bg_hmarkers, 255, 0).astype(np.uint8),
                "Vessel markers": np.where(vessel_hmarkers, 255, 0).astype(np.uint8),
                "Corrected vessel markers": np.where(
                    vessel_hmarkers_compensated, 255, 0
                ).astype(np.uint8),
            },
            title=title or "Compensated vessel markers",
            axis=1,
        )

    return vessel_hmarkers_compensated, vessel_hmarkers_bg_mask


@no_modification_to_args
def find_label_mask(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    sobel_threshold: int = 16,
    color_low: int | tuple[int, int, int] | str = 150,
    color_high: int | tuple[int, int, int] | str = 220,
    check_color_in_gray: bool = False,
    contours: list[np.ndarray] | None = None,
    vessel_mask: np.ndarray | None = None,
    cap_bottom: int | None = None,
    debug: bool = False,
    title: str | None = None,
) -> tuple[np.ndarray, list[list[int, int]]]:
    """
    Find the mask of any potential *white* label on the vessel.

    > NOTE: NOT WORKING VERY WELL YET

    :param np.ndarray image:
        the image, grayscale or color (BGR)
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param int sobel_threshold:
        threshold for Sobel gradient markers. Either the vertical gradient (sobely) or the horizontal gradient (sobelx)
        should be larger than this threshold for a pixel to be considered as a part of a label boundary.
    :param int | tuple[int, int, int] | str color_low:
        the lower bound of the label color.

        - If a single int, it will be used as the lower bound for all channels.
        - If a tuple of three ints, it will be used as the lower bound for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str color_high:
        the upper bound of the label color. Similar to `color_low`, but for the upper bound.
    :param bool check_color_in_gray:
        whether to check the color in a grayscale image. In other words, if the image is colored,
        it will be converted to grayscale and the color will be checked in the grayscale image.
    :param list[np.ndarray] | None contours:
        the contours of the vessel in the order of (top, bottom, left, right).
        If None, the contours will be calculated using the `get_vessel_contour` function.
    :param np.ndarray | None vessel_mask:
        the mask of the vessel in the image.
        If None, it will be calculated using the `get_vessel_mask` function.
    :param int | None cap_bottom:
        the row index of the bottom of the cap (0-indexed from the top of the image).
        If None, it will be calculated using the `get_cap_range` function.
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        title for the debug images in the debug window
    :return np.ndarray:
        the mask of the label in the image (pixels inside the label are True)
    :return list[list[int, int]]:
        the polylines for each individual label in the image
    """
    available_nchannels = [1, 3] if image.ndim == 3 else [1]
    if isinstance(color_low, str):  # compatibility with the API string input
        color_low = tuple(int(x) for x in color_low.strip(_BRACKETS).split(","))
        assert len(color_low) in available_nchannels, "Invalid value for `color_low`"
        color_low = color_low[0] if len(color_low) == 1 else color_low
    if isinstance(color_high, str):
        color_high = tuple(int(x) for x in color_high.strip(_BRACKETS).split(","))
        assert len(color_high) in available_nchannels, "Invalid value for `color_high`"
        color_high = color_high[0] if len(color_high) == 1 else color_high

    gray = get_gray_image(image)
    if image.ndim == 3:
        LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        _image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        LAB = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)

    check_color_in_gray = check_color_in_gray if image.ndim == 3 else True
    image4color_analysis = gray.copy() if check_color_in_gray else image.copy()

    color_low, color_high = (
        np.minimum(color_low, color_high).flatten(),
        np.maximum(color_low, color_high).flatten(),
    )
    # regulate color_low and color_high to be within np.uint8 range
    color_low: np.ndarray = np.clip(color_low, 0, 255).astype(int)
    color_high: np.ndarray = np.clip(color_high, 0, 255).astype(int)
    if check_color_in_gray and not len(color_low) == 1:
        color_low = np.mean(color_low).astype(int)
        color_high = np.mean(color_high).astype(int)
    elif not check_color_in_gray and len(color_low) == 1:
        color_low = np.repeat(color_low, 3)
        color_high = np.repeat(color_high, 3)
    color_mid = ((color_low + color_high) // 2).astype(int)
    color_tol = np.maximum(np.abs(color_high - color_low) // 2, 10).astype(int)

    if contours is None:
        contours = get_vessel_contour(
            image,
            enhanced_image=enhanced_image,
            where=["top", "bottom", "left", "right"],
        )
    top, bottom, left, right = contours

    if vessel_mask is None:
        vessel_mask = get_vessel_mask(
            image,
            enhanced_image=enhanced_image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )

    if cap_bottom is None:
        _, cap_bottom, *_ = get_cap_range(
            image, enhanced_image=enhanced_image, mask=vessel_mask
        )
    vessel_mask[: cap_bottom + 1] = False

    # dimensions of the glass part of the vessel
    Wvessel = math.ceil(np.percentile(right[:, 1] - left[:, 1], 50))
    vessel_bottom = math.ceil(np.percentile(bottom[:, 0], 50))
    Hvessel = vessel_bottom - cap_bottom + 1
    Avessel = Wvessel * Hvessel

    # erode the image to get rid of the text
    size = max(3, int(min(gray.shape) / 200.0))
    kernel = np.ones((size, size), np.uint8)
    # text_color_cutoff_grayscale
    text_mask_raw = gray <= np.maximum(40, color_low).mean()
    # find contours with connected components
    text_mask_small = (text_mask_raw * 127 + 128).astype(np.uint8)
    text_mask_small = np.where(vessel_mask, text_mask_small, 0)
    connectivity = 8
    (numLabels, labels, stats, _) = cv2.connectedComponentsWithStats(
        text_mask_small, connectivity, cv2.CV_32S
    )
    # only areas with a small area are considered as text
    _idxs = np.arange(1, numLabels)  # exclude the background label (0)
    _idxs = list(filter(lambda x: stats[x, cv2.CC_STAT_AREA] <= Avessel // 20, _idxs))
    # get a mask for these filtered labels
    text_mask_small = np.isin(labels, _idxs).astype(bool)
    text_mask_raw = np.bitwise_and(text_mask_raw, text_mask_small)

    text_mask_dilated = cv2.dilate(
        text_mask_raw.astype(np.uint8),
        kernel,
        iterations=math.ceil(min(gray.shape) / 200.0),
    ).astype(bool)
    text_mask = np.bitwise_and(text_mask_dilated, vessel_mask)
    non_text_mask = np.bitwise_and(vessel_mask, np.bitwise_not(text_mask))

    # remove some of the non-text areas from the mask to prevent over-dilation of black text
    possible_label_mask = cv2.inRange(image4color_analysis, color_low, color_high)
    possible_label_mask = np.bitwise_and(possible_label_mask.astype(bool), vessel_mask)
    possible_label_mask = np.bitwise_or(possible_label_mask, text_mask)

    presobel = gray.copy()
    sobelx = cv2.Sobel(presobel, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.minimum(np.absolute(sobelx), 255).astype(np.uint8)
    sobely = cv2.Sobel(presobel, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.minimum(np.absolute(sobely), 255).astype(np.uint8)
    sobel_raw = np.maximum(sobelx, sobely)
    sobel_raw_mask = sobel_raw >= sobel_threshold
    # edges are marked as black(1, True) as boundaries (texts are filled with white(0, False))
    sobel = np.bitwise_and(sobel_raw_mask, non_text_mask)
    # label areas are marked as 1, non-label areas are marked as 0
    possible_label_mask = np.bitwise_and(possible_label_mask, np.bitwise_not(sobel))
    possible_label_region = possible_label_mask.astype(np.uint8) * 255

    # find contours with connected components
    connectivity = 4
    (numLabels, labels, stats, _) = cv2.connectedComponentsWithStats(
        possible_label_region, connectivity, cv2.CV_32S
    )

    # sort from largest area to smallest
    labelIDs = np.arange(1, numLabels)
    labelIDs = sorted(labelIDs, key=lambda x: -stats[x, cv2.CC_STAT_AREA], reverse=True)
    unchecked_labelIDs = deepcopy(labelIDs)

    label_masks = np.zeros_like(gray, dtype=bool)
    # polylines of all labels
    label_mask_polylines = []

    # labels that are too small or too large will be ignored
    height_cutoff = max(Hvessel // 50, Wvessel // 10) + 1
    width_cutoff = Wvessel // 5 + 1
    while len(unchecked_labelIDs) > 0:
        i = unchecked_labelIDs.pop()

        # first, check if the label is too small or too large
        _, _, w, h, area = stats[i]
        if h <= height_cutoff or w <= width_cutoff:
            continue
        if area >= Avessel // 3 or h * w >= Avessel // 2:
            if debug:
                print(
                    f"{title}-{i}: area too large, {area} vs. {Avessel // 3} ({Avessel} // 3) "
                    f"| {h * w} vs. {Avessel // 2} ({Avessel} // 2)"
                )
            continue

        # get the average color and variance in this area
        mask = (labels == i).astype(bool)

        if debug:
            display_image(
                {
                    "Image": image,
                    "Enhanced": enhanced_image,
                    "Text mask": text_mask.astype(np.uint8) * 255,
                    "Color mask": possible_label_mask.astype(np.uint8) * 255,
                    "Sobel mask": sobel_raw_mask.astype(np.uint8) * 255,
                    "Label region": possible_label_region,
                    "Mask": mask.astype(np.uint8) * 255,
                },
                title=f"{title}-{i}: label mask",
                axis=1,
            )

        x1, y1, x2, y2, mask_cropped = get_trimmed_mask_bbox(mask)
        mask_trimmed = np.zeros_like(mask, dtype=bool)
        mask_trimmed[y1 : y2 + 1, x1 : x2 + 1] = mask_cropped

        # preferably, the label should not be too close to the bottom of the vessel
        # unless its length is comparable to the vessel height
        _boundary_distance = max(Hvessel // 20, Wvessel // 4)
        if vessel_bottom - (y1 + y2) // 2 <= min(Hvessel // 6, Wvessel // 2) or (
            vessel_bottom - y2 <= _boundary_distance
            and cap_bottom - y1 <= _boundary_distance
        ):
            if debug:
                print(
                    f"{title}-{i}: area too close to the bottom, "
                    f"{vessel_bottom - (y1 + y2) // 2} vs. {_boundary_distance}"
                )
            continue
        # if the label if too close to the top, no need to consider its effect on the vessel
        # (should not fill liquid to this high, noise is common around vessel top)
        if y2 - cap_bottom <= max(Hvessel // 6, Wvessel // 4):
            if debug:
                print(
                    f"{title}-{i}: area too close to the top, "
                    f"{y1 - cap_bottom} vs. {_boundary_distance}"
                )
            continue
        if (x2 - x1) <= width_cutoff:
            if debug:
                print(f"{title}-{i}: width too small, {x2 - x1} vs. {width_cutoff}")
            continue
        if (x2 - x1) >= Wvessel * 0.8:
            if debug:
                print(f"{title}-{i}: width too large, {x2 - x1} vs. {Wvessel * 0.8}")
            continue

        # find the bounding polyline of mask region
        mask_polylines = mask_to_polylines(
            mask_cropped, epsilon_factor=0.001, convex_hull=True, debug=False
        )
        mask_polylines = np.array(mask_polylines) + np.array([[x1, y1]])

        # fill the mask region using the bounding polyline
        mask_refined_filled = polylines_to_mask([mask_polylines], gray.shape)
        mask_refined_filled_area = np.sum(mask_refined_filled)

        _overlap_mask = np.bitwise_and(mask_refined_filled, label_masks)
        if np.sum(_overlap_mask) >= mask_refined_filled_area * 0.3:
            if debug:
                print(
                    f"{title}-{i}: mask already claimed by other labels, refined area {mask_refined_filled_area}"
                    f" vs. current mask area {np.sum(label_masks)}, overlap area {np.sum(_overlap_mask)}"
                )
            continue

        # within the solid mask, find the color-matching mask
        mask_refined_filled_label = np.bitwise_and(
            mask_refined_filled, possible_label_mask
        )
        mask_refined_filled_label_area = np.sum(mask_refined_filled_label)

        # checked refined mask and dimensions
        if mask_refined_filled_label_area <= mask_refined_filled_area * 0.65:
            if debug:
                print(
                    f"{title}-{i}: area not well filled due to noise, label area {mask_refined_filled_label_area} vs. "
                    f"filled solid area {mask_refined_filled_area} ({mask_refined_filled_label_area/mask_refined_filled_area:.2f})"
                )
            continue
        _height = np.sum(mask_refined_filled, axis=0)
        _height = _height[_height > 0]
        _height_median = np.median(_height)
        _width = np.sum(mask_refined_filled, axis=1)
        _width = _width[_width > 0]
        _width_median = np.median(_width)
        if _height_median <= height_cutoff or _width_median <= width_cutoff:
            if debug:
                print(
                    f"{title}-{i}: area not well filled due to noise,  median height {_height_median:.2f} "
                    f"vs. {height_cutoff}, median width {_width_median:.2f} vs. {width_cutoff}"
                )
            continue
        _height_avg, _height_std = np.mean(_height), np.std(_height)
        _width_avg, _width_std = np.mean(_width), np.std(_width)
        _cutoff = 0.5
        if _height_std / _height_avg >= _cutoff or _width_std / _width_avg >= _cutoff:
            if debug:
                print(
                    f"{title}-{i}: area shape not well defined (irregular shape detected), "
                    f"height std {round(_height_std, 2)} / avg {_height_avg:.2f} >= {_cutoff:.2f} or "
                    f"width std {round(_width_std, 2)} / avg {_width_avg:.2f} >= {_cutoff:.2f}"
                )
            continue

        # checked color and variance
        color_avg, color_std = cv2.meanStdDev(
            image4color_analysis, mask=mask_refined_filled_label.astype(np.uint8)
        )
        color_avg = color_avg.flatten().astype(int)
        color_std = color_std.flatten().astype(int)
        color_std_tolerance = np.minimum(color_tol / 1.5, 30).astype(int)
        if np.any(np.abs(color_avg - color_mid) > color_tol) or np.any(
            color_std > color_std_tolerance
        ):
            if debug:
                print(
                    f"{title}-{i}: color not matched, {color_avg.flatten()} ± {color_std.flatten()}"
                    f"vs. {color_mid} ± {color_std_tolerance}"
                )
            continue

        # check color and variance in LAB space
        _, color_std_LAB = cv2.meanStdDev(
            LAB, mask=mask_refined_filled_label.astype(np.uint8)
        )
        if np.any(
            np.mean(color_std_LAB.flatten()[1:]) >= np.mean(color_std_tolerance) / 1.5
        ):
            if debug:
                print(
                    f"{title}-{i}: color not matched in LAB, "
                    f"{color_std_LAB.flatten()[:2].mean()} vs. {np.mean(color_std_tolerance)/1.5}"
                )
            continue

        mask_polylines_eroded = get_scaled_polylines(
            mask_polylines, scale=0.9, dimensions=gray.shape
        )
        mask_eroded = polylines_to_mask([mask_polylines_eroded], dimensions=gray.shape)
        mask_eroded_label = np.bitwise_and(mask_eroded, possible_label_mask)
        mask_eroded_edge = np.bitwise_and(mask_eroded, sobel_raw_mask)

        _ratio = 0.05
        mask_eroded_area = np.sum(mask_eroded)
        mask_eroded_edge_area = np.sum(mask_eroded_edge)
        if np.sum(mask_eroded_edge_area) < _ratio * mask_eroded_area:
            if debug:
                print(
                    f"V {title}-{i}: smoothness check passed, {np.sum(mask_eroded_edge_area)} vs. "
                    f"{_ratio * mask_eroded_area:.2f} ({mask_eroded_area} * {_ratio:.2f})"
                )
        else:
            # any smooth area within a label should be considered as label
            mask_edgezone_color_avg = max(
                np.percentile(gray[mask_eroded_edge], 50), gray[mask_eroded_edge].mean()
            )
            mask_smoothzone_color_avg = (
                min(
                    np.percentile(gray[mask_eroded_label], 50),
                    gray[mask_eroded_label].mean(),
                )
                - 5
            )
            if mask_edgezone_color_avg >= mask_smoothzone_color_avg:
                if debug:
                    print(
                        f"{title}-{i}: non-label color too bright, "
                        f"{mask_edgezone_color_avg:.2f} >= {mask_smoothzone_color_avg:.2f}"
                    )
                continue

            if debug:
                print(
                    f"V {title}-{i}: non-label color adequately dark, "
                    f"{mask_edgezone_color_avg:.2f} < {mask_smoothzone_color_avg:.2f}"
                )
                display_image(
                    {
                        "Original": image,
                        "Gray": gray,
                        "Total": np.where(mask_refined_filled, gray, 0),
                        "Smooth": np.where(mask_eroded_label, gray, 0),
                        "Edge": np.where(mask_eroded_edge, gray, 0),
                    },
                    title=f"Label smoothness {i}: {title or 'Label contour'}",
                    axis=1,
                )

        label_masks = np.bitwise_or(label_masks, mask_refined_filled)
        label_mask_polylines.append(mask_polylines.tolist())

    label_masks = np.bitwise_and(label_masks, vessel_mask)
    if debug:
        _gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _polylines = [np.array(poly).reshape(-1, 2) for poly in label_mask_polylines]
        cv2.polylines(_gray, _polylines, True, (0, 255, 0), 2)
        cv2.polylines(possible_label_region, _polylines, True, 127, 2)
        display_image(
            {
                title or "Original": image,
                "Gray": _gray,
                "Text mask": text_mask.astype(np.uint8) * 255,
                "Color mask": possible_label_mask.astype(np.uint8) * 255,
                "Label region": possible_label_region,
                "Label mask": np.where(label_masks, gray, 0),
            },
            title=title or "All label mask",
            axis=1,
        )

    return label_masks, label_mask_polylines


@no_modification_to_args
def get_interfaces(
    image: np.ndarray,
    over_exposure_threshold: int = 200,
    contours: list[np.ndarray] | None = None,
    vessel_mask: np.ndarray | None = None,
    bg_correction: bool = True,
    bg_tolerance: int | tuple[int, int, int] | str = 30,
    bg_resolution: int | float = 0.02,
    bg_sobel_ratio: float = 0.55,
    cap_ratio: float | tuple[float] | str = (0.1, 0.3),
    cap_target: int | tuple[int, int, int] | str = -1,
    cap_tolerance: int | tuple[int, int, int] | str = 40,
    vessel_sobel_threshold: int = 63,
    sobel_threshold: int = 24,
    sobel_xy_ratio: float = 0.75,
    dilation: bool = True,
    phase_resolution: float = 0.06,
    boundary_resolution: float | tuple[float] | str = (0.1, 0.15),
    label_correction: bool = True,
    label_low: int | tuple[int, int, int] | str = 150,
    label_high: int | tuple[int, int, int] | str = 220,
    label_check_gray: bool = False,
    interface_signal: float = 0.6,
    debug: bool = False,
    title: str | None = None,
) -> tuple[
    list[int], tuple[int, int, list[int] | int], bool, bool, list[list[int, int]]
]:
    """
    Find the liquid interfaces in an image. The image should be a capped transparent vessel with liquid inside.

    :param np.ndarray image:
        the image, grayscale or color (BGR)
    :param int over_exposure_threshold:
        the threshold for identifying over-exposed images.
        If the mean brightness of the image, e.g., in grayscale values, is greater than this value,
        the image will be considered over-exposed. Over-exposed images will be enhanced to improve the detection of interfaces.
        Check `interface_detection.utils.increase_image_contrast` for details.
        You can set a value larger than 255 to disable this feature.
    :param list[np.ndarray] | None contours:
        the contours of the vessel in the order of (top, bottom, left, right).
        If None, the contours will be calculated using the `get_vessel_contour` function.
    :param np.ndarray | None vessel_mask:
        the mask of the vessel in the image.
        If None, it will be calculated using the `get_vessel_mask` function.
    :param bool bg_correction:
        whether to compensate for the background. If True, background-induced horizontal edges/interfaces will be removed.
    :param int | tuple[int, int, int] | str bg_tolerance:
        the tolerance for performing background correction. See `get_bg_correction` for details.

        - If a single int, it will be used as the tolerance for all channels.
        - If a tuple of three ints, it will be used as the tolerance for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param int | float bg_resolution:
        the resolution of the background interfaces. This is used to determine when the background outside the vessel is creating horizontal markers,
        how far above or below should we check within the vessel to find the background interface. This is because the vessel and the liquid within
        the vessel will cause displacements and distortions in the background.

        - If `bg_resolution` is an integer, it will be used as the number of pixels to check above and below the horizontal marker.
        - If `bg_resolution` is a float, it will be converted to an integer by multiplying it with the height of the vessel.
    :param float bg_sobel_ratio:
        for background correction, if the vertical Sobel gradient (sobely) is smaller than this ratio of the vertical Sobel gradient (sobely)
        in the background, then it is assumed this is a background-induced gradient.

        This parameter should be between 0.5 and 1.0, and is proportional to the background correction intensity. Typically, if the image is
        in a noisy background, this value should be lower so that true interfaces are not removed.
        Under a relatively clean background, this value can be higher (e.g., 0.8) for more aggressive background correction.

        See `get_bg_correction` for details.
    :param float | tuple[float] | str cap_ratio:
        the min/max ratio of the cap height to the vessel height. See `get_cap_range` for details.

        - If a single float, it will be used as the ratio for both min and max.
        - If a tuple of two floats, it will be used as the min and max ratio.
        - If a string, it will be converted to a float or a tuple of two floats (comma separated).
            This is for compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str cap_target:
        the target color of the cap, either a single value (grayscale) or a tuple of values for each channel (BGR).
        See `get_cap_range` for details.

        - If a single int, it will be used as the target color for all channels.
        - If a tuple of three ints, it will be used as the target color for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str cap_tolerance:
        the tolerance for the target color of the cap, either a single value (inferred to `cap_target`) or a tuple of values for each channel (BGR).
        See `get_cap_range` for details.

        - If a single int, it will be used as the tolerance for all channels.
        - If a tuple of three ints, it will be used as the tolerance for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param int vessel_sobel_threshold:
        the threshold for Sobel gradient markers for outlining the vessel.
        Only pixels with a gradient larger than this will be considered as a part of the vessel boundary.
        Use smaller values if the vessel has similar color to the background, e.g., 31.
        For dark backgrounds where the vessel is distinct, use larger values, e.g., 63 (default).
    :param int sobel_threshold:
        threshold for Sobel gradient markers for interface detection.
        Only pixels with a gradient larger than this will be considered as a part of an interface.
    :param float sobel_xy_ratio:
        horizontal gradient (sobelx) should be less than this ratio of the vertical gradient (sobely) for a pixel to be considered as a part of a horizontal interface.
    :param bool dilation:
        whether to dilate the horizontal markers to make them more robust (could also enable noise to be considered as interfaces).
        In other words, if True, the horizontal markers will be dilated to include neighboring pixels.
    :param float phase_resolution:
        the resolution of the phase (phase boundaries). A phase must be at least this ratio of the vessel height (excl. cap) to be considered.
        This is used to filter out cases when a single phase boundary creates too many interfaces when it is seen from an angle.
        Small angles will be ignored since its span is too small (less than `phase_resolution`).
    :param float | tuple[float] | str boundary_resolution:
        the resolution of the boundary.
        Any interface within this ratio of the vessel height (excl. cap) to the cap bottom (top of vessel body) or the vessel bottom will be ignored.
        Could be a single value or a tuple of values for the top and bottom boundaries.

        - If a single float, it will be used as the ratio for both top and bottom.
        - If a tuple of two floats, the first value is used to constrain the top boundary and the second value is used to constrain the bottom boundary.
            For example, (0.1, 0.15) means the top 10% and the bottom 15% of the vessel (excluding the cap) will be ignored for interface detection.
        - If a string, it will be converted to a float or a tuple of two floats (comma separated).
            This is for compatibility with URL query parameters.
    :param bool label_correction:
        whether to find label and make correction. If True, the label mask will be used to correct the interface detection.
        See also `find_label_mask`.
    :param int | tuple[int, int, int] | str label_low:
        the lower bound of the label color, either a single value (grayscale) or a tuple of values for each channel (BGR).
        See `find_label_mask` for details.

        - If a single int, it will be used as the lower bound for all channels.
        - If a tuple of three ints, it will be used as the lower bound for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param int | tuple[int, int, int] | str label_high:
        the upper bound of the label color, either a single value (grayscale) or a tuple of values for each channel (BGR).
        See `find_label_mask` for details.

        - If a single int, it will be used as the upper bound for all channels.
        - If a tuple of three ints, it will be used as the upper bound for each channel (BGR).
        - If a string, it will be converted to an int or a tuple of three ints (comma separated).
            This is for compatibility with URL query parameters.
    :param bool label_check_gray:
        whether to check the color in the grayscale image. If True, the label color will be checked in the grayscale image.
        Check `find_label_mask` for details.
    :param float interface_signal:
        the interface intensity signal per row for a row to be considered as a part of an interface.
        If the percentage of the interface signal in a row is larger than this value, the row will be considered as a part of an interface.
        Should be between 0.5 and 0.9. A low value will cause more noise to be detected as interfaces.
        A high value can potentially miss some interfaces if the image is not `perfect`.
    :param bool debug:
        whether to show debug images. If True, the debug images will be shown in a cv2 window.
    :param str | None title:
        the title of the debug images to be shown in the cv2 window.

    :return list[int]:
        the detected *liquid* interfaces in the image. The list contains the y-coordinates of the interfaces.
        The y-coordinates are in the range of [0, image.shape[0]) and are sorted in ascending order.
    :return tuple[int, int, list[int] | int]:
        the cap bottom, the cap top, and the color of the cap. See the output of `get_cap_range` for details.
    :return bool:
        if the liquid within the vessel show a continuous phase (miscible).
        When more than one (N >= 2) interfaces are detected, immiscibility is assumed and a False is returned.
    :return bool:
        if further human attention is needed. If True, the image should be checked manually.
        This is when the number of interfaces is less than 1 (no interface detection) or more than 2 (liquid phase >= 2).
    :return list[list[int, int]]:
        the list of polyline points for each individual label mask. See the output of `find_label_mask` for details.

    :raises ValueError:
        if the image is empty or not in the correct format (grayscale or color).
    :raises TypeError:
        if the input image is not a numpy array.
    """
    if isinstance(boundary_resolution, str):  # compatibility with the API string input
        boundary_resolution = tuple(
            float(x) for x in boundary_resolution.strip(_BRACKETS).split(",")
        )
        boundary_resolution = (
            boundary_resolution[0]
            if len(boundary_resolution) == 1
            else boundary_resolution
        )

    image_resize_factor = 1
    # the max height of the images is 1080 pixels
    if image.shape[0] > 1080:
        image_resize_factor = image.shape[0] / 1080
        image = cv2.resize(
            image, (math.ceil(image.shape[1] / image_resize_factor), 1080)
        )

    gray = get_gray_image(image)
    if gray.size == 0:
        raise ValueError("The image is empty.")

    # increase image contrast, image could be grayscale or color
    enhanced_image = increase_image_contrast(
        image=image, thresh=over_exposure_threshold
    )
    enhanced_gray = get_gray_image(enhanced_image)

    extra_enhanced_image = increase_image_contrast(image=image, thresh=0)

    if contours is None:
        contours = get_vessel_contour(
            image,
            enhanced_image=extra_enhanced_image,
            where=["top", "bottom", "left", "right"],
            sobel_threshold=vessel_sobel_threshold,
        )

    top, bottom, left, right = contours
    vessel_bottom = math.ceil(np.percentile(bottom[:, 0], 50))
    Wvessel_rows = right[:, 1] - left[:, 1] + 1  # vessel width for each row
    _min_w = math.ceil(np.percentile(Wvessel_rows, 50))
    Wvessel_rows = np.maximum(Wvessel_rows, _min_w)
    Wvessel = math.ceil(np.percentile(Wvessel_rows, 50))

    if vessel_mask is None:
        vessel_mask = get_vessel_mask(
            image,
            enhanced_image=extra_enhanced_image,
            sobel_threshold=vessel_sobel_threshold,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )
    cap_top, cap_bottom, cap_color, cap_mask = get_cap_range(
        image,
        enhanced_image=extra_enhanced_image,
        mask=vessel_mask,
        cap_ratio=cap_ratio,
        target=cap_target,
        tolerance=cap_tolerance,
    )
    Hvessel = vessel_bottom - cap_bottom + 1

    # vertical gradient, horizontal edge
    sobely = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.minimum(np.absolute(sobely), 255).astype(np.uint8)
    # horizontal gradient, vertical edge
    sobelx = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.minimum(np.absolute(sobelx), 255).astype(np.uint8)
    # a horizontal edge should have a larger vertical gradient than horizontal gradient
    possible_interface_mask = sobelx <= sobely * sobel_xy_ratio
    hmarkers = np.bitwise_and(sobely >= sobel_threshold, possible_interface_mask)

    if dilation_kernel := dilation:
        # WARNING: dilation also enables noise to be considered as interfaces
        niter = 2
        dilation_kernel = np.ones((1, 3), np.uint8)
        hmarkers = hmarkers.astype(np.uint8)
        hmarkers = cv2.erode(hmarkers, dilation_kernel, iterations=niter)
        hmarkers = cv2.dilate(hmarkers, dilation_kernel, iterations=niter).astype(bool)

    if bg_correction:
        vessel_hmarkers, vessel_hmarkers_bg_mask = compensate_for_background(
            image,
            enhanced_image=enhanced_image,
            contours=contours,
            vessel_mask=vessel_mask,
            sobel_threshold=sobel_threshold,
            sobel_xy_ratio=sobel_xy_ratio,
            dilation=dilation_kernel,
            bg_resolution=bg_resolution,
            bg_sobel_ratio=bg_sobel_ratio,
            color_tolerance=bg_tolerance,
            debug=False,
        )
    else:
        # vessel horizontal markers after background correction
        vessel_hmarkers = np.bitwise_and(hmarkers, vessel_mask)
        vessel_hmarkers_bg_mask = np.zeros_like(gray, dtype=bool)

    if dilation:
        niter = max(2, math.ceil(gray.shape[1] / 200.0))
        vessel_hmarkers: np.ndarray = cv2.dilate(
            vessel_hmarkers.astype(np.uint8), dilation_kernel, iterations=niter
        ).astype(bool)
        vessel_hmarkers_bg_mask = np.bitwise_and(
            vessel_hmarkers_bg_mask, np.bitwise_not(vessel_hmarkers)
        )

    # compensate row_total_sum with label mask
    if label_correction:
        label_mask, label_mask_points = find_label_mask(
            image,
            enhanced_image=extra_enhanced_image,
            color_low=label_low,
            color_high=label_high,
            check_color_in_gray=label_check_gray,
            contours=contours,
            vessel_mask=vessel_mask,
            cap_bottom=cap_bottom,
            debug=False,
        )
        label_compensation_rate = (
            0.8
            * np.sum(label_mask, axis=1)
            / np.maximum(np.sum(vessel_mask, axis=1), 1)
        )
        label_compensation_rate = 1.0 / (1.0 - np.minimum(label_compensation_rate, 0.9))
    else:
        label_mask, label_mask_points = np.zeros_like(gray, dtype=bool), []
        label_compensation_rate = 1
    row_total_sum = (
        np.bitwise_and(vessel_hmarkers, np.bitwise_not(label_mask)).sum(axis=1)
        * label_compensation_rate
    )

    row_is_interface_ratio = row_total_sum / np.maximum(Wvessel_rows, 1)
    row_is_interface = row_is_interface_ratio >= np.clip(0.5, interface_signal, 0.9)

    boundary_resolution = (
        (boundary_resolution, boundary_resolution)
        if not isinstance(boundary_resolution, (tuple, list))
        else boundary_resolution
    )
    bd_res_top, bd_res_bottom = int(boundary_resolution[0] * Hvessel), max(
        int(boundary_resolution[1] * Hvessel), int(Wvessel * 0.3)
    )
    bd_res_top, bd_res_bottom = max(bd_res_top, 1), max(
        min(bd_res_bottom, Hvessel // 2), 1
    )  # at least 1 pixel
    interface_bound_top, interface_bound_bottom = (
        cap_bottom + bd_res_top,
        vessel_bottom - bd_res_bottom,
    )
    row_is_interface[:interface_bound_top] = False
    row_is_interface[interface_bound_bottom:] = False

    # get the row indices of the interfaces
    interface_rows = np.where(row_is_interface)[0]

    if len(interface_rows) == 0:
        interfaces, interfaces_expanded = [], []
    else:
        full_res = math.ceil(Hvessel * phase_resolution)
        # first, an initial guess/grouping of the interfaces
        res = math.ceil(Hvessel * min(0.0125, phase_resolution * 0.25))
        res_for_addition = np.array([-res, res]).reshape(1, -1)
        domains = interface_rows.reshape(-1, 1) + res_for_addition
        domains = union_of_domains(domains) - res_for_addition
        if debug:
            print(f"[scaled] Initial interface domains: {domains.tolist()}")

        # now we have some minimal groups of the interfaces
        # check if we can further merge the domains (reduce groups)
        new_domains = np.zeros((0, 2), dtype=int)
        for i in range(len(domains)):
            if i == 0:
                new_domains = domains[i : i + 1]
                continue
            if domains[i, 0] - new_domains[-1, 1] < res or (
                domains[i, 1] - new_domains[-1, 0] < full_res
            ):
                # check if the interface signal between the two domains is strong enough
                # in other words, the in-between space is likely a tilted interface
                # if not, we skip the domain merge
                _intermediate_ratios = row_is_interface_ratio[
                    new_domains[-1, 1] + 1 : domains[i, 0]
                ]
                if (
                    len(_intermediate_ratios) == 0
                    or np.percentile(_intermediate_ratios, 75) >= 0.25
                ):
                    new_domains[-1, 1] = domains[i, 1]
                    continue
            new_domains = np.vstack((new_domains, domains[i : i + 1]))
        domains = new_domains
        if debug:
            print(f"[scaled] Merged interface domains: {domains.tolist()}")

        domains[:, 0] = np.maximum(domains[:, 0], interface_bound_top)
        domains[:, 1] = np.minimum(domains[:, 1], interface_bound_bottom)
        interfaces = np.average(domains, axis=1).astype(int).tolist()
        interfaces_expanded = union_of_domains(
            np.array(interfaces),
            expansion=1,
            min_domain=0,
            max_domain=gray.shape[0] - 1,
            return_flattened=True,
        )

    # return results: whether the two phases are miscible, whether further human attention is needed
    n_interfaces = len(interfaces)
    if n_interfaces == 0:
        warnings.warn(
            f"Are you trying to fool me with an empty vial? No interfaces found in {title or 'the image'}."
        )
        results = [True, True]
    elif n_interfaces == 1:
        if interfaces[0] > vessel_bottom * 0.5 + cap_bottom * 0.5:
            warnings.warn(
                f"Only one interface found in {title or 'the image'}. This should be the air-liquid interface. "
                f"Did you fill the vial to its half?"
            )
            results = [True, True]
        results = [True, False]
    elif n_interfaces == 2:
        if (
            interfaces[1] - interfaces[0] < Hvessel * phase_resolution * 2
            and interfaces[1] < cap_bottom + Hvessel * phase_resolution * 2
        ):
            warnings.warn(
                f"Two interfaces were found but they are too close to the top and to each other in {title or 'the image'}. "
                f"Are the both air-liquid interfaces?"
            )
            results = [False, True]
        results = [False, False]
    else:  # if more than three interfaces are found, it is highly likely that two phases are not miscible
        results = [False, False]

    if debug:  # display the results for debugging
        print(f"[scaled] Interface rows: {interface_rows}")
        print(f"[scaled] All interfaces: {[cap_top, cap_bottom]} + {interfaces}")
        cap_liners = union_of_domains(
            np.array([cap_top, cap_bottom]),
            expansion=1,
            min_domain=0,
            max_domain=gray.shape[0] - 1,
            return_flattened=True,
        )
        image_with_marked_cap_bottom = display_image(
            [image], [(cap_liners, (0, 255, 0))], noshow=True, return_image=True
        )

        # the original copy with no background correction
        bg_correction_img = np.where(hmarkers.reshape(*gray.shape, 1), 63, 0)
        bg_correction_img = np.where(
            vessel_hmarkers.reshape(*gray.shape, 1), 255, bg_correction_img
        )
        bg_correction_img = np.where(
            vessel_hmarkers_bg_mask.reshape(*gray.shape, 1),
            (255, 0, 0),
            bg_correction_img,
        ).astype(np.uint8)

        _polylines = [np.array(poly).reshape(-1, 2) for poly in label_mask_points]
        cv2.polylines(bg_correction_img, _polylines, True, (0, 255, 0), 2)
        cv2.polylines(image_with_marked_cap_bottom, _polylines, True, (0, 255, 0), 2)

        miscible_or_not = "Miscible" if results[1] else "NOT miscible"
        display_image(
            {
                title or "Original": image.copy(),
                "Enhanced": enhanced_image.copy(),
                "hmarkers": hmarkers.astype(np.uint8) * 255,
                "SobelY w. bg correction": bg_correction_img,
                # "Label mask": np.where(label_mask, gray, 0),
                "All interfaces": image.copy(),
                miscible_or_not: image_with_marked_cap_bottom,
            },
            line_markers={
                "All interfaces": (interface_rows, (0, 0, 255)),
                miscible_or_not: (interfaces_expanded, (0, 0, 255)),
            },
            title=None,
            axis=1,
        )

    if image_resize_factor > 1:
        interfaces = (np.array(interfaces) * image_resize_factor).astype(int).tolist()
        cap_top = int(cap_top * image_resize_factor)
        cap_bottom = int(cap_bottom * image_resize_factor)
        label_mask_points = [
            (np.array(polylines) * image_resize_factor).astype(int).tolist()
            for polylines in label_mask_points
        ]

    if debug:  # display the results for debugging
        interface_rows = np.array(interface_rows) * image_resize_factor
        interface_rows = interface_rows.astype(int).tolist()
        print(f"[original] Interface rows: {interface_rows}")
        print(f"[original] All interfaces: {[cap_top, cap_bottom]} + {interfaces}")

    return interfaces, (cap_top, cap_bottom, cap_color), *results, label_mask_points
