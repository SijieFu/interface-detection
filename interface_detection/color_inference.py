"""
Infer color of requested object in the image
"""

from typing import Iterable

import cv2
import numpy as np

from .utils import find_longest_in_sequence
from .utils import increase_image_contrast
from .utils import display_image, no_modification_to_args
from .utils import get_gray_image


@no_modification_to_args
def infer_cap_color(
    image: np.ndarray,
    enhanced_image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    sobel_threshold: int = 63,
    cap_ratio: float | tuple[float] = (0.1, 0.3),
    color_tolerance: int | tuple[int, int, int] = 40,
    debug: bool = False,
    title: str | None = None,
) -> tuple[tuple[int] | int, int, int, np.ndarray]:
    """
    Infer the color of the cap of the vial in the image. Theoretically, this function can also be used to roughly infer the top and bottom of the cap.

    :param np.ndarray image:
        The image of the vial, either in BGR or grayscale.
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param np.ndarray | None mask:
        The mask of the vial, if available; if not, the whole image will be used. This parameter is used to refine vial height estimation.
    :param int sobel_threshold:
        The threshold for the Sobel gradient to identify edges. This is used to identify the cap region.
    :param float | tuple[float] cap_ratio:
        The ratio of the cap height to the total height of the vial. If a tuple, the first element is the lower bound and the second element is the upper bound.
    :param int | tuple[int, int, int] color_tolerance:
        experimental, for caps with non-uniform colors, the tolerance for color difference for cap detection/masking.

        - if int, the tolerance for all channels will be the same;
        - if tuple, the tolerance for each channel will be different.
    :param bool debug:
        whether to show debug images
    :param str | None title:
        the title of the debug image to show in the debug window.
    :return tuple[int] | int:
        The color of the cap in BGR format or grayscale depending on the input image.
    :return int:
        The row index of the upper bound of the cap (cap top should be above this row).
    :return int:
        The row index of the lower bound of the cap (cap bottom should be below this row).
    :return np.ndarray:
        The 2D mask of the cap in the image, a 2D binary mask.
    """
    if mask is None:
        mask_2d = np.ones(image.shape[:2], dtype=bool)
        vessel_top = 0
        H = image.shape[0]
    else:
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        vessel_top = np.percentile(np.argmax(mask_2d, axis=0), 50).astype(int)

        # get the average of values between 30 and 70 percentiles
        heights = np.sum(mask_2d, axis=0)
        p1, p2 = np.percentile(heights, [30, 70])
        p1, p2 = int(p1), int(p2)
        # estimate height of the vessel
        H = np.mean(heights[np.bitwise_and(heights >= p1, heights <= p2)])

    cap_ratio = (
        (cap_ratio / 2.0, cap_ratio) if isinstance(cap_ratio, float) else cap_ratio
    )
    # the estimated min and max height of the cap
    Hcap_min, Hcap_max = int(H * cap_ratio[0]), int(H * cap_ratio[1])

    # detect vessel top
    if enhanced_image is None:
        gray = get_gray_image(image)
    else:
        gray = get_gray_image(enhanced_image)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.minimum(np.abs(sobely), 255).astype(np.uint8)
    sobely = np.where(mask_2d, sobely, 0)
    sobely = sobely > sobel_threshold
    kernel = np.ones((3, 3), dtype=np.uint8)
    sobely = cv2.dilate(sobely.astype(np.uint8), kernel, iterations=2).astype(bool)
    sobely = np.where(mask_2d, sobely, False)
    row_can_be_cap_by_sobel = np.sum(sobely, axis=1) >= sobely.shape[1] // 4

    # find the longest zero sequence - this identifies the body of the cap
    _row_upper, _row_lower = find_longest_in_sequence(
        row_can_be_cap_by_sobel[
            vessel_top
            + Hcap_min // 2 : vessel_top
            + min((Hcap_min + Hcap_max) // 2, 2 * Hcap_min)
        ],
        value=0,
    )
    if _row_upper == -1 and _row_lower == -1:
        cap_row_upper, cap_row_lower = (vessel_top, vessel_top + Hcap_min)
    else:
        cap_row_upper, cap_row_lower = vessel_top + _row_upper, vessel_top + _row_lower

    ndims = image.ndim
    _move_left = int(0.15 * gray.shape[1])
    cropped_cap = image[cap_row_upper : cap_row_lower + 1, _move_left:-_move_left]
    cap_color = np.percentile(cropped_cap, 50, axis=(0, 1)).astype(int)

    if ndims == 3:
        if not isinstance(color_tolerance, Iterable):
            color_tolerance = np.repeat(color_tolerance, 3)
        color_tolerance = np.array(color_tolerance).reshape(1, 1, 3)
    elif ndims == 2:
        if isinstance(color_tolerance, Iterable):
            raise ValueError(
                "color and color_tolerance must be a single value for grayscale images"
            )
    else:
        raise ValueError(
            "Dimension mismatch: `image` and `color_tolerance` should be compatible"
        )

    if ndims == 3:
        cap_mask = np.all(
            np.abs(image - cap_color.reshape(1, 1, 3)) <= color_tolerance, axis=-1
        )
        cap_color = tuple(cap_color)
    else:
        cap_mask = np.abs(image - cap_color) <= color_tolerance
        cap_color = int(cap_color)

    cap_mask[: vessel_top + 1] = False
    cap_mask[cap_row_lower + 1 :] = False

    if debug:
        bg_color = 0 if np.mean(cap_color) > 127 else 255
        display_image(
            {
                "Original": image,
                f"Color: {cap_color}": np.where(cap_mask, image, bg_color),
            },
            line_markers={
                f"Color: {cap_color}": ([cap_row_upper, cap_row_lower], (0, 0, 255))
            },
            title=title,
        )

    return cap_color, cap_row_upper, cap_row_lower, cap_mask


def _get_mask_within_tolerence(
    image: np.ndarray,
    color: int | tuple[int, int, int] | np.ndarray,
    color_tolerance: int | tuple[int, int, int] | np.ndarray,
) -> np.ndarray:
    """
    Get a mask for a BGR or grayscale image based on the color and color tolerance.

    :param np.ndarray image:
        The input image in BGR format. Must be a 3D array in the shape of (H, W, 3) or a 2D array for grayscale.
    :param int | tuple[int, int, int] | np.ndarray color:
        The target color in BGR format (for 3D images) or grayscale (for 2D images).
        If a single integer is provided, it is assumed to be the grayscale value.
    :param int | tuple[int, int, int] | np.ndarray color_tolerance:
        The tolerance for the color difference. If a single integer is provided, it is assumed to be the grayscale tolerance.
        If a tuple is provided, it should match the number of channels in the image (3 for BGR, 1 for grayscale).
    :return np.ndarray:
        A 2D boolean mask where True indicates pixels that match the color within the tolerance.
    """
    color = np.array(color).astype(int).flatten()
    color = np.clip(color, 0, 255)
    color_tolerance = np.array(color_tolerance).astype(int).flatten()
    color_tolerance = np.clip(color_tolerance, 0, 255)

    if image.ndim == 3:
        if len(color) == 1:
            color = np.repeat(color, 3)
        color = np.array(color).reshape(1, 1, 3)
        if len(color_tolerance) == 1:
            color_tolerance = np.repeat(color_tolerance, 3)
        color_tolerance = np.array(color_tolerance).reshape(1, 1, 3)
        cap_mask = np.all(np.abs(image - color) <= color_tolerance, axis=-1)
    elif image.ndim == 2:
        if len(color) > 1 or len(color_tolerance) > 1:
            raise ValueError(
                "`color` and `color_tolerance` must be a single value for grayscale images"
            )
        cap_mask = np.abs(image - color[0]) <= color_tolerance[0]
    else:
        raise ValueError("image must be either grayscale (2D array) or BGR (3D array)")

    return cap_mask


@no_modification_to_args
def get_cap_mask_by_color(
    image: np.ndarray,
    color: int | tuple[int, int, int],
    enhanced_image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    sobel_threshold: int = 63,
    cap_ratio: float | tuple[float] = (0.1, 0.3),
    color_tolerance: int | tuple[int, int, int] = 40,
    debug: bool = False,
    title: str | None = None,
) -> np.ndarray:
    """
    Get the mask of the cap in the image by color. Horizontal gradients are used
    to identify cap boundaries and these boundaries are excluded from the mask.

    :param np.ndarray image:
        The image of the vial/vessel, either in BGR or grayscale.
    :param int | tuple[int, int, int] color:
        The color of the cap in BGR format or grayscale depending on the input image.
        Check `infer_cap_color` for more details.
    :param np.ndarray | None enhanced_image:
        if provided as a numpy array, it will be used as the enhanced image to calculate the gradient
        and identify edges. Same as `image`, it can be grayscale or color (BGR), if provided.
        If None, the original image will be used.
    :param np.ndarray | None mask:
        The mask of the vial, if available; if not, the whole image will be used.
        This parameter is used to refine the search for the cap.
    :param int sobel_threshold:
        The threshold for the Sobel gradient to identify edges. This is used to identify the cap boundaries.
        Cap boundaries are excluded from the final cap mask.
    :param float | tuple[float] cap_ratio:
        The ratio of the cap height to the total height of the vial.
        If a tuple, the first element is the lower bound and the second element is the upper bound. In other words,
        the bottom of the cap should lie between the lower and upper bounds of the cap height.
    :param int | tuple[int, int, int] color_tolerance:
        experimental, for caps with non-uniform colors, the tolerance for color difference for cap detection/masking.

        - if int, the tolerance for all channels will be the same;
        - if tuple, the tolerance for each channel will be different.
    :param bool debug:
        whether to show debug images
    :param str | None title:
        the title of the debug image to show in the debug window.
    :return np.ndarray:
        The 2D mask of the cap in the image, a 2D binary mask.
    """
    if mask is None:
        mask_2d = np.ones(image.shape[:2], dtype=bool)
        vessel_top = 0
        H = image.shape[0]
    else:
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        vessel_top = np.percentile(np.argmax(mask_2d, axis=0), 50).astype(int)

        # get the average of values between 30 and 70 percentiles
        heights = np.sum(mask_2d, axis=0)
        p1, p2 = np.percentile(heights, [30, 70])
        p1, p2 = int(p1), int(p2)
        # estimate height of the vessel
        H = np.mean(heights[np.bitwise_and(heights >= p1, heights <= p2)])

    cap_ratio = (
        (cap_ratio / 2.0, cap_ratio) if isinstance(cap_ratio, float) else cap_ratio
    )
    # the estimated min and max height of the cap
    Hcap_min, Hcap_max = int(H * cap_ratio[0]), int(H * cap_ratio[1])

    cap_mask = _get_mask_within_tolerence(
        image=image, color=color, color_tolerance=color_tolerance
    )
    _image = increase_image_contrast(image=image, thresh=0)
    _color = np.median(_image[cap_mask], axis=0).astype(int)
    _cap_mask = _get_mask_within_tolerence(
        image=_image, color=_color, color_tolerance=color_tolerance
    )
    cap_mask = np.bitwise_or(cap_mask, _cap_mask)

    # detect vessel top
    if enhanced_image is None:
        gray = get_gray_image(image)
    else:
        gray = get_gray_image(enhanced_image)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.minimum(np.abs(sobely), 255).astype(np.uint8)
    sobely = np.where(mask_2d, sobely, 0)
    sobely = sobely > sobel_threshold
    kernel = np.ones(3, np.uint8)
    sobely = cv2.erode(sobely.astype(np.uint8), kernel, iterations=2)
    sobely = cv2.dilate(
        sobely, kernel, iterations=min(5, 2 + gray.shape[1] // 200)
    ).astype(bool)
    sobely = np.where(mask_2d, sobely, False)
    sobely[: vessel_top + Hcap_min + 1] = False
    cap_mask = np.where(sobely, False, cap_mask)

    cap_mask = np.where(mask_2d, cap_mask, False)
    cap_mask[: vessel_top + 1] = False
    cap_mask[vessel_top + Hcap_max :] = False

    if debug:
        bg_color = 0 if np.mean(color) > 127 else 255
        display_image(
            {
                "Original": image,
                f"Color: {color}": np.where(cap_mask, image, bg_color),
            },
            title=title,
        )

    return cap_mask
