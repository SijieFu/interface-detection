"""
Transforms a mask to 4 mask's contours: top, bottom, left, right
"""

import math
from typing import Iterable

import numpy as np
from scipy.stats import linregress
import cv2

from .utils import display_image, no_modification_to_args
from .utils import find_intersection_point, find_longest_in_sequence

_BRACKETS = r"()[]{}"


@no_modification_to_args
def get_trimmed_mask_bbox(
    mask: np.ndarray, trim: bool = True, debug: bool = False, title: str | None = None
) -> tuple[int, int, int, int]:
    """
    Trim the mask to remove the empty space and weird outlier shapes around the mask and return the bounding box

    :param np.ndarray mask:
        the 2D mask of type bool to trim
    :param bool trim:
        whether or not to trim the mask
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image
    :return int:
        the x1 coordinate of the bounding box
    :return int:
        the y1 coordinate of the bounding box
    :return int:
        the x2 coordinate of the bounding box
    :return int:
        the y2 coordinate of the bounding box
    :return np.ndarray:
        the trimmed/cropped mask within the bounding box
    """
    mask = mask[:, :, 0].astype(bool) if mask.ndim == 3 else mask.astype(bool)
    if not mask.ndim == 2:
        raise ValueError("Invalid mask shape, expected 2D mask")
    if not np.any(mask):
        raise ValueError("Invalid mask, the mask is empty or all False")

    # get the bbox of the central part of the mask
    vmask, hmask = np.any(mask, axis=0), np.any(mask, axis=1)
    x1, x2 = np.argmax(vmask), mask.shape[1] - np.argmax(np.flip(vmask)) - 1
    y1, y2 = np.argmax(hmask), mask.shape[0] - np.argmax(np.flip(hmask)) - 1

    if not trim:  # return the bbox without trimming
        return x1, y1, x2, y2, mask[y1 : y2 + 1, x1 : x2 + 1]

    mask_cropped = mask[y1 : y2 + 1, x1 : x2 + 1]
    for _ in range(3):
        nrows, ncols = mask_cropped.shape

        # vertical signals, whether to trim the left or right part of the mask
        mask_cropped_vsignal = np.sum(mask_cropped, axis=0).astype(np.int32)
        vsignal_high, vsignal_low = np.percentile(mask_cropped_vsignal, [70, 10])
        vsignal_cutoff = min(int(0.2 * vsignal_high), int(2 * vsignal_low))
        vaverage_1 = np.cumsum(mask_cropped_vsignal) / np.arange(1, ncols + 1)
        vaverage_2 = (np.cumsum(mask_cropped_vsignal[::-1]) / np.arange(1, ncols + 1))[
            ::-1
        ]
        cols_with_high_signals = np.minimum(vaverage_1, vaverage_2) >= vsignal_cutoff
        vstart, vend = find_longest_in_sequence(cols_with_high_signals, value=0)
        cols_min, cols_max = 0, ncols - 1
        _x = np.arange(nrows)
        if (
            vstart >= 0
            and vstart <= ncols * 0.1
            and np.mean(cols_with_high_signals[: vend + 1]) < 0.4
        ):
            # vend bounce a bit further left?
            vsignal_crop_high = np.percentile(
                mask_cropped_vsignal[vstart : vend + 1], 85
            ).astype(int)
            move_left = np.argmax(
                mask_cropped_vsignal[vstart : vend + 1][::-1] < vsignal_crop_high
            )
            vend -= move_left
            # is trimming the left part worth it? i.e. making `cols_min = vend`
            _y0 = np.argmax(mask_cropped, axis=1)
            _y1 = np.maximum(0, _y0 - vend)
            if (
                np.abs(np.corrcoef(_x, _y1)[0, 1]) - np.abs(np.corrcoef(_x, _y0)[0, 1])
                >= 0.01
            ):
                cols_min = vend
        elif (
            vend >= 0
            and vend >= ncols * 0.9
            and np.mean(cols_with_high_signals[vstart:]) < 0.4
        ):
            # vstart bounce a bit further right?
            vsignal_crop_high = np.percentile(
                mask_cropped_vsignal[vstart : vend + 1], 85
            ).astype(int)
            move_right = np.argmax(
                mask_cropped_vsignal[vstart : vend + 1] < vsignal_crop_high
            )
            vstart += move_right
            # is trimming the right part worth it? i.e. making `cols_max = vstart`
            _y0 = np.argmax(np.flip(mask_cropped, axis=1), axis=1)
            _y1 = np.maximum(0, _y0 - (ncols - 1 - vstart))
            if (
                np.abs(np.corrcoef(_x, _y1)[0, 1]) - np.abs(np.corrcoef(_x, _y0)[0, 1])
                >= 0.01
            ):
                cols_max = vstart

        # horizontal signals, whether to trim the top or bottom part of the mask
        mask_cropped_hsignal = np.sum(mask_cropped, axis=1).astype(np.int32)
        hsignal_high, hsignal_low = np.percentile(mask_cropped_hsignal, [70, 10])
        hsignal_cutoff = min(int(0.2 * hsignal_high), int(2 * hsignal_low))
        haverage_1 = np.cumsum(mask_cropped_hsignal) / np.arange(1, nrows + 1)
        haverage_2 = (np.cumsum(mask_cropped_hsignal[::-1]) / np.arange(1, nrows + 1))[
            ::-1
        ]
        rows_with_high_signals = np.minimum(haverage_1, haverage_2) >= hsignal_cutoff
        hstart, hend = find_longest_in_sequence(rows_with_high_signals, value=0)
        rows_min, rows_max = 0, nrows - 1
        _x = np.arange(ncols)
        if (
            hstart >= 0
            and hstart <= nrows * 0.1
            and np.mean(rows_with_high_signals[: hend + 1]) < 0.4
        ):
            # hend bounce a bit further up?
            hsignal_crop_high = np.percentile(
                mask_cropped_hsignal[hstart : hend + 1], 85
            ).astype(int)
            move_up = np.argmax(
                mask_cropped_hsignal[hstart : hend + 1][::-1] < hsignal_crop_high
            )
            hend -= move_up
            # is trimming the top part worth it? i.e. making `rows_min = hend`
            _y0 = np.argmax(mask_cropped, axis=0)
            _y1 = np.maximum(0, _y0 - hend)
            if (
                np.abs(np.corrcoef(_x, _y1)[0, 1]) - np.abs(np.corrcoef(_x, _y0)[0, 1])
                >= 0.01
            ):
                rows_min = hend
        elif (
            hend >= 0
            and hend >= nrows * 0.9
            and np.mean(rows_with_high_signals[hstart:]) < 0.4
        ):
            # hstart bounce a bit further down?
            hsignal_crop_high = np.percentile(
                mask_cropped_hsignal[hstart : hend + 1], 85
            ).astype(int)
            move_down = np.argmax(
                mask_cropped_hsignal[hstart : hend + 1] < hsignal_crop_high
            )
            hstart += move_down
            # is trimming the bottom part worth it? i.e. making `rows_max = hstart`
            _y0 = np.argmax(np.flip(mask_cropped, axis=0), axis=0)
            _y1 = np.maximum(0, _y0 - (nrows - 1 - hstart))
            if (
                np.abs(np.corrcoef(_x, _y1)[0, 1]) - np.abs(np.corrcoef(_x, _y0)[0, 1])
                >= 0.01
            ):
                rows_max = hstart

        mask_cropped = mask_cropped[rows_min : rows_max + 1, cols_min : cols_max + 1]
        x1, x2 = x1 + cols_min, x1 + cols_max
        y1, y2 = y1 + rows_min, y1 + rows_max
        if rows_min == 0 and rows_max == nrows - 1 and cols_min == 0 and ncols - 1:
            break

    if debug:
        mask_copy = mask.copy().astype(np.uint8) * 255
        mask_copy[y1 : y2 + 1, x1 : x2 + 1] = 127
        display_image(mask_copy, title=f"Trimmed mask: {title or 'trim_mask'}")

    return x1, y1, x2, y2, mask_cropped


@no_modification_to_args
def mask_to_contours(
    mask: np.ndarray,
    where: str | Iterable[str] | None = None,
    percentiles: tuple[int] | str | None = None,
    debug: bool = False,
    title: str | None = None,
) -> list[np.ndarray] | np.ndarray:
    """
    Transform a mask to 4 mask contours: top, bottom, left, right

    :param np.ndarray mask:
        the mask to transform
    :param str | Iterable[str] | None where:
        where to transform the mask to contours; can be "top", "bottom", "left", "right" or a list of them
    :param tuple[int] | str | None percentiles:
        the percentiles to use for the transformation. For example, for a top contour, the y-coordinates of the contour will be
        identified. But some noise may be present in the mask. The percentiles will be used to filter the outlier coordinates outside the
        defined percentiles. The y-coordinates within the percentiles will be used to find the slope and intercept of the line.
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image
    :return np.ndarray | list[np.ndarray]:
        contours of the mask in the order of top, bottom, left, right, if `where` is None
        or in the order of the locations in `where` if `where` is a list
        or a single contour if `where` only contains one location
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
            percentiles = [
                percentiles[i : i + 2] for i in range(0, len(percentiles), 2)
            ]

    if percentiles is None:
        percentiles = [(0, 30)] * len(locations)
    else:
        assert len(percentiles) == len(locations) and all(
            len(p) == 2 for p in percentiles
        ), "Invalid value for `percentiles`"

    mask = mask[:, :, 0].astype(bool) if mask.ndim == 3 else mask.astype(bool)
    x1, y1, x2, y2, mask_cropped = get_trimmed_mask_bbox(mask, trim=True)

    contours = []
    for loc, ps in zip(locations, percentiles):
        # make transformations --> find left contour of the transformed mask
        require_transpose = True if loc.lower() in ["top", "bottom"] else False
        require_flip = True if loc.lower() in ["bottom", "right"] else False

        tmask = mask_cropped.copy()  # temporary mask
        tmask = tmask.T if require_transpose else tmask
        tmask = np.flip(tmask, axis=1) if require_flip else tmask
        tmask_nrows, tmask_ncols = tmask.shape

        contour_left = np.minimum(np.argmax(tmask, axis=1), tmask_nrows - 1)
        _x1, _x2 = np.percentile(contour_left, ps)
        _x1, _x2 = math.floor(_x1), math.ceil(_x2)
        contour_left = np.hstack(
            [np.arange(tmask_nrows).reshape(-1, 1), contour_left.reshape(-1, 1)]
        )
        contour_left = contour_left[
            np.bitwise_and(contour_left[:, 1] >= _x1, contour_left[:, 1] <= _x2)
        ]
        if len(contour_left) == 0:
            raise ValueError(f"Invalid mask for {loc} transformation")
        elif len(contour_left) == 1:
            slope, intercept = 0, contour_left[0, 1]
        else:
            slope, intercept, _, _, _ = linregress(*contour_left.T)

        xdim_original, xanchor = (
            (mask.shape[1], x1) if require_transpose else (mask.shape[0], y1)
        )
        row_indices = np.arange(xdim_original).reshape(-1, 1) - xanchor
        col_indices = row_indices * slope + intercept
        contour_left = np.hstack([row_indices, col_indices])

        # return the original indices
        contour_left[:, 1] = (
            tmask_ncols - 1 - contour_left[:, 1] if require_flip else contour_left[:, 1]
        )
        contour_left = (
            np.flip(contour_left, axis=1) if require_transpose else contour_left
        )
        contour_left[:, 0] += y1
        contour_left[:, 1] += x1

        contour_left = np.maximum(contour_left, 0)
        contour_left = np.minimum(contour_left, np.array(mask.shape).reshape(1, -1) - 1)
        contours.append(contour_left.astype(np.int32))

    if debug:
        mask_with_contours = mask.copy().astype(np.uint8) * 255
        for contour in contours:
            mask_with_contours[contour[:, 0], contour[:, 1]] = 255
        # display_image(mask_with_contours, title=title)
        mask_to_polylines(contours=contours, mask=mask, debug=debug, title=title)

    return contours[0] if isinstance(where, str) else contours


@no_modification_to_args
def mask_to_polylines(
    mask: np.ndarray | None = None,
    epsilon_factor: float = 0.02,
    contours: list[np.ndarray] | None = None,
    original_mask: np.ndarray | None = None,
    percentiles: tuple[int] | str | None = None,
    convex_hull: bool = False,
    debug: bool = False,
    title: str | None = None,
) -> list[list[int]]:
    """
    Find polylines to form a polygon as the best approximation of the mask's contours

    :param np.ndarray | None mask:
        the mask to transform (typically a trimmed mask from `get_trimmed_mask_bbox` function)
    :param float epsilon_factor:
        the factor to use for the epsilon in the `cv2.approxPolyDP` function
    :param list[np.ndarray] | None contours:
        the mask's contours from `mask_to_contours` function with `top`, `bottom`, `left`, `right` locations
    :param np.ndarray | None original_mask:
        the original mask ( must be provided if `contours` is None)
    :param tuple[int] | str | None percentiles:
        the percentiles to use for the transformation. Must be provided if `contours` is None.
        Check `mask_to_contours` function for more details.
    :param bool convex_hull:
        whether to use the convex hull of the mask's contours
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image to show in debug window
    :return list[list[int]]:
        the list of polylines in the order of top, bottom, left, right.
        For each set of polylines, the points are in the order of (x, y) coordinates
    """
    if mask is None:
        if contours is None:
            assert (
                original_mask is not None
            ), "Invalid input: You must provide either `mask` or one of `contours` and `original_mask`"
            contours = mask_to_contours(
                original_mask,
                where=["top", "bottom", "left", "right"],
                percentiles=percentiles,
            )
        mask = contours_to_mask(*contours, mask=original_mask).astype(np.uint8) * 255
    else:
        mask = mask[:, :, 0].astype(bool) if mask.ndim == 3 else mask.astype(bool)
        mask = mask.astype(np.uint8) * 255

    # New method: use cv2 to find the polygons
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    if convex_hull:
        max_contour = cv2.convexHull(max_contour)
    epsilon = epsilon_factor * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    if debug:
        cv2.polylines(mask, [approx], isClosed=True, color=127, thickness=2)
        display_image(mask, title=f"Polygon approx.: {title}")

    return approx.reshape(-1, 2).astype(int).tolist()


@no_modification_to_args
def polylines_to_mask(
    polylines: list[np.ndarray | list[list[int]]] | np.ndarray | list[list[int]],
    dimensions: tuple[int, int] | None = None,
    debug: bool = False,
    title: str | None = None,
) -> np.ndarray:
    """
    Transform polylines to a mask

    :param list[np.ndarray | list[list[int]]]| np.ndarray | list[list[int]] polylines:
        the list of polylines to transform (the output of `mask_to_polylines` returns one polyline).
        Check the argument `pts` in `cv2.fillPoly` for more details.
        You can also pass a single polyline as a numpy array or a list of (x, y) coordinates,
        but this is not recommended (you should pass a list of polylines instead).
    :param tuple[int, int] | None dimensions:
        the dimensions of the mask to create, if None, the dimensions will be the max of the polylines
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image to show in debug window
    :return np.ndarray:
        the transformed mask (filled boolean mask)
    """
    if dimensions is None:
        dimensions = (np.max(polylines, axis=0) + 1).astype(int)
    mask = np.zeros(dimensions)

    if isinstance(polylines, np.ndarray) or (
        isinstance(polylines, list) and not isinstance(polylines[0][0], Iterable)
    ):
        polylines = [polylines]

    for poly in polylines:
        cv2.fillPoly(mask, [np.array(poly).reshape(-1, 1, 2)], 1)
    mask = mask.astype(bool)

    if debug:
        display_image(
            mask.astype(np.uint8) * 255,
            title=f"Mask from polylines: {title or 'polylines_to_mask'}",
        )

    return mask


@no_modification_to_args
def contours_to_mask(
    top: np.ndarray | None = None,
    bottom: np.ndarray | None = None,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    debug: bool = False,
    title: str | None = None,
) -> np.ndarray:
    """
    Transform 4 mask contours of top, bottom, left, right to a mask

    :param np.ndarray | None top:
        the top contour. If None, the top contour will be ignored.
    :param np.ndarray | None bottom:
        the bottom contour. If None, the bottom contour will be ignored.
    :param np.ndarray | None left:
        the left contour. If None, the left contour will be ignored.
    :param np.ndarray | None right:
        the right contour. If None, the right contour will be ignored.
    :param np.ndarray | None mask:
        the mask to transform. If None, the mask will be created from the contours
        and its dimensions will be inferred from the contours (min/max x/y coordinates).
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image to show in debug window
    :return np.ndarray:
        the transformed mask (filled boolean mask)
    """
    if mask is None:
        assert (top is not None or bottom is not None) and (
            left is not None or right is not None
        ), "Invalid input"
        ncols = top[-1, 1] + 1 if top is not None else bottom[-1, 1] + 1
        nrows = left[-1, 0] + 1 if left is not None else right[-1, 0] + 1
        crop_mask = np.ones((nrows, ncols), dtype=bool)
    else:
        crop_mask = mask[:, :, 0].astype(bool) if mask.ndim == 3 else mask.astype(bool)

    contours = {"top": top, "bottom": bottom, "left": left, "right": right}
    for loc, contour in contours.items():
        if contour is None:
            continue
        assert (
            contour.ndim == 2 and contour.shape[1] == 2
        ), f"Invalid contour shape for {loc} contour: {contour.shape}"
        xdim, ydim = crop_mask.shape

        requires_transpose = True if loc in ["top", "bottom"] else False
        requires_flip = True if loc in ["bottom", "right"] else False
        if requires_transpose:  # only transpose the last two dimensions
            contour = np.flip(contour, axis=1)  # exchange x and y indices
            xdim, ydim = ydim, xdim
        if requires_flip:
            contour[:, 1] = ydim - 1 - contour[:, 1]

        mask = np.tile(np.arange(ydim).reshape(1, -1), (xdim, 1))
        mask = mask > contour[:, 1].reshape(-1, 1)

        # transform back to the original shape
        if requires_flip:
            mask = np.flip(mask, axis=1)
            contour[:, 1] = ydim - 1 - contour[:, 1]
        if requires_transpose:
            mask = mask.T
            contour = np.flip(contour, axis=1)

        crop_mask = np.bitwise_and(crop_mask, mask)

    if debug:
        mask_copy = crop_mask.astype(np.uint8) * 255
        for contour in contours.values():
            if contour is not None:
                mask_copy[contour[:, 0], contour[:, 1]] = 127
        display_image(
            mask_copy, title=f"Mask from contours: {title or 'contours_to_mask'}"
        )

    if isinstance(mask, np.ndarray) and mask.ndim == 3:
        crop_mask = np.stack([crop_mask] * mask.shape[-1], axis=-1)
    return crop_mask


@no_modification_to_args
def get_scaled_polylines(
    polylines: np.ndarray | list[list[int]],
    scale: float,
    dimensions: tuple[int, int] | None = None,
    debug: bool = False,
    title: str | None = None,
) -> np.ndarray:
    """
    Scale a contour or a set of polylines. Here, scale means enlarge or shrink the contour/polylines from the center of the contour/polylines.

    - NOTE: the contour/polylines must be closed.
    - NOTE: the returned contour/polylines does not neccearily have the same number of points as the input contour/polylines

    :param np.ndarray | list[list[int]] polylines:
        the closed contour/polyline to scale
    :param float scale:
        the scale factor
    :param tuple[int, int] | None dimensions:
        the dimensions to confine the scaled contour (the contour cannot outgrow the dimensions).
        If None, the scaled polylines can grow without any restriction.
    :param bool debug:
        whether to show debug images in a window
    :param str | None title:
        the title of the debug image to show in debug window
    :return np.ndarray:
        the scaled contour/polylines. The shape is (N, 2) where N is the number of points in the contour/polylines.
        The data type is int, and the points are in the order of (x, y) coordinates.
    """
    polylines = np.array(polylines, dtype=int)
    assert (
        polylines.ndim == 2 and polylines.shape[0] > 0 and polylines.shape[1] == 2
    ), "Invalid contour shape"
    assert scale > 0, "Invalid scale factor"
    if np.isclose(scale, 1, atol=1e-3):
        return polylines

    filled_mask = polylines_to_mask([polylines], dimensions=dimensions)
    mass_y, mass_x = np.mean(np.where(filled_mask), axis=1)

    scaled_polylines = scale * (polylines - np.array([[mass_x, mass_y]])) + np.array(
        [[mass_x, mass_y]]
    )
    scaled_polylines = np.maximum(scaled_polylines.astype(int), 0)
    if dimensions is not None:
        dimensions = np.array(dimensions).flatten()[
            ::-1
        ]  # reverse the dimensions for x, y format
        assert np.all(
            dimensions > np.max(polylines, axis=0)
        ), "Invalid dimensions: the provided dimensions are smaller than the original contour"
        scaled_polylines = np.minimum(scaled_polylines, dimensions.reshape(1, -1) - 1)

    if debug:
        if dimensions is None:
            _scale = max(1, scale)
            dimensions = np.ceil(_scale * (polylines.max(axis=0)[::-1] + 1)).astype(int)
        background = np.zeros((*dimensions, 3), dtype=np.uint8)
        cv2.polylines(
            background,
            [polylines.reshape(-1, 1, 2)],
            isClosed=True,
            color=(255, 255, 255),
            thickness=2,
        )
        cv2.polylines(
            background,
            [scaled_polylines.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            background,
            f"Scaled",
            (scaled_polylines[0, 1], scaled_polylines[0, 0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        display_image(
            background, title=f"`get_scaled_contour`: {title or 'scaled_contour'}"
        )

    return scaled_polylines.astype(int)


def _get_angle_between_pts(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray) -> float:
    """
    Calculate the angle between three points pt1, pt2, pt3.
    The angle is calculated at pt2 and returned in degrees from (-180, 180).
    `-180` degrees means float point precision, and the angle is still clockwise.
    If `absolute` is True, the angle is returned in the range of 0 to 360 degrees.

    :param np.ndarray pt1:
        first point (x, y)
    :param np.ndarray pt2:
        second point (x, y)
    :param np.ndarray pt3:
        third point (x, y)
    :return float:
        angle in degrees
    """
    pt1 = np.array(pt1, dtype=float).flatten()
    pt2 = np.array(pt2, dtype=float).flatten()
    pt3 = np.array(pt3, dtype=float).flatten()

    if not all(x.size == 2 for x in [pt1, pt2, pt3]):
        raise ValueError("All points must be 2D points (x, y)")

    if np.allclose(pt1, pt2) or np.allclose(pt2, pt3):
        raise ValueError("Points must be distinct")
    if np.allclose(pt1, pt3):
        # pt1 and pt3 are the same point, so the angle is 0
        return 0.0

    v1 = pt1 - pt2
    v2 = pt3 - pt2
    angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    return float(np.degrees(angle))


def _check_polyline_angle(
    polyline: list[list[int]] | np.ndarray,
    degrees: float = 90.0,
    tolerance: float = 20.0,
) -> bool:
    """
    For a given **convex** polyline, check if the angle between the points is within the tolerance.

    :param list[list[int]] | np.ndarray polyline:
        an ArrayLike object representing a polyline with the shape (N, 2) where N is the number of points in the polyline.
        There should be at least 3 points in the polyline to check the angle.
        This is designed to process the output of `mask_to_polylines` function.
    :param float degrees:
        the angle in degrees to check. Should be between 0 and 180 degrees.
    :param float tolerance:
        the tolerance in degrees to check the angle against. Default is 20 degrees.
    """
    polyline = np.array(polyline, dtype=float).reshape(-1, 2)
    polyline = np.unique(polyline, axis=0)  # remove duplicate points
    if polyline.shape[0] < 3:
        raise ValueError(
            "The polyline must have at least 3 unique points to check the angle"
        )

    # pad the polyline with the first and last points to make it cyclic
    polyline = np.vstack([polyline[-1:], polyline, polyline[:1]])

    tolerance = abs(tolerance)
    lower_bound = max(0, degrees - tolerance)
    upper_bound = min(180, degrees + tolerance)

    for i in range(1, polyline.shape[0] - 1):
        angle = _get_angle_between_pts(polyline[i - 1], polyline[i], polyline[i + 1])
        print(angle)
        if not (lower_bound <= abs(angle) <= upper_bound):
            return False
    return True
