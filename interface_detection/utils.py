import math
from copy import deepcopy
from typing import Callable, Iterable, Optional
from typing import ParamSpec, TypeVar
from functools import wraps

import numpy as np
import cv2

P = ParamSpec("P")
R = TypeVar("R")


def no_modification_to_args(
    func: Callable[P, R], types: Iterable[type] = (list, dict)
) -> Callable[P, R]:
    """
    A decorator to ensure that the input arguments of a function are not modified in place after the function is called

    :param Callable[P, R] func:
        the function to decorate
    :param Iterable[type] types:
        the types of the input arguments to check to disallow in-place modification
    :return Callable[P, R]:
        the decorated function
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if types is not None:
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, types):
                    args[i] = deepcopy(arg)
            for key, val in kwargs.items():
                if isinstance(val, types):
                    kwargs[key] = deepcopy(val)
        return func(*args, **kwargs)

    return wrapper


def get_gray_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale

    :param np.ndarray image:
        the input image, should be a 2D array for grayscale or a 3D array for color (BGR)
    :return np.ndarray:
        the grayscaled image with the same shape as the input image
    """
    if image.ndim == 2:
        return image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image must be a 2D or 3D array with 3 channels (BGR)")


def increase_image_contrast(image: np.ndarray, thresh: int = 200) -> np.ndarray:
    """
    Increase the contrast of an image if the mean of the image is greater than a threshold

    :param np.ndarray image:
        the input image, should be a 2D array for grayscale or a 3D array for color (BGR)
    :param int thresh:
        the threshold for the mean of the grayscaled image to increase the contrast
    :return np.ndarray:
        the contrast-enhanced image with the same shape as the input image
    """
    gray = get_gray_image(image)
    if gray.mean() < thresh:
        return image.copy()

    # increase contrast
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = cv2.equalizeHist(l)
    hls = cv2.merge((h, l, s))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


@no_modification_to_args
def display_image(
    img: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray],
    line_markers: (
        dict[str, tuple[np.ndarray, Optional[int | tuple[int] | np.ndarray]]] | list
    ) = None,
    title: str = "",
    axis: int = 1,
    noshow: bool = False,
    return_image: bool = False,
) -> None | np.ndarray:
    """
    Display an image or a list of images in a window

    :param dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray] img:

        - a dictionary of images: key as the name of the image, value as the image itself
        - a list of images: the index of the image will be used as the name of the image
        - a tuple of images: the index of the image will be used as the name of the image
    :param dict[str, tuple[np.ndarray, Optional[int | tuple[int] | np.ndarray]]] | list line_markers:
        a dictionary of line markers: key as the name of the image, value as a tuple of (lines to mark, color)
    :param str title:
        title of the window
    :param int axis: axis along which to concatenate the images

        - axis=1: horizontal stack (default)
        - axis=0: vertical stack
    :param bool noshow:
        whether to show the image in a window or not (default: False)
    :param bool return_image:
        whether to return the image or not (default: False)
    :return None | np.ndarray:
        the concatenated image if return_image is True, otherwise None
    """
    if isinstance(img, np.ndarray):
        img = [img]
    if isinstance(img, (list, tuple)):
        img = {f"{i}": img[i] for i in range(len(img))}
        if line_markers is not None:
            line_markers = (
                [line_markers] * len(img)
                if isinstance(line_markers, (int, tuple))
                else line_markers
            )
            assert isinstance(line_markers, (list, tuple)) and len(line_markers) == len(
                img
            ), "line_markers must be a list with the same length as img"
            line_markers = {f"{i}": line_markers[i] for i in range(len(img))}
    for key, val in img.items():
        if str(val.dtype) in ["bool", "np.bool_"]:
            val = val.astype(np.uint8) * 255
            img[key] = val
        if val.ndim == 2:
            img[key] = cv2.cvtColor(val, cv2.COLOR_GRAY2BGR)
        if line_markers is not None and key in line_markers:
            lines_to_mark = line_markers[key][0]
            if len(line_markers[key]) == 1 or line_markers[key][1] is None:
                color = (0, 0, 255)
            else:
                color = line_markers[key][1]
            # default color will be red; in addition, if provided color is int, it will be used as a grayscale value
            if isinstance(color, int):
                color = (color, color, color)
            elif isinstance(color, tuple) and len(color) == 3:
                pass
            elif isinstance(color, (np.ndarray, list)):
                color = np.asarray(color)
                if color.ndim == 1 or color.shape[1] == 1:
                    color = color.flatten().reshape(-1, 1)
                    assert len(lines_to_mark) == len(
                        color
                    ), "color must have the same length as lines_to_mark"
                    color = np.tile(color, (1, 3))
                else:
                    assert color.shape[1] == 3, "color must be a 3-tuple or a 1D array"
                    assert len(lines_to_mark) == len(
                        color
                    ), "color must have the same length as lines_to_mark"
            else:
                raise ValueError(
                    "color must be an integer, a 3-tuple, a 1D array or a 2D array with 3 columns"
                )

            img[key][lines_to_mark] = color

    # horizontal stack of images
    max_dims = np.max([val.shape for val in img.values()], axis=0)
    for key, val in img.items():
        temp_val = np.zeros(max_dims, dtype=val.dtype)
        temp_val[: val.shape[0], : val.shape[1]] = val
        img[key] = temp_val
    all_img = np.concatenate([val for val in img.values()], axis=axis).astype(np.uint8)
    if not noshow:
        prefix = f"{title}: " if title else ""
        all_img_names = prefix + " | ".join([key for key in img.keys()])
        cv2.namedWindow(all_img_names, cv2.WINDOW_NORMAL)
        cv2.imshow(all_img_names, all_img)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if return_image:
        return all_img


def union_of_domains(
    domains: np.ndarray,
    expansion: int | float = 0,
    min_domain: int | float | None = None,
    max_domain: int | float | None = None,
    return_flattened: bool = False,
) -> np.ndarray:
    """
    Find the union of a set of domains

    :param np.ndarray domains:
        a list of domains in a 2D array, where each row is a domain with two columns of (start, end)
    :param int | float expansion:
        the amount by which to expand/flex the domains from both ends, i.e., (start-expansion, end+expansion).
        If expansion is 0, the domains will not be expanded.
    :param int | float | None min_domain:
        the minimum value of the domain to constrain any expansion
    :param int | float | None max_domain:
        the maximum value of the domain to constrain any expansion
    :param bool return_flattened:
        whether to return the flattened union of the domains (list of included intergers in the union).
        For example, [[2, 4], [6, 7]] will return [2, 3, 4, 6, 7] if return_flattened is True.
    :return np.ndarray:
        the union of the domains, e.g., [[2, 4], [6, 7]],
        or the included integers in the union of the domains, e.g., [2, 3, 4, 6, 7]
    """
    assert (
        isinstance(expansion, (int, float)) and expansion >= 0
    ), "expansion must be a non-negative number"
    # sort the domains according to the first column
    domains = np.asarray(domains)
    if domains.size == 0:
        return np.array([])

    if domains.ndim == 1 or domains.shape[1] == 1:
        if expansion <= 0:
            raise ValueError(
                "expansion must be greater than 0 for 1D domains, or use a 2D array"
            )
        domains = domains.reshape(-1, 1)
        domains = np.concatenate([domains, domains], axis=1)

    if domains.ndim != 2:
        raise ValueError(
            "domains must be a 2D array or a 1D array when `expansion` is greater than 0"
        )

    if np.any(domains[:, 0] > domains[:, 1]):
        raise ValueError("each domain must be a valid interval, i.e., start <= end")

    domains = domains[np.argsort(domains[:, 0])]
    collapsed_domains = []
    for start, end in domains:
        if collapsed_domains and start - expansion <= collapsed_domains[-1][1]:
            _max = max(collapsed_domains[-1][1], end + expansion)
            collapsed_domains[-1][1] = (
                _max if max_domain is None else min(_max, max_domain)
            )
        else:
            start_ = (
                max(min_domain, start - expansion)
                if min_domain is not None
                else start - expansion
            )
            end_ = (
                min(max_domain, end + expansion)
                if max_domain is not None
                else end + expansion
            )
            collapsed_domains.append([start_, end_])
    if return_flattened and len(collapsed_domains) > 0:
        return np.concatenate(
            [
                np.arange(math.floor(start), math.floor(end + 1))
                for start, end in collapsed_domains
            ]
        ).astype(int)
    elif return_flattened and len(collapsed_domains) == 0:
        return np.array([])
    return np.array(collapsed_domains)


def number_to_excel_column(n: int | str) -> str | int:
    """
    Convert a number to an Excel column name or vice versa

    :param int n: the number to convert, or the Excel column name to convert
    :return str | int: the Excel column name, or the number from the Excel column name

    For example:
    ```
    1 <-> A
    2 <-> B
    ...
    26 <-> Z
    27 <-> AA
    28 <-> AB
    ```
    """
    if isinstance(n, int):
        assert n > 0, "n must be a positive integer"
        result = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(65 + remainder) + result
        return result
    elif isinstance(n, str):
        s = n.upper()
        return sum((ord(c) - 64) * 26**i for i, c in enumerate(reversed(s)))


def excel_column_to_number(col: str) -> int:
    """
    Convert an Excel column name to a number. See `number_to_excel_column` for more details.

    :param str col: the Excel column name, such as "A", "B", "C", ..., "Z", "AA", "AB", ...
    :return int: the number corresponding to the Excel column name
    """
    col = col.upper()
    return sum((ord(c) - 64) * 26**i for i, c in enumerate(reversed(col)))


def find_longest_in_sequence(
    arr: np.ndarray, value: int | float | bool = 0
) -> tuple[int, int]:
    """
    Find the longest sequence of a defined value in a 1-d binary array

    :param np.ndarray arr:
        the input 1D array
    :param int | float | bool value:
        the value to find the longest sequence of. Match will be done using `==` operator.
    :return tuple[int, int]:
        the start and end indices of the longest sequence.
        (-1, -1) if no sequence of the given value is found.
    """
    arr = np.asarray(arr).reshape(-1)
    if len(arr) == 0:
        return -1, -1

    arr_match = arr == value
    if np.all(arr_match):
        return 0, len(arr) - 1
    elif np.all(~arr_match):
        return -1, -1  # no sequence of the given value

    arr_mismatch = np.bitwise_not(arr_match)
    # the following finds the longest zero sequence
    _cumsum = np.cumsum(arr_mismatch)
    _max_consecutive_num = np.argmax(np.bincount(_cumsum))
    _consecutive_zeros = np.bitwise_and(arr_match, _cumsum == _max_consecutive_num)
    start = int(np.argmax(_consecutive_zeros))
    end = start + int(np.sum(_consecutive_zeros)) - 1
    return start, end


def find_intersection_point(
    line1_pt1: list | tuple | np.ndarray,
    line1_pt2: list | tuple | np.ndarray,
    line2_pt1: list | tuple | np.ndarray,
    line2_pt2: list | tuple | np.ndarray,
    dtype: type = float,
) -> tuple[float, float]:
    """
    Find the intersection point of two lines

    References
    ---------
    1. https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

    :param list | tuple | np.ndarray line1_pt1: the first point of the first line
    :param list | tuple | np.ndarray line1_pt2: the second point of the first line
    :param list | tuple | np.ndarray line2_pt1: the first point of the second line
    :param list | tuple | np.ndarray line2_pt2: the second point of the second line
    :param float dtype: the data type of the intersection point. Be aware of overflow if using int.
    :return tuple[float, float]:
        the intersection point of the two lines

    :raises ValueError: if the lines are parallel or if one of the lines is a point
    """
    if np.allclose(line1_pt1, line1_pt2) or np.allclose(line2_pt1, line2_pt2):
        raise ValueError("The lines must not be a point")

    def _det(_a, _b):
        return dtype(_a[0]) * dtype(_b[1]) - dtype(_a[1]) * dtype(_b[0])

    xdiff = np.array(
        [line1_pt1[0] - line1_pt2[0], line2_pt1[0] - line2_pt2[0]], dtype=dtype
    )
    ydiff = np.array(
        [line1_pt1[1] - line1_pt2[1], line2_pt1[1] - line2_pt2[1]], dtype=dtype
    )

    div = _det(xdiff, ydiff)
    if np.isclose(div, 0):
        raise ValueError("The lines must not be parallel")

    d = [_det(line1_pt1, line1_pt2), _det(line2_pt1, line2_pt2)]
    x = _det(d, xdiff) / div
    y = _det(d, ydiff) / div
    return dtype(x), dtype(y)


if __name__ == "__main__":

    domains = np.array(
        [
            [7, 9],
            [8, 9],
            [0, 4],
            [3, 6],
            [10, 11],
        ]
    )
    print(union_of_domains(domains))

    l1pt1, l1pt2 = [0, 0], [0, 10]
    l2pt1, l2pt2 = [0, 8], [10, 8]
    intersection = find_intersection_point(l1pt1, l1pt2, l2pt1, l2pt2)
    assert np.allclose(intersection, [0, 8]), "Invalid intersection point"
