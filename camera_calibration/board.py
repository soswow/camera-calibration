"""Shared physical ChArUco definition for detection and printing."""

import cv2
import numpy as np


def make_board(
    squares_x,
    squares_y,
    square_size,
    marker_size,
    dictionary,
    *,
    legacy_pattern=False,
    first_marker_id=0,
):
    if (
        min(squares_x, squares_y) < 2
        or not 0 < marker_size < square_size
        or not np.isfinite([square_size, marker_size]).all()
    ):
        raise ValueError("Invalid board dimensions or marker/square size")
    if first_marker_id < 0:
        raise ValueError("First marker ID must be nonnegative")
    count = (squares_x * squares_y) // 2
    if first_marker_id + count > len(dictionary.bytesList):
        raise ValueError("Dictionary does not contain enough marker IDs for this board")
    ids = np.arange(first_marker_id, first_marker_id + count, dtype=np.int32)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), float(square_size), float(marker_size), dictionary, ids
    )
    board.setLegacyPattern(bool(legacy_pattern))
    return board
