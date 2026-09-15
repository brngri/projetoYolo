# -*- coding: utf-8 -*-
"""Pós-processamento: maioria, morfológico, buracos, área, circularidade."""

import logging

import numpy as np
import cv2
from scipy.ndimage import generic_filter

logger = logging.getLogger(__name__)


def majority_filter(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Filtro de maioria (moda) em janela size x size."""
    if size < 3:
        return mask

    def _mode(arr):
        return np.bincount(arr.astype(np.int64)).argmax()

    result = generic_filter(mask, _mode, size=size, mode='constant', cval=0)
    return (result > 0.5).astype(np.uint8)


def apply_morphological_closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Fechamento morfológico para unir regiões próximas."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Preenche buracos internos (componentes de fundo que não tocam a borda)."""
    mask_bin = (mask > 0).astype(np.uint8)
    inv = (1 - mask_bin).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    holes = np.zeros_like(mask_bin, dtype=np.uint8)

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w_ = stats[i, cv2.CC_STAT_WIDTH]
        h_ = stats[i, cv2.CC_STAT_HEIGHT]
        right = left + w_ - 1
        bottom = top + h_ - 1
        if (left == 0 or right == mask.shape[1] - 1
                or top == 0 or bottom == mask.shape[0] - 1):
            continue
        holes[labels == i] = 1

    filled = mask_bin | holes
    return filled.astype(np.uint8)


def filter_components(
    mask: np.ndarray,
    min_area: int = 0,
    circularity_filter: bool = False,
    circularity_threshold: float = 0.5,
) -> np.ndarray:
    """Remove componentes pequenos e/ou pouco circulares."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    filtered = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        if circularity_filter:
            comp = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], closed=True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue

        filtered[labels == i] = 1

    return filtered


def postprocess_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """Pipeline completo de pós-processamento conforme config."""
    pp = config.get("postprocess", {})

    if pp.get("majority_filter", False):
        mask = majority_filter(mask, pp.get("majority_size", 3))

    if pp.get("morphological_closing", False):
        mask = apply_morphological_closing(mask, pp.get("closing_kernel", 5))

    if pp.get("fill_holes", False):
        mask = fill_holes(mask)

    min_area = pp.get("min_area", 0)
    circ_filter = pp.get("circularity_filter", False)
    circ_thresh = pp.get("circularity_threshold", 0.5)
    if min_area > 0 or circ_filter:
        mask = filter_components(mask, min_area, circ_filter, circ_thresh)

    return mask