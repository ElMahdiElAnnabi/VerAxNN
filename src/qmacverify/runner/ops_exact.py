from __future__ import annotations

from typing import Tuple

import numpy as np

from qmacverify.runner.int_math import clamp, requant_mul_shift, INT8_MAX, INT8_MIN


def conv2d_int8(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> np.ndarray:
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    pad_h, pad_w = padding
    out_h = (h + 2 * pad_h - kh) // stride[0] + 1
    out_w = (w + 2 * pad_w - kw) // stride[1] + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.int32)

    for batch in range(n):
        for oc in range(c_out):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = int(bias[oc])
                    for ic in range(c_in):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * stride[0] + ky
                                ix = ox * stride[1] + kx
                                acc += int(x_pad[batch, ic, iy, ix]) * int(weight[oc, ic, ky, kx])
                    out[batch, oc, oy, ox] = acc
    return out


def linear_int8(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], weight.shape[0]), dtype=np.int32)
    for b in range(x.shape[0]):
        for o in range(weight.shape[0]):
            acc = int(bias[o])
            for i in range(weight.shape[1]):
                acc += int(x[b, i]) * int(weight[o, i])
            out[b, o] = acc
    return out


def relu_int32(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def add_int32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.int32) + b.astype(np.int32)


def flatten(x: np.ndarray, start_dim: int = 1, end_dim: int = -1) -> np.ndarray:
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    new_shape = list(x.shape[:start_dim]) + [-1] + list(x.shape[end_dim + 1 :])
    return x.reshape(new_shape)


def avg_pool2d_int32(
    x: np.ndarray,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> np.ndarray:
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    out = np.zeros((n, c, out_h, out_w), dtype=np.int32)
    area = kh * kw
    for b in range(n):
        for ch in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy * sh + ky
                            ix = ox * sw + kx
                            acc += int(x_pad[b, ch, iy, ix])
                    out[b, ch, oy, ox] = int(acc // area)
    return out


def max_pool2d_int32(
    x: np.ndarray,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> np.ndarray:
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (ph, ph), (pw, pw)),
        mode="constant",
        constant_values=np.iinfo(np.int32).min,
    )
    out = np.zeros((n, c, out_h, out_w), dtype=np.int32)
    for b in range(n):
        for ch in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    max_val = np.iinfo(np.int32).min
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy * sh + ky
                            ix = ox * sw + kx
                            max_val = max(max_val, int(x_pad[b, ch, iy, ix]))
                    out[b, ch, oy, ox] = max_val
    return out


def adaptive_avg_pool2d_int32(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    n, c, h, w = x.shape
    out_h, out_w = output_size
    out = np.zeros((n, c, out_h, out_w), dtype=np.int32)
    for b in range(n):
        for ch in range(c):
            for oy in range(out_h):
                y_start = (oy * h) // out_h
                y_end = ((oy + 1) * h + out_h - 1) // out_h
                for ox in range(out_w):
                    x_start = (ox * w) // out_w
                    x_end = ((ox + 1) * w + out_w - 1) // out_w
                    window = x[b, ch, y_start:y_end, x_start:x_end]
                    out[b, ch, oy, ox] = int(np.mean(window, dtype=np.float64))
    return out


def adaptive_max_pool2d_int32(x: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    n, c, h, w = x.shape
    out_h, out_w = output_size
    out = np.zeros((n, c, out_h, out_w), dtype=np.int32)
    for b in range(n):
        for ch in range(c):
            for oy in range(out_h):
                y_start = (oy * h) // out_h
                y_end = ((oy + 1) * h + out_h - 1) // out_h
                for ox in range(out_w):
                    x_start = (ox * w) // out_w
                    x_end = ((ox + 1) * w + out_w - 1) // out_w
                    window = x[b, ch, y_start:y_end, x_start:x_end]
                    out[b, ch, oy, ox] = int(np.max(window))
    return out


def requant_int8(x: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    return requant_mul_shift(x, multiplier, shift, INT8_MIN, INT8_MAX)
