# -*- coding: utf-8 -*-
"""Shared camera/base coordinate helpers for Task3 arm planning."""

import numpy as np


def pixel_to_camera(u, v, depth_mm, intrinsics):
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (float(u) - cx) * float(depth_mm) / fx
    y = (float(v) - cy) * float(depth_mm) / fy
    z = float(depth_mm)
    return np.array([x, y, z], dtype=np.float64)


def transform_point(T, point):
    p = np.asarray(point, dtype=np.float64).reshape(3)
    hp = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (np.asarray(T, dtype=np.float64) @ hp)[:3]
