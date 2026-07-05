# -*- coding: utf-8 -*-
import math


def Arm(x=None, y=None):
    if x is None:
        x = int(input("x:"))
    if y is None:
        y = int(input("y:"))

    pi = 3.14
    l1 = 105
    l2 = 100
    l3 = 120
    theta = math.radians(0)

    bx = float(x) - l3 * math.cos(theta)
    by = float(y) - l3 * math.sin(theta)
    lp = bx * bx + by * by
    if lp <= 1e-6:
        raise ValueError("Target is too close to arm base")

    beta_arg = (l1 * l1 + lp - l2 * l2) / (2 * l1 * math.sqrt(lp))
    q2_arg = (l1 * l1 + l2 * l2 - lp) / (2 * l1 * l2)
    beta_arg = max(-1.0, min(1.0, beta_arg))
    q2_arg = max(-1.0, min(1.0, q2_arg))

    alpha = math.atan2(by, bx)
    beta = math.acos(beta_arg)
    q1 = -(pi / 2.0 - alpha - beta)
    q2 = math.acos(q2_arg) - pi
    q3 = -q1 - q2 - pi / 2

    angle_5 = int(2047 + int(math.degrees(q1) * 11.375))
    angle_4 = int(2047 + int(math.degrees(q2) * 11.375))
    angle_3 = int(2047 - int(math.degrees(q3) * 11.375))
    return angle_3, angle_4, angle_5
