#!/usr/bin/env python3
"""Generate a racetrack coil layout and write it to CSV."""

import csv
import math
from pathlib import Path
import numpy as np


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


import math
import numpy as np


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


def build_coils(L=2.0, D=1.0):
    """
    Correct racetrack coil ordering:
        TOP straight:    left → right
        RIGHT arc:       top → bottom
        BOTTOM straight: right → left
        LEFT arc:        bottom → top
    L2 coils in pairs (±δ) using true coil width DZ * Nz.
    """

    coils = []

    def add(x, y, angle_rad, ctype):
        coils.append(
            {
                "Xc": x,
                "Yc": y,
                "angle": math.degrees(angle_rad),
                "type": ctype,
            }
        )

    # ---------------------------------------------------------
    # L2 coil physical width → angular half-offset δ
    # ---------------------------------------------------------
    DZ_L2 = 0.03  # m tangential coil thickness
    Nz_L2 = 2  # layers in tangential direction
    W = DZ_L2 * Nz_L2  # 0.06 m arc width

    R = D / 2  # arc radius = 0.5 m
    dtheta = W / R  # angular coil width
    delta = dtheta / 2  # half angular width of L2 coil

    # ---------------------------------------------------------
    # STRAIGHT SECTIONS (4 evenly spaced coils)
    # ---------------------------------------------------------
    mx = midpoints(-L / 2, L / 2, 4)
    straight_types = ["Brown", "OM", "OM", "Brown"]

    # 1. TOP STRAIGHT: left → right
    for x, ctype in zip(mx, straight_types):
        add(x, +D / 2, +math.pi / 2, ctype)

    # ---------------------------------------------------------
    # CURVED SECTIONS (10 coils from 7 arc centers)
    # ---------------------------------------------------------

    # 7 equally spaced center angles
    t_start = -math.pi / 2  # bottom
    t_stop = +math.pi / 2  # top
    centers = np.linspace(t_start, t_stop, 7)

    # pattern:
    #   Blue, L2pair, Blue, L2pair, Blue, L2pair, Blue
    center_types = ["Blue", "L2pair", "Blue", "L2pair", "Blue", "L2pair", "Blue"]

    r = R

    # ---------------------------------------------------------
    # 2. RIGHT ARC: **top → bottom**
    # ---------------------------------------------------------

    cx = +L / 2

    for t_center, typ in zip(centers[::-1], center_types[::-1]):
        if typ == "Blue":
            add(cx + r * math.cos(t_center), r * math.sin(t_center), t_center, "Blue")

        else:  # L2pair
            # maintain same ordering as we move top→bottom
            tA = t_center + delta  # upper
            tB = t_center - delta  # lower

            add(cx + r * math.cos(tA), r * math.sin(tA), tA, "L2")

            add(cx + r * math.cos(tB), r * math.sin(tB), tB, "L2")

    # ---------------------------------------------------------
    # 3. BOTTOM STRAIGHT: right → left
    # ---------------------------------------------------------

    for x, ctype in zip(mx[::-1], straight_types):
        add(x, -D / 2, -math.pi / 2, ctype)

    # ---------------------------------------------------------
    # 4. LEFT ARC: **bottom → top**
    # ---------------------------------------------------------

    cx = -L / 2

    for t_center, typ in zip(centers, center_types):
        if typ == "Blue":
            add(
                cx - r * math.cos(t_center),
                r * math.sin(t_center),
                math.pi - t_center,
                "Blue",
            )

        else:  # L2pair
            tA = t_center - delta  # lower
            tB = t_center + delta  # upper

            add(cx - r * math.cos(tA), r * math.sin(tA), math.pi - tA, "L2")

            add(cx - r * math.cos(tB), r * math.sin(tB), math.pi - tB, "L2")

    return coils


def write_csv(coils, path):
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Xc", "Yc", "angle", "type"])
        for coil in coils:
            writer.writerow(
                [
                    f"{coil['Xc']:.6f}",
                    f"{coil['Yc']:.6f}",
                    f"{coil['angle']:.6f}",
                    coil["type"],
                ]
            )


def main():
    coils = build_coils()
    out_path = Path(__file__).with_name("test-coil-shapes_case3.csv")
    write_csv(coils, out_path)
    print(f"Wrote {len(coils)} coils to {out_path}")


if __name__ == "__main__":
    main()
