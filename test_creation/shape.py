#!/usr/bin/env python3
"""Generate a racetrack coil layout and write it to CSV."""

import csv
import math
from pathlib import Path


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


def build_coils(L=2.0, D=1.0, Nm=6, Ns=10):
    mx = midpoints(-L / 2, L / 2, Nm)
    ts = midpoints(-math.pi / 2, math.pi / 2, Ns)
    coils = []

    def add(x, y, angle, ctype):
        coils.append({"Xc": x, "Yc": y, "angle": angle, "type": ctype})

    for x in mx:  # upper straight (left -> right)
        add(x, D / 2, 90.0, "OM")

    for t in ts:  # right curved (bottom -> top)
        add(
            L / 2 + (D / 2) * math.cos(t),
            (D / 2) * math.sin(t),
            math.degrees(t),
            "L2",
        )

    for x in reversed(mx):  # lower straight (right -> left)
        add(x, -D / 2, -90.0, "OM")

    for t in reversed(ts):  # left curved (top -> bottom)
        add(
            -L / 2 - (D / 2) * math.cos(t),
            (D / 2) * math.sin(t),
            180.0 - math.degrees(t),
            "L2",
        )

    # sort by polar angle about the origin to enforce monotonic progression
    for coil in coils:
        phi = math.atan2(coil["Yc"], coil["Xc"])
        coil["phi"] = (phi + 2 * math.pi) % (2 * math.pi)
    coils.sort(key=lambda c: c["phi"])
    for coil in coils:
        coil.pop("phi", None)
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
    out_path = Path(__file__).with_name("test-coil-shapes.csv")
    write_csv(coils, out_path)
    print(f"Wrote {len(coils)} coils to {out_path}")


if __name__ == "__main__":
    main()
