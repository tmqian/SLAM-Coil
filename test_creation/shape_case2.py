#!/usr/bin/env python3
"""Generate a racetrack coil layout and write it to CSV."""

import csv
import math
from pathlib import Path


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


def build_coils(L=2.0, D=1.0, Ns=10):
    # 9 coils on curved arcs
    Ns_alt = 9

    # 4 coils per straight section, evenly spaced
    mx = midpoints(-L / 2, L / 2, 4)
    straight_types = ["Brown", "OM", "OM", "Brown"]

    # original 10 arc points, we use endpoints only
    ts_old = midpoints(-math.pi / 2, math.pi / 2, Ns)

    # new 9 angles between preserved endpoints
    ts_new = [ts_old[0]] + [
        ts_old[0] + i*(ts_old[-1]-ts_old[0])/(Ns_alt-1)
        for i in range(1, Ns_alt-1)
    ] + [ts_old[-1]]

    arc_types = ["Blue","L2","Blue","L2","Blue","L2","Blue","L2","Blue"]

    coils = []

    def add(x, y, angle, ctype):
        coils.append({"Xc": x, "Yc": y, "angle": angle, "type": ctype})

    # ───── TOP STRAIGHT: left → right ─────
    for x, ctype in zip(mx, straight_types):
        add(x, D/2, 90.0, ctype)

    # ───── RIGHT ARC: bottom → top ─────
    for t, ctype in zip(ts_new, arc_types):
        add(L/2 + (D/2)*math.cos(t),
            (D/2)*math.sin(t),
            math.degrees(t),
            ctype)

    # ───── BOTTOM STRAIGHT: right → left ─────
    for x, ctype in zip(reversed(mx), straight_types):
        add(x, -D/2, -90.0, ctype)

    # ───── LEFT ARC: top → bottom ─────
    for t, ctype in zip(reversed(ts_new), arc_types):
        add(-L/2 - (D/2)*math.cos(t),
            (D/2)*math.sin(t),
            180.0 - math.degrees(t),
            ctype)

    # order coils by polar angle
    for coil in coils:
        phi = math.atan2(coil["Yc"], coil["Xc"])
        coil["phi"] = (phi + 2*math.pi) % (2*math.pi)
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
    out_path = Path(__file__).with_name("test-coil-shapes_case2.csv")
    write_csv(coils, out_path)
    print(f"Wrote {len(coils)} coils to {out_path}")


if __name__ == "__main__":
    main()
