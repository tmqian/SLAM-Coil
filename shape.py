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

    coils += [{"Xc": x, "Yc": D / 2, "angle": 90.0, "type": "OM"} for x in mx]
    coils += [
        {
            "Xc": L / 2 + (D / 2) * math.cos(t),
            "Yc": (D / 2) * math.sin(t),
            "angle": math.degrees(t),
            "type": "L2",
        }
        for t in ts
    ]
    coils += [{"Xc": x, "Yc": -D / 2, "angle": -90.0, "type": "OM"} for x in mx]
    coils += [
        {
            "Xc": -L / 2 - (D / 2) * math.cos(t),
            "Yc": (D / 2) * math.sin(t),
            "angle": 180.0 - math.degrees(t),
            "type": "L2",
        }
        for t in ts
    ]
    return coils


def write_csv(coils, path):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["Xc", "Yc", "angle", "type"])
        writer.writeheader()
        for row in coils:
            writer.writerow(
                {
                    "Xc": f"{row['Xc']:.6f}",
                    "Yc": f"{row['Yc']:.6f}",
                    "angle": f"{row['angle']:.6f}",
                    "type": row["type"],
                }
            )


def main():
    coils = build_coils()
    out_path = Path(__file__).with_name("test-coil-shapes.csv")
    write_csv(coils, out_path)
    print(f"Wrote {len(coils)} coils to {out_path}")


if __name__ == "__main__":
    main()
