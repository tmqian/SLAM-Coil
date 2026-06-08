# pyright: standard
# Generate a racetrack coil layout and write it to CSV.

import csv
import math
from os import name
from pathlib import Path
import numpy as np


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


import math
import numpy as np
from field import *


class Racetrack:
    def __init__(
        self,
        Mirror_Length,
        Stellerator_Radius,
        straight_types,
        center_types,
        extra_R_6pan=0,
        extra_R_blue=0,
        straight_displacements=None,
        center_displacements=None,
        filename=None,
    ):
        self.Mirror_Length = Mirror_Length
        self.Stellerator_Radius = Stellerator_Radius
        self.straight_types = straight_types
        self.center_types = center_types
        self.extra_R_6pan = extra_R_6pan
        self.extra_R_blue = extra_R_blue
        self.filename = filename
        self.coils = None
        self.straight_displacements = straight_displacements or {
            typ: 0 for typ in straight_types
        }
        self.center_displacements = center_displacements or {
            typ: 0 for typ in center_types
        }
        self.mirror_shift = 0

        for typ in straight_types:
            if typ not in self.straight_displacements:
                self.straight_displacements[typ] = 0

        for typ in center_types:
            if typ not in self.center_displacements:
                self.center_displacements[typ] = 0

    def build_coils(self):
        self.coils: List[Coil] = []

        def add(x, y, angle_rad, ctype):
            self.coils.append(Coil(x, y, math.degrees(angle_rad), ctype))

        # ---------------------------------------------------------
        # Straight sections:
        # ---------------------------------------------------------

        num_straight = len(self.straight_types)
        mx = midpoints(-self.Mirror_Length / 2, self.Mirror_Length / 2, num_straight)

        # 1. TOP STRAIGHT: left → right
        for x, ctype in zip(mx, self.straight_types):
            add(x, self.Stellerator_Radius + self.mirror_shift, math.pi / 2, ctype)

        # ---------------------------------------------------------
        # CURVED SECTIONS:
        # ---------------------------------------------------------

        num_curve = len(self.center_types)

        t_start = -math.pi / 2 + math.pi / (num_curve * 2)
        t_stop = math.pi / 2 - math.pi / (num_curve * 2)

        centers = np.linspace(t_start, t_stop, num_curve)

        # ---------------------------------------------------------
        # 2. RIGHT ARC:
        # ---------------------------------------------------------
        cx = self.Mirror_Length / 2

        for t_center, typ in zip(centers[::-1], self.center_types[::-1]):

            if typ == "6pan":
                add(
                    cx
                    + (self.Stellerator_Radius + self.extra_R_6pan)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_6pan) * math.sin(t_center),
                    t_center,
                    f"{typ}",
                )
            elif typ == "6panCenter":
                add(
                    cx
                    + (self.Stellerator_Radius + self.extra_R_6pan)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_6pan) * math.sin(t_center),
                    t_center,
                    f"{typ}",
                )
            elif typ == "Blue":
                add(
                    cx
                    + (self.Stellerator_Radius + self.extra_R_blue)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_blue) * math.sin(t_center),
                    t_center,
                    f"{typ}",
                )
            elif typ == "BlueCenter":
                add(
                    cx
                    + (self.Stellerator_Radius + self.extra_R_blue)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_blue) * math.sin(t_center),
                    t_center,
                    f"{typ}",
                )
            else:
                add(
                    cx + self.Stellerator_Radius * math.cos(t_center),
                    self.Stellerator_Radius * math.sin(t_center),
                    t_center,
                    f"{typ}",
                )

        # ---------------------------------------------------------
        # 3. BOTTOM STRAIGHT
        # ---------------------------------------------------------
        for x, ctype in zip(mx[::-1], self.straight_types):
            add(x, -self.Stellerator_Radius - self.mirror_shift, -math.pi / 2, ctype)

        # ---------------------------------------------------------
        # 4. LEFT ARC
        # ---------------------------------------------------------

        for t_center, typ in zip(centers, self.center_types):

            if typ == "6pan":
                add(
                    -cx
                    - (self.Stellerator_Radius + self.extra_R_6pan)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_6pan) * math.sin(t_center),
                    math.pi - t_center,
                    f"{typ}",
                )
            elif typ == "6panCenter":
                add(
                    -cx
                    - (self.Stellerator_Radius + self.extra_R_6pan)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_6pan) * math.sin(t_center),
                    math.pi - t_center,
                    f"{typ}",
                )
            elif typ == "Blue":
                add(
                    -cx
                    - (self.Stellerator_Radius + self.extra_R_blue)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_blue) * math.sin(t_center),
                    math.pi - t_center,
                    f"{typ}",
                )
            elif typ == "BlueCenter":
                add(
                    -cx
                    - (self.Stellerator_Radius + self.extra_R_blue)
                    * math.cos(t_center),
                    (self.Stellerator_Radius + self.extra_R_blue) * math.sin(t_center),
                    math.pi - t_center,
                    f"{typ}",
                )
            else:
                add(
                    -cx - self.Stellerator_Radius * math.cos(t_center),
                    self.Stellerator_Radius * math.sin(t_center),
                    math.pi - t_center,
                    f"{typ}",
                )

        # shift coils in sstraight sections if needed
        for coil in self.coils:
            if coil.type in self.straight_types:
                if coil.Xc > 0:
                    coil.Xc += self.straight_displacements[coil.type]
                elif coil.Xc < 0:
                    coil.Xc -= self.straight_displacements[coil.type]

        # shift coils in curved sections if needed
        for coil in self.coils:
            if coil.type in self.center_types:
                X = coil.Xc
                Y = coil.Yc
                dtheta = math.pi / 180 * self.center_displacements[coil.type]
                if coil.Xc > 0 and coil.Yc > 0:
                    coil.Xc = (
                        (X - self.Mirror_Length / 2) * math.cos(dtheta)
                        - Y * math.sin(dtheta)
                        + self.Mirror_Length / 2
                    )
                    coil.Yc = (X - self.Mirror_Length / 2) * math.sin(
                        dtheta
                    ) + Y * math.cos(dtheta)
                    coil.angle += self.center_displacements[coil.type]
                elif coil.Xc < 0 and coil.Yc > 0:
                    coil.Xc = (
                        (X + self.Mirror_Length / 2) * math.cos(-dtheta)
                        - Y * math.sin(-dtheta)
                        - self.Mirror_Length / 2
                    )
                    coil.Yc = (X + self.Mirror_Length / 2) * math.sin(
                        -dtheta
                    ) + Y * math.cos(-dtheta)
                    coil.angle -= self.center_displacements[coil.type]
                elif coil.Xc < 0 and coil.Yc < 0:
                    coil.Xc = (
                        (X + self.Mirror_Length / 2) * math.cos(dtheta)
                        - Y * math.sin(dtheta)
                        - self.Mirror_Length / 2
                    )
                    coil.Yc = (X + self.Mirror_Length / 2) * math.sin(
                        dtheta
                    ) + Y * math.cos(dtheta)
                    coil.angle += self.center_displacements[coil.type]
                elif coil.Xc > 0 and coil.Yc < 0:
                    coil.Xc = (
                        (X - self.Mirror_Length / 2) * math.cos(-dtheta)
                        - Y * math.sin(-dtheta)
                        + self.Mirror_Length / 2
                    )
                    coil.Yc = (X - self.Mirror_Length / 2) * math.sin(
                        -dtheta
                    ) + Y * math.cos(-dtheta)
                    coil.angle -= self.center_displacements[coil.type]

    def write_csv(self):
        path = Path("./test_creation") / self.filename
        if self.coils is None:
            raise ValueError("Coils have not been built yet. Call build_coils() first.")
        with path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Xc", "Yc", "angle", "type"])
            for coil in self.coils:
                writer.writerow(
                    [
                        f"{coil.Xc:.6f}",
                        f"{coil.Yc:.6f}",
                        f"{coil.angle:.6f}",
                        coil.type,
                    ]
                )
        print(f"Wrote {len(self.coils)} coils to {path}")
