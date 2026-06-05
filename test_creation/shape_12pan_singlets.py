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



'''

Inputs 


'''

#large
#straight_types = ["Brown", "OM","OM","OM", "Brown"]
#center_types = ["Blue", "L2block","Blue", "L2block", "Blue", "L2block", "Blue","L2block","Blue"]

#Mirror_Length = 2
#Stellerator_Radius = 1.2

# small 
#straight_types = ["L2", "L2","L2", "L2","L2", "L2"]
#center_types = ["Blue","Blue", "Blue","Blue","Blue"]

#Mirror_Length = 1
#Stellerator_Radius = .9
#filename = "small_test_L2_mirror.csv"

#high field low current 
straight_types = ["Brown", "OM","OM","OM", "Brown"]
center_types = ["Blue", "12pan","BlueCenter","12panCenter", "BlueCenter", "12pan", "Blue"]

Mirror_Length = 1.2
Stellerator_Radius = 1
#filename = "medium_Lm_1p5.csv"
filename = "../test_files/racetrack_12pan.csv"

#L2 Stell 
#straight_types = ["Blue","Blue", "OM","OM","OM", "Blue","Blue"]
#center_types = ["L2block","L2block", "L2block","L2block","L2block","L2block"]

#Mirror_Length = 1.5
#Stellerator_Radius = .5
#filename = "L2Stell_Lm_1.csv"


def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


def build_coils(L=Mirror_Length, D=Stellerator_Radius):
    """
    Hardcoded: Racetrack coil layout with L2 TRIPLETS 
    
    """
    coils = []

    def add(x, y, angle_rad, ctype):
        coils.append({
            "Xc": x,
            "Yc": y,
            "angle": math.degrees(angle_rad),
            "type": ctype,
        })


    R = D / 2          # racetrack arc radius = 0.5 m

    # ---------------------------------------------------------
    # Straight sections: coils evenly spaced
    # ---------------------------------------------------------
    num_straight = len(straight_types)   
    mx = midpoints(-L/2, L/2, num_straight)

    # 1. TOP STRAIGHT: left → right
    for x, ctype in zip(mx, straight_types):
        add(x, +D/2, +math.pi/2, ctype)

    # ---------------------------------------------------------
    # CURVED SECTIONS:
    # ---------------------------------------------------------
    
    num_curve = len(center_types)   

    t_start = -math.pi/2 + math.pi/(num_curve*2)
    t_stop  = +math.pi/2 - math.pi/(num_curve*2)
    
    centers = np.linspace(t_start, t_stop, num_curve)

    # ---------------------------------------------------------
    # 2. RIGHT ARC:
    # ---------------------------------------------------------
    cx = +L/2

    for t_center, typ in zip(centers[::-1], center_types[::-1]):

        add(cx + R*math.cos(t_center),
                R*math.sin(t_center),
                t_center,
                f"{typ}")

    # ---------------------------------------------------------
    # 3. BOTTOM STRAIGHT
    # ---------------------------------------------------------
    for x, ctype in zip(mx[::-1], straight_types):
        add(x, -D/2, -math.pi/2, ctype)

    # ---------------------------------------------------------
    # 4. LEFT ARC
    # ---------------------------------------------------------
    cx = -L/2

    for t_center, typ in zip(centers, center_types):

        add(cx - R*math.cos(t_center),
                R*math.sin(t_center),
                math.pi - t_center,
                f"{typ}")

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
    out_path = Path(__file__).parent / filename
    write_csv(coils, out_path)
    print(f"Wrote {len(coils)} coils to {out_path}")


if __name__ == "__main__":
    main()


