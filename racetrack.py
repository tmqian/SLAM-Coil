"""Generate a racetrack coil layout and write it to CSV."""

import csv
import math
from os import name
from pathlib import Path
import numpy as np
import re

def midpoints(start, stop, count):
    step = (stop - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


import math
import numpy as np
from field import *

class Racetrack:
    def __init__(self, Mirror_Length, Stellerator_Radius, straight_types, center_types, straight_displacements=None, center_displacements=None, filename=None, toroid_trans=0):
        self.Mirror_Length = Mirror_Length
        self.Stellerator_Radius = Stellerator_Radius
        self.straight_types = straight_types
        self.center_types = center_types
        self.filename = filename
        self.toroid_trans = toroid_trans
        self.coils = None
        self.straight_displacements = straight_displacements or {typ: 0 for typ in straight_types}
        self.center_displacements = center_displacements or {typ: 0 for typ in center_types}
        self.mirror_shift = 0
        self.ports = None

        for typ in straight_types:
            if typ not in self.straight_displacements:
                self.straight_displacements[typ] = 0

        for typ in center_types:
            if typ not in self.center_displacements:
                self.center_displacements[typ] = 0


    
    def build_coils(self):
        self.coils = []

        def add(x, y, angle_rad, ctype, id = 0):
            self.coils.append(Coil(x, y, math.degrees(angle_rad), ctype))
            self.coils[-1].id = id

        # ---------------------------------------------------------
        # Straight sections: 
        # ---------------------------------------------------------

        num_straight = len(self.straight_types)   
        mx = midpoints(-self.Mirror_Length/2, self.Mirror_Length/2, num_straight)

        # 1. TOP STRAIGHT: left → right
        for x, ctype in zip(mx, self.straight_types):
            match = re.search(r"\d+$", ctype)
            id = int(match.group()) if match else 0
            typ_base = re.sub(r"\d+$", "", ctype) if match else ctype
        
            add(x, self.Stellerator_Radius + self.mirror_shift, math.pi/2, f"{typ_base}", id = id)
        

        # ---------------------------------------------------------
        # CURVED SECTIONS:
        # ---------------------------------------------------------
        
        num_curve = len(self.center_types) 

        t_start = -math.pi/2 #+ math.pi/(num_curve*2)
        t_stop  = math.pi/2 #- math.pi/(num_curve*2)
        
        centers = np.linspace(t_start, t_stop, num_curve)

        # ---------------------------------------------------------
        # 2. RIGHT ARC:
        # ---------------------------------------------------------
        cx = self.Mirror_Length/2

        for t_center, typ in zip(centers[::-1], self.center_types[::-1]):
        
            match = re.search(r"\d+$", typ)
            id = int(match.group()) if match else 0
            typ_base = re.sub(r"\d+$", "", typ) if match else typ
        
            add(cx + self.Stellerator_Radius*math.cos(t_center),
                    self.Stellerator_Radius*math.sin(t_center),
                    t_center,
                    f"{typ_base}", id = id)

        # ---------------------------------------------------------
        # 3. BOTTOM STRAIGHT
        # ---------------------------------------------------------
        for x, ctype in zip(mx[::-1], self.straight_types):
            match = re.search(r"\d+$", ctype)
            id = int(match.group()) if match else 0
            typ_base = re.sub(r"\d+$", "", ctype) if match else ctype
            add(x, -self.Stellerator_Radius - self.mirror_shift, -math.pi/2, f"{typ_base}", id = id)

        # ---------------------------------------------------------
        # 4. LEFT ARC
        # ---------------------------------------------------------

        for t_center, typ in zip(centers, self.center_types):

            match = re.search(r"\d+$", typ)
            id = int(match.group()) if match else 0
            typ_base = re.sub(r"\d+$", "", typ) if match else typ

            add(-cx - self.Stellerator_Radius*math.cos(t_center),
                    self.Stellerator_Radius*math.sin(t_center),
                    math.pi - t_center,
                    f"{typ_base}", id = id)
                
        #shift coils in sstraight sections if needed
        for coil in self.coils:
            if coil.type in self.straight_types:
                if coil.Xc > 0:
                    coil.Xc += self.straight_displacements[coil.type]
                elif coil.Xc < 0:
                    coil.Xc -= self.straight_displacements[coil.type]

        #shift coils in curved sections if needed

        for coil in self.coils:
            #shift all coils of a type if needed
            if coil.type in self.center_types:
                    X = coil.Xc
                    Y = coil.Yc
                    dtheta = math.pi/180 * self.center_displacements[coil.type]
                    if coil.Xc > 0 and coil.Yc > 0:
                        coil.Xc = (X - self.Mirror_Length/2) *math.cos(dtheta) - Y * math.sin(dtheta) + self.Mirror_Length/2
                        coil.Yc = (X - self.Mirror_Length/2) * math.sin(dtheta) + Y * math.cos(dtheta)
                        coil.angle += self.center_displacements[coil.type]
                    elif coil.Xc < 0 and coil.Yc > 0:
                        coil.Xc = (X + self.Mirror_Length/2) *math.cos(-dtheta) - Y * math.sin(-dtheta) - self.Mirror_Length/2
                        coil.Yc = (X + self.Mirror_Length/2) * math.sin(-dtheta) + Y * math.cos(-dtheta)
                        coil.angle -= self.center_displacements[coil.type]
                    elif coil.Xc < 0 and coil.Yc < 0:
                        coil.Xc = (X + self.Mirror_Length/2) *math.cos(dtheta) - Y * math.sin(dtheta) - self.Mirror_Length/2
                        coil.Yc = (X + self.Mirror_Length/2) * math.sin(dtheta) + Y * math.cos(dtheta)
                        coil.angle += self.center_displacements[coil.type]
                    elif coil.Xc > 0 and coil.Yc < 0:
                        coil.Xc = (X - self.Mirror_Length/2) *math.cos(-dtheta) - Y * math.sin(-dtheta) + self.Mirror_Length/2
                        coil.Yc = (X - self.Mirror_Length/2) * math.sin(-dtheta) + Y * math.cos(-dtheta)
                        coil.angle -= self.center_displacements[coil.type]
                    if coil.Xc > 0:
                        coil.Xc += self.toroid_trans
                    elif coil.Xc < 0:
                        coil.Xc -= self.toroid_trans
            #shift coils by id from new positions if needed
            if coil.id != 0:
                ctype = coil.type + str(coil.id)
                if ctype in self.center_types:
                    if coil.Xc > 0:
                        coil.Xc -= self.toroid_trans
                    elif coil.Xc < 0:
                        coil.Xc += self.toroid_trans
                    X = coil.Xc
                    Y = coil.Yc
                    dtheta = math.pi/180 * self.center_displacements[ctype]
                    if coil.Xc > 0 and coil.Yc > 0:
                        coil.Xc = (X - self.Mirror_Length/2) *math.cos(dtheta) - Y * math.sin(dtheta) + self.Mirror_Length/2 + self.toroid_trans
                        coil.Yc = (X - self.Mirror_Length/2) * math.sin(dtheta) + Y * math.cos(dtheta)
                        coil.angle += self.center_displacements[ctype]
                    elif coil.Xc < 0 and coil.Yc > 0:
                        coil.Xc = (X + self.Mirror_Length/2) *math.cos(-dtheta) - Y * math.sin(-dtheta) - self.Mirror_Length/2 - self.toroid_trans
                        coil.Yc = (X + self.Mirror_Length/2) * math.sin(-dtheta) + Y * math.cos(-dtheta)
                        coil.angle -= self.center_displacements[ctype]
                    elif coil.Xc < 0 and coil.Yc < 0:
                        coil.Xc = (X + self.Mirror_Length/2) *math.cos(dtheta) - Y * math.sin(dtheta) - self.Mirror_Length/2 - self.toroid_trans
                        coil.Yc = (X + self.Mirror_Length/2) * math.sin(dtheta) + Y * math.cos(dtheta)
                        coil.angle += self.center_displacements[ctype]
                    elif coil.Xc > 0 and coil.Yc < 0:
                        coil.Xc = (X - self.Mirror_Length/2) *math.cos(-dtheta) - Y * math.sin(-dtheta) + self.Mirror_Length/2 + self.toroid_trans
                        coil.Yc = (X - self.Mirror_Length/2) * math.sin(-dtheta) + Y * math.cos(-dtheta)
                        coil.angle -= self.center_displacements[ctype]
                    if coil.Xc > 0:
                        coil.Xc += self.toroid_trans
                    elif coil.Xc < 0:
                        coil.Xc -= self.toroid_trans


        #Filter coil types by removing trailing digits for port calculations
        seen = set()
        new_center = []
        for s in self.center_types:
            base = re.sub(r"\d+$", "", s) if re.search(r"\d+$", s) else s
            if base not in seen:
                seen.add(base)
                new_center.append(base)
        self.center_types = new_center

        seen = set()
        new_straight = []
        for s in self.straight_types:
            base = re.sub(r"\d+$", "", s) if re.search(r"\d+$", s) else s
            if base not in seen:
                seen.add(base)
                new_straight.append(base)
        self.straight_types = new_straight



    def write_csv(self):
        path = Path(__file__).parent / self.filename
        if self.coils is None:
            raise ValueError("Coils have not been built yet. Call build_coils() first.")
        with path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Xc", "Yc", "angle", "type"])
            for coil in self.coils:
                if coil.type[-1].isdigit() and (coil.type != 'L2'):
                    coil.type = coil.type[:-1]
                writer.writerow(
                    [
                        f"{coil.Xc:.6f}",
                        f"{coil.Yc:.6f}",
                        f"{coil.angle:.6f}",
                        coil.type,
                    ]
                )
        print(f"Wrote {len(self.coils)} coils to {path}")

    def build_ports(self, straight=True, center=True, r = None, rho= None):

        r = self.Stellerator_Radius if r is None else r
        rho = self.Stellerator_Radius if rho is None else rho

        if self.coils is None:
            raise ValueError("Coils have not been built yet. Call build_coils() first.")
        
        self.ports = []
        s_ports_center = []
        s_ports_straight = []

        # Separate into the 4 sections by position
        right_arc       = [c.angle for c in self.coils if c.type in self.center_types and c.Xc > 0]
        left_arc        = [c.angle for c in self.coils if c.type in self.center_types and c.Xc < 0]
        top_straight    = [c.Xc for c in self.coils if c.type in self.straight_types and c.Yc > 0]
        bottom_straight = [c.Xc for c in self.coils if c.type in self.straight_types and c.Yc < 0]
        dz_right = np.array([c.DZ for c in self.coils if c.type in self.center_types and c.Xc > 0])
        dz_left = np.array([c.DZ for c in self.coils if c.type in self.center_types and c.Xc < 0])
        dz_top = np.array([c.DZ for c in self.coils if c.type in self.straight_types and c.Yc > 0])
        dz_bottom = np.array([c.DZ for c in self.coils if c.type in self.straight_types and c.Yc < 0])

        if center:
            angle_right = (np.diff(np.array(right_arc))/2 + np.array(right_arc[:-1])) % 360
            angle_left = (np.diff(np.array(left_arc))/2 + np.array(left_arc[:-1])) % 360

            s_port_right = r * (np.radians(np.abs(np.diff(np.array(right_arc)))) - np.abs(np.arcsin(dz_right[:-1]/(2*r))) - np.abs(np.arcsin(dz_right[1:]/(2*r))))
            s_port_left = r * (np.radians(np.abs(np.diff(np.array(left_arc)))) - np.abs(np.arcsin(dz_left[:-1]/(2*r))) - np.abs(np.arcsin(dz_left[1:]/(2*r))))
            s_ports_center = np.array(s_port_right.tolist() + s_port_left.tolist())

            for angle in angle_right:
                x = self.Mirror_Length/2 + r * math.cos(math.radians(angle))
                y = r * math.sin(math.radians(angle))
                self.ports.append((x, y))

            for angle in angle_left:
                x = -self.Mirror_Length/2 + r * math.cos(math.radians(angle))
                y = r * math.sin(math.radians(angle))
                self.ports.append((x, y))

        if straight:
            xpos_top = np.diff(np.array(top_straight))/2 + np.array(top_straight[:-1])
            xpos_bottom = np.diff(np.array(bottom_straight))/2 + np.array(bottom_straight[:-1])
            for x in xpos_top:
                y = rho
                self.ports.append((x, y))
            for x in xpos_bottom:
                y = -rho
                self.ports.append((x, y))

        self.ports = np.array(self.ports)

        return np.array(angle_right.tolist() + angle_left.tolist()), np.abs(s_ports_center*100),  np.array(xpos_top)

