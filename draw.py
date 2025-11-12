import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

class Coil:

    def __init__(self, Xc=1,Yc=1, angle=90, ID=0.1, OD=0.2, DZ=0.05, Nr=1, Nz=1):
        '''
        (Xc,Yc) is the COM of the coil
        ID,OD are the inner and outer diameters
        DZ is the axial width of the coil
        angle is of the plane of the coil in degrees, 0 is +xaxis
        '''

        self.Xc = Xc
        self.Yc = Yc

        self.angle = angle
        self.ID = ID
        self.OD = OD
        self.DZ = DZ

    def draw(self, ax):

        # class matplotlib.patches.Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)

        # get rectangular dimensions
        dr = (self.OD - self.ID)/2
        dz = self.DZ

        # get center of coil and inner radius
        r = self.ID/2
        xc = self.Xc
        yc = self.Yc

        # get midpoint of inner radius
        phi = np.deg2rad(self.angle)
        xi = xc + r*np.cos(phi)
        yi = yc + r*np.sin(phi)

        # get corner for patches.Rectangle object
        xr = xi + dz*np.cos(phi)
        yr = yi + dz*np.sin(phi)
        xy = (xr,yr)

        # draw patch
        rect = Rectangle(xy, dz, dr, angle=self.angle)
        ax.add_patch(rect)

# main
import sys
fin = sys.argv[1]

import pandas as pd
df = pd.read_csv(fin).dropna(how='all')
df.columns = df.columns.str.strip()  # fix headers
coils = [Coil(**row) for row in df.to_dict('records')]

fig,axs = plt.subplots()
for c in coils:
    c.draw(axs)

import pdb
pdb.set_trace()
