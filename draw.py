import numpy as np
import matplotlib.pyplot as plt

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

        # get corner for patches.Rectangle object
        xr = xc + r
        yr = yc - dz/2
        xy = (xr,yr)

        # draw patch
        print(xy, dz, dr)
        rect = Rectangle(xy, dr, dz, angle=self.angle, rotation_point=(xc,yc))
        ax.add_patch(rect)

        plt.plot(xc,yc,'C0.')
        #plt.plot(xi,yi,'C1.')

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

plt.xlim(0,2)
plt.ylim(0,2)
plt.grid()
plt.show()

import pdb
pdb.set_trace()
