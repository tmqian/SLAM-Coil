from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

_COIL_MODELS = None


def get_coil_models():
    """Load coil geometry templates from coil-model.csv once per process."""
    global _COIL_MODELS
    if _COIL_MODELS is None:
        model_path = Path(__file__).with_name('coil-model.csv')
        df = pd.read_csv(model_path).dropna(how='all')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['type'])
        df['type'] = df['type'].apply(lambda x: x.strip().upper() if isinstance(x, str) else x)
        _COIL_MODELS = {
            row['type']: {
                'ID': float(row['ID']),
                'OD': float(row['OD']),
                'DZ': float(row['DZ']),
                'Nr': int(row['Nr']),
                'Nz': int(row['Nz']),
            }
            for _, row in df.iterrows()
        }
    return _COIL_MODELS

class Coil:

    def __init__(self, Xc=1, Yc=1, angle=90, type=None):
        '''
        (Xc,Yc) is the COM of the coil
        angle is of the plane of the coil in degrees, 0 is +xaxis
        type selects geometry defined in coil-model.csv (e.g. "OM", "L2")
        '''

        self.Xc = Xc
        self.Yc = Yc

        self.angle = angle
        if not isinstance(type, str):
            raise ValueError("Coil type string is required (e.g. 'OM', 'L2').")
        self.type = type.strip().upper()

        models = get_coil_models()
        if self.type not in models:
            raise ValueError(f"Unknown coil type '{self.type}'. Add it to coil-model.csv.")
        model = models[self.type]

        self.ID = float(model['ID'])
        self.OD = float(model['OD'])
        self.DZ = float(model['DZ'])
        self.Nr = int(model['Nr'])
        self.Nz = int(model['Nz'])

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

        # draw main cross section
        print(xy, dz, dr)
        rect = Rectangle(xy, dr, dz, angle=self.angle, rotation_point=(xc,yc))
        ax.add_patch(rect)

        # draw second cross section
        rect = Rectangle(xy, dr, dz, angle=self.angle+180, rotation_point=(xc,yc), alpha=0.5)
        ax.add_patch(rect)

        plt.plot(xc,yc,'C0.')
        #plt.plot(xi,yi,'C1.')

# main
import sys
fin = sys.argv[1]
df = pd.read_csv(fin).dropna(how='all')
df.columns = df.columns.str.strip()  # fix headers
coils = [Coil(**row) for row in df.to_dict('records')]

fig,axs = plt.subplots()
for c in coils:
    c.draw(axs)

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.grid()
plt.show()
