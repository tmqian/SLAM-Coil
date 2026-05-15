import sys

import numpy as np
import pandas as pd

from field import *

fin = sys.argv[1]
df = pd.read_csv(fin).dropna(how='all')
df.columns = df.columns.str.strip()  # fix headers
coils = [Coil(**row) for row in df.to_dict('records')]

for c in coils:
    c.get_length()

axis_xy = df[['Xc', 'Yc']].to_numpy()
axis_path = interpolate_axis(axis_xy, AXIS_SAMPLES_PER_SEGMENT)
axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
B_total = np.zeros((len(axis_points), 3))
for coil in coils:
    B_total += coil.magnetic_field(axis_points)
B_mag = np.linalg.norm(B_total, axis=1)

