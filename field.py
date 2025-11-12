from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

from matplotlib.patches import Rectangle

_COIL_MODELS = None
MU0 = 4 * math.pi * 1e-7
AXIS_SAMPLES_PER_SEGMENT = 25
GRID_RES_X = 80
GRID_RES_Y = 80
PLANE_HALF_WIDTH = 2.0  # meters around the axis center for 2D field view


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
                'current': float(row.get('current', 0.0)),
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
        self.current = float(model.get('current', 0.0))
        self._radial_filaments = self._build_midpoints(self.ID / 2, self.OD / 2, self.Nr)
        self._axial_filaments = self._build_midpoints(-self.DZ / 2, self.DZ / 2, self.Nz)

    @staticmethod
    def _build_midpoints(start, stop, count):
        if count <= 1:
            return np.array([(start + stop) / 2], dtype=float)
        edges = np.linspace(start, stop, count + 1)
        return 0.5 * (edges[1:] + edges[:-1])

    def draw(self, ax, color='C0', linewidth=1.5, show_center=False):
        """Plot the rectangular coil outline as in draw.py."""
        dr = (self.OD - self.ID) / 2
        dz = self.DZ
        r = self.ID / 2
        xc = self.Xc
        yc = self.Yc

        xr = xc + r
        yr = yc - dz / 2
        xy = (xr, yr)

        rect = Rectangle(
            xy,
            dr,
            dz,
            angle=self.angle,
            rotation_point=(xc, yc),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)

        rect_opposite = Rectangle(
            xy,
            dr,
            dz,
            angle=self.angle + 180,
            rotation_point=(xc, yc),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle='--',
        )
        ax.add_patch(rect_opposite)

        if show_center:
            ax.plot(xc, yc, marker='o', color=color, ms=3)

    def magnetic_field(self, points, current=None):
        """
        Compute the magnetic field vector at one or more 3D points.

        Parameters
        ----------
        points : array_like
            Single 3-vector or Nx3 array of evaluation points in meters.
        current : float
            Total coil current in Amperes.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 3) containing (Bx, By, Bz) in Tesla.
        """
        pts = np.asarray(points, dtype=float)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts[None, :]
        if pts.shape[1] != 3:
            raise ValueError("points array must have shape (N, 3)")

        B_total = np.zeros_like(pts)
        center = np.array([self.Xc, self.Yc, 0.0], dtype=float)
        total_current = self.current if current is None else current
        filament_current = total_current  # each filament represents one turn carrying full current

        for radius in self._radial_filaments:
            for z_offset in self._axial_filaments:
                rel = pts - (center + np.array([0.0, 0.0, z_offset]))
                rho = np.hypot(rel[:, 0], rel[:, 1])
                z = rel[:, 2]
                Br, Bz = self._loop_field(radius, rho, z, filament_current)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cos_phi = np.divide(rel[:, 0], rho, out=np.zeros_like(rel[:, 0]), where=rho != 0)
                    sin_phi = np.divide(rel[:, 1], rho, out=np.zeros_like(rel[:, 1]), where=rho != 0)
                B_total[:, 0] += Br * cos_phi
                B_total[:, 1] += Br * sin_phi
                B_total[:, 2] += Bz

        return B_total[0] if single_point else B_total

    @staticmethod
    def _loop_field(radius, rho, z, current):
        """Return (Br,Bz) for a thin circular loop with elliptic integrals."""
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)
        denom = (radius + rho) ** 2 + z ** 2
        k_sq = np.where(denom == 0, 0.0, 4 * radius * rho / denom)
        K = ellipk(k_sq)
        E = ellipe(k_sq)
        common = MU0 * current / (2 * math.pi * np.sqrt(denom))
        denom2 = (radius - rho) ** 2 + z ** 2
        denom2 = np.where(denom2 == 0, np.finfo(float).eps, denom2)
        Br = np.zeros_like(rho)
        nz = rho != 0
        Br[nz] = common[nz] * z[nz] / rho[nz] * (
            -K[nz] + (radius ** 2 + rho[nz] ** 2 + z[nz] ** 2) / denom2[nz] * E[nz]
        )
        Bz = common * (K + (radius ** 2 - rho ** 2 - z ** 2) / denom2 * E)
        on_axis = rho == 0
        if np.any(on_axis):
            Bz[on_axis] = MU0 * current * radius ** 2 / (2 * (radius ** 2 + z[on_axis] ** 2) ** 1.5)
        return Br, Bz


def interpolate_axis(points, samples_per_segment=AXIS_SAMPLES_PER_SEGMENT):
    """Linearly interpolate between COM points to approximate the magnetic axis."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        for t in ts:
            segments.append(start + t * (end - start))
    segments.append(points[-1])
    return np.vstack(segments)


def cumulative_distance(points):
    """Return cumulative path length along a polyline."""
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def planar_field_grid(coils, x_range, y_range, nx=GRID_RES_X, ny=GRID_RES_Y, z_plane=0.0):
    """Compute B-field vectors on a uniform XY grid at a fixed z-plane."""
    xs = np.linspace(*x_range, nx)
    ys = np.linspace(*y_range, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z_plane)])
    B_total = np.zeros((len(pts), 3))
    for coil in coils:
        B_total += coil.magnetic_field(pts)
    Bx = B_total[:, 0].reshape(ny, nx)
    By = B_total[:, 1].reshape(ny, nx)
    Bmag = np.linalg.norm(B_total, axis=1).reshape(ny, nx)
    return xs, ys, Bx, By, Bmag

# main
import sys
fin = sys.argv[1]
df = pd.read_csv(fin).dropna(how='all')
df.columns = df.columns.str.strip()  # fix headers
coils = [Coil(**row) for row in df.to_dict('records')]

fig,axs = plt.subplots()

axis_xy = df[['Xc', 'Yc']].to_numpy()
axis_path = interpolate_axis(axis_xy, AXIS_SAMPLES_PER_SEGMENT)
axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
B_total = np.zeros((len(axis_points), 3))
for coil in coils:
    B_total += coil.magnetic_field(axis_points)
B_mag = np.linalg.norm(B_total, axis=1)
s_coord = cumulative_distance(axis_path)

# Contour plot of |B| in axis neighborhood
axis_x = axis_path[:, 0]
axis_y = axis_path[:, 1]
axis_x_mid = 0.5 * (axis_x.min() + axis_x.max())
axis_y_mid = 0.5 * (axis_y.min() + axis_y.max())
x_range = (axis_x_mid - PLANE_HALF_WIDTH, axis_x_mid + PLANE_HALF_WIDTH)
y_range = (axis_y_mid - PLANE_HALF_WIDTH, axis_y_mid + PLANE_HALF_WIDTH)
xs, ys, _, _, Bplane = planar_field_grid(coils, x_range, y_range)
contour = axs.contourf(xs, ys, Bplane, levels=20, cmap='jet', alpha=0.7)
plt.colorbar(contour, ax=axs, label='|B| in plane (T)')

for c in coils:
    c.draw(axs, color='black', linewidth=1.0)

axs.set_xlim(-2, 2)
axs.set_ylim(-2, 2)
axs.grid(True)

fig2, ax2 = plt.subplots()
ax2.plot(s_coord, B_mag, lw=2)
ax2.set_xlabel('Axis distance s (m)')
ax2.set_ylabel('|B| (T)')
ax2.set_title('Magnetic field magnitude along axis')
ax2.grid(True)

plt.show()
