'''
Includes all useful functions and classes for the magnetic field.
'''
from pathlib import Path
import math

import matplotlib
matplotlib.use('TkAgg')

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import ellipk, ellipe
from scipy.signal import find_peaks

from matplotlib.patches import Rectangle

import matplotlib
matplotlib.use('TkAgg')

_COIL_MODELS = None
MU0 = 4 * math.pi * 1e-7
AXIS_SAMPLES_PER_SEGMENT = 25
GRID_RES_X = 80
GRID_RES_Y = 80
X_RANGE = (-2, 2)
Y_RANGE = (-2, 2)
CU_DENSITY = 8960  #kg/m^3
COIL_MODEL_FILE = 'coil_models/coil_model.csv'


def get_coil_models():
    """Load coil geometry templates from coil-model.csv once per process."""
    global _COIL_MODELS
    if _COIL_MODELS is None:
        model_path = Path(__file__).parent / COIL_MODEL_FILE
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
                'nr': int(row.get('nr', row['Nr'])),
                'nz': int(row.get('nz', row['Nz'])),
                'parallel': int(row.get('parallel', row['parallel'])),
                'partitions': int(row.get('partitions', row['partitions']))
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
        self.nr = max(1, int(model.get('nr', self.Nr)))
        self.nz = max(1, int(model.get('nz', self.Nz)))
        self.current = float(model.get('current', 0.0))
        self._radial_filaments = self._build_midpoints(self.ID / 2, self.OD / 2, self.nr)
        self._axial_filaments = self._build_midpoints(-self.DZ / 2, self.DZ / 2, self.nz)
        theta = math.radians(self.angle)
        self._basis_e1 = np.array([math.cos(theta), math.sin(theta), 0.0])
        self._basis_e2 = np.array([0.0, 0.0, 1.0])
        self._basis_e3 = np.cross(self._basis_e1, self._basis_e2)
        self._basis = np.stack([self._basis_e1, self._basis_e2, self._basis_e3], axis=1)
        self._basis_T = self._basis.T
        self.parallel =  False if int(model['parallel']) == 0 else True
        self.parallel_partitions = int(model['partitions'])

    @staticmethod
    def _build_midpoints(start, stop, count):
        if count <= 1:
            return np.array([(start + stop) / 2], dtype=float)
        edges = np.linspace(start, stop, count + 1)
        return 0.5 * (edges[1:] + edges[:-1])

    def draw(self, ax, color='C0', linewidth=1.5, show_center=False,
             label=None, add_to_legend=False, length_units='m'):
        """Plot the rectangular coil outline as in draw.py.
        
        Parameters
        ----------
        label : str or None
            Legend label for this coil type (e.g. 'L2'). Only used if add_to_legend=True.
        add_to_legend : bool
            If True, registers exactly one legend entry per unique label per Axes."""
    
        dr = (self.OD - self.ID) / 2
        dz = self.DZ
        r = self.ID / 2
        xc = self.Xc
        yc = self.Yc

        if length_units == 'cm':
            dr = dr * 100
            dz = dz * 100
            r = r * 100
            xc = xc * 100
            yc = yc * 100

        xr = xc + r
        yr = yc - dz / 2
        xy = (xr, yr)

        # Only attach label once per axes to avoid duplicate legend entries
        rect_label = None
        if add_to_legend and label:
            if not hasattr(ax, "_coil_legend_labels"):
                ax._coil_legend_labels = set()
            if label not in ax._coil_legend_labels:
                rect_label = label
                ax._coil_legend_labels.add(label)

        rect = Rectangle(
            xy,
            dr,
            dz,
            angle=self.angle,
            rotation_point=(xc, yc),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=rect_label,  # <-- minimal addition
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
        if self.parallel == True and self.parallel_partitions != 0:
            total_current = total_current / self.parallel_partitions
        num_filaments = len(self._radial_filaments) * len(self._axial_filaments)
        turns_total = max(1, self.Nr * self.Nz)
        weight = turns_total / num_filaments
        filament_current = total_current * weight
        basis = self._basis
        basis_T = self._basis_T

        for radius in self._radial_filaments:
            for z_offset in self._axial_filaments:
                shifted_center = center + z_offset * self._basis_e2
                rel_global = pts - shifted_center
                rel_local = rel_global @ basis
                rho = np.hypot(rel_local[:, 0], rel_local[:, 1])
                z_local = rel_local[:, 2]
                Br, Bz = self._loop_field(radius, rho, z_local, filament_current)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cos_phi = np.divide(rel_local[:, 0], rho, out=np.zeros_like(rel_local[:, 0]), where=rho != 0)
                    sin_phi = np.divide(rel_local[:, 1], rho, out=np.zeros_like(rel_local[:, 1]), where=rho != 0)
                B_local = np.zeros_like(rel_local)
                B_local[:, 0] = Br * cos_phi
                B_local[:, 1] = Br * sin_phi
                B_local[:, 2] = Bz
                B_total += B_local @ basis_T

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

    def get_length(self, print_length=True):

        '''
        Calculate the coil length.
        '''

        D_avg = (self.ID + self.OD)/2
        L_pancake = D_avg * np.pi * self.Nz

        L_coil = L_pancake * self.Nr

        if print_length == True:
            print(f"  {self.type}: {L_coil:.2f} m") 
        return L_coil
    
    def get_volume(self, print_volume=True):

        '''
        Calculate the maximum coil volume.
        '''

        V_coil = self.DZ * np.pi * ((self.OD/2)**2 - (self.ID/2)**2)

        if print_volume == True:
            print(f"  {self.type}: {V_coil:.4f} m^3")
        return V_coil


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


###########################################
#    Useful functions for other files.    #
###########################################

def print_total_coil_params(coils):
    '''Prints total coil length, volume, and mass.'''
    num_coils = len(coils)
    coil_lengths = np.zeros(num_coils)
    coil_volumes = np.zeros(num_coils)
    for i in range(num_coils):
        coil_lengths[i] = coils[i].get_length(False)
        coil_volumes[i] = coils[i].get_volume(False)
    total_length = np.sum(coil_lengths)
    total_volume = np.sum(coil_volumes)
    total_mass = total_volume * CU_DENSITY
    print(f"\n  Total Coil Length: {total_length:.2f} m")
    print(f"  Total Coil Volume: {total_volume:.2f} m^3")
    print(f"  Total Coil Mass: {total_mass:.2f} kg")


def get_coil_info(test_file, interpolate=True, L=1.5, R=0.5):
    '''Returns a list of coils and the magnetic axis
       path as an array of points in the xy-plane.'''
    df = pd.read_csv(test_file).dropna(how='all')
    df.columns = df.columns.str.strip()  # fix headers
    coils = [Coil(**row) for row in df.to_dict('records')]
    axis_xy = df[['Xc', 'Yc']].to_numpy()
    axis_path = axis_xy
    if interpolate == True:
        axis_path = interpolate_axis(axis_xy, AXIS_SAMPLES_PER_SEGMENT)
    else:
        axis_samples_per_segment = 5
        axis_xy = np.zeros((56, 2))
        for i in range(9):
            x = -L/2 + L/8*i
            axis_xy[i,:] = np.array([x, R])
            axis_xy[i+28,:] = np.array([-x, -R])
        for i in range(19):
            angle = np.pi/2 - np.pi/20 - np.pi/20*i
            x = L/2 + R*np.cos(angle)
            y = R*np.sin(angle)
            axis_xy[i+9,:] = np.array([x, y])
            axis_xy[i+37,:] = np.array([-x, -y])
        axis_path = interpolate_axis(axis_xy, axis_samples_per_segment)

    return coils, axis_path


################
#     Plots    #
################

def contour_plot(fig, ax, coils, axis_path, show_labels=True, length_units='m', field_units='T', levels=32, extend='neither', coil_colors={}):
    '''Creates a contour plot of |B| on the xy-plane.
       Side effects on the inputs fig and ax.'''
    x_range = X_RANGE
    y_range = Y_RANGE
    if length_units == 'cm':
        x_range = tuple(x * 100 for x in X_RANGE)
        y_range = tuple(y * 100 for y in Y_RANGE)

    xs, ys, BX_full, BY_full, Bplane = planar_field_grid(coils, X_RANGE, Y_RANGE)
    if length_units == 'cm':
        xs = xs * 100
        ys = ys * 100
    if field_units == 'G':
        Bplane = Bplane * 10000

    contour = ax.contourf(xs, ys, Bplane, levels=levels, cmap='jet', alpha=0.7, extend=extend)
    fig.colorbar(contour, ax=ax, label=f'|B| in plane ({field_units})')

    for coil in coils:
        coil.draw(
        ax,
        color=coil_colors.get(coil.type, 'black'),
        linewidth=1.5,
        label=f"{coil.type} Coil: {coil.current:>1,.0f} A",
        add_to_legend=True,
        length_units=length_units
    )

    if length_units == 'm':
        ax.plot(axis_path[:, 0], axis_path[:, 1], 'w-', lw=2, label='Magnetic axis')
    elif length_units == 'cm':
        ax.plot(axis_path[:, 0]*100, axis_path[:, 1]*100, 'w-', lw=2, label='Magnetic axis')
    
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect('equal', adjustable='box')
    if show_labels == True:
        ax.set_title('Planar |B| with coil outlines')
        ax.set_xlabel(f'X ({length_units})')
        ax.set_ylabel(f'Y ({length_units})')
    ax.grid(True)


def axis_field_plot(ax, coils, axis_path, show_labels=True, length_units='m', field_units='T', color='blue', label=''):
    '''Returns |B| along the magnetic axis.
       Side effects on the input ax.'''
    axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
    B_total = np.zeros((len(axis_points), 3))
    for coil in coils:
        B_total += coil.magnetic_field(axis_points)
    B_mag = np.linalg.norm(B_total, axis=1)
    s_coord = cumulative_distance(axis_path)

    if length_units == 'cm':
        s_coord = s_coord * 100
    if field_units == 'G':
        B_mag = B_mag * 10000

    if len(s_coord) == 1:
        ax.plot(s_coord, B_mag, 'o', color=color, label=label)
    else:
        ax.plot(s_coord, B_mag, lw=2, color=color, label=label)

    if show_labels == True:
        ax.set_xlabel(f'Axis distance s ({length_units})')
        ax.set_ylabel(f'|B| ({field_units})')
        ax.set_title('|B| along magnetic axis: Rm = {:.2f}'.format(B_mag.max()/B_mag.min()))
    ax.grid(True)
    ax.set_ylim(bottom=0)
    return B_mag


def field_streamplot(fig, ax, coils, show_labels=True, length_units='m', field_units='T', color=''):
    '''Returns a streamplot over the domain.
       Side effects on the inputs fig and ax.'''
    x_range = X_RANGE
    y_range = Y_RANGE
    if length_units == 'cm':
        x_range = tuple(x * 100 for x in X_RANGE)
        y_range = tuple(y * 100 for y in Y_RANGE)

    xs_stream, ys_stream, BX_stream, BY_stream, _ = planar_field_grid(
        coils, X_RANGE, Y_RANGE, nx=GRID_RES_X * 3, ny=GRID_RES_Y * 3
    )

    if length_units == 'cm':
        xs_stream = xs_stream * 100
        ys_stream = ys_stream * 100
    if field_units == 'G':
        BX_stream = BX_stream * 10000
        BY_stream = BY_stream * 10000
        
    speed_stream = np.hypot(BX_stream, BY_stream)
    if color == '':
        stream = ax.streamplot(
            xs_stream, ys_stream, BX_stream, BY_stream, color=speed_stream, cmap='jet', density=2.0
        )
    else:
        stream = ax.streamplot(
            xs_stream, ys_stream, BX_stream, BY_stream, color=color, density=0.5
        )

    if color == '':
        fig.colorbar(stream.lines, ax=ax, label=f'|B| in plane ({field_units})')
        
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect('equal', adjustable='box')
    if show_labels == True:
        ax.set_title('In-plane B field lines (full domain)')
        ax.set_xlabel(f'X ({length_units})')
        ax.set_ylabel(f'Y ({length_units})')
    ax.grid(True)


def axis_field_plot_by_coil(ax, coils, axis_path, show_labels=True, length_units='m', field_units='T', coil_colors={}):
    '''Returns |B| along the magnetic axis, split into colors based on coil type.
       Side effects on the input ax.'''
    axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
    s_coord = cumulative_distance(axis_path)

    # group coils by type
    coils_by_type = {}
    for c in coils:
        coils_by_type.setdefault(c.type, []).append(c)

    if length_units == 'cm':
        s_coord = s_coord * 100

    # compute and plot |B| contribution from each type
    for ctype, c_list in coils_by_type.items():
        B_type = np.zeros((len(axis_points), 3))
        for c in c_list:
            B_type += c.magnetic_field(axis_points)

        B_type_mag = np.linalg.norm(B_type, axis=1)

        if field_units == 'G':
            B_type_mag = B_type_mag * 10000

        if len(s_coord) == 1:
            ax.plot(s_coord, B_type_mag, 'o', color=coil_colors.get(ctype, 'black'), label=ctype)
        else:
            ax.plot(
                s_coord,
                B_type_mag,
                lw=2,
                color=coil_colors.get(ctype, 'black'),
                label=ctype
            )

    if show_labels == True:
        ax.set_title("|B| on axis by coil type")
        ax.set_xlabel(f"Axis distance s ({length_units})")
        ax.set_ylabel(f"|B| ({field_units})")

    ax.grid(True)
    ax.legend()