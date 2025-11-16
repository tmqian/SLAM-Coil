# ---------- Just like field.py, but vectorized coils so we can quickly trace field lines ----------
from pathlib import Path
import math
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from jaxellip import ellipk, ellipe
import jax
from matplotlib.patches import Rectangle


# START INPUTS # 
_COIL_MODELS = None
MU0 = 4 * math.pi * 1e-7
AXIS_SAMPLES_PER_SEGMENT = 25
GRID_RES_X = 80
GRID_RES_Y = 80
PLANE_HALF_WIDTH = 2.0  # meters around the axis center for 2D field view

### field line trace 
# a line 
N = 9 # two on both side will be masked to show termination 
#y0 = np.linspace(.3,.7,N) # trace starts to take longer, too close to coils
y0 = np.linspace(.35,.65,N) 
z0 =  np.ones_like(y0) *0
x0 =  np.ones_like(y0) *0

# END INPUTS #



# ================================================================
#   Full JAX-Compatible Coil Field Solver
#   No for loops, no indexing, vectorized over all filaments
#   Compatible with jit, vmap, grad
# ================================================================

# ================================================================
#   Load Coil Templates (outside JIT; runs once)
# ================================================================

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
                'nr': int(row.get('nr', row['Nr'])),
                'nz': int(row.get('nz', row['Nz'])),
            }
            for _, row in df.iterrows()
        }
    return _COIL_MODELS


# ================================================================
#   JAX midpoint helper, used when initilizing Coils 
# ================================================================

def build_midpoints(start, stop, count):
    start = jnp.asarray(start, float)
    stop = jnp.asarray(stop, float)
    if count <= 1:
        return jnp.array([(start + stop) / 2])
    edges = jnp.linspace(start, stop, count + 1)
    return 0.5 * (edges[1:] + edges[:-1])


# ================================================================
#   JAX Loop Field (Br, Bz) for thin circular loop
# ================================================================

def loop_field(radius, rho, z, current):
    radius = jnp.asarray(radius, float)
    rho = jnp.asarray(rho, float)
    z = jnp.asarray(z, float)

    denom = (radius + rho)**2 + z**2
    k_sq = jnp.where(denom == 0, 0.0, 4 * radius * rho / denom)

    K = ellipk(k_sq)
    E = ellipe(k_sq)
    common = MU0 * current / (2 * math.pi * jnp.sqrt(denom))

    denom2 = (radius - rho)**2 + z**2
    denom2 = jnp.where(denom2 == 0, jnp.finfo(float).eps, denom2)

    Br = common * z / jnp.where(rho == 0, 1, rho) * (
        -K + (radius**2 + rho**2 + z**2) / denom2 * E
    )

    Bz = common * (K + (radius**2 - rho**2 - z**2)/denom2 * E)

    # On-axis fix
    Bz = jnp.where(
        rho == 0,
        MU0 * current * radius**2 / (2 * (radius**2 + z**2)**1.5),
        Bz,
    )

    return Br, Bz


# ================================================================
#   Coil Class with (PyTree)
# ================================================================

@jax.tree_util.register_pytree_node_class
class Coil:

    def __init__(self, Xc, Yc, angle, type):
        models = get_coil_models()
        if type.upper() not in models:
            raise ValueError(f"Unknown coil type: {type}")
        m = models[type.upper()]

        self.Xc = float(Xc)
        self.Yc = float(Yc)
        self.angle = float(angle)
        self.type = type.upper()

        self.ID = m["ID"]
        self.OD = m["OD"]
        self.DZ = m["DZ"]
        self.Nr = m["Nr"]
        self.Nz = m["Nz"]
        self.nr = m["nr"]
        self.nz = m["nz"]
        self.current = m["current"]

        # midpoints
        self.radial_filaments = build_midpoints(self.ID/2, self.OD/2, self.nr)
        self.axial_filaments  = build_midpoints(-self.DZ/2, self.DZ/2, self.nz)

        # basis vectors
        theta = math.radians(self.angle)
        e1 = jnp.array([math.cos(theta), math.sin(theta), 0.0])
        e2 = jnp.array([0.0, 0.0, 1.0])
        e3 = jnp.cross(e1, e2)

        self.basis = jnp.stack([e1, e2, e3], axis=1)
        self.basis_T = self.basis.T


    # -------------------------
    # PyTree: flatten/unflatten
    # -------------------------

    def tree_flatten(self):
        children = (
            jnp.array(self.Xc),
            jnp.array(self.Yc),
            jnp.array(self.angle),
            jnp.array(self.ID),
            jnp.array(self.OD),
            jnp.array(self.DZ),
            jnp.array(self.Nr),
            jnp.array(self.Nz),
            jnp.array(self.nr),
            jnp.array(self.nz),
            jnp.array(self.current),
            self.radial_filaments,
            self.axial_filaments,
            self.basis,
            self.basis_T,
        )
        aux = self.type
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            Xc, Yc, angle,
            ID, OD, DZ,
            Nr, Nz, nr, nz,
            current,
            radial_filaments,
            axial_filaments,
            basis,
            basis_T,
        ) = children

        # We reconstruct manually
        obj = cls(float(Xc), float(Yc), float(angle), aux)
        obj.radial_filaments = radial_filaments
        obj.axial_filaments  = axial_filaments
        obj.basis = basis
        obj.basis_T = basis_T
        return obj


    # ===========================================================
    #   JAX-vectorized Magnetic Field Calculation
    # ===========================================================

    def magnetic_field(self, points, current=None):
        pts = jnp.asarray(points, float)
        if pts.ndim == 1:
            pts = pts[None, :]

        center = jnp.array([self.Xc, self.Yc, 0.0])
        basis = self.basis
        basis_T = self.basis_T

        I = self.current if current is None else float(current)

        R = self.radial_filaments
        Z = self.axial_filaments

        # Cartesian product -> all filament pairs
        Rf, Zf = jnp.meshgrid(R, Z, indexing="ij")
        Rf = Rf.ravel()
        Zf = Zf.ravel()

        turns_total = max(1, self.Nr * self.Nz)
        weight = turns_total / Rf.size
        I_fil = I * weight

        def field_of_filament(radius, zoff):
            shifted = center + zoff * basis[:,1]
            rel = pts - shifted
            rel_local = rel @ basis

            rho = jnp.hypot(rel_local[:,0], rel_local[:,1])
            zloc = rel_local[:,2]

            Br, Bz = loop_field(radius, rho, zloc, I_fil)

            cos_phi = jnp.where(rho > 0, rel_local[:,0]/rho, 0)
            sin_phi = jnp.where(rho > 0, rel_local[:,1]/rho, 0)

            B_loc = jnp.stack([Br*cos_phi, Br*sin_phi, Bz], axis=1)
            return B_loc @ basis_T

        B_all = jax.vmap(field_of_filament)(Rf, Zf)   # (F, N, 3)
        B_total = jnp.sum(B_all, axis=0)              # (N, 3)

        return B_total[0] if points.ndim == 1 else B_total

    # 
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
    
def interpolate_axis(points, samples_per_segment=AXIS_SAMPLES_PER_SEGMENT):
    """Linearly interpolate between COM points to approximate the magnetic axis."""
    points = jnp.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        ts = jnp.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        for t in ts:
            segments.append(start + t * (end - start))
    segments.append(points[-1])
    return jnp.vstack(segments)


def cumulative_distance(points):
    """Return cumulative path length along a polyline."""
    diffs = jnp.diff(points, axis=0)
    seg_lengths = jnp.linalg.norm(diffs, axis=1)
    return jnp.concatenate((jnp.array([0.0]), jnp.cumsum(seg_lengths)))


def planar_field_grid(coils, x_range, y_range, nx=GRID_RES_X, ny=GRID_RES_Y, z_plane=0.0):
    """Compute B-field vectors on a uniform XY grid at a fixed z-plane."""
    xs = jnp.linspace(*x_range, nx)
    ys = jnp.linspace(*y_range, ny)
    XX, YY = jnp.meshgrid(xs, ys)
    pts = jnp.column_stack([XX.ravel(), YY.ravel(), jnp.full(XX.size, z_plane)])
    B_total = jnp.zeros((len(pts), 3))
    for coil in coils:
        B_total += coil.magnetic_field(pts)
    Bx = B_total[:, 0].reshape(ny, nx)
    By = B_total[:, 1].reshape(ny, nx)
    Bmag = jnp.linalg.norm(B_total, axis=1).reshape(ny, nx)
    return xs, ys, Bx, By, Bmag


# -------------- and a field line tracer -----------

from diffrax import (
    ODETerm,
    SaveAt,
    PIDController,
    diffeqsolve,
    Tsit5,
    Event,     # IMPORTANT: diffrax 0.7.0 event API
)


def field_line_trace_xyz(
    x0, y0, z0,
    *,
    field,                         # object with .magnetic_field(pts) -> (N,3)
    params=None,
    source_grid=None,
    s_total=1.0,
    direction="forward",
    rtol=1e-8, atol=1e-8,
    min_step_size=1e-8,
    max_steps_per_meter=20000,
    bounds_X=(-jnp.inf, jnp.inf),
    bounds_Y=(-jnp.inf, jnp.inf),
    bounds_Z=(-jnp.inf, jnp.inf),
    chunk_size=None,
    eps_B=0.0,
    solver=Tsit5(),
):
    """
    Trace magnetic field lines using arc-length parametrization:

         dX/ds = Bx/|B|
         dY/ds = By/|B|
         dZ/ds = Bz/|B|

    Works with batch seeds: x0, y0, z0 may be scalars or arrays of same shape S.
    Returns:
        ts   : (nsteps,)
        path : (nsteps, *S, 3)
    """

    # ---------- batch seeds ----------
    x0 = jnp.asarray(x0)
    y0 = jnp.asarray(y0)
    z0 = jnp.asarray(z0)
    assert x0.shape == y0.shape == z0.shape

    S = x0.shape         # logical seed shape
    N = x0.size          # flatten → N seeds

    # Component-first state: (3, N)
    y_init = jnp.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=0)


    # ---------- ODE RHS ----------
    def odefun(s, state, args):
        X, Y, Z = state  # each (N,)
        pts = jnp.stack([X, Y, Z], axis=-1)   # (N,3)

        #B = field.magnetic_field(pts)        # (N,3)
        #B = field(pts)        # (N,3)
        B = magnetic_field_from_coils(field,pts)
        
        Bn = jnp.linalg.norm(B, axis=-1)     # (N,)

        if eps_B > 0:
            Bn = jnp.maximum(Bn, eps_B)
        else:
            Bn = jnp.where(Bn == 0.0, 1.0, Bn)

        dX = B[:, 0] / Bn
        dY = B[:, 1] / Bn
        dZ = B[:, 2] / Bn

        return jnp.stack([dX, dY, dZ], axis=0)  # (3,N)


    term = ODETerm(odefun)
    stepsize = PIDController(rtol=rtol, atol=atol, dtmin=jnp.abs(min_step_size))
    save_steps = SaveAt(steps=True)


    # ---------- termination event (bounds) ----------
    # DIFFRAX 0.7.0 REQUIRES: return scalar float, not bool.
    def cond_fn(t, y, args, **kw):
        X, Y, Z = y
        out = (
            (X < bounds_X[0]) | (X > bounds_X[1]) |
            (Y < bounds_Y[0]) | (Y > bounds_Y[1]) |
            (Z < bounds_Z[0]) | (Z > bounds_Z[1])
        )
        return jnp.any(out).astype(float)

    ev = Event(cond_fn=cond_fn)


    # ---------- single-direction solve ----------
    def _solve(sign):
        s_mag = float(abs(s_total))

        t0 = 0.0
        t1 = float(sign) * s_mag
        dt0 = float(sign) * float(jnp.abs(min_step_size))

        max_steps = int(max(1.0, s_mag) * max_steps_per_meter)

        sol = diffeqsolve(
            term,
            solver,
            t0=t0, t1=t1,
            y0=y_init,
            dt0=dt0,
            stepsize_controller=stepsize,
            saveat=save_steps,
            max_steps=max_steps,
            event=ev,   # IMPORTANT for diffrax 0.7.0
        )

        ts = sol.ts                     # (nsteps,)
        ys = sol.ys                     # (nsteps, 3, N)

        # reshape back to (nsteps, *S, 3)
        path = jnp.moveaxis(ys, 1, -1).reshape(len(ts), *S, 3)
        return ts, path


    # ---------- direction dispatch ----------
    d = direction.lower()

    if d == "forward":
        return _solve(+1)

    elif d == "backward":
        return _solve(-1)

    elif d == "both":
        ts_b, pb = _solve(-1)
        ts_f, pf = _solve(+1)

        # reverse backward half, drop duplicate start point, make ts negative
        ts = jnp.concatenate([-ts_b[::-1][:-1], ts_f])
        path = jnp.concatenate([pb[::-1][:-1], pf], axis=0)
        return ts, path

    else:
        raise ValueError("direction must be 'forward', 'backward', or 'both'")

## This is a simple for loop sum. We could get huge speed ups if we summed the field 
def magnetic_field_from_coils(coils, points):
    """
    Evaluate total magnetic field from one coil or a list/tuple of coils.

    coils : Coil or list/tuple of Coil
    points : (N,3)

    returns : (N,3)
    """
    points = jnp.asarray(points)

    if isinstance(coils, Coil):
        return coils.magnetic_field(points)

    # sum over multiple coils
    B_total = jnp.zeros((points.shape[0], 3))

    for coil in coils:
        B_total = B_total + coil.magnetic_field(points)

    return B_total

# main
import sys
fin = sys.argv[1]
df = pd.read_csv(fin).dropna(how='all')

df.columns = df.columns.str.strip()  # fix headers
coils = [Coil(**row) for row in df.to_dict('records')]

axis_xy = df[['Xc', 'Yc']].to_numpy()
axis_path = interpolate_axis(axis_xy, AXIS_SAMPLES_PER_SEGMENT)
axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
B_total = np.zeros((len(axis_points), 3))
for coil in coils:
    B_total += coil.magnetic_field(axis_points)
B_mag = np.linalg.norm(B_total, axis=1)
s_coord = cumulative_distance(axis_path)

axis_x = axis_path[:, 0]
axis_y = axis_path[:, 1]
axis_x_mid = 0.5 * (axis_x.min() + axis_x.max())
axis_y_mid = 0.5 * (axis_y.min() + axis_y.max())


## run field line trace 
ts, path = field_line_trace_xyz(
    x0, y0, z0,
    field=coils,          
    s_total= 5,            # arc-length to trace (meters)
    direction="both",       # "forward", "backward", or "both"
    bounds_X=(-1e9, 1e9),   
    bounds_Y=(-1e9, 1e9),
    bounds_Z=(-1e9, 1e9),
)
# most of the path is 'inf', this is because of the diffrax adaptive solver


### PLOTTING###
# Figure 1: contour plot with coil outlines (full domain)
x_range = (axis_x_mid - PLANE_HALF_WIDTH, axis_x_mid + PLANE_HALF_WIDTH)
y_range = (axis_y_mid - PLANE_HALF_WIDTH, axis_y_mid + PLANE_HALF_WIDTH)
xs, ys, BX_full, BY_full, Bplane = planar_field_grid(coils, x_range, y_range)
fig1, ax1 = plt.subplots(figsize=(9, 9))
contour = ax1.contourf(xs, ys, Bplane, levels=32, cmap='jet', alpha=0.7)
plt.colorbar(contour, ax=ax1, label='|B| in plane (T)')
for c in coils:
    c.draw(ax1, color='black', linewidth=1.0)
ax1.plot(axis_path[:, 0], axis_path[:, 1], 'w-', lw=2, label='Magnetic axis')
ax1.legend(loc='upper right')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Planar |B| with coil outlines')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.grid(True)

# Figure 2: |B| along axis (wider aspect)
fig2, ax2 = plt.subplots(figsize=(12, 4.5))
ax2.plot(s_coord, B_mag, lw=2)
ax2.set_xlabel('Axis distance s (m)')
ax2.set_ylabel('|B| (T)')
ax2.set_title('|B| along magnetic axis')
ax2.grid(True)

# Figure 3: Streamplot over full domain with higher density
xs_stream, ys_stream, BX_stream, BY_stream, _ = planar_field_grid(
    coils, x_range, y_range, nx=GRID_RES_X * 3, ny=GRID_RES_Y * 3
)
speed_stream = np.hypot(BX_stream, BY_stream)

## Now needs conversion to numpy: streamplots do not work with JAX
xs_stream = np.asarray(xs_stream)
ys_stream = np.asarray(ys_stream)
BX_stream = np.asarray(BX_stream)
BY_stream = np.asarray(BY_stream)
speed_stream = np.asarray(speed_stream)

fig3, ax3 = plt.subplots(figsize=(10, 10))
stream = ax3.streamplot(
    xs_stream, ys_stream, BX_stream, BY_stream, color=speed_stream, cmap='jet', density=2.0
)
fig3.colorbar(stream.lines, ax=ax3, label='|B| in plane (T)')
ax3.set_xlim(*x_range)
ax3.set_ylim(*y_range)
ax3.set_aspect('equal', adjustable='box')
ax3.set_title('In-plane B field lines (full domain)')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.grid(True)


# Figure 4: Field line Trace 

fig4, ax4 = plt.subplots(figsize=(9, 9))

# plot first two and last two field lines only from x = -1 to x = 1
for i in range(N):
    x = path[:, i, 0]
    y = path[:, i, 1]

    if i in (0, 1, N-1, N-2):
        # bottom
        mask_bottom = (jnp.abs(x) < 1) & (y < 0)
        ax4.plot(jnp.where(mask_bottom, x, jnp.nan),
                 jnp.where(mask_bottom, y, jnp.nan),
                 color="darkgray")

        # top
        mask_top = (jnp.abs(x) < 1) & (y > 0)
        ax4.plot(jnp.where(mask_top, x, jnp.nan),
                 jnp.where(mask_top, y, jnp.nan),
                 color="darkgray")

    else:
        ax4.plot(x, y, color="darkgray")
# for legend 
ax4.plot(x[0], y[0], color="darkgray",alpha = 1, label = "Fieldlines")

x_range = (axis_x_mid - PLANE_HALF_WIDTH, axis_x_mid + PLANE_HALF_WIDTH)
y_range = (axis_y_mid - PLANE_HALF_WIDTH, axis_y_mid + PLANE_HALF_WIDTH)
xs, ys, BX_full, BY_full, Bplane = planar_field_grid(coils, x_range, y_range)
contour = ax4.contourf(xs, ys, Bplane, levels=32, cmap='jet', alpha=0.7)
plt.colorbar(contour, ax=ax4, label='|B| in plane (T)')

for c in coils:
    c.draw(ax4, color='black', linewidth=1.0)
 
ax4.legend(loc='upper right')
ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal', adjustable='box')
ax4.set_title('Field Line Trace with coil outlines')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.grid(True)

plt.show()

