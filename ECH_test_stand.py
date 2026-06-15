# pyright: standard
import matplotlib.pyplot as plt
import numpy as np

from field import get_coil_info

CURRENT = 250  # Amps
THRESHOLD = 875.0  # Gauss
COIL_WIDTH = 0.076
VESSEL_WIDTH = 0.205  # 20.5 cm
VESSEL_CENTER_X = 0.0


# Get axis path
coils, _ = get_coil_info(
    "test_files/ECH_test_stand.csv", interpolate=True, L=1.5, R=0.42
)

# Compute on-axis field for a range -1m to 1m in x
axis_points_x = np.linspace(-1, 1, 201)
axis_points = np.column_stack(
    [axis_points_x, np.zeros_like(axis_points_x), np.zeros_like(axis_points_x)]
)
B_field = np.zeros_like(axis_points)
for coil in coils:
    # For each coil, compute the field at each poing and sum them
    B_field += coil.magnetic_field(axis_points, CURRENT)

B_mag = np.linalg.norm(B_field, axis=1)
fig, ax = plt.subplots(figsize=(12, 4.5))

# Shade vacuum vessel region centered on x=0
ax.axvspan(
    VESSEL_CENTER_X - VESSEL_WIDTH / 2,
    VESSEL_CENTER_X + VESSEL_WIDTH / 2,
    color="red",
    alpha=0.12,
    linewidth=0,
    zorder=0,
    label="Vacuum vessel",
)


# Shade approximate coil-width regions on the axis plot
for i, coil in enumerate(coils):
    center_x = float(coil.Xc)
    half_width = COIL_WIDTH / 2
    ax.axvspan(
        center_x - half_width,
        center_x + half_width,
        color="gray",
        alpha=0.18,
        linewidth=0,
        zorder=0,
        label="Coil width" if i == 0 else None,
    )

ax.plot(axis_points_x, B_mag * 10000, color="blue", zorder=2, label="|B| on axis")

# Mark threshold crossings
y = B_mag * 10000
ax.axhline(
    THRESHOLD,
    color="red",
    linestyle="--",
    linewidth=0.8,
    label=f"Threshold ({THRESHOLD:.0f} G)",
)
# find sign changes relative to threshold
diff = y - THRESHOLD
signs = np.sign(diff)
crossing_labels = []
cross_idxs = np.where(signs[:-1] * signs[1:] < 0)[0]
for i, idx in enumerate(cross_idxs):
    # linear interpolation to estimate crossing x
    x1, x2 = axis_points_x[idx], axis_points_x[idx + 1]
    y1, y2 = y[idx], y[idx + 1]
    if y2 == y1:
        xc = x1
    else:
        t = (THRESHOLD - y1) / (y2 - y1)
        xc = x1 + t * (x2 - x1)
    yc = THRESHOLD
    ax.scatter([xc], [yc], color="red", zorder=5)
    crossing_labels.append(float(xc))

ax.set_xlabel("x (m)")
ax.set_ylabel("|B| (G)")
ax.set_title("ECH Test Stand: |B| along axis")
ax.grid()
ax.legend(loc="lower right", framealpha=0.95)

# Add consolidated informational box for geometry, current, and crossings
vessel_left = VESSEL_CENTER_X - VESSEL_WIDTH / 2
vessel_right = VESSEL_CENTER_X + VESSEL_WIDTH / 2
info_lines = [
    r"$\mathbf{Setup}$",
    f"Current: {CURRENT:.0f} A",
    f"Threshold: {THRESHOLD:.0f} G",
    "",
    r"$\mathbf{Geometry}$",
    f"Vessel: {vessel_left:.4f} to {vessel_right:.4f} m",
]
for i, coil in enumerate(coils):
    coil_left = coil.Xc - COIL_WIDTH / 2
    coil_right = coil.Xc + COIL_WIDTH / 2
    info_lines.append(f"Coil Set {i + 1}: {coil_left:.4f} to {coil_right:.4f} m")

if crossing_labels:
    info_lines.extend(
        [
            "",
            r"$\mathbf{Resonances}$",
            *[f"x = {xc:+.3f} m" for xc in crossing_labels],
        ]
    )

info_text = "\n".join(info_lines)
info_props = dict(
    boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.92
)
ax.text(
    0.98,
    0.97,
    info_text,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="top",
    horizontalalignment="right",
    multialignment="left",
    bbox=info_props,
)

fig.savefig("generated/plots/ECH_test_stand_axis.png", dpi=300)
