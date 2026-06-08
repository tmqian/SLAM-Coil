# pyright: standard

# I'm looking at the effects changing the mirror length and stellarator radius have on the coil ripple.

from field import *

fin = sys.argv[1]
coils, axis_path = get_coil_info(fin, interpolate=True, L=1.5, R=0.42)

coil_colors = {
    "BROWN": "brown",
    "L2": "gold",
    "BLUE": "blue",
    "BLUECENTER": "red",
    "OM": "green",
    "12PAN": "black",
    "12PANCENTER": "gold",
    "6PAN": "black",
    "6PANCENTER": "gold",
}

# Array of 3 plots using GridSpec
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
# 2 rows × 3 cols:
# col0 = contour
# col1 = colorbar (skinny)
# col2 = right panels
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 0.05, 1.0], height_ratios=[1.0, 1.0])

ax_contour = fig.add_subplot(gs[:, :2])  # left spans both rows
ax_axis = fig.add_subplot(gs[0, 2])  # top-right
ax_blank = fig.add_subplot(gs[1, 2])  # bottom-right

# -------------------------
# Left: planar |B| contour
# -------------------------
contour_plot(fig, ax_contour, coils, axis_path, coil_colors=coil_colors)
ax_contour.legend()


# -------------------------
# Top-right: |B| along axis (total)
# -------------------------
B_mag = axis_field_plot(ax_axis, coils, axis_path)
B_filtered = B_mag[B_mag > B_mag.max() / 2]
peaks, _ = find_peaks(B_filtered)
troughs, _ = find_peaks(-B_filtered)
print("Flatness of ripple peaks: ", np.abs(np.diff(B_filtered[peaks])).max())
print("Flatness of ripple troughs: ", np.abs(np.diff(B_filtered[troughs])).max())
B_filtered = B_filtered[peaks[0] : peaks[-1]]
# =====================================================
# Bottom-right: |B| on axis from each COIL TYPE
# =====================================================
for coil in coils:
    if coil.type == "BLUECENTER":
        coil.type = "BLUE"
    elif coil.type == "12PANCENTER":
        coil.type = "12PAN"
    elif coil.type == "6PANCENTER":
        coil.type = "6PAN"
axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

ax_contour.legend()
plt.show()
