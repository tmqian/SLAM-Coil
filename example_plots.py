'''
Showcases an arrangement of plots.
'''
from field import *


fin = sys.argv[1]
coils, axis_path = get_coil_info(fin)

coil_colors = {'BROWN':'brown', 'L2':'gold', 'BLUE':'blue', 'OM':'green'}

fig1, ax1 = plt.subplots(figsize=(9, 9))
contour_plot(fig1, ax1, coils, axis_path, coil_colors=coil_colors)
ax1.legend(loc="upper right")

fig2, ax2 = plt.subplots(figsize=(12, 4.5))
axis_field_plot(ax2, coils, axis_path)

fig3, ax3 = plt.subplots(figsize=(10, 10))
field_streamplot(fig3, ax3, coils)


# Array of 3 plots using GridSpec
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
# 2 rows × 3 cols:
# col0 = contour
# col1 = colorbar (skinny)
# col2 = right panels
gs = GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1.5, 0.05, 1.0],   
    height_ratios=[1.0, 1.0]
)

ax_contour = fig.add_subplot(gs[:, :2])   # left spans both rows
ax_axis    = fig.add_subplot(gs[0, 2])   # top-right
ax_blank   = fig.add_subplot(gs[1, 2])   # bottom-right

# -------------------------
# Left: planar |B| contour
# -------------------------
contour_plot(fig, ax_contour, coils, axis_path, coil_colors=coil_colors)

# -------------------------
# Top-right: |B| along axis (total)
# -------------------------
axis_field_plot(ax_axis, coils, axis_path)

# =====================================================
# Bottom-right: |B| on axis from each COIL TYPE
# =====================================================
axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

plt.show()