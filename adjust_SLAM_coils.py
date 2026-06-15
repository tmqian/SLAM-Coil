
from racetrack import *

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OM","OM", "Brown"]
center_types = ['Blue', 'Custom1', 'Custom2', 'BlueInner', 'CustomCenter1', 'CustomCenter2', 'BlueCenter', 'CustomCenter2', 'CustomCenter1', 'BlueInner', 'Custom2', 'Custom1', 'Blue']

Mirror_Length = 1.5
Stellerator_Radius = 0.5
filename = "test_files/racetrack_4x5_5Blue_smallerID.csv"
sd = {'Brown': 0, 'OM': 0}
disp_angle = 0
cd = {'Blue': 0, 'Custom1':-disp_angle, 'Custom2':disp_angle, 'CustomCenter1':-disp_angle, 'CustomCenter2':disp_angle}

rt = Racetrack(Mirror_Length, 
                         Stellerator_Radius,
                         straight_types,
                         center_types,
                         straight_displacements=sd,center_displacements=cd,
                         filename=filename)
rt.build_coils()
rt.write_csv()

plot = input("Do you want to plot the racetrack? (y/n): ")
if plot.lower() == 'y':
    coils, axis_path = get_coil_info(filename, interpolate=False, L=Mirror_Length, R=Stellerator_Radius)


    coil_colors = {'Brown':'brown', 'L2Center':'gold', 'L2': 'black', 'Blue':'blue', 'BlueCenter':'red', 'BlueInner':'pink', 'OM':'green', 'CustomCenter':'gold', 'Custom':'black'}

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
    ax_contour.legend()


    # -------------------------
    # Top-right: |B| along axis (total)
    # -------------------------
    axis_field_plot(ax_axis, coils, axis_path)

    # =====================================================
    # Bottom-right: |B| on axis from each COIL TYPE
    # =====================================================
    for coil in coils:
        if coil.type.endswith('Center'):
            coil.type = coil.type[:-6]
        elif coil.type.endswith('Inner'):
            coil.type = coil.type[:-5]
    axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

    ax_contour.legend()
    plt.show()