from field import *
from racetrack import *

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OM","OM", "Brown"]
center_types = ["Blue", "12pan","BlueCenter","12panCenter", "BlueCenter", "12pan", "Blue"]

Mirror_Length = 1.5
Stellerator_Radius = 0.5
#filename = "medium_Lm_1p5.csv"
filename = "../test_files/racetrack_6pan.csv"
sd = {'Brown': 0.01, 'OM': 0.02}

rt = racetrack(Mirror_Length, 
                         Stellerator_Radius,
                         straight_types,
                         center_types,
                         straight_displacements=sd,
                         filename=filename)
rt.build_coils()
rt.write_csv()

plot = input("Do you want to plot the coils? (y/n): ")
if plot.lower() == 'y':
    coils, axis_path = get_coil_info(filename[1:], interpolate=False, L=Mirror_Length, R=Stellerator_Radius)
    axis_path = np.vstack([axis_path, axis_path[0:1]]) #close loop

    coil_colors = {'BROWN':'brown', 'L2':'gold', 'BLUE':'blue', 'BLUECENTER':'red', 'OM':'green', '12PAN':'black', '12PANCENTER':'gold', '6PAN':'black', '6PANCENTER':'gold'}

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
    B_mag = axis_field_plot(ax_axis, coils, axis_path)

    # =====================================================
    # Bottom-right: |B| on axis from each COIL TYPE
    # =====================================================
    for coil in coils:
        if coil.type == 'BLUECENTER':
            coil.type = 'BLUE'
        elif coil.type == '12PANCENTER':
            coil.type = '12PAN'
        elif coil.type == '6PANCENTER':
            coil.type = '6PAN'
    axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

    ax_contour.legend()
    plt.show()