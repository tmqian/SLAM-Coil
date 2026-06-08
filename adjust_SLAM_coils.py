
from racetrack import *

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OM","OM", "Brown"]
center_types = ["Blue", "3pan","4panO2","3pan" ,"BlueCenter", "3panCenter", "8pan","3panCenter", "BlueCenter", "3pan","4panO2","3pan" , "Blue"]

Mirror_Length = 1.5
Mirror_Length = 1.5
Stellerator_Radius = 0.5
#filename = "medium_Lm_1p5.csv"
filename = "../test_files/racetrack.csv"
sd = {'Brown': 0, 'OM': 0}
cd = {'Blue': 3.5, '3pan': 1.6, 'BlueCenter': 1.1, '4panO2': 1.6, '3panCenter': 0}

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
    coils, axis_path = get_coil_info(filename[1:], interpolate=False, L=Mirror_Length, R=Stellerator_Radius)


    coil_colors = {'BROWN':'brown', 'L2':'gold', 'BLUE':'blue', 'BLUECENTER':'red', 'OM':'green', '3PANCENTER':'pink', '4PANO2':'gray', '3PAN':'gold', '6PAN':'black', '6PANCENTER':'gold'}

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
        if coil.type == 'BLUECENTER':
            coil.type = 'BLUE'
        elif coil.type == '12PANCENTER':
            coil.type = '12PAN'
        elif coil.type == '6PANCENTER':
            coil.type = '6PAN'
    axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

    ax_contour.legend()
    plt.show()