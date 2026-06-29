
from racetrack import *
from optimize_ripple import *

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']

Mirror_Length = 1.5
Stellerator_Radius = 0.5
filename = "test_files/racetrack4.csv"
sd = {'Brown': -0.15, 'OM': 0}
disp_angle = -0.4 # positive toward blue, negative away from blue
cd = {'Blue': 0, 'Lani1':-disp_angle, 'Lani2':disp_angle, 'LaniCenter1':-disp_angle, 'LaniCenter2':disp_angle}

rt = Racetrack(Mirror_Length, 
                         Stellerator_Radius,
                         straight_types,
                         center_types,
                         straight_displacements=None,center_displacements=cd,
                         filename=filename)
rt.build_coils() 


for coil in rt.coils:
    print(coil.type)

#Optimize(plot=False, rt=rt, coil_ref='Lani', coil_idx=0, target_B=0.25)

center, center_space, straight = rt.build_ports(r = 0.47, rho=0.5)
angles = np.array([c.angle for c in rt.coils if c.type in rt.center_types]) % 360


# print(f"Port angles: {angles}")
# print(f"Port center spaces: {center_space}")
# print(f"Port center positions: {center}")

#Write ports to file
with open('ports.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ports (deg)', 'space (cm)'])
    for port, space in zip(center, center_space):
        writer.writerow([f"{port:.6f}", f"{space:.6f}"])


_, axis_path = get_coil_info(filename, interpolate=False, L=Mirror_Length, R=Stellerator_Radius)
axis_path = np.vstack([axis_path, axis_path[0]])  # close the loop

# B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path)

# s_ends, idx_ends = get_coil_scoord(rt.coils, axis_path, 'Blue')

# B_mag_center = B_mag[idx_ends[0]:idx_ends[1]]
# s_coord_center = s_coord[idx_ends[0]:idx_ends[1]]

# peak_idx, _ = find_peaks(B_mag_center)
# trough_idx, _ = find_peaks(-B_mag_center)

# ripple = (np.max(B_mag_center[peak_idx]) - np.min(B_mag_center[trough_idx])) / np.max(B_mag_center)
# print(f"Ripple for Center coil: {ripple:.4%}")

plot = input("Do you want to plot the racetrack? (y/n): ")
if plot.lower() == 'y':

    coil_colors = {'Brown':'brown', 'L2':'gold', 'Blue':'blue',
               'BlueCenter':'red', 'BlueInner': 'violet', 'OM':'green', 'LaniCenter':'pink',
                 'Lani':'gray', '3pan':'gold', '6pan':'black', '6panCenter':'gold',
                 '2panBu': 'Cyan', '2panBuCenter': 'magenta', '2panBuOut': 'orange'}
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
    contour_plot(fig, ax_contour, rt.coils, axis_path, coil_colors=coil_colors)
    ax_contour.legend()
    ax_contour.plot(rt.ports[:, 0], rt.ports[:, 1], 'o', label='Ports', color='black')


    # -------------------------
    # Top-right: |B| along axis (total)
    # -------------------------
    axis_field_plot(ax_axis, rt.coils, axis_path)
    #ax_axis.plot(scoord[idx_4panS[1]:idx_4panS[2]], B_ripple, color='purple')


    # =====================================================
    # Bottom-right: |B| on axis from each COIL TYPE
    # =====================================================
    for coil in rt.coils:
        if coil.type.endswith('Center'):
            coil.type = coil.type[:-6]
        elif coil.type.endswith('Inner'):
            coil.type = coil.type[:-5]
    axis_field_plot_by_coil(ax_blank, rt.coils, axis_path, coil_colors=coil_colors)

    ax_contour.legend()
    plt.show()