
from racetrack import *

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
#center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']
center_types = ['Blue', 'L1', 'L2', 'BlueInner', 'LCenter1', 'LCenter2', 'BlueCenter', 'LCenter2', 'LCenter1', 'BlueInner', 'L2', 'L1', 'Blue']

Mirror_Length = 1.5
Stellerator_Radius = 0.5
filename = "test_files/racetrack_5Blue_withL2.csv"
sd = {'Brown': -0.15, 'OM': 0}
disp_angle = 1.46 # positive away from blue, negative toward blue
cd = {'Blue': 0, 'L1':-disp_angle, 'L2':disp_angle, 'L3':disp_angle, 'LCenter1':-disp_angle, 'LCenter2':disp_angle, 'LCenter3':disp_angle,
      'Lani1':-disp_angle, 'Lani2':disp_angle, 'LaniCenter1':-disp_angle, 'LaniCenter2':disp_angle}
toroid_trans = 0.06

rt = Racetrack(Mirror_Length, 
                         Stellerator_Radius,
                         straight_types,
                         center_types,
                         straight_displacements=None,center_displacements=cd,
                         filename=filename,
                         toroid_trans=toroid_trans)
rt.build_coils()


rt.write_csv()
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


coils, axis_path = get_coil_info(filename, interpolate=False, L=Mirror_Length, R=Stellerator_Radius, toroid_trans=toroid_trans)
axis_path = np.vstack([axis_path, axis_path[0]])  # close the loop


# blue_scoord, idx_4panS = get_coil_scoord(rt.coils, axis_path, '4panS')
# print(f"4panS coil s-coordinates: {blue_scoord}")
# B_ripple, scoord = get_Bmag_on_axis(rt.coils, axis_path)
# B_ripple = B_ripple[idx_4panS[1]:idx_4panS[2]]
# plt.plot(scoord[idx_4panS[1]:idx_4panS[2]], B_ripple, label='4panS Coil B-field', linestyle='-', color='purple')
# ripple = (B_ripple.max() - B_ripple.min()) / B_ripple.mean()
# print(f"Ripple for 4panS coil: {ripple:.2%}")
# plt.xlabel('s (m)')
# plt.ylabel('|B| (T)')

plot = input("Do you want to plot the racetrack? (y/n): ")
if plot.lower() == 'y':

    coil_colors = {'Brown':'brown', 'L2':'gold', 'Blue':'blue',
               'BlueCenter':'red', 'BlueInner': 'violet', 'OM':'green', 'LaniCenter':'pink', 'LCenter':'pink',
                 'Lani':'gray', 'L':'gray', '3pan':'gold', '6pan':'black', '6panCenter':'gold',
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
    contour_plot(fig, ax_contour, coils, axis_path, coil_colors=coil_colors, just_coils=False)
    ax_contour.legend()
    #ax_contour.plot(rt.ports[:, 0], rt.ports[:, 1], 'o', label='Ports', color='black')
    sign_x = [1, 1, -1, -1]
    sign_y = [1, -1, 1, -1]
    # old_port_angles =np.array([15, 36, 66])
    new_port_angles = np.array([20.8, 62.3])
    new_port_angles = np.array([22.5, 67.5])
    #new_port_angles = np.array([7.79, 37.21, 52.79, 82.21])
    # old_ports_x = 1.5/2 + 0.5*np.cos(old_port_angles/180*np.pi)
    # old_ports_y = 0.5*np.sin(old_port_angles/180*np.pi)
    new_ports_x = 1.5/2 + 0.5*np.cos(new_port_angles/180*np.pi) + toroid_trans
    new_ports_y = 0.5*np.sin(new_port_angles/180*np.pi)
    # old_ports_x = np.concatenate((old_ports_x, old_ports_x*sign_x[1], old_ports_x*sign_x[2], old_ports_x*sign_x[3]))
    # old_ports_y = np.concatenate((old_ports_y, old_ports_y*sign_y[1], old_ports_y*sign_y[2], old_ports_y*sign_y[3]))
    new_ports_x = np.concatenate((new_ports_x, new_ports_x*sign_x[1], new_ports_x*sign_x[2], new_ports_x*sign_x[3]))
    new_ports_y = np.concatenate((new_ports_y, new_ports_y*sign_y[1], new_ports_y*sign_y[2], new_ports_y*sign_y[3]))
    # ax_contour.plot(old_ports_x, old_ports_y, 'o', label='Old Ports', color='gray')
    ax_contour.plot(new_ports_x, new_ports_y, 'o', label='New Ports', color='black')


    # -------------------------
    # Top-right: |B| along axis (total)
    # -------------------------
    axis_field_plot(ax_axis, coils, axis_path, label='B-field')
    #ax_axis.plot(scoord[idx_4panS[1]:idx_4panS[2]], B_ripple, color='purple')


    # =====================================================
    # Bottom-right: |B| on axis from each COIL TYPE
    # =====================================================
    for coil in coils:
        if coil.type.endswith('Center'):
            coil.type = coil.type[:-6]
        elif coil.type.endswith('Inner'):
            coil.type = coil.type[:-5]
    axis_field_plot_by_coil(ax_blank, coils, axis_path, coil_colors=coil_colors)

    brown_locations = np.array([0.15+toroid_trans, 1.5-0.15+toroid_trans, 1.5+np.pi*0.5+0.15+toroid_trans*3, 1.5+np.pi*0.5+1.5-0.15+toroid_trans*3])
    brown_edges = np.concatenate((brown_locations-0.0443, brown_locations+0.0443))
    for i in range(len(brown_edges)-1):
        ax_axis.axvline(x=brown_edges[i+1], ls='--', color='brown', lw=1)
        ax_blank.axvline(x=brown_edges[i+1], ls='--', color='brown', lw=1)

    ax_axis.axvline(x=brown_edges[0], ls='--', color='brown', lw=1, label='Brown Coil Edges')
    ax_blank.axvline(x=brown_edges[0], ls='--', color='brown', lw=1, label='Brown Coil Edges')
    ax_axis.legend(loc='lower left')
    ax_contour.legend()
    ax_blank.legend()
    plt.show()