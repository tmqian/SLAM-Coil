
from racetrack import *
import os
from optimize_ripple import Optimize

"""Testing ways to make it easer to adjust coil positions"""

straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
#center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']
center_types = ['Blue', 'L1', 'L2', 'BlueInner', 'LCenter1', 'LCenter2', 'BlueCenter', 'LCenter2', 'LCenter1', 'BlueInner', 'L2', 'L1', 'Blue']
#center_types = ['Blue', 'Lani', 'LaniInner1', 'BlueInner', 'LaniInner2', 'LaniCenter', 'BlueCenter', 'LaniCenter', 'LaniInner2', 'BlueInner', 'LaniInner1', 'Lani', 'Blue']
# center_types = ['Blue', 'LaniPP', 'LaniPPInner1','BlueInner', 'LaniPPInner2', 'LaniPPCenter', 'BlueCenter', 'LaniPPCenter', 'LaniPPInner2', 'BlueInner', 'LaniPPInner1', 'LaniPP', 'Blue']


conv_types = None
conv_displacements = {'ConvB1': -0.03, 'ConvB2': 0}
Mirror_Length = 1.5
Stellerator_Radius = 0.5
filename = "test_files/racetrack_lani_optim.csv"
sd = {'Brown': 0.044, 'OM': 0}
start_angle = -3
end_angle = 3
num_steps = 50

step_size = (end_angle - start_angle) / (num_steps - 1)
disp_angle = start_angle

toroid_trans = 0.01
optimize = False

ripple_pts = []
alpha_pts = []

step_idx = 0
while step_idx < num_steps:
    cd = {'Blue': 0, 'L1':-disp_angle, 'L2':disp_angle, 'L3':disp_angle, 'LCenter1':-disp_angle, 'LCenter2':disp_angle, 'LCenter3':disp_angle,
      'Lani1':-disp_angle, 'Lani2':disp_angle, 'LaniCenter1':-disp_angle, 'LaniCenter2':disp_angle,
      'Lani':-disp_angle, 'LaniInner1':disp_angle, 'LaniInner2':-disp_angle, 'LaniCenter':disp_angle}

    rt = Racetrack(Mirror_Length, 
                            Stellerator_Radius,
                            straight_types,
                            conv_types,
                            center_types,
                            straight_displacements=sd,center_displacements=cd,conv_displacements=conv_displacements,
                            filename=filename,
                            toroid_trans=toroid_trans)
    rt.Blue90 = True
    rt.build_coils()
    if optimize:
        Optimize(plot=True, rt=rt, coil_ref='Blue', coil_idx=0, target_B=0.25)

    L_angles = []
    for coil in rt.coils:
        if coil.type == "L":
            L_angles.append(coil.angle % 360)
    print(L_angles)

    L_coils = [coil for coil in rt.coils if coil.type == "L"]

    alpha = L_coils[0].angle % 360 - 67.5
    print(alpha)

    alpha_pts.append(alpha)



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

    coils = rt.coils
    axis_path = rt.axis_path

    _, idx_blue = get_coil_scoord(rt.coils, axis_path, 'Blue')
    B_ripple, s = get_Bmag_on_axis(rt.coils, axis_path)
    B_ripple = B_ripple[idx_blue[0]:idx_blue[1]]
    s_coord = s[idx_blue[0]:idx_blue[1]]
    B_peaks = find_peaks(B_ripple)[0]
    B_trough = find_peaks(-B_ripple)[0]

    ripple = (B_ripple[B_peaks].max() - B_ripple[B_trough].min()) / B_ripple.max()
    ripple_pts.append(ripple)

    print("ripple: ", ripple)
    disp_angle += step_size
    step_idx += 1

plt.plot(alpha_pts, ripple_pts, marker='o')
plt.xlabel("L Coil Angle (deg)")
plt.ylabel("Ripple")
plt.title("Ripple vs L Coil Angle")
plt.plot()
plt.show()