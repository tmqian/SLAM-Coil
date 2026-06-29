from field import *
from racetrack import *

straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']

Mirror_Length = 1.5
Stellerator_Radius = 0.5
filename = "/home/benis/Research/Wippl /SLAM/SLAM-Coil/test_files/racetrack_4x5_5Blue_smallerID.csv"
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
    print(f" type: {coil.type}, current: {coil.current}, group: {coil.group}, id: {coil.id}")

fig, ax = plt.subplots(figsize=(8, 5))
_, axis_path = get_coil_info(filename, interpolate=False, L=Mirror_Length, R=Stellerator_Radius)
axis_path = np.vstack([axis_path, axis_path[0]])  # close the loop

B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path)
displace = True
if displace:
    for coil in rt.coils:
        if coil.type == 'OM':
            coil.Xc += 0.08 if coil.Xc < 0 else -0.08
            
    B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path)

OM_current = 270
rt.set_coil_current('OMCenter', OM_current)
# OM_coords, idx_OM = get_coil_scoord(rt.coils, axis_path, 'OM')
# for i in range(len(idx_OM)):
#     ax.axvspan(OM_coords[idx_OM[i][0]], OM_coords[idx_OM[i][1]], color='green', alpha=0.3, label='OM Coil' if i == 0 else "")

B_mag1, _ = get_Bmag_on_axis(rt.coils, axis_path)
plt.plot(s_coord, B_mag, linewidth=2, label='Original OM current at 300 A')
plt.plot(s_coord, B_mag1, linewidth=2, label=f'Lowered central OM current to {OM_current} A', color  = 'red', linestyle='--')
plt.xlabel('s (m)')
plt.ylabel('B (T)')
plt.title('|B| On Axis')
plt.show()