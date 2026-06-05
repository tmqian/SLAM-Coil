from field import *
from racetrack import *

straight_types = ["Brown", "OM","OM","OM", "Brown"]
center_types = ["Blue", "12pan","BlueCenter", "12panCenter", "BlueCenter", "12pan", "Blue"]

Mirror_Length = 1.5
Stellerator_Radius = 0.5
sd = {'Brown': 0.00, 'OM': 0.0}
cd = {'Blue': 0, '12pan': 0, 'BlueCenter': 0}

rt = Racetrack(Mirror_Length, 
                         Stellerator_Radius,
                         straight_types,
                         center_types,
                         straight_displacements=None,center_displacements=cd)
rt.build_coils()
axis_path = get_axis_path(rt.coils, L=Mirror_Length, R=Stellerator_Radius, interpolate=False)
B_mag, s_coord = get_Bmag_on_axis(rt.coils, axis_path)

s_ends, idx_ends = get_coil_scoord(rt.coils, axis_path, 'Blue')

B_mag_center1 = B_mag[idx_ends[0]:idx_ends[1]]
s_coord_center1 = s_coord[idx_ends[0]:idx_ends[1]]
B_mag_center2 = B_mag[idx_ends[2]:idx_ends[3]]
s_coord_center2 = s_coord[idx_ends[2]:idx_ends[3]]
