from racetrack import * 
from field import *

mirror_ripple = False
coil_size = True
mlsr = False









#---------------------------------------Plot Mirror Ripple Minimization------------------------------------
if mirror_ripple:
    fig, ax = plt.subplots(figsize=(9,6))

    straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
    center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']
    #center_types = ['Blue', 'L1', 'L2', 'BlueInner', 'LCenter1', 'LCenter2', 'BlueCenter', 'LCenter2', 'LCenter1', 'BlueInner', 'L2', 'L1', 'Blue']

    Mirror_Length = 1.5
    Stellerator_Radius = 0.5
    filename = "test_files/racetrack_5Blue_withL2.csv"
    sd = {'Brown': 0, 'OM': 0}
    disp_angle = -0.4 # positive away from blue, negative toward blue
    cd = {'Blue': 0, 'L1':-disp_angle, 'L2':disp_angle, 'L3':disp_angle, 'LCenter1':-disp_angle, 'LCenter2':disp_angle, 'LCenter3':disp_angle,
        'Lani1':-disp_angle, 'Lani2':disp_angle, 'LaniCenter1':-disp_angle, 'LaniCenter2':disp_angle}
    toroid_trans = 0.00

    rt = Racetrack(Mirror_Length, 
                            Stellerator_Radius,
                            straight_types,
                            center_types,
                            straight_displacements=None,center_displacements=cd,
                            filename=filename,
                            toroid_trans=toroid_trans)
    rt.build_coils()

    B_mag, s_coord = get_Bmag_on_axis(rt.coils, rt.axis_path)
    rt.set_coil_currents('OMCenter', 270)
    B_mag1, _ = get_Bmag_on_axis(rt.coils, rt.axis_path)
    sd = {'Brown': 0, 'OM': -0.03}
    rt.reset(Mirror_Length, Stellerator_Radius, straight_types, center_types, straight_displacements=sd, center_displacements=cd, filename=filename, toroid_trans=toroid_trans)
    rt.build_coils()
    rt.set_coil_currents('OMCenter', 260)
    rt.set_coil_currents('Brown', 290)
    B_mag2, _ = get_Bmag_on_axis(rt.coils, rt.axis_path)
    rt.reset(Mirror_Length, Stellerator_Radius, straight_types, center_types, straight_displacements=sd, center_displacements=cd, filename=filename, toroid_trans=toroid_trans)
    rt.build_coils()
    Bmag_3, _ = get_Bmag_on_axis(rt.coils, rt.axis_path)

    plt.plot(s_coord, B_mag, label='Base Case', color='blue')
    plt.plot(s_coord, B_mag1, label='Lower OMCenter current', color='red', linestyle='--')
    plt.plot(s_coord, Bmag_3, label='OM displacements', color='Orange', linestyle='--')
    plt.plot(s_coord, B_mag2, label='OM displacement and lower OMCenter + Brown current', color='green', linestyle='-')

    plt.xlabel('s (m)')
    plt.ylabel('|B| (T)')
    plt.legend()
        
    plt.show()

#----------------------------------------------------------------------plot isometric coil size comparison---------------------------------------------------------------------------
if coil_size:
    OM = Coil(-0.5, 0, 90, "OM")
    Brown = Coil(-0.25, 0, 90, "Brown")
    Blue = Coil(0, 0, 90, "Blue")
    L2 = Coil(0.25, 0, 90, "L2")
    Lani = Coil(0.5, 0, 90, "Lani")

    fig, ax = plt.subplots(figsize=(8, 8))

    OM.draw(ax, color='green', label='OM: ID = 45cm', add_to_legend=True)
    Brown.draw(ax, color='brown', label='Brown: ID = 33cm', add_to_legend=True)
    Blue.draw(ax, color='blue', label='Blue: ID = 31cm', add_to_legend=True)
    L2.draw(ax, color='yellow', label='L2: ID = 25cm', add_to_legend=True)
    Lani.draw(ax, color='pink', label='NewCoil', add_to_legend=True)
    plt.legend()
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.6, 0.6)
    plt.show()

#------------------------------Mirror length and stellerator radius tests--------------------------------
if mlsr:
    straight_types = ["Brown", "OM","OMCenter","OM", "Brown"]
    center_types = ['Blue', 'Lani1', 'Lani2', 'BlueInner', 'LaniCenter1', 'LaniCenter2', 'BlueCenter', 'LaniCenter2', 'LaniCenter1', 'BlueInner', 'Lani2', 'Lani1', 'Blue']
    #center_types = ['Blue', 'L1', 'L2', 'BlueInner', 'LCenter1', 'LCenter2', 'BlueCenter', 'LCenter2', 'LCenter1', 'BlueInner', 'L2', 'L1', 'Blue']

    Mirror_Length = 1.5
    Stellerator_Radius = 0.5
    filename = "test_files/racetrack_5Blue_withL2.csv"
    sd = {'Brown': 0, 'OM': 0}
    disp_angle = -0.4 # positive away from blue, negative toward blue
    cd = {'Blue': 0, 'L1':-disp_angle, 'L2':disp_angle, 'L3':disp_angle, 'LCenter1':-disp_angle, 'LCenter2':disp_angle, 'LCenter3':disp_angle,
        'Lani1':-disp_angle, 'Lani2':disp_angle, 'LaniCenter1':-disp_angle, 'LaniCenter2':disp_angle}
    toroid_trans = 0.00

    rt = Racetrack(Mirror_Length, 
                            Stellerator_Radius,
                            straight_types,
                            center_types,
                            straight_displacements=None,center_displacements=cd,
                            filename=filename,
                            toroid_trans=toroid_trans)
    rt.build_coils()

    B_mag_base, s_coord_base = get_Bmag_on_axis(rt.coils, rt.axis_path)

    rt.reset(1.2, Stellerator_Radius, straight_types, center_types, straight_displacements=None, center_displacements=cd, filename=filename)
    rt.build_coils()
    B_mag_shorter, s_coord_shorter = get_Bmag_on_axis(rt.coils, rt.axis_path)


    rt.reset(1.8, Stellerator_Radius, straight_types, center_types, straight_displacements=None, center_displacements=cd, filename=filename)
    rt.build_coils()
    B_mag_longer, s_coord_longer = get_Bmag_on_axis(rt.coils, rt.axis_path)

    rt.reset(Mirror_Length, 0.45, straight_types, center_types, straight_displacements=None, center_displacements=cd, filename=filename)
    rt.build_coils()
    B_mag_smaller, s_coord_smaller = get_Bmag_on_axis(rt.coils, rt.axis_path)

    rt.reset(Mirror_Length, 0.55, straight_types, center_types, straight_displacements=None, center_displacements=cd, filename=filename)
    rt.build_coils()
    B_mag_larger, s_coord_larger = get_Bmag_on_axis(rt.coils, rt.axis_path) 

    fig, ax = plt.subplots(1, 2, figsize=(24,15))

    s_coord_max = 1

    # Plotting B-field for different mirror lengths
    ax[0].plot(s_coord_base/s_coord_max, B_mag_base, label='Base Case (Mirror Length = 1.5 m)', color='blue')
    ax[0].plot(s_coord_base/s_coord_max, B_mag_shorter, label='Shorter Mirror Length (1.2 m)', color='red', linestyle='--')
    ax[0].plot(s_coord_base/s_coord_max, B_mag_longer, label='Longer Mirror Length (1.8 m)', color='green', linestyle='--')
    ax[0].set_xlabel('s (m)', fontsize=14)
    ax[0].set_ylabel('|B| (T)', fontsize=14)
    ax[0].set_title('Effect of Mirror Length on B-field', fontsize=16)
    ax[0].legend(fontsize=14)

    # Plotting B-field for different stellerator radii
    ax[1].plot(s_coord_base/s_coord_max, B_mag_base, label='Base Case (Stellerator Radius = 0.5 m)', color='blue')
    ax[1].plot(s_coord_base/s_coord_max, B_mag_smaller, label='Smaller Stellerator Radius (0.45 m)', color='orange', linestyle='--')
    ax[1].plot(s_coord_base/s_coord_max, B_mag_larger, label='Larger Stellerator Radius (0.55 m)', color='purple', linestyle='--')
    ax[1].set_xlabel('s (m)', fontsize=14)
    ax[1].set_ylabel('|B| (T)', fontsize=14)
    ax[1].set_title('Effect of Stellerator Radius on B-field', fontsize=16)
    ax[1].legend(fontsize=14) 

    plt.show()
