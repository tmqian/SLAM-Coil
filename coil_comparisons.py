'''
Plots |B| for different coil types on the same space.
'''
from field import *


fig = plt.figure(figsize=(16, 8), constrained_layout=False)

gs = GridSpec(4, 8, figure=fig, left=0, right=0.98, bottom=0.1, top=0.95,
              width_ratios=[1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0],
              height_ratios=[0.6, 0.7, 0.1, 1.0]
              )
ax_text = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,2:6])
ax2 = [0, 0, 0]
ax2[0] = fig.add_subplot(gs[3,:2])
ax2[1] = fig.add_subplot(gs[3,3:5])
ax2[2] = fig.add_subplot(gs[3,6:])

fig.suptitle("SLAM Coil Comparisons", fontsize=16)

ax_text.axis('off')
ax_text.text(0.5, 0.5, "3 L2 Coils: 8 radial, 12 axial  (Current: 121.0 A,  Power: 1023 W,  Length: 113.1 m,  Mass: 51 kg)" \
            "\nHSX Copper: 6 radial, 8 axial  (Current: 246.9 A,  Power: 1152 W,  Length: 57.3 m,  Mass: 26.4 kg)" \
            "\nHSX Copper: 4 radial, 12 axial  (Current: 231.2 A,  Power: 957 W,  Length: 54.3 m,  Mass: 25.0 kg)",
            ha='center', va='center', fontsize=14)

for k in range(3):

    fin = sys.argv[k+1]
    coils, axis_path = get_coil_info(fin)
    mag_axis = np.linspace(-0.5, 0.5, 51)
    axis_path = np.column_stack([mag_axis, np.zeros(len(mag_axis))])

    if k == 0:
        axis_field_plot(ax1, coils, axis_path, length_units='cm', field_units='G', color='blue', label='3 L2')
    elif k == 1:
        axis_field_plot(ax1, coils, axis_path, length_units='cm', field_units='G', color='red', label='HSX 8 axial')
    elif k == 2:
        axis_field_plot(ax1, coils, axis_path, length_units='cm', field_units='G', color='green', label='HSX 12 axial')

    contour_plot(fig, ax2[k], coils, axis_path, length_units='cm', field_units='G', levels=np.arange(0, 660, 20), extend='max')
    field_streamplot(fig, ax2[k], coils, show_labels=False, length_units='cm', field_units='G', color='white')


ax1.set_title('|B| along magnetic axis')
ax2[0].set_title('3 L2')
ax2[1].set_title('HSX 8 axial')
ax2[2].set_title('HSX 12 axial')
ax1.legend(loc='upper right')
plt.show()