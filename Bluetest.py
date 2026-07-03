from field import *

blue_brown = Coil(-0.7, 0, 90, type='Blue')
blue = Coil(0.7, 0, 90, type='RBlue')

blue_brown.current = 1
blue.current = 1

coils = [blue_brown, blue]
axis_path = get_axis_path(coils)[:-1]
B_mag, s_coord = get_Bmag_on_axis(coils, axis_path)
x = np.linspace(-0.7, 0.7, np.size(s_coord))

# Find maximum B value for x < 0
mask = x < 0
B_per_I_blue_brown = np.max(B_mag[mask])*10000
B_per_I_blue = np.max(B_mag[~mask])*10000

plt.plot(x, B_mag, color = 'red', label='|B| on Axis')
plt.axvline(x=blue_brown.Xc, color='brown', linestyle='--', label='Center of Blue-Brown Coil')
plt.axvline(x=blue.Xc, color='blue', linestyle='--', label='Center of Blue Coil')

plt.axvspan(xmin=blue_brown.Xc-0.0443, xmax=blue_brown.Xc+0.0443, color='brown', alpha=0.3, label=f'Blue-Brown Coil: {B_per_I_blue_brown:.2f} G/A')
plt.axvspan(xmin=blue.Xc-0.0443, xmax=blue.Xc+0.0443, color='blue', alpha=0.3, label=f'Blue Coil: {B_per_I_blue:.2f} G/A')


plt.xlabel('s (m)')
plt.ylabel('|B| (T)')
plt.title('|B| on Axis for Two Coils')
plt.legend()
plt.show()