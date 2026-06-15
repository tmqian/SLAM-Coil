# pyright: standard
# Determines useful information from a single coil with 1 A through it.


from field import *

fin = sys.argv[1]
coils, axis_path = get_coil_info(fin)

if len(coils) != 1:
    sys.exit()

# Determine B-field in center of coil based on Current
coil = coils[0]
coil.current = 1.0
axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
B_total = coil.magnetic_field(axis_points)
B_mag = np.linalg.norm(B_total, axis=1)
G_per_A = np.max(B_mag) * 10000
print(f"\n  B-field per Unit Current: {G_per_A:.2f} G/A")

coil_length = coil.get_length(False)
print(f"  Length: {coil_length:.2f} m")

resistance = coil_length * 0.00033
print(f"  Resistance: {resistance * 1000:.2f} mOhms")

# Assume pure copper
coil_volume = coil_length * (0.008**2 - np.pi * 0.002**2)
coil_mass = coil_volume * CU_DENSITY
print(f"  Mass: {coil_mass:.2f} kg")

power = coil.current**2 * resistance
print(f"  Power: {power:.4f} W")

temp_change = power / coil_mass / 385
print(f"  Temperature Change: {temp_change:.9f} C/s\n")
