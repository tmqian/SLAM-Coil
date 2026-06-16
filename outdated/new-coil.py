import csv
import pandas
import numpy as np
from field import *

typ = '2panBv1'
ID= 0.3
Nr = 23
Nz= 2
dr = 0.00252
dz = 0.0381

OD = ID + 2*Nr*dr
DZ = Nz*dz

data = {
    'type': [typ],
    'ID': [ID],
    'OD': [OD],
    'DZ': [DZ],
    'Nr': [Nr],
    'Nz': [Nz],
    'current': [100.0],
    'nr': [20],
    'nz': [4*Nz],
    'parallel': [0],
    'partitions': [0]
}

try:
    existing_df = pandas.read_csv('coil_models/coil_model.csv')
    if typ in existing_df['type'].values:
        print(f"Error: Coil type '{typ}' already exists in the database.")
    else:
        df = pandas.DataFrame(data)
        existing_df = pandas.concat([existing_df, df], ignore_index=True)
        existing_df.to_csv('coil_models/coil_model.csv', index=False)
        print(f"Row added successfully: {df.to_dict('records')[0]}")
except FileNotFoundError:
    df = pandas.DataFrame(data)
    df.to_csv('coil_models/coil_model.csv', index=False)
    print(f"Row added successfully: {df.to_dict('records')[0]}")

coils = [Coil(type = typ)]
axis_path = get_axis_path(coils)
coil = coils[0]
coil.current = 1.0
axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])
B_total = coil.magnetic_field(axis_points)
B_mag = np.linalg.norm(B_total, axis=1)
G_per_A = np.max(B_mag) * 10000
print(f"\n  B-field per Unit Current: {G_per_A:.2f} G/A")

coil_length = coil.get_length(False)
print(f"  Length: {coil_length:.2f} m")

resistance = coil_length * 1.68e-8 / (dr*dz - np.pi * 0.002**2)
print(f"  Resistance: {resistance*1000:.2f} mOhms")

# Assume pure copper
coil_volume = coil_length * (dz*dr - np.pi * 0.002**2)
coil_mass = coil_volume * CU_DENSITY
print(f"  Mass: {coil_mass:.2f} kg")
coil.current = 1000
power = coil.current**2 * resistance
print(f"  Power: {power:.4f} W")

temp_change = power / coil_mass / 385
print(f"  Temperature Change: {temp_change:.9f} C/s\n")


mass_flow = power / (385 * 10)
print(f"  Mass Flow Rate for 10 C/s cooling: {mass_flow:.6f} kg/s")

fluid_velocity = 4*mass_flow / (np.pi * 0.004**2 * 1000)
print(f"  Coolant Velocity for 10 C/s cooling: {fluid_velocity:.4f} m/s")

dynamic_viscosity = 1e-3  # Pa·s for water at room temp
Reynolds_number = fluid_velocity * 0.004 * 1000 / dynamic_viscosity  # Assuming density of water ~1000 kg/m^3
print(f"  Reynolds Number for 10 C/s cooling: {Reynolds_number:.2f}")

f = 0.3164 / Reynolds_number**0.25
print(f"  Darcy Friction Factor for 10 C/s cooling: {f:.6f}")

dP = f * (coil_length / 0.004) * 0.5 * 1000 * fluid_velocity**2
print(f"  Pressure Drop for 10 C/s cooling: {dP:.2f} Pa")

