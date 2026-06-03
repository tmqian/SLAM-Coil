import pandas as pd
import numpy as np
import field
from field import Coil, interpolate_axis, AXIS_SAMPLES_PER_SEGMENT, CU_DENSITY

"""This script generates a CSV of possible coil geometries and their properties (G/A, length, mass, etc.) for a single coil design. It iterates over a range of outer diameters (OD), inner diameters (ID), and axial lengths (DZ) to compute the resulting magnetic field strength per ampere (G/A) at the center of the coil, as well as other relevant properties. The results are saved to 'possible-coil.csv' and visualized with plots."""

results = []
for OD in [0.4, 0.5]:
    for ID in np.arange(0.31, OD - 0.01, 0.01):
        for DZ in np.arange(0.01, 0.1, 0.01):

            coil_models = pd.read_csv('coil_models/coil_model-tests.csv')

            nr = round((OD - ID) / 0.02, 10)
            nz = round(DZ / 0.01, 10)

            if not nr.is_integer() or not nz.is_integer():
                continue

            coil_idx = coil_models[coil_models['type'].str.strip() == 'coil'].index[0]
            coil_models.loc[coil_idx, 'OD'] = OD
            coil_models.loc[coil_idx, 'ID'] = ID
            coil_models.loc[coil_idx, 'DZ'] = DZ
            coil_models.loc[coil_idx, 'Nr'] = nr
            coil_models.loc[coil_idx, 'Nz'] = nz
            coil_models.to_csv('coil_models/coil_model-tests.csv', index=False)


            field._COIL_MODELS = None

            # main from field.py
            import sys
            fin = sys.argv[1]
            df = pd.read_csv(fin).dropna(how='all')
            df.columns = df.columns.str.strip()  # fix headers
            coils = [Coil(**row) for row in df.to_dict('records')]

            axis_xy = df[['Xc', 'Yc']].to_numpy()
            axis_path = interpolate_axis(axis_xy, AXIS_SAMPLES_PER_SEGMENT)
            axis_points = np.column_stack([axis_path, np.zeros(len(axis_path))])

            if len(coils) != 1:
                continue

            # Determine B-field in center of coil based on Current
            coil = coils[0]
            coil.current = 1.0
            B_total = coil.magnetic_field(axis_points)
            B_mag = np.linalg.norm(B_total, axis=1)
            G_per_A = np.max(B_mag) * 10000

            coil_length = coil.get_length(False)
            resistance = coil_length * 0.00033

            coil_volume = coil_length * (0.008**2 - np.pi * 0.002**2)
            coil_mass = coil_volume * CU_DENSITY

            power = coil.current**2 * resistance
            temp_change = power / coil_mass / 385

            results.append({
                'OD': round(OD, 4),
                'ID': round(ID, 4),
                'DZ': round(DZ, 4),
                'Nr': nr,
                'Nz': nz,
                'G_per_A': round(G_per_A, 4),
                'Length_m': round(coil_length, 4),
                'Resistance_mOhm': round(resistance * 1000, 4),
                'Mass_kg': round(coil_mass, 4),
                'Power_W': round(power, 6),
                'TempChange_C_per_s': round(temp_change, 9),
            })

results_df = pd.DataFrame(results).sort_values('G_per_A').reset_index(drop=True)
results_df.to_csv('possible-coil.csv', index=False)
print(f"Saved {len(results)} results to possible-coil.csv")



import matplotlib.pyplot as plt
for OD in [0.4, 0.5]:
    subset = results_df[results_df['OD'] == OD]
    pivot = subset.pivot(index='ID', columns='DZ', values='G_per_A')
    pivot_length = subset.pivot(index='ID', columns='DZ', values='Length_m')

    fig, ax = plt.subplots()
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='inferno')
    fig.colorbar(im, ax=ax, label='G/A')

    threshold = [5, 10, 15, 20, 30, 40, 50, 60]

    contour = ax.contour(pivot_length.columns, pivot_length.index, pivot_length.values, levels=threshold, colors='white', linestyles='--')
    ax.clabel(contour, fmt=lambda x: f'{x} m', inline=True, fontsize=8)
    ax.set_xlabel('Axial Length, Dz (m)')
    ax.set_ylabel('Inner Diameter, ID (m)')
    ax.set_title('G/A for Different Coil Geometries')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(subset['Length_m'], subset['G_per_A'], c=subset['Mass_kg'], cmap='viridis', s=100, edgecolors='k')
    ax.set_xlabel('Coil Length (m)')
    ax.set_ylabel('G/A')
    ax.set_title('G/A vs Coil Length')
    plt.colorbar(ax.collections[0], ax=ax, label='Mass (kg)')
    plt.tight_layout()
    plt.show()
