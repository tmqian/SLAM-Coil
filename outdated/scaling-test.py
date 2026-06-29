import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('possible-coil.csv')
# Convert to numeric, turning bad data into NaN
df['DZ'] = pd.to_numeric(df['DZ'], errors='coerce')

# Drop any NaNs if you want to be safe, then filter
df = df.dropna(subset=['DZ'])
df = df[df['DZ'] <= 0.4]

x = 1 / np.sqrt(df['ID'] * df['OD'])
y = df['G_per_A']

fig, ax = plt.subplots(figsize=(12, 8))

# Expand domain to avoid points at edges
x_margin = (x.max() - x.min()) * 0.1
y_margin = (y.max() - y.min()) * 0.1
ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

# Add shading for different ID regions
id_shading = ax.tricontourf(x, y, df['ID'], levels=10, alpha=0.3, cmap='viridis')
cbar_id = fig.colorbar(id_shading, ax=ax, label='ID (m)', pad=0.15)

scatter = ax.scatter(x, y, c=df['DZ'], cmap='rainbow', s=50, edgecolors='white')
fig.colorbar(scatter, ax=ax, label='DZ (m)')

contour = ax.tricontour(x, y, df['Length_m'], levels=8, colors='black', linestyles='-', linewidths=1)
texts = ax.clabel(contour, fmt=lambda v: f'{v:.0f} m', inline=True, fontsize=8)
for t in texts:
    t.set_fontweight('bold')

ax.set_xlabel('1 / sqrt(ID * OD)  (m⁻¹)')
ax.set_ylabel('G/A')
ax.set_title('G/A vs 1/√(ID·OD)')
ax.grid(True)
plt.tight_layout()
plt.show()

