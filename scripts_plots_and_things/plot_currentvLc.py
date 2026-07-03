import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Lc_curve.csv')
df = df.sort_values('Current').reset_index(drop=True)

current = df['Current'].values
blue_xc = df['Blue_Xc'].values
brown_xc = df['Brown_Xc'].values

Dz_brown = 0.0886

Lc = blue_xc - (brown_xc - Dz_brown/2)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# ax1.plot(Lc, current, label='Lc (Blue - Brown)', color='blue', marker='o')
ax2.plot(Lc, blue_xc, label='Blue Xc', color='navy', marker='o')
ax2.plot(Lc, brown_xc, label='Brown Xc', color='brown', marker='o')
ax1.plot(Lc, current/1.28, label='scaled down current by 1.28', color='orange', marker='o')

ax1.set_xlabel('Lc (m)')
ax1.set_ylabel('Current (A)')
ax2.set_ylabel('Xc (m)')
ax1.set_title('Lc and Xc vs Current')
ax1.grid()

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.savefig('Lc_vs_Current.png', dpi=300)
plt.show()
