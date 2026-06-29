# pyright: standard
import matplotlib.pyplot as plt
import numpy as np

from field import axis_field_plot, contour_plot, get_coil_info

CURRENT = 250  # Amps
THRESHOLD = 875.0  # Gauss


# Get axis path
coils, path = get_coil_info("test_files/OMirror.csv", interpolate=True, L=1.5, R=0.42)
path = path[:-2]

coils[0].current *= 2
coils[-1].current *= 2

fig, ax = plt.subplots()
contour_plot(
    fig,
    ax,
    coils,
    path,
)

plt.savefig("generated/plots/OM_Contour.png")

# Graph field on-axis
fix, ax = plt.subplots()
axis_field_plot(
    ax,
    coils,
    path,
)

plt.savefig("generated/plots/OM_OnAxis.png")
