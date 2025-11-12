#!/usr/bin/env python3
"""Visualize rectangular coil cross-sections and their filament dots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _unit_radial(angle_deg: float) -> np.ndarray:
    """Return a unit vector in XY pointing toward the coil's ID."""
    angle_rad = np.deg2rad(angle_deg)
    vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
    return vec / np.linalg.norm(vec)


def _rectangle_vertices(center: np.ndarray, u_r: np.ndarray, radial_width: float, axial_width: float) -> list[np.ndarray]:
    """Return the four 3D vertices of the radial-axial rectangle."""
    half_r = radial_width / 2.0
    half_z = axial_width / 2.0
    e_z = np.array([0.0, 0.0, 1.0], dtype=float)
    corners = []
    for dr, dz in [(-half_r, -half_z), (half_r, -half_z), (half_r, half_z), (-half_r, half_z)]:
        corners.append(center + dr * u_r + dz * e_z)
    return corners


def _filament_points(center: np.ndarray, u_r: np.ndarray, radial_width: float, axial_width: float, nr: int, nz: int) -> np.ndarray:
    """Return NR x NZ filament coordinates across the radial/axial grid."""
    half_r = radial_width / 2.0
    half_z = axial_width / 2.0
    e_z = np.array([0.0, 0.0, 1.0], dtype=float)

    def _linspace(count: int, half_width: float) -> np.ndarray:
        if count <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(-half_width, half_width, count)

    offsets_r = _linspace(nr, half_r)
    offsets_z = _linspace(nz, half_z)
    points = []
    for dr in offsets_r:
        for dz in offsets_z:
            points.append(center + dr * u_r + dz * e_z)
    return np.array(points)


def plot_coils(ax, coils: pd.DataFrame) -> None:
    colors = plt.cm.tab10.colors
    for idx, row in coils.iterrows():
        center = np.array([row["X0"], row["Y0"], 0.0], dtype=float)
        u_r = _unit_radial(row["angle"])
        radial_width = (row["OD"] - row["ID"]) / 2.0
        axial_width = row["DZ"]
        nr = max(int(row["Nr"]), 1)
        nz = max(int(row["Nz"]), 1)

        color = colors[idx % len(colors)]
        vertices = _rectangle_vertices(center, u_r, radial_width, axial_width)
        poly = Poly3DCollection([vertices], facecolor=color, alpha=0.35, edgecolor=color, linewidths=1.2)
        ax.add_collection3d(poly)

        filaments = _filament_points(center, u_r, radial_width, axial_width, nr, nz)
        ax.scatter(filaments[:, 0], filaments[:, 1], filaments[:, 2], color=color, s=12)
        ax.text(center[0], center[1], center[2], f"Coil {idx}", color=color)


def _set_equal_scale(ax) -> None:
    """Force equal scaling across X/Y/Z for accurate box proportions."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    ranges = np.array([x_limits, y_limits, z_limits])
    spans = ranges[:, 1] - ranges[:, 0]
    centers = np.mean(ranges, axis=1)
    max_span = max(spans)

    for center, setter in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], strict=True):
        setter(center - max_span / 2.0, center + max_span / 2.0)


def load_coils(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(how="all")
    df.columns = [col.strip() for col in df.columns]
    numeric_cols = ["X0", "Y0", "angle", "ID", "OD", "DZ", "Nr", "Nz"]
    return df[numeric_cols].apply(pd.to_numeric)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coil boxes and filament locations from a CSV file.")
    parser.add_argument("csv", nargs="?", default="test-coil.csv", help="Path to the coil definition CSV.")
    parser.add_argument("--output", "-o", help="Optional path to save the figure instead of displaying it.")
    args = parser.parse_args()

    coils = load_coils(Path(args.csv))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_coils(ax, coils)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Coil Cross-Sections and Filaments")
    ax.view_init(elev=20, azim=-60)
    _set_equal_scale(ax)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
