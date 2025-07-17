# -*- coding: utf-8 -*-
"""
Visualization Module for the Koch Snowflake Simulation

This script provides functions to create high-quality plots and animations
of the simulation results, including the vibrational eigenmodes and the
time-dependent wave propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plot_eigenmodes(verts, tris, eigenvalues, eigenvectors, num_modes_to_plot, koch_iterations):
    """
    Plots the computed eigenmodes (vibrational patterns) of the snowflake.

    Each subplot shows a different mode shape, ordered by frequency, along
    with its calculated vibrational frequency.

    Args:
        verts (numpy.ndarray): Mesh vertices.
        tris (numpy.ndarray): Mesh elements (triangles).
        eigenvalues (numpy.ndarray): The computed eigenvalues (lambda = omega^2).
        eigenvectors (numpy.ndarray): The computed eigenvectors (mode shapes).
        num_modes_to_plot (int): The number of modes to display.
        koch_iterations (int): The iteration number of the snowflake, for the title.
    """
    print("Generating eigenmode plot...")

    # Frequencies f = omega / (2*pi) = sqrt(lambda) / (2*pi)
    # We add a small epsilon to avoid sqrt of negative numbers due to numerical noise
    frequencies = np.sqrt(np.maximum(0, eigenvalues)) / (2 * np.pi)

    # Determine a good grid size for the subplots
    cols = int(np.ceil(np.sqrt(num_modes_to_plot)))
    rows = int(np.ceil(num_modes_to_plot / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5),
                             constrained_layout=True)
    axes = axes.flatten()

    for i in range(num_modes_to_plot):
        if i >= len(eigenvalues):
            axes[i].set_axis_off()  # Turn off empty subplots
            continue

        ax = axes[i]
        mode_shape = eigenvectors[:, i]

        # Normalize the mode shape for consistent coloring across different modes
        vmax = np.max(np.abs(mode_shape))
        if vmax < 1e-9: vmax = 1e-9

        ax.set_aspect('equal')
        ax.set_axis_off()

        # tripcolor with Gouraud shading creates a beautiful, smooth plot
        ax.tripcolor(verts[:, 0], verts[:, 1], tris, mode_shape,
                     cmap='seismic', vmin=-vmax, vmax=vmax, shading='gouraud')

        # Use LaTeX for a professional-looking title
        title = f"Mode {i + 1}\n$f={frequencies[i]:.3f}$ Hz"
        ax.set_title(title)

    # Hide any unused subplots at the end
    for i in range(num_modes_to_plot, len(axes)):
        axes[i].set_axis_off()

    fig.suptitle(f'Vibrational Modes of a Koch Snowflake Drum (N={koch_iterations})', fontsize=16)
    plt.show()

