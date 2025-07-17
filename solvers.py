# -*- coding: utf-8 -*-
"""
Numerical Solvers for the Wave Equation

This module provides high-performance solvers for:
1.  The generalized eigenvalue problem (K*v = lambda*M*v) to find the
    vibrational modes of the system.
2.  The time-domain simulation of the wave equation (M*u_tt + K*u = 0)
    using an efficient leapfrog integration scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse.linalg import eigsh, splu


def solve_eigenproblem(K, M, num_modes):
    """
    Solves the generalized eigenvalue problem K*v = lambda*M*v.

    This finds the natural vibrational frequencies (eigenvalues) and mode
    shapes (eigenvectors) of the snowflake drum.

    Args:
        K (scipy.sparse.csc_matrix): Stiffness matrix.
        M (scipy.sparse.csc_matrix): Mass matrix.
        num_modes (int): The number of lowest-frequency eigenmodes to compute.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Eigenvalues (squared angular frequencies, lambda).
            - numpy.ndarray: Eigenvectors (mode shapes), with shape (N, num_modes).
    """
    print(f"Solving eigenvalue problem for {num_modes} smallest non-trivial modes...")

    # We want the smallest non-trivial eigenvalues. 'SM' stands for Smallest
    # Magnitude. Using sigma=0 is a highly effective shift-invert strategy
    # for finding eigenvalues near zero quickly and accurately.
    try:
        # We ask for a few more modes to discard any trivial (zero) modes if they appear.
        eigenvalues, eigenvectors = eigsh(K, k=num_modes + 2, M=M, which='SM', sigma=0)
    except Exception as e:
        print(f"eigsh with sigma=0 failed ({e}), trying without shift-invert...")
        eigenvalues, eigenvectors = eigsh(K, k=num_modes + 2, M=M, which='SM')

    # The eigenvalues from the solver are lambda = omega^2.
    # We sort them to ensure they are in ascending order of frequency.
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Return the requested number of modes, discarding the lowest ones which might be zero.
    # This is a common practice to get the first few non-trivial vibrational modes.
    if len(eigenvalues) > num_modes:
        eigenvalues = eigenvalues[:num_modes]
        eigenvectors = eigenvectors[:, :num_modes]

    print("Eigenvalue problem solved.")
    return eigenvalues, eigenvectors


def run_simulation(K, M, verts, tris, initial_u, initial_v, total_time, num_steps):
    """
    Runs the time-domain simulation using a leapfrog integrator.

    The equation to solve is M*u_tt + K*u = 0.
    The leapfrog scheme is chosen for its efficiency and stability (it is
    symplectic and conserves energy over long simulations).

    Args:
        K, M: Stiffness and Mass matrices.
        verts, tris: Mesh data for plotting.
        initial_u, initial_v: Initial displacement and velocity vectors.
        total_time (float): Total simulation time.
        num_steps (int): Number of time steps to simulate.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    print("Initializing time-domain simulation...")

    dt = total_time / num_steps

    # For the equation a = M_inv * f, we never compute M_inv explicitly.
    # Instead, we pre-factorize M and use a sparse solver, which is much
    # faster and more numerically stable.
    M_inv_solver = splu(M.tocsc())

    u_current = initial_u.copy()
    v_current = initial_v.copy()

    # The leapfrog method requires an initial half-step for the velocity.
    # a_0 = -M_inv * K * u_0
    acceleration = M_inv_solver.solve(-K @ u_current)
    v_current -= 0.5 * dt * acceleration

    # --- Set up the animation ---
    fig, ax = plt.subplots(figsize=(10, 8.5))
    fig.tight_layout()

    # Use a diverging colormap, ideal for visualizing vibrations (positive/negative displacement)
    cmap = 'seismic'

    # Add a colorbar placeholder that we can update
    vmax = np.max(np.abs(u_current))
    if vmax < 1e-9: vmax = 1e-9
    tripcolor = ax.tripcolor(verts[:, 0], verts[:, 1], tris, u_current,
                             cmap=cmap, vmin=-vmax, vmax=vmax, shading='gouraud')
    cbar = fig.colorbar(tripcolor, ax=ax, fraction=0.046, pad=0.04)

    def update_frame(frame):
        nonlocal u_current, v_current

        # Leapfrog integration step:
        # 1. Compute acceleration: a_n = -M_inv * K * u_n
        acceleration = M_inv_solver.solve(-K @ u_current)
        # 2. Update velocity: v_{n+1/2} = v_{n-1/2} + dt * a_n
        v_current += dt * acceleration
        # 3. Update position: u_{n+1} = u_n + dt * v_{n+1/2}
        u_current += dt * v_current

        # --- Render the frame ---
        ax.clear()
        ax.set_aspect('equal')
        ax.set_axis_off()

        # Dynamically adjust the color limits based on the current wave amplitude
        vmax = np.max(np.abs(u_current))
        if vmax < 1e-9: vmax = 1e-9

        # Use Gouraud shading for a smooth appearance
        ax.tripcolor(verts[:, 0], verts[:, 1], tris, u_current,
                     cmap=cmap, vmin=-vmax, vmax=vmax, shading='gouraud')

        ax.set_title(f"Wave Propagation on Koch Snowflake\nTime: {frame * dt:.2f}s")

        if frame % 10 == 0:
            print(f"  Simulating frame {frame}/{num_steps}", end='\r')

        # Returning the artists to be redrawn
        return ax.collections

    # Create the animation object. `blit=False` is safer when clearing axes.
    ani = animation.FuncAnimation(fig, update_frame, frames=num_steps,
                                  interval=20, blit=False)
    return ani
