# -*- coding: utf-8 -*-
"""
Main Driver Script for the Koch Snowflake Wave Simulation

This script orchestrates the entire process of simulating wave propagation on a
Koch snowflake domain. It serves as the entry point for the project.

Workflow:
1.  Sets up global configuration parameters for the simulation.
2.  Generates the Koch snowflake boundary points using the 'koch_geometry' module.
3.  Creates a high-quality triangular mesh of the domain using the 'fem' module.
4.  Assembles the sparse mass and stiffness matrices using the 'fem' module.
5.  Solves the eigenvalue problem to find the snowflake's vibrational modes
    and frequencies using the 'solvers' module.
6.  Plots the beautiful eigenmodes using the 'visualization' module.
7.  Runs the time-domain simulation of a wave pulse using the 'solvers' module.
8.  Saves the resulting wave animation to a file.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import project modules
import koch_geometry
import fem
import solvers
import visualization

# --- Configuration ---
KOCH_ITERATIONS = 4          # Fractal depth of the snowflake (3-5 is reasonable)
MESH_MAX_AREA = 1e-4         # Controls mesh density (smaller is finer)
WAVE_SPEED = 1.0             # Speed of wave propagation (c)
SIMULATION_TIME = 5.0        # Total simulation time in seconds
TIME_STEPS = 500             # Number of time steps
INITIAL_CONDITION_TYPE = 'gaussian' # 'gaussian' or 'eigenmode'
EIGENMODE_TO_SIMULATE = 5    # Which eigenmode to use for initial condition (if selected)
NUM_EIGENMODES_TO_COMPUTE = 12 # Number of eigenmodes to calculate and plot
SAVE_ANIMATION = True        # Whether to save the animation to a file
ANIMATION_FILENAME = 'koch_snowflake_wave.mp4'

def main():
    """Main function to run the entire simulation pipeline."""
    
    print("--- Starting Koch Snowflake Wave Simulation ---")

    # 1. Generate snowflake boundary
    print("\nStep 1: Generating Koch snowflake geometry...")
    snowflake_boundary_points = koch_geometry.get_koch_snowflake_points(KOCH_ITERATIONS)

    # 2. Create the mesh
    print("\nStep 2: Creating triangular mesh...")
    mesh_verts, mesh_tris = fem.create_mesh(snowflake_boundary_points, MESH_MAX_AREA)

    # 3. Assemble FEM matrices
    print("\nStep 3: Assembling FEM matrices...")
    K, M = fem.assemble_matrices(mesh_verts, mesh_tris, snowflake_boundary_points)

    # 4. Solve the eigenvalue problem
    print("\nStep 4: Solving eigenvalue problem...")
    eigenvalues, eigenvectors = solvers.solve_eigenproblem(K, M, NUM_EIGENMODES_TO_COMPUTE)

    # 5. Plot the eigenmodes
    print("\nStep 5: Plotting eigenmodes...")
    visualization.plot_eigenmodes(
        mesh_verts, mesh_tris, eigenvalues, eigenvectors, 
        NUM_EIGENMODES_TO_COMPUTE, KOCH_ITERATIONS
    )
    
    # 6. Set up and run the time-domain simulation
    print("\nStep 6: Running time-domain simulation...")
    u0 = np.zeros(len(mesh_verts))
    v0 = np.zeros(len(mesh_verts))

    if INITIAL_CONDITION_TYPE == 'gaussian':
        # A Gaussian pulse at the center
        center = np.array([0, 0])
        dist_sq = np.sum((mesh_verts - center)**2, axis=1)
        u0 = np.exp(-dist_sq / 0.01)
    elif INITIAL_CONDITION_TYPE == 'eigenmode':
        # Start with the shape of a specific eigenmode
        if EIGENMODE_TO_SIMULATE < len(eigenvalues):
            u0 = eigenvectors[:, EIGENMODE_TO_SIMULATE]
        else:
            print(f"Warning: Eigenmode {EIGENMODE_TO_SIMULATE} not available. Defaulting to Gaussian.")
            center = np.array([0, 0])
            dist_sq = np.sum((mesh_verts - center)**2, axis=1)
            u0 = np.exp(-dist_sq / 0.01)

    # Apply boundary conditions to initial state
    boundary_nodes = fem.get_boundary_nodes(mesh_verts, snowflake_boundary_points)
    u0[boundary_nodes] = 0.0
    
    wave_animation = solvers.run_simulation(
        K, M, mesh_verts, mesh_tris, u0, v0, 
        SIMULATION_TIME, TIME_STEPS
    )

    # 7. Save or show the animation
    if SAVE_ANIMATION:
        print(f"\nStep 7: Saving animation to '{ANIMATION_FILENAME}'...")
        try:
            wave_animation.save(
                ANIMATION_FILENAME, 
                writer='ffmpeg', 
                fps=30,
                extra_args=['-vcodec', 'libx264']
            )
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Displaying animation instead.")
            plt.show()
    else:
        print("\nStep 7: Displaying animation...")
        plt.show()

    print("\n--- Simulation Complete ---")


if __name__ == '__main__':
    main()
