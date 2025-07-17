# -*- coding: utf-8 -*-
"""
Finite Element Method (FEM) Module

This script contains functions for mesh generation and the assembly of
FEM matrices (mass and stiffness) for solving the 2D wave equation.
"""

import numpy as np
import meshpy.triangle as triangle
from scipy.sparse import lil_matrix, csc_matrix


def create_mesh(boundary_points, max_area):
    """
    Creates a high-quality triangular mesh for the given domain.

    Uses the 'triangle' library to perform Delaunay triangulation with
    constraints on triangle quality and area.

    Args:
        boundary_points (numpy.ndarray): (N, 2) array of vertices defining
                                         the domain boundary.
        max_area (float): The maximum area for any triangle in the mesh.
                          A smaller value leads to a finer, denser mesh.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Mesh vertices (nodes).
            - numpy.ndarray: Mesh elements (triangles), as indices into the
                             vertex array.
    """
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(boundary_points)

    # Define segments connecting the boundary points to form a closed loop
    segments = [(i, i + 1) for i in range(len(boundary_points) - 1)]
    segments.append((len(boundary_points) - 1, 0))
    mesh_info.set_facets(segments)

    # Build the mesh.
    # 'p' -> Planar Straight Line Graph (the boundary).
    # 'q' -> Quality mesh generation (avoids skinny triangles).
    # 'a' -> Imposes a maximum triangle area constraint.
    # 'z' -> Zero-based indexing for elements.
    mesh = triangle.build(mesh_info, quality_meshing=True, max_volume=max_area)

    verts = np.array(mesh.points)
    tris = np.array(mesh.elements)

    print(f"Mesh generated: {len(verts)} vertices, {len(tris)} triangles.")
    return verts, tris


def get_boundary_nodes(mesh_verts, boundary_points):
    """
    Identifies which nodes in the mesh lie on the boundary.

    Args:
        mesh_verts (numpy.ndarray): All vertices in the mesh.
        boundary_points (numpy.ndarray): The original boundary points.

    Returns:
        numpy.ndarray: An array of indices corresponding to the boundary nodes.
    """
    # A robust way to find boundary nodes is to check which mesh vertices
    # are also present in the original boundary definition.
    # Using a tolerance `rtol` for floating point comparison.
    is_boundary = np.array([np.any(np.all(np.isclose(v, boundary_points), axis=1)) for v in mesh_verts])
    return np.where(is_boundary)[0]


def assemble_matrices(verts, tris, boundary_points):
    """
    Assembles the FEM mass (M) and stiffness (K) matrices.

    This uses linear Lagrangian basis functions (P1 "hat" functions) on each
    triangular element. The resulting matrices are sparse, which is crucial
    for performance and memory efficiency.

    Args:
        verts (numpy.ndarray): Mesh vertices.
        tris (numpy.ndarray): Mesh elements (triangles).
        boundary_points (numpy.ndarray): The boundary points used to identify
                                         boundary nodes for applying Dirichlet
                                         conditions.

    Returns:
        tuple: A tuple containing:
            - scipy.sparse.csc_matrix: The sparse stiffness matrix (K).
            - scipy.sparse.csc_matrix: The sparse mass matrix (M).
    """
    num_verts = len(verts)
    # Use LIL format for efficient incremental construction
    K = lil_matrix((num_verts, num_verts))
    M = lil_matrix((num_verts, num_verts))

    # Loop over each triangle in the mesh
    for tri_indices in tris:
        # Get coordinates of the triangle's three vertices
        p = verts[tri_indices]

        # Calculate area using the explicit formula (more direct than determinant)
        area = 0.5 * np.abs(
            p[0, 0] * (p[1, 1] - p[2, 1]) + p[1, 0] * (p[2, 1] - p[0, 1]) + p[2, 0] * (p[0, 1] - p[1, 1]))
        if area < 1e-14:  # Skip degenerate triangles
            continue

        # For a linear basis function N_i = a_i + b_i*x + c_i*y, the gradient is [b_i, c_i].
        # The b_i and c_i coefficients can be calculated directly from vertex coordinates.
        b_vec = np.array([p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]])
        c_vec = np.array([p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]])

        # Element stiffness matrix Ke_ij = integral(grad(N_i) . grad(N_j)) dA
        # This simplifies to (1 / 4A) * (b_i*b_j + c_i*c_j) for i,j in {0,1,2}
        Ke = (1.0 / (4.0 * area)) * (np.outer(b_vec, b_vec) + np.outer(c_vec, c_vec))

        # Element mass matrix (Me_ij = integral(N_i * N_j) dA)
        # This is a "consistent" mass matrix, which is more accurate than lumping.
        Me = area / 12.0 * (np.ones((3, 3)) + np.eye(3))

        # Add the 3x3 element matrices to the global (num_verts x num_verts) matrices
        for i in range(3):
            for j in range(3):
                K[tri_indices[i], tri_indices[j]] += Ke[i, j]
                M[tri_indices[i], tri_indices[j]] += Me[i, j]

    # Identify boundary nodes to apply Dirichlet boundary conditions (u=0)
    boundary_nodes = get_boundary_nodes(verts, boundary_points)

    # Modify K and M to enforce u=0 on the boundary.
    # This is a standard technique to constrain the system.
    for node_idx in boundary_nodes:
        K[node_idx, :] = 0
        K[:, node_idx] = 0
        K[node_idx, node_idx] = 1.0
        M[node_idx, :] = 0
        M[node_idx, node_idx] = 1.0  # Keeps M non-singular

    print("FEM matrices assembled and boundary conditions applied.")
    # Convert to CSC format for fast arithmetic operations
    return csc_matrix(K), csc_matrix(M)
