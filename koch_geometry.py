# -*- coding: utf-8 -*-
"""
Module for Generating Koch Snowflake Geometry

This script provides a function to generate the vertices of a Koch snowflake,
a classic fractal curve. The generation is done using complex numbers for
elegant rotation and scaling operations.
"""

import numpy as np

def get_koch_snowflake_points(iterations):
    """
    Generates the vertices of a Koch snowflake.

    The process starts with an equilateral triangle and iteratively refines
    each line segment by adding a triangular bump.

    Args:
        iterations (int): The number of iterations for the fractal. A higher
                          number results in a more detailed snowflake.

    Returns:
        numpy.ndarray: A 2D array of (x, y) coordinates for the points
                       on the snowflake boundary, ordered sequentially.
    """
    def rotate(p, angle, center):
        """Helper function to rotate a complex number `p` around a `center`."""
        return (p - center) * np.exp(1j * angle) + center

    # Start with an equilateral triangle centered at the origin
    # Using complex numbers simplifies rotations and translations
    points = [np.exp(1j * 2 * np.pi * i / 3) for i in range(3)]
    points.append(points[0]) # Close the loop
    points = np.array(points)

    print(f"Generating snowflake with {iterations} iterations...")
    for n in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            
            new_points.append(p1)
            
            # Trisect the segment from p1 to p2
            s1 = p1 + (p2 - p1) / 3.0
            s2 = p1 + 2.0 * (p2 - p1) / 3.0
            
            # Create the tip of the new equilateral triangle.
            # This involves rotating the segment (s2 - s1) by -60 degrees (-pi/3)
            # around the point s1.
            tip = rotate(s2, -np.pi / 3.0, s1)
            
            new_points.extend([s1, tip, s2])
        
        new_points.append(points[-1]) # Add the final point
        points = np.array(new_points)
        print(f"  Iteration {n+1}: {len(points)-1} segments")

    # Convert complex numbers to a (N, 2) numpy array of (x, y) coordinates
    final_points = np.array([points.real, points.imag]).T
    
    # Scale the snowflake to fit nicely in a [-1, 1] x [-1, 1] box
    max_val = np.max(np.abs(final_points))
    if max_val > 0:
        final_points /= max_val
        
    print(f"Snowflake geometry generated with {len(final_points)} vertices.")
    return final_points
