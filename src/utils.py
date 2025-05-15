import numpy as np
import pandas as pd

from shapely.geometry import Polygon, Point

# ======================================
# Cell Level VSI
# ======================================


def extract_cell_vsi(
    boundary_df, 
    integrity,
    strength,
    integrity_size=1800
):
    """
    Extracts the cell integrity and strength arrays based on polygonal boundaries.

    Parameters:
        boundary_df (pd.DataFrame): DataFrame with 'boundaryX' and 'boundaryY' columns.
        integrity (np.ndarray): 2D array of integrity values.
        strength (np.ndarray): 2D array of signal strength values.
        integrity_size (int): Size of the output grid (assumed square).

    Returns:
        tuple: (cell_integrity, cell_strength) as 2D numpy arrays.
    """

    cell_integrity = np.zeros((integrity_size, integrity_size))
    cell_strength = np.zeros((integrity_size, integrity_size))

    for idx, row in boundary_df.iterrows():
        x_coords = np.array(row['boundaryX'])
        y_coords = np.array(row['boundaryY'])

        # Filter out NaN values
        valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if len(x_coords) < 3:
            continue  # Not enough points to form a polygon

        polygon = Polygon(zip(x_coords, y_coords))
        if not polygon.is_valid:
            continue  # Skip invalid polygons

        x_min, x_max = int(np.floor(polygon.bounds[0])), int(np.ceil(polygon.bounds[2]))
        y_min, y_max = int(np.floor(polygon.bounds[1])), int(np.ceil(polygon.bounds[3]))

        x_min, x_max = max(0, x_min), min(integrity_size, x_max)
        y_min, y_max = max(0, y_min), min(integrity_size, y_max)

        y_indices, x_indices = np.meshgrid(range(y_min, y_max), range(x_min, x_max), indexing='ij')
        points = np.column_stack([x_indices.ravel(), y_indices.ravel()])

        epsilon = 0.7
        mask = np.array([
            polygon.contains(Point(x, y)) or
            polygon.touches(Point(x, y)) or
            polygon.boundary.distance(Point(x, y)) < epsilon
            for x, y in points
        ])
        mask = mask.reshape(y_indices.shape)

        subgrid_int = integrity[y_min:y_max, x_min:x_max]
        subgrid_str = strength[y_min:y_max, x_min:x_max]

        cell_integrity[y_min:y_max, x_min:x_max][mask] = subgrid_int[mask]
        cell_strength[y_min:y_max, x_min:x_max][mask] = subgrid_str[mask]

    return cell_integrity, cell_strength