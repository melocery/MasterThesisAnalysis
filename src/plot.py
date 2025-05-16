import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, leaves_list

from sklearn.neighbors import NearestNeighbors

import umap
from sklearn.decomposition import PCA

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

_BIH_CMAP = LinearSegmentedColormap.from_list(
    "BIH",
    [
        "#430541",
        "mediumvioletred",
        "violet",
        "powderblue",
        "powderblue",
        "white",
        "white",
    ][::-1],
)

HEATMAP_CMAP = sns.color_palette("RdYlBu_r", as_cmap=True)

# ======================================
# Vertical Signal Integrity Map
# ======================================
def plot_VSI_map(
    cell_integrity,
    cell_strength,
    signal_threshold=3.0,
    figure_height=10,
    cmap="BIH",
    side_display=None,  # "hist", "colorbar", or None
    boundary_df=None,
    plot_boundarys=False,
    boundary_color="yellow",
    boundary_width=1.5,
    cell_centroid=None,
    plot_centroid=False,
    x_range=None,
    y_range=None,
    show=False
):
    """
    Visualize the VSI (signal integrity) map with optional histogram/colorbar and overlays.

    Parameters:
        cell_integrity (2D np.ndarray): Signal integrity matrix.
        cell_strength (2D np.ndarray): Signal strength matrix.
        signal_threshold (float): Threshold for alpha masking.
        figure_height (float): Height of the figure in inches.
        cmap (str or colormap): Colormap name or object.
        side_display (str or None): "hist", "colorbar", or None.
        boundary_df (pd.DataFrame): Optional DataFrame with boundaryX and boundaryY columns.
        plot_boundarys (bool): Whether to draw boundaries.
        boundary_color (str): Color of the boundary lines.
        boundary_width (float): Width of the boundary lines.
        cell_centroid (pd.DataFrame): DataFrame with x and y coordinates.
        plot_centroid (bool): Whether to overlay centroids.
        x_range, y_range (list or tuple): Display range for x and y axes.
    """
    if not (isinstance(cell_integrity, np.ndarray) and cell_integrity.ndim == 2):
        raise ValueError("cell_integrity must be a 2D numpy array.")
    if not (isinstance(cell_strength, np.ndarray) and cell_strength.ndim == 2):
        raise ValueError("cell_strength must be a 2D numpy array.")

    aspect_ratio = cell_integrity.shape[0] / cell_integrity.shape[1]

    with plt.style.context("dark_background"):
        # Handle colormap
        if cmap == "BIH":
            try:
                cmap = _BIH_CMAP
            except NameError:
                raise ValueError("BIH colormap is not defined.")

        side_display = str(side_display).lower() if side_display else None
        show_hist = side_display == "hist"
        show_colorbar = side_display == "colorbar"

        # Define figure layout
        if show_hist:
            fig, ax = plt.subplots(
                1, 2,
                figsize=(figure_height / aspect_ratio * 1.4, figure_height),
                gridspec_kw={"width_ratios": [6, 1]},
                dpi=600
            )
        else:
            fig, ax = plt.subplots(
                1, 1,
                figsize=(figure_height / aspect_ratio, figure_height),
                dpi=600
            )
            ax = [ax]

        # Main heatmap
        img = ax[0].imshow(
            cell_integrity,
            cmap=cmap,
            alpha=((cell_strength / signal_threshold).clip(0, 1) ** 2),
            vmin=0,
            vmax=1,
        )
        ax[0].invert_yaxis()
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Display region (whole image or custom ROI)
        if x_range is not None:
            ax[0].set_xlim(*x_range)
        else:
            ax[0].set_xlim(0, cell_integrity.shape[1])
        if y_range is not None:
            ax[0].set_ylim(*y_range)
        else:
            ax[0].set_ylim(0, cell_integrity.shape[0])

        # Optional centroids
        if plot_centroid and cell_centroid is not None:
            ax[0].scatter(cell_centroid['x'], cell_centroid['y'], s=1, c='orange', alpha=0.1)

        # Optional boundaries
        if plot_boundarys and boundary_df is not None:
            for _, row in boundary_df.iterrows():
                ax[0].plot(row['boundaryX'], row['boundaryY'],
                           c=boundary_color, linewidth=boundary_width)

        # Optional histogram
        if show_hist:
            vals, bins = np.histogram(
                cell_integrity[cell_strength > signal_threshold],
                bins=50, range=(0, 1), density=True
            )
            bars = ax[1].barh(bins[1:-1], vals[1:], height=0.01)
            for i, bar in enumerate(bars):
                bar.set_color(cmap(bins[1:-1][i]))

            ax[1].set_ylim(0, 1)
            ax[1].invert_xaxis()
            ax[1].set_xticks([])
            ax[1].yaxis.tick_right()
            ax[1].spines[["top", "bottom", "left"]].set_visible(False)
            ax[1].set_ylabel("signal integrity")
            ax[1].yaxis.set_label_position("right")

        elif show_colorbar:
            fig.colorbar(img, ax=ax[0], shrink=0.8)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_named_squares(ax, square_list, default_color="yellow", default_size=100):
    """
    add squares and annotations

    square_list: list of dicts like:
        {"x": ..., "y": ..., "name": ..., "size": ..., "color": ...}
    """
    for sq in square_list:
        x = sq.get("x")
        y = sq.get("y")
        size = sq.get("size", default_size)
        name = sq.get("name", "")
        color = sq.get("color", default_color)

        rect = patches.Rectangle(
            (x, y), size, size,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x + size / 2, y + size + 10,
            name,
            color=color,
            fontsize=6,
            fontweight='bold',
            ha='center'
        )

def plot_vsi_with_named_squares(
    cell_integrity,
    cell_strength,
    named_squares=None,
    **kwargs
):
    fig = plot_VSI_map(cell_integrity, cell_strength, **kwargs)
    if named_squares:
        ax = fig.axes[0]
        plot_named_squares(ax, named_squares)

    return fig

# ======================================
# integrity comparison
# ======================================
def plot_histogram(ax, cell_integrity, cell_strength, signal_threshold, cmap, label):
    """
    Plot a histogram with color gradients based on a colormap.

    Parameters:
        ax: matplotlib axes object.
        cell_integrity: 1D array of signal integrity values.
        cell_strength: 1D array of signal strength values.
        signal_threshold: Threshold to filter cell_strength.
        cmap: Colormap for gradient coloring.
        label: Label for the x-axis.
    """
    # Calculate histogram values
    vals, bins = np.histogram(
        cell_integrity[cell_strength > signal_threshold],
        bins=50,
        range=(0, 1),
        density=True,
    )
    
    # Plot histogram
    n, bins, patches = ax.hist(
        cell_integrity[cell_strength > signal_threshold],
        bins=50,
        range=(0, 1),
        density=True,
        edgecolor='black',
        alpha=0.8
    )
    
    # Apply colormap
    for i, patch in enumerate(patches):
        patch.set_facecolor(cmap(i / len(patches)))
    
    # Customize appearance
    ax.set_xlim(0, 1)
    # ax.set_ylim(, 32)
    ax.set_yscale('log', base=2)
    ax.set_ylabel("Density")
    ax.set_xlabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(labelright=False)
    ax.xaxis.set_tick_params(labelsize=8)
    
    return vals, bins

def plot_vsi_distribution_comparison(
    cell_integrity_1,
    cell_strength_1,
    cell_integrity_2,
    cell_strength_2,
    signal_threshold=3.0,
    figure_height=10,
    cmap="BIH",
):
    """
    Compare histograms and cumulative densities of two datasets.

    Parameters:
        cell_integrity_1, cell_strength_1: Data for dataset 1.
        cell_integrity_2, cell_strength_2: Data for dataset 2.
        signal_threshold: Threshold for filtering data.
        figure_height: Height of the figure.
        cmap: Colormap for histogram gradients.
    """
    # Validate inputs
    for data, name in [
        (cell_integrity_1, "cell_integrity_1"),
        (cell_strength_1, "cell_strength_1"),
        (cell_integrity_2, "cell_integrity_2"),
        (cell_strength_2, "cell_strength_2"),
    ]:
        if not (isinstance(data, np.ndarray) and data.ndim == 2):
            raise ValueError(f"{name} must be a 2D numpy array.")

    with plt.style.context("default"):
        # Define colormap
        if cmap == "BIH":
            try:
                cmap = _BIH_CMAP
            except NameError:
                raise ValueError("BIH colormap is not defined.")
        
        # Create figure and subplots
        fig, ax = plt.subplots(2, 1, figsize=(figure_height, figure_height), dpi=600)
            
        # Plot histograms
        vals1, bins1 = plot_histogram(
            ax[0], cell_integrity_1, cell_strength_1, signal_threshold, cmap, label="MOD1 Signal Integrity"
        )
        vals2, bins2 = plot_histogram(
            ax[1], cell_integrity_2, cell_strength_2, signal_threshold, cmap, label="MOD2 Signal Integrity"
        )

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()

    return vals1, bins1, vals2, bins2


def plot_normalized_histogram(vals1, vals2, bins, epsilon, cmap=_BIH_CMAP, xlab="Signal Integrity", ylab="VSI Density of MOD2/MOD1"):
    vals = vals2 / (vals1 + epsilon)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)

    # Create the histogram bars
    bars = ax.bar(bin_centers, vals, width=np.diff(bins), edgecolor="black", alpha=0.7, linewidth=0.3)

    # Apply colormap
    for i, bar in enumerate(bars):
        bar.set_facecolor(cmap(i / len(bars)))  # Set color based on the colormap

    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(labelright=False)
    ax.xaxis.set_tick_params(labelsize=8)
    # Set the y-axis scale to log
    ax.set_yscale('log')
    # Add a horizontal dashed line at y = 1 (10^0)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    plt.show()


# plot_qq_integrity: Create a Q-Q plot comparing two histograms with limited (0,1) range and 1:1 aspect ratio.
def histogram_to_cdf(vals, bins):
    """
    Convert histogram data into a cumulative distribution function (CDF).

    Parameters:
        vals (array-like): Histogram counts.
        bins (array-like): Bin edges.

    Returns:
        tuple: (bin centers, CDF values) excluding the first bin edge.
    """
    cdf = np.cumsum(vals) / np.sum(vals)  # Normalize cumulative sum to [0, 1]
    return bins[1:], cdf  # Ignore first bin edge to match histogram length

def get_quantiles(cdf_x, cdf_y, quantiles):
    """
    Interpolate data values at specified quantiles using a CDF.

    Parameters:
        cdf_x (array-like): Bin centers (data values).
        cdf_y (array-like): CDF values.
        quantiles (array-like): Quantiles to evaluate (between 0 and 1).

    Returns:
        array: Data values corresponding to the input quantiles.
    """
    interp_func = interp1d(
        cdf_y, cdf_x, 
        bounds_error=False, 
        fill_value="extrapolate", 
        assume_sorted=True
    )
    return interp_func(quantiles)

def plot_vsi_qqplot(vals1, bins1, vals2, bins2, 
                      use_cmap=False, cmap="viridis", 
                      xlab="Quantiles of MOD1",
                      ylab="Quantiles of MOD2",
                      title="Q-Q Plot"
                      ):
    """
    Create a Q-Q plot comparing two histograms based on their quantiles.

    Parameters:
        vals1, bins1: Histogram values and bin edges for dataset 1.
        vals2, bins2: Histogram values and bin edges for dataset 2.
        use_cmap (bool): Whether to color points based on quantile position.
        cmap (str or Colormap): Colormap to apply (if use_cmap is True).

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Compute CDFs for both histograms
    x1, cdf1 = histogram_to_cdf(vals1, bins1)
    x2, cdf2 = histogram_to_cdf(vals2, bins2)

    # Define a list of target quantiles (from 0 to 1)
    quantiles = np.linspace(0, 1, 100)

    # Interpolate the corresponding values at each quantile
    q1 = get_quantiles(x1, cdf1, quantiles)
    q2 = get_quantiles(x2, cdf2, quantiles)

    # Ensure all values lie within the [0, 1] range
    q1 = np.clip(q1, 0, 1)
    q2 = np.clip(q2, 0, 1)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Plot the Q-Q data
    if use_cmap:
        norm = Normalize(vmin=0, vmax=1)
        ax.scatter(
            q1, q2, c=quantiles, cmap=cmap, alpha=0.6, s=17,
            norm=norm, linewidths=0.1, edgecolors='black', label=title
        )
    else:
        ax.scatter(q1, q2, color="purple", alpha=0.6, s=17, label=title)

    # Draw the y = x reference line
    ax.plot([0, 1], [0, 1], "k--", label="y = x")

    # Format plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ======================================
# Cell types
# ======================================
def plot_celltypes(
    cell_type, 
    boundary_df,
    MOD_boundary,
    x_range=None, 
    y_range=None 
):      
    """
    Plot Banksy cell type clusters with boundary overlays.

    Parameters:
    - cell_type: DataFrame with columns ['x', 'y', 'banksy_cluster']
    - boundary_df: DataFrame with boundary coordinates to plot in gray
    - MOD_boundary: DataFrame with MOD boundary coordinates to plot in teal
    - x_range, y_range: Viewport limits in μm
    - cmap: Optional dict mapping cluster labels to colors; if None, will generate default
    """
    # colormap, fix MOD subtype colors
    all_labels = cell_type['banksy_cluster'].unique()
    # Preferred hard-coded label colors
    color_map = {8: 'red', 7: 'orange'}
    # Assign tab20b colors to other clusters
    unique_other_labels = [label for label in all_labels if label not in color_map]
    colormap = plt.cm.get_cmap('tab20b', len(unique_other_labels))
    other_colors = {
        label: colormap(i) for i, label in enumerate(sorted(unique_other_labels))
    }
    cmap = {**color_map, **other_colors}

    # Map colors to data points
    colors = cell_type['banksy_cluster'].map(cmap)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)

    # Scatter plot for cells
    ax.scatter(
        cell_type["x"],
        cell_type["y"],
        s=71,
        c=colors
    )

    # Overlay boundaries
    for _, row in boundary_df.iterrows():
        ax.plot(row['boundaryX'], row['boundaryY'], c='grey', lw=1)
    for _, row in MOD_boundary.iterrows():
        ax.plot(row['boundaryX'], row['boundaryY'], c='#00bfae', lw=1)

    # Display region (whole image or custom ROI)
    if x_range is not None:
        ax[0].set_xlim(*x_range)
    else:
        ax[0].set_xlim(0, cell_type.shape[1])
    if y_range is not None:
        ax[0].set_ylim(*y_range)
    else:
        ax[0].set_ylim(0, cell_type.shape[0])
    
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


# ======================================
# Transcript Neighborhood
# ======================================

def plot_circular_neighborhood(
    signals_df, centroid_df, MOD_boundaries, boundaries_df,
    x_range, y_range, diameters=[6, 8, 10, 12],
    true_boundary=True, plot_top20=False, top20=None
):

    def filter_in_bounds(df):
        return df[
            (df["x"] >= x_range[0]) & (df["x"] <= x_range[1]) &
            (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
        ]

    signals_filtered = filter_in_bounds(signals_df)
    centroid_filtered = filter_in_bounds(centroid_df)
    MOD_filtered = filter_in_bounds(MOD_boundaries)
    boundaries_filtered = filter_in_bounds(boundaries_df)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)

    # Plot signal points
    if plot_top20 and top20 is not None:
        marker_styles = ['o', 's', 'D', '^']
        colors = sns.color_palette("tab10", 5)
        style_dict = {
            gene: (marker_styles[i % len(marker_styles)], colors[i % len(colors)])
            for i, gene in enumerate(top20)
        }
        # Background (other genes)
        ax.scatter(signals_filtered['x'], signals_filtered['y'], s=3, color='lightgrey', alpha=0.5, label="Other Genes")
        # Top 20 genes
        for gene, (marker, color) in style_dict.items():
            subset = signals_filtered[signals_filtered['gene'] == gene]
            ax.scatter(subset['x'], subset['y'], s=3, color=color, marker=marker, alpha=0.8, label=gene)
    else:
        # Continuous color mapping
        norm = Normalize(vmin=0, vmax=2000)
        cmap = plt.get_cmap("Oranges")
        ax.scatter(signals_filtered["x"], signals_filtered["y"], s=3, c=cmap(norm(signals_filtered["Total_brightness"])))
        # Colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.45, pad=0.02, anchor=(0.0, 0.3))
        cbar.set_label("Signal Brightness")

    # Plot centroids
    ax.scatter(centroid_filtered["x"], centroid_filtered["y"], s=15, c='blue', marker="x", label="Cell Centroids")

    # Plot boundaries
    if true_boundary:
        for _, row in boundaries_filtered.iterrows():
            ax.plot(row['boundaryX'], row['boundaryY'], c='grey', lw=1)
        for _, row in MOD_filtered.iterrows():
            ax.plot(row['boundaryX'], row['boundaryY'], c='#00bfae', lw=1)
        # Legend handles
        ax.plot([], [], color='grey', lw=1, label="Other Cells Boundary")
        ax.plot([], [], color='#00bfae', lw=1, label="MOD Cells Boundary")

    # Plot concentric rings
    cmap_rings = plt.get_cmap('tab20')
    for idx, diameter in enumerate(diameters):
        color = cmap_rings(idx)
        for _, row in centroid_filtered.iterrows():
            circle = Circle(
                (row["x"], row["y"]),
                radius=diameter / 2,
                color=color, fill=False,
                linewidth=0.7, alpha=0.7
            )
            ax.add_patch(circle)
        ax.plot([], [], color=color, label=f'Diameter={diameter}')

    # Final touches
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False, markerscale=1.5, ncol=1)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def compute_knn(coordinate_df, query_points, k):
    """
    Compute the k-nearest neighbors for a set of query points.

    Parameters:
        coordinate_df (DataFrame): DataFrame containing x, y, z coordinates.
        query_points (ndarray): Array of query points (Nx3).
        k (int): Number of neighbors to find.

    Returns:
        List of dictionaries with neighbor indices and distances.
    """
    if k > len(coordinate_df):
        raise ValueError(f"k ({k}) cannot be greater than the number of points in coordinate_df ({len(coordinate_df)})")

    coordinates = coordinate_df[['x', 'y', 'z']].values

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coordinates)
    distances, indices = nbrs.kneighbors(query_points)

    return [
        {
            'query_point': query.tolist(),
            'neighbor_indices': idx.tolist(),
            'neighbor_distances': dist.tolist()
        }
        for query, dist, idx in zip(query_points, distances, indices)
    ]


def plot_knn_neighborhood(signals_df, centroid_df, MOD_boundaries, boundaries_df, x_range, y_range, 
                                         neighbors=[20, 40, 80, 160, 220], true_boundary=True, 
                                         plot_top20=False, top20=None):
    """
    Plot focus points, boundaries, and k-NN-based convex hulls around centroids.

    Parameters:
        signals_df (DataFrame): DataFrame containing signal points.
        centroid_df (DataFrame): DataFrame containing centroid positions.
        MOD_boundaries (DataFrame): DataFrame with MOD boundary data.
        boundaries_df (DataFrame): DataFrame with other boundary data.
        x_range (tuple): X-axis range.
        y_range (tuple): Y-axis range.
        neighbors (list): List of k-values for kNN hull drawing.
        true_boundary (bool): Whether to plot true cell boundaries.
        plot_top20 (bool): Whether to highlight top 20 genes.
        top20 (list): List of top 20 gene names.
    """
    # Filter all dataframes by range
    def filter_df(df):
        return df[(df["x"] >= x_range[0]) & (df["x"] <= x_range[1]) &
                  (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])]

    signals_filtered = filter_df(signals_df)
    centroid_filtered = filter_df(centroid_df)
    MOD_filtered = filter_df(MOD_boundaries)
    boundaries_filtered = filter_df(boundaries_df)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    norm = Normalize(vmin=0, vmax=2000)
    cmap = plt.cm.Oranges

    if plot_top20 and top20 is not None:
        marker_styles = ['o', 's', 'D', '^']
        colors = sns.color_palette("tab10", 5)
        top20_dict = {gene: (marker_styles[i % 4], colors[i % 5]) for i, gene in enumerate(top20)}

        ax.scatter(signals_filtered['x'], signals_filtered['y'], s=3, color='lightgrey', alpha=0.5, label="Other Genes")

        for gene, (marker, color) in top20_dict.items():
            subset = signals_filtered[signals_filtered['gene'] == gene]
            ax.scatter(subset['x'], subset['y'], s=3, color=color, marker=marker, alpha=0.8, label=gene)
    else:
        ax.scatter(signals_filtered['x'], signals_filtered['y'], s=3, c=cmap(norm(signals_filtered['Total_brightness'])))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.45, pad=0.02, anchor=(0.0, 0.3))
        cbar.set_label("Signal Brightness")

    # Plot centroids
    ax.scatter(centroid_filtered['x'], centroid_filtered['y'], s=15, c='blue', label="Cell Centroids", marker='x')

    # Plot true boundaries if requested
    if true_boundary:
        for df, color, label in [(boundaries_filtered, 'grey', "Other Cells Boundary"), (MOD_filtered, '#00bfae', "MOD Cells Boundary")]:
            for _, row in df.iterrows():
                ax.plot(row['boundaryX'], row['boundaryY'], c=color, lw=1)
            ax.plot([], [], color=color, lw=1, label=label)

    # Plot convex hulls around kNN
    cmap_rings = mpl.colormaps['tab20']
    for idx, k in enumerate(neighbors):
        color = cmap_rings(idx / len(neighbors))
        label_added = False
        for _, centroid in centroid_filtered.iterrows():
            query_point = np.array([[centroid['x'], centroid['y'], 4.5]])
            knn = compute_knn(signals_df, query_point, k)
            neighbor_pts = signals_df.iloc[knn[0]['neighbor_indices']][['x', 'y']].values

            if len(neighbor_pts) < 3:
                continue

            hull = ConvexHull(neighbor_pts)
            for simplex in hull.simplices:
                ax.plot(neighbor_pts[simplex, 0], neighbor_pts[simplex, 1], color=color, lw=1,
                        label=f"k={k} NN" if not label_added else None)
                label_added = True

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False, markerscale=1.5, ncol=1)
    plt.show()


# ======================================
# Marker Signals Spatial Distribution
# ======================================

def plot_marker_signals(signal_df, centroid_df, title=None, 
                        color="MOD1", xlim=(-10, 1810), ylim=(-10, 1810), 
                        plot_rect=False, rect=[750, 100, 220, 1400]):
    """
    Plot signal scatter and centroid markers with optional highlight rectangle.
    
    Parameters:
    - signal_df: DataFrame with 'x', 'y', and 'Total_brightness' columns
    - centroid_df: DataFrame with 'x' and 'y' columns for centroid locations
    - title: Title of the plot (string)
    - color: 'MOD1' for orange-red, otherwise blue
    - xlim: Tuple specifying x-axis limits
    - ylim: Tuple specifying y-axis limits
    - plot_rect: Boolean, whether to draw a rectangle
    - rect: List [x, y, width, height] for the rectangle
    """

    norm = Normalize(0, 2000)

    if color == "MOD1":
        color_s = plt.cm.Oranges(norm(signal_df['Total_brightness']))
        color_c = 'red'
    else:
        color_s = plt.cm.Blues(norm(signal_df['Total_brightness']))
        color_c = 'blue'

    plt.figure(figsize=(8, 8), dpi=600)

    plt.scatter(
        signal_df["x"],
        signal_df["y"],
        s=1,
        c=color_s, 
        alpha=0.8,
    )

    for _, row in centroid_df.iterrows():
        plt.text(
            row["x"], 
            row["y"], 
            "x", 
            fontsize=7, 
            color=color_c,
            ha='center', 
            va='center'
        )

    if plot_rect and rect is not None and len(rect) == 4:
        x_rect, y_rect, width, height = rect
        ax = plt.gca()
        rectangle_patch = patches.Rectangle(
            (x_rect, y_rect), width, height,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rectangle_patch)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel("(μm)")
    plt.ylabel("(μm)")
    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title, fontsize=15)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()


# ======================================
# Marker Heatmap
# ======================================
def plot_annotate_heatmap(cluster_data, cluster_labels, gene_groups=None, zscore=True, cmap=HEATMAP_CMAP,
                          box_specs=None, cluster_text_y=-1.2, show_cluster=True,
                          show_cluster_lines=True, DE_g_line=True):
    """
    Plot expression heatmap with cluster annotations and optional decorations.

    Parameters:
    - cluster_data: DataFrame, rows = cells, columns = genes (+ optional 'Cell_class')
    - cluster_labels: 1D array-like, cluster labels for each cell
    - gene_groups: optional DataFrame with 'cluster' column for gene clustering
    - zscore: Boolean, apply z-score normalization by gene
    - cmap: Colormap to use
    - box_specs: List of dictionaries, each with keys: x_offset, width, color, linestyle
    - cluster_text_y: Y position to write cluster labels
    - show_cluster: Boolean, whether to show cluster labels
    - show_cluster_lines: Boolean, draw dashed lines between clusters
    - DE_g_line: Boolean, draw a horizontal line at y=9
    """

    # Data preprocessing
    cluster_data = cluster_data.copy()
    cluster_data['Cell_class'] = cluster_labels

    cluster_data = cluster_data.sort_values(by='Cell_class')
    cell_class_col = cluster_data['Cell_class']
    numeric_data = cluster_data.drop(columns=['Cell_class']).apply(pd.to_numeric, errors='coerce')
    numeric_data = numeric_data.dropna(axis=1).loc[:, ~numeric_data.T.duplicated()]
    cluster_data = pd.concat([cell_class_col, numeric_data], axis=1)

    expression_data = cluster_data.drop('Cell_class', axis=1).T
    expression_data = expression_data.loc[:, ~expression_data.columns.duplicated()]

    cluster_labels_sorted = cluster_data['Cell_class'].values
    unique_labels = sorted(set(cluster_labels_sorted))

    # Normalize
    if zscore:
        expression_data = expression_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        vmin, vmax = -3, 3
    else:
        vmin, vmax = 0, 5

    # Reorder cells within cluster
    new_order = []
    for label in unique_labels:
        indices = np.where(cluster_labels_sorted == label)[0]
        subset = expression_data.iloc[:, indices]
        if subset.shape[1] > 1:
            linkage_matrix = linkage(subset.T, method='ward')
            sorted_indices = indices[leaves_list(linkage_matrix)]
        else:
            sorted_indices = indices
        new_order.extend(sorted_indices)

    reordered_expression_data = expression_data.iloc[:, new_order]
    reordered_cluster_labels = cluster_labels_sorted[new_order]

    # Reorder genes
    if gene_groups is not None:
        new_gene_order = []
        for label in sorted(set(gene_groups['cluster'])):
            gene_indices = np.where(gene_groups == label)[0]
            subset = reordered_expression_data.iloc[gene_indices, :]
            if subset.shape[0] > 1:
                linkage_matrix = linkage(subset, method='average')
                sorted_gene_indices = gene_indices[leaves_list(linkage_matrix)]
            else:
                sorted_gene_indices = gene_indices
            new_gene_order.extend(sorted_gene_indices)
    else:
        new_gene_order = leaves_list(linkage(reordered_expression_data, method='average'))

    reordered_expression_data = reordered_expression_data.iloc[new_gene_order, :]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 10), dpi=600)
    sns.heatmap(
        reordered_expression_data,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        xticklabels=False,
        yticklabels=True,
        cbar=True,
        ax=ax,
        cbar_kws={"shrink": 0.5}
    )

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Annotate cluster
    cluster_boundaries = []
    for label in unique_labels:
        indices = np.where(reordered_cluster_labels == label)[0]
        if len(indices) == 0:
            continue
        start, end = indices[0], indices[-1]
        x_pos = (start + end) / 2
        if show_cluster:
            ax.text(x_pos, cluster_text_y, str(label), ha='center', va='center', rotation=90, fontsize=15)
        cluster_boundaries.append(end)

    if show_cluster_lines:
        for boundary in cluster_boundaries[:-1]:
            ax.axvline(x=boundary + 0.5, color='purple', linestyle='--', linewidth=1)

    if DE_g_line:
        ax.hlines(y=9, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='green', linestyle='--', linewidth=1)

    if box_specs is not None:
        ax_pos = ax.get_position()
        y_rect = ax_pos.y0 - 0.1
        height = ax_pos.height * 18.15
        for box in box_specs:
            rect = patches.Rectangle(
                (ax_pos.x0 + box["x_offset"], y_rect),
                box["width"], height,
                linewidth=1.5,
                edgecolor=box["color"],
                facecolor='none',
                linestyle=box.get("linestyle", '--')
            )
            ax.add_patch(rect)

    plt.show()


def plot_neuron_cluster_heatmap(re_IN, re_IN_clu, DE_g=True, cmap=HEATMAP_CMAP, figures=(15, 25), DE_g_x=9):
    """
    Plot heatmap of expression matrix with cluster labels and optional DE gene divider.

    Parameters:
    - re_IN: Gene x Cell expression matrix (already ordered).
    - re_IN_clu: Cluster label array aligned to columns of re_IN.
    - DE_g: Whether to plot vertical line indicating DE gene boundary.
    - cmap: Colormap used in heatmap.
    - figures: Tuple indicating figure size (width, height).
    - DE_g_x: X-position for vertical DE gene divider.
    """
    unique_labels = sorted(set(re_IN_clu))

    plt.figure(figsize=figures, dpi=600)
    ax = sns.heatmap(
        re_IN.T,
        vmin=-3, vmax=3,
        annot=False, fmt="g", xticklabels=True, yticklabels=False,
        cmap=cmap,
        cbar=True,
        cbar_kws={"shrink": 0.5}
    )

    try:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_position([
            ax.get_position().x1 + 0.01,
            ax.get_position().y1 - 0.3,
            0.02,
            0.2
        ])
    except:
        print("Warning: Failed to reposition colorbar.")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=0)

    cluster_boundaries = []
    for label in unique_labels:
        class_indices = np.where(re_IN_clu == label)[0]
        if len(class_indices) == 0:
            continue

        start_idx, end_idx = class_indices[0], class_indices[-1]
        x_pos = (start_idx + end_idx) / 2
        ax.text(-1.5, x_pos, str(label), ha='center', va='center', rotation=0, fontsize=11, color='black')
        cluster_boundaries.append(end_idx)

    yticks = ax.get_yticks()
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]
    adjusted_positions = {}

    for i, (ytick, label) in enumerate(zip(yticks, yticklabels)):
        if i > 0 and abs(ytick - yticks[i - 1]) < 5:
            new_y = yticks[i - 1] - 5
            adjusted_positions[label] = new_y
        else:
            adjusted_positions[label] = ytick

    for label, new_y in adjusted_positions.items():
        if new_y != yticks[yticklabels.index(label)]:
            orig_y = yticks[yticklabels.index(label)]
            ax.annotate(
                label,
                xy=(-1.5, orig_y), xytext=(-3, new_y),
                ha='right', va='center', fontsize=10, color='black',
                arrowprops=dict(arrowstyle="-", color="gray", linewidth=1.0, alpha=0.6)
            )
        else:
            ax.text(-1.5, new_y, label, ha='right', va='center', fontsize=10, color='black')

    for boundary in cluster_boundaries:
        ax.axhline(y=boundary, color='purple', linestyle='--', linewidth=1)

    if DE_g:
        ax.vlines(x=DE_g_x, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='green', linestyle='--', linewidth=1)

    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# ======================================
# PCA and UMAP
# ======================================
def plot_pca_variance_ratio(data, n_components=14, title="Explained Variance by PC"):
    """
    Plot the explained variance ratio of PCA components.

    Parameters:
    - data (pd.DataFrame or np.ndarray): Input data (cells x features)
    - n_components (int): Number of PCA components to calculate
    - title (str): Title for the plot
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(
        np.arange(1, len(explained_variance) + 1),
        explained_variance,
        marker='o',
        linestyle='--',
        color='mediumvioletred'
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def plot_pca_cumulative_variance(data, n_components=14, title="Cumulative Explained Variance"):
    """
    Plot cumulative explained variance from PCA.

    Parameters:
    - data (pd.DataFrame or np.ndarray): Input data (cells x features)
    - n_components (int): Number of PCA components to calculate
    - title (str): Title for the plot
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o',
        linestyle='-',
        color='mediumvioletred'
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


def plot_umap_from_pca(data, n_PCs=5, title="UMAP after PCA", color='mediumvioletred'):
    """
    Perform PCA and UMAP projection, then plot UMAP result.

    Parameters:
    - data (pd.DataFrame or np.ndarray): Input data (cells x features)
    - n_PCs (int): Number of principal components to use
    - title (str): Title for the plot
    - color (str): Color of scatter points
    """
    pca = PCA(n_components=n_PCs, random_state=42)
    pca_result = pca.fit_transform(data)

    umap_result = umap.UMAP(n_components=2).fit_transform(pca_result)

    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(
        umap_result[:, 0],
        umap_result[:, 1],
        alpha=0.7,
        s=11,
        edgecolors='none',
        color=color
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
