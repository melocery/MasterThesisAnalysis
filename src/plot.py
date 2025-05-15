import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

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
# Vertical Signal Integrity Map
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
    if plot_top20 and top20:
        marker_styles = ['o', 's', 'D', '^']
        colors = sns.color_palette("tab10", 5)
        style_dict = {
            gene: (marker_styles[i % len(marker_styles)], colors[i % len(colors)])
            for i, gene in enumerate(top20)
        }

        # Background (other genes)
        ax.scatter(
            signals_filtered['x'], signals_filtered['y'],
            s=3, color='lightgrey', alpha=0.5, label="Other Genes"
        )

        # Top 20 genes
        for gene, (marker, color) in style_dict.items():
            subset = signals_filtered[signals_filtered['gene'] == gene]
            ax.scatter(
                subset['x'], subset['y'],
                s=3, color=color, marker=marker, alpha=0.8, label=gene
            )
    else:
        # Continuous color mapping
        norm = Normalize(vmin=0, vmax=2000)
        cmap = plt.get_cmap("Oranges")
        ax.scatter(
            signals_filtered["x"], signals_filtered["y"],
            s=3, c=cmap(norm(signals_filtered["Total_brightness"]))
        )
        # Colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, shrink=0.5, pad=0.02, anchor=(0.0, 0.3)
        )
        cbar.set_label("Signal Brightness")

    # Plot centroids
    ax.scatter(
        centroid_filtered["x"], centroid_filtered["y"],
        s=15, c='blue', marker="x", label="Cell Centroids"
    )

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
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=10, frameon=False, markerscale=1.5
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
