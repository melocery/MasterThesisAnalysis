import numpy as np
import pandas as pd
import anndata as ad
from natsort import natsorted
from pandas.api.types import CategoricalDtype

from shapely.geometry import Polygon
from matplotlib.path import Path

from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list

from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg as la

import umap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


_BIH_CMAP = LinearSegmentedColormap.from_list(
    "BIH",
    [
        "mediumvioletred",
        "violet",
        "powderblue",
        "powderblue",
    ][::-1],
)

_BIH_CMAP_re = LinearSegmentedColormap.from_list(
    "BIH",
    [
        "powderblue",
        "powderblue",
        "violet",
        "mediumvioletred",
    ][::-1],
)

# ======================================
# Load Data
# ======================================
def load_merfish_signal_df(file_path):
    """
    Load and process MERFISH signal coordinate data.

    Parameters:
    - file_path (str or Path): Path to the MERFISH CSV file.

    Returns:
    - pd.DataFrame: Cleaned signal coordinate dataframe with adjusted coordinates.
    """
    columns = [
        "Centroid_X",
        "Centroid_Y",
        "Centroid_Z",
        "Gene_name",
        "Cell_name",
        "Total_brightness",
        "Area",
        "Error_bit",
        "Error_direction",
    ]

    df = pd.read_csv(file_path, usecols=columns).rename(
        columns={
            "Centroid_X": "x",
            "Centroid_Y": "y",
            "Centroid_Z": "z",
            "Gene_name": "gene",
        }
    )

    # Remove dummy molecules
    df = df.loc[~df["gene"].str.contains("Blank|NegControl")].copy()

    # Convert gene column to category dtype
    df["gene"] = df["gene"].astype("category")

    # Shift coordinates to remove negative values
    df_x_min = df["x"].min()
    df_y_min = df["y"].min()
    df["x"] = df["x"] - df_x_min
    df["y"] = df["y"] - df_y_min

    return df, df_x_min, df_y_min

def load_banksy_result(banksy_path, coordinate_x_m=0, coordinate_y_m=0, bregma_value=-0.24):
    """
    Load and preprocess Banksy clustering result.

    Parameters:
    - banksy_path (Path or str): Path to 'banksy_cluster.txt'.
    - coordinate_x_m (float): Minimum x coordinate for shifting.
    - coordinate_y_m (float): Minimum y coordinate for shifting.
    - bregma_value (float): Bregma slice to select (default: -0.24).

    Returns:
    - pd.DataFrame: Processed Banksy clustering result for specified Bregma slice.
    """
    columns = ["Centroid_X", "Centroid_Y", "Bregma", "lam0.2"]

    banksy_result = pd.read_csv(
        banksy_path,
        usecols=columns,
        sep="\t"
    ).rename(
        columns={
            "Centroid_X": "x",
            "Centroid_Y": "y",
            "Bregma": "Bregma",
            "lam0.2": "banksy_cluster",
        }
    )

    # Select the specified Bregma slice
    banksy_result = banksy_result[banksy_result["Bregma"] == bregma_value]

    # Shift coordinates
    banksy_result["x"] = banksy_result["x"] - coordinate_x_m
    banksy_result["y"] = banksy_result["y"] - coordinate_y_m

    return banksy_result.copy()


def load_merfish_data(data_path, banksy_result_df, coordinate_x_m=0, coordinate_y_m=0, animal_id=1, bregma_value=-0.24):
    """
    Load and process MERFISH single-cell data, assign BankSY clusters, shift coordinates.

    Parameters:
    - data_path (Path or str): Path to 'merfish_all_cells.csv'.
    - banksy_result_df (pd.DataFrame): Preloaded BankSY result with 'banksy_cluster'.
    - coordinate_x_m (float): Minimum x for alignment shift.
    - coordinate_y_m (float): Minimum y for alignment shift.
    - animal_id (int): ID of the animal to filter on.
    - bregma_value (float): Bregma slice to filter.

    Returns:
    - pd.DataFrame: Processed MERFISH data.
    """
    
    df = pd.read_csv(data_path).rename(columns={"Centroid_X": "x", "Centroid_Y": "y"})

    # Drop unwanted columns
    df = df.drop(columns=[col for col in df.columns if col == 'Fos' or col.startswith('Blank_')])

    # Filter data
    df = df[df["Cell_class"] != "Ambiguous"]
    df = df[df["Animal_ID"] == animal_id]
    df = df[df["Bregma"] == bregma_value]

    # Coordinate shift
    df["x"] -= coordinate_x_m
    df["y"] -= coordinate_y_m

    # Assign BankSY cluster (merge safer than indexing by order)
    df = df.merge(
        banksy_result_df[["x", "y", "banksy_cluster"]],
        on=["x", "y"],
        how="left"
    )
    df = df.rename(columns={"banksy_cluster": "banksy"})

    # Class mapping
    cell_class_m = {
        'Astrocyte': 'Astrocyte',
        'Endothelial 1': 'Endothelial',
        'Endothelial 2': 'Endothelial',
        'Endothelial 3': 'Endothelial',
        'Ependymal': 'Ependymal',
        'Excitatory': 'Excitatory',
        'Inhibitory': 'Inhibitory',
        'Microglia': 'Microglia',
        'OD Immature 1': 'OD Immature',
        'OD Immature 2': 'OD Immature',
        'OD Mature 1': 'OD Mature',
        'OD Mature 2': 'OD Mature',
        'OD Mature 3': 'OD Mature',
        'OD Mature 4': 'OD Mature',
        'Pericytes': 'Pericytes'
    }

    df["Cell_class"] = df["Cell_class"].map(cell_class_m).fillna("Other")

    # Sort for nicer visualization/grouping
    df = df.sort_values(by="Cell_class")

    return df.copy()


def load_boundaries_data(boundary_path, merfish_data_df, coordinate_x_m=0, coordinate_y_m=0):
    """
    Load and process boundary information, align with MERFISH cells.

    Parameters:
    - boundary_path (Path or str): Path to 'cellboundaries_example_animal.csv'.
    - merfish_data_df (pd.DataFrame): Preprocessed MERFISH data with Cell_ID.
    - coordinate_x_m (float): Minimum x shift.
    - coordinate_y_m (float): Minimum y shift.

    Returns:
    - pd.DataFrame: Processed boundary data with shifted coordinates and metadata.
    """
    df = pd.read_csv(boundary_path)
    df = df.dropna(subset=["boundaryX", "boundaryY"])

    # Filter and merge metadata
    cell_ids = merfish_data_df["Cell_ID"]
    df = df[df["feature_uID"].isin(cell_ids)]
    df = df.merge(
        merfish_data_df[["Cell_ID", "x", "y", "banksy", "Cell_class"]],
        left_on="feature_uID",
        right_on="Cell_ID",
        how="inner"
    ).drop(columns=["Cell_ID"])

    # Convert boundary strings to lists and apply shift
    df["boundaryX"] = df["boundaryX"].apply(lambda x: [float(i) for i in x.split(";")] if isinstance(x, str) else x)
    df["boundaryY"] = df["boundaryY"].apply(lambda x: [float(i) for i in x.split(";")] if isinstance(x, str) else x)

    df["boundaryX"] = df["boundaryX"].apply(lambda x: [i - coordinate_x_m for i in x] if isinstance(x, list) else x)
    df["boundaryY"] = df["boundaryY"].apply(lambda x: [i - coordinate_y_m for i in x] if isinstance(x, list) else x)

    return df.copy()


def load_scRNA_data(mtx_path, barcodes_path, genes_path, meta_path, cell_class_filter, neuron_cluster = False):
    # Read count matrix and metadata files
    X = mmread(mtx_path).tocsr()
    cell_ids = pd.read_csv(barcodes_path, sep="\t", header=None)[0].values
    gene_names = pd.read_csv(genes_path, sep="\t", header=None)[1].values

    # Create AnnData object (cells x genes)
    adata = ad.AnnData(X=X.T)  # transpose to cells x genes
    adata.var_names = gene_names
    adata.obs_names = cell_ids
    adata.var_names_make_unique()  # ensure unique gene names

    # Load metadata and filter cells by specified cell classes
    meta = pd.read_excel(meta_path).rename(columns={
        "Cell name": "Cell_name",
        "Sex": "Sex",
        "Replicate number": "Rep",
        "Cell class (determined from clustering of all cells)": "Cell_class",
        "Non-neuronal cluster (determined from clustering of all cells)": "Non_neuronal_cluster",
        "Neuronal cluster (determined from clustering of inhibitory or excitatory neurons)": "Neuronal_cluster"
    })
    meta = meta.set_index("Cell_name")
    meta = meta.loc[meta["Cell_class"].isin(cell_class_filter.keys())]
    adata = adata[adata.obs_names.isin(meta.index)].copy()
    if neuron_cluster:
        adata.obs = meta.loc[adata.obs_names, ["Cell_class", "Neuronal_cluster"]].copy()
    else:
        adata.obs = meta.loc[adata.obs_names, ["Cell_class"]].copy()
    adata.obs["Cell_class"] = adata.obs["Cell_class"].map(cell_class_filter)

    cell_class_order = list(cell_class_filter.values())
    cat_dtype = CategoricalDtype(categories=cell_class_order, ordered=True)
    adata.obs["Cell_class"] = adata.obs["Cell_class"].astype(cat_dtype)

    # Filter cells with mitochondrial RNA fraction >= 20%
    mt_mask = adata.var_names.str.startswith("mt")
    mt_fraction = np.array(adata[:, mt_mask].X.sum(axis=1)).flatten() / (np.array(adata.X.sum(axis=1)).flatten() + 1e-6)
    adata = adata[mt_fraction < 0.2, :].copy()

    # Keep cells with more than 1000 detected genes
    nonzero_counts = np.array((adata.X != 0).sum(axis=1)).flatten()
    adata = adata[nonzero_counts > 1000, :].copy()

    # Filter out genes starting with "Blank"
    blank_mask = ~adata.var_names.str.startswith("Blank")
    adata = adata[:, blank_mask].copy()

    # Normalize counts per cell to 10,000 (keeping sparse matrix)
    sc_total = np.array(adata.X.sum(axis=1)).flatten() + 1e-6
    normalizer = sparse.diags(10_000 / sc_total)
    adata.X = normalizer.dot(adata.X)

    # Log-transform (using sparse matrix log1p)
    adata.X = adata.X.log1p()

    return adata


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

    # Initialize output arrays
    cell_integrity = np.zeros((integrity_size, integrity_size))
    cell_strength = np.zeros((integrity_size, integrity_size))

    for idx, row in boundary_df.iterrows():
        x_coords = np.array(row['boundaryX'])
        y_coords = np.array(row['boundaryY'])

        # Remove NaN values to clean the coordinates
        valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if len(x_coords) < 3:
            continue  # Not enough points to form a valid polygon

        polygon = Polygon(zip(x_coords, y_coords))
        if not polygon.is_valid:
            continue  # Skip invalid polygons

        # Compute bounding box and clip it within the image bounds
        x_min, x_max = int(np.floor(polygon.bounds[0])), int(np.ceil(polygon.bounds[2]))
        y_min, y_max = int(np.floor(polygon.bounds[1])), int(np.ceil(polygon.bounds[3]))

        x_min, x_max = max(0, x_min), min(integrity_size, x_max)
        y_min, y_max = max(0, y_min), min(integrity_size, y_max)

        # Generate the grid of points inside the bounding box
        y_indices, x_indices = np.meshgrid(range(y_min, y_max), range(x_min, x_max), indexing='ij')
        points = np.column_stack([x_indices.ravel(), y_indices.ravel()])

        path = Path(np.column_stack((x_coords, y_coords)))
        epsilon = 0.7
        mask = path.contains_points(points, radius=epsilon)
        mask = mask.reshape(y_indices.shape)

        # Extract sub-regions of the integrity and strength maps
        subgrid_int = integrity[y_min:y_max, x_min:x_max]
        subgrid_str = strength[y_min:y_max, x_min:x_max]

        cell_integrity[y_min:y_max, x_min:x_max] = np.where(
            mask, subgrid_int, cell_integrity[y_min:y_max, x_min:x_max]
        )
        cell_strength[y_min:y_max, x_min:x_max] = np.where(
            mask, subgrid_str, cell_strength[y_min:y_max, x_min:x_max]
        )

    return cell_integrity, cell_strength



# ======================================
# Neuron Cluster
# ======================================
def order_neuron_clusters(cluster_data, cluster_labels, gene_groups=None, zscore=True):
    """
    Reorder expression matrix by clustering within cell classes and optionally gene groups.

    Parameters:
    - cluster_data: DataFrame containing gene expression data.
    - cluster_labels: Cluster assignments for each cell.
    - gene_groups: Optional DataFrame with gene group information (e.g., {'cluster': ...}).
    - zscore: Whether to z-score normalize each gene across cells.

    Returns:
    - reordered_expression_data: Reordered expression matrix.
    - reordered_cluster_labels: Reordered cluster labels matching columns of expression matrix.
    """
    cluster_data = cluster_data.copy()
    cluster_data['Cell_class'] = cluster_labels

    cell_class_col = cluster_data['Cell_class']
    numeric_data = cluster_data.drop(columns=['Cell_class']).apply(pd.to_numeric, errors='coerce')
    numeric_data = numeric_data.dropna(axis=1, thresh=int(0.9 * len(numeric_data)))
    numeric_data = numeric_data.loc[:, ~numeric_data.T.duplicated()]
    cluster_data = pd.concat([cell_class_col, numeric_data], axis=1)

    expression_data = cluster_data.drop('Cell_class', axis=1).T
    expression_data = expression_data.loc[:, ~expression_data.columns.duplicated()]
    cluster_labels_sorted = cluster_data['Cell_class'].values

    if zscore:
        def safe_zscore(x):
            return (x - x.mean()) / x.std() if x.std() > 0 else x * 0
        expression_data = expression_data.apply(safe_zscore, axis=1)

    new_order = []
    unique_labels = natsorted(set(cluster_labels_sorted))

    for label in unique_labels:
        class_indices = np.where(cluster_labels_sorted == label)[0]
        if len(class_indices) == 0:
            continue

        subset = expression_data.iloc[:, class_indices]
        if subset.shape[1] > 1:
            linkage_matrix = linkage(subset.T, method='ward')
            sorted_indices = leaves_list(linkage_matrix)
            sorted_indices = class_indices[sorted_indices]
        else:
            sorted_indices = class_indices

        new_order.extend(sorted_indices)

    reordered_expression_data = expression_data.iloc[:, new_order]
    reordered_cluster_labels = cluster_labels_sorted[new_order]

    if gene_groups is not None:
        new_gene_order = []
        unique_gene_labels = sorted(set(gene_groups['cluster']))

        for label in unique_gene_labels:
            gene_indices = np.where(gene_groups['cluster'] == label)[0]
            if len(gene_indices) == 0:
                continue

            subset = reordered_expression_data.iloc[gene_indices, :]
            if subset.shape[0] > 1:
                linkage_matrix = linkage(subset, method='average')
                sorted_gene_indices = leaves_list(linkage_matrix)
                sorted_gene_indices = gene_indices[sorted_gene_indices]
            else:
                sorted_gene_indices = gene_indices

            new_gene_order.extend(sorted_gene_indices)
    else:
        linkage_matrix = linkage(reordered_expression_data, method='average')
        new_gene_order = leaves_list(linkage_matrix)

    reordered_expression_data = reordered_expression_data.iloc[new_gene_order, :]

    return reordered_expression_data, reordered_cluster_labels


def get_cluster_boundaries(cluster_labels):
    """
    Return a DataFrame with start, end, and size of each cluster in order.
    
    Parameters:
    - cluster_labels: list or array of cluster assignments (ordered).
    
    Returns:
    - pd.DataFrame with cluster, start_idx, end_idx, size
    """
    boundaries = []
    unique_labels = natsorted(set(cluster_labels))

    for label in unique_labels:
        class_indices = np.where(cluster_labels == label)[0]
        if len(class_indices) == 0:
            continue
        boundaries.append({
            'cluster': label,
            'start_idx': class_indices[0],
            'end_idx': class_indices[-1],
            'size': len(class_indices)
        })
    
    return pd.DataFrame(boundaries)


# ======================================
# VSI at the location of transcripts
# ======================================
def marker_transcripts_vsi(signal_df, signal_strength, signal_integrity, gene):
    # Filter to keep only rows for the given gene
    gene_signal = signal_df[signal_df['gene'].isin(gene)].copy()

    # Initialize empty masks with same shape as signal arrays
    mask = np.zeros_like(signal_strength, dtype=bool)

    # Ensure coordinates are within bounds
    valid_coords = (
        (gene_signal['x'] >= 0) & (gene_signal['x'] < signal_strength.shape[1]) &
        (gene_signal['y'] >= 0) & (gene_signal['y'] < signal_strength.shape[0])
    )
    gene_signal = gene_signal[valid_coords]

    # Get x and y coordinates
    xs = gene_signal['x'].astype(int).values
    ys = gene_signal['y'].astype(int).values

    # Mark positions of this gene as True in the mask
    mask[ys, xs] = True

    # Apply the mask to extract the 2D arrays
    gene_strength = np.where(mask, signal_strength, 0)
    gene_integrity = np.where(mask, signal_integrity, 0)

    return gene_strength, gene_integrity


# ======================================
# Correlated Genes for Each Marker
# ======================================
def find_corr_genes(gene, sc_data):
    cor = pd.Series(index=sc_data.index, dtype=np.float64)
    x = pd.to_numeric(sc_data.loc[gene].values.flatten(), errors='coerce')
    
    for other_gene in sc_data.index:
        if other_gene.startswith("Blank") or other_gene == 'Cell_class':
            continue

        y = pd.to_numeric(sc_data.loc[other_gene].values.flatten(), errors='coerce')
        if np.std(x) == 0 or np.std(y) == 0:
            cor[other_gene] = np.nan
            continue

        cor[other_gene], _ = spearmanr(x, y)

    return cor.dropna().sort_values(ascending=False)

def process_related_genes(DE_genes, sc_data, top_n=26):
    """
    For a list of DE genes, find top correlated genes per gene.

    Parameters:
    - DE_genes (list): List of gene names.
    - sc_data (pd.DataFrame): Expression matrix (genes x cells).
    - top_n (int): Number of top correlated genes to retrieve per input gene.

    Returns:
    - list: Unique list of top correlated genes.
    """
    related_genes = set()

    for gene in DE_genes:
        if gene not in sc_data.index:
            continue
        correlated = find_corr_genes(gene, sc_data)
        top_genes = correlated.iloc[:top_n].index.tolist()
        related_genes.update(top_genes)

    return list(related_genes)


# ======================================
# scRNA-seq Data Clustering
# ======================================

def kmeans_clustering(data_for_clustering, k=2, n_PCs=5, cmap_re=False):
    """
    Perform KMeans clustering and visualize results using UMAP.

    Parameters:
    - data_for_clustering (ndarray or DataFrame): Input data (cells x features)
    - k (int): Number of clusters
    - n_PCs (int): Number of principal components for UMAP visualization

    Returns:
    - labels (ndarray): Cluster assignments
    - centroids (ndarray): Cluster centers
    """
    data_for_clustering = data_for_clustering.copy()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_for_clustering)
    centroids = kmeans.cluster_centers_

    pca_result = PCA(n_components=n_PCs, random_state=42).fit_transform(data_for_clustering)
    umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(pca_result)

    if cmap_re:
        cm = _BIH_CMAP_re
    else:
        cm = _BIH_CMAP

    plt.figure(figsize=(8/2.54, 6/2.54), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=labels, cmap=cm, alpha=0.6, s=5, edgecolors='none', rasterized=True
    )
    plt.title("K-means Clustering", fontsize=7)
    plt.xlabel("UMAP 1", fontsize=6)
    plt.ylabel("UMAP 2", fontsize=6)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=6, width=0.3)
    ax.yaxis.set_tick_params(labelsize=6, width=0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    plt.show()

    return labels, centroids


def leiden_clustering(data_for_clustering, k=15, resolution=0.5, n_PCs=5, cmap_re=False):
    """
    Perform Leiden clustering and visualize results using UMAP.

    Parameters:
    - data_for_clustering (ndarray or DataFrame): Input data (cells x features)
    - k (int): Number of neighbors for kNN graph
    - resolution (float): Leiden resolution parameter
    - n_PCs (int): Number of principal components for UMAP visualization

    Returns:
    - cluster_assignments (ndarray): Cluster labels
    """
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(data_for_clustering)
    knn_graph = nbrs.kneighbors_graph(data_for_clustering, mode='connectivity')

    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources, targets))
    graph = ig.Graph(edges=edges, directed=False)
    graph.es['weight'] = knn_graph.data

    partition = la.find_partition(
        graph,
        partition_type=la.RBConfigurationVertexPartition,
        resolution_parameter=resolution
    )
    cluster_assignments = np.array(partition.membership)

    pca_result = PCA(n_components=n_PCs, random_state=42).fit_transform(data_for_clustering)
    umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(pca_result)

    if cmap_re:
        cm = _BIH_CMAP_re
    else:
        cm = _BIH_CMAP

    plt.figure(figsize=(8/2.54, 6/2.54), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=cluster_assignments, cmap=cm, alpha=0.6, s=5, edgecolors='none', rasterized=True
    )
    plt.title("Leiden Clustering", fontsize=7)
    plt.xlabel("UMAP 1", fontsize=6)
    plt.ylabel("UMAP 2", fontsize=6)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=6, width=0.3)
    ax.yaxis.set_tick_params(labelsize=6, width=0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    plt.show()

    return cluster_assignments


def hierarchical_clustering(data_for_clustering, k=2, n_PCs=5, cmap_re=False):
    """
    Perform hierarchical clustering and visualize results using UMAP.

    Parameters:
    - data_for_clustering (ndarray or DataFrame): Input data (cells x features)
    - k (int): Number of clusters
    - n_PCs (int): Number of principal components for UMAP visualization

    Returns:
    - labels (ndarray): Cluster assignments
    """
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data_for_clustering)

    pca_result = PCA(n_components=n_PCs, random_state=42).fit_transform(data_for_clustering)
    umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(pca_result)

    if cmap_re:
        cm = _BIH_CMAP_re
    else:
        cm = _BIH_CMAP

    plt.figure(figsize=(8/2.54, 6/2.54), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=labels, cmap=cm, alpha=0.6, s=5, edgecolors='none', rasterized=True
    )
    plt.title("Hierarchical Clustering", fontsize=7)
    plt.xlabel("UMAP 1", fontsize=6)
    plt.ylabel("UMAP 2", fontsize=6)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=6, width=0.3)
    ax.yaxis.set_tick_params(labelsize=6, width=0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)    
    plt.show()

    return labels



