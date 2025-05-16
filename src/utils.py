import numpy as np
import pandas as pd
import anndata as ad
from natsort import natsorted

from shapely.geometry import Polygon, Point

from scipy.io import mmread
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg as la

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
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

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

    # Scale data without centering to keep sparsity
    scaler = MaxAbsScaler()
    adata.X = scaler.fit_transform(adata.X)

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

def process_related_genes_check(DE_genes, sc_data, top_n=26):
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
    failed_genes = []
    too_few_corr = {}

    print(f"=== Debug Summary ===")
    print(f"🧬 Total DE genes given: {len(DE_genes)}")

    for gene in sorted(set(DE_genes)):
        if gene not in sc_data.index:
            failed_genes.append(gene)
            print(f"❌ {gene} not in sc_data, skipped.")
            continue

        correlated = find_corr_genes(gene, sc_data)
        if gene in correlated.index:
            correlated = correlated.drop(labels=[gene], errors='ignore')

        top_genes = correlated.iloc[:top_n].index.tolist()
        related_genes.update(top_genes)

        if len(top_genes) < top_n:
            too_few_corr[gene] = len(top_genes)
            print(f"⚠️ {gene}: Only got {len(top_genes)} correlated genes (expected {top_n})")
        else:
            print(f"✅ {gene}: Added {len(top_genes)} genes")

    print(f"\n=== Summary ===")
    print(f"✅ Processed DE genes: {len(DE_genes) - len(failed_genes)}")
    print(f"❌ Skipped genes: {len(failed_genes)} — {failed_genes}")
    print(f"⚠️ DE genes with < {top_n} correlated genes: {len(too_few_corr)} — {too_few_corr}")
    print(f"📦 Total unique related genes returned: {len(related_genes)}")

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

    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=labels, cmap=cm, alpha=0.6, s=11, edgecolors='none'
    )
    plt.title("K-means Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=cluster_assignments, cmap=cm, alpha=0.6, s=11, edgecolors='none'
    )
    plt.title("Leiden Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=labels, cmap=cm, alpha=0.6, s=11, edgecolors='none'
    )
    plt.title("Hierarchical Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    return labels
