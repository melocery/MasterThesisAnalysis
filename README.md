## M.Sc. Thesis Analysis

**Title**: *Cell Overlaps in Tissue Sections Convolute de Novo Cell-Type Discovery in Spatial Transcriptomics Data*

**Summary**:

- Analyzed vertical signal integrity (VSI) in [mouse hypothalamus data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) (Moffitt et al., 2018).
- Investigated the impact of VSI on novel cell-type identification.

  - BANKSY identified two mature oligodendrocyte (MOD) subtypes: MOD\_wm and MOD\_gm (referred to as MOD1 and MOD2 here) (Singhal et al., 2024).
  - Question: Are these truly distinct subtypes?

**Tool**: [`ovrl.py`](https://github.com/HiDiHlabs/ovrl.py)

**Data**:
[Supplemental Data: Segmentation-free inference of cell types from in situ transcriptomics data](https://zenodo.org/records/3478502)

* `merfish_barcodes_example.csv`: mRNA spot coordinates


### Slice Equivalence

Compared slices at Bregma = -0.24 and Bregma = 0.16/0.26 (main slices used in BANKSY):

- Overview of all cell types
- Stacked bar plots showing cell type proportions
- UMAP embeddings from BANKSY
- MOD spatial patterns: presence of MOD1 and MOD2


### VSI Analysis

Assessed VSI using `ovrl.py`:

- Low VSI may cause misclassification or false discovery of new cell types
- Borders tend to have lower VSI, affecting boundary accuracy
- MOD1 cells generally show higher VSI than MOD2


### Transcript Neighborhoods

Explored local transcript environments:

- Circular region-based method
- k-nearest neighbors (kNN) method


### Spatial Expression of MOD Markers

Spatial distribution of MOD1 and MOD2 marker transcripts:

- Overall marker localization
- Individual marker expression patterns


### Marker Expression in scRNA-seq

Expression patterns in matched scRNA-seq data:  

- in MOD cells:  
  - Some show minimal expression in MOD cells  
  - No clear separation into MOD1 and MOD2  
- in non-OD cells:  
  - Expressed in neurons and astrocytes  
  - Some associated with Moffitt et al.'s neuron clusters  


### Marker-Associated VSI

Assessed VSI at specific marker locations:  

- Marker-specific VSI distributions
- Neuron-related markers exhibit broader distributions


### VSI Under Marker Exclusion Scenarios

Compared VSI after excluding transcripts under different conditions:

- Excluding MOD1 markers
- Excluding MOD2 markers
- Excluding all MOD markers

**Result**: Excluding MOD2 markers significantly improved VSI.
