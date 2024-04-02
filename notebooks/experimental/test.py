from seagal import seagal

sg = seagal.SEAGAL(
    visium_path=r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\external"
)
seagal.group_adata_by_genes(sg, inplace=True)


seagal.clustermap(sg, figsize=(6, 6), use_grouped=True)
seagal.save_coexpr(sg, path="./spatial_ct_colocolize.csv", use_grouped=True)
f, _ = seagal.hotspot(sg, "B cells", "T cells", use_grouped=True, cmap="bwr")
f.savefig("./B&T.pdf", dpi=100, bbox_inches="tight")
f = seagal.volcano(sg, use_grouped=True)
seagal.spatial_pattern_genes(sg, I=0.4, topK=None)
seagal.spatial_association(sg, grouped_only=False)
f = seagal.volcano(sg, use_grouped=False)
seagal.genemodules(sg)
seagal.module_pattern(sg)
seagal.module_hotspot(sg, cmap="bwr", vmax=4, vmin=-4)


# Test on spatially variable genes

import scanpy as sc
import pandas as pd

adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_02_29-02_29_42_PM\ST\Files\ST_SpatiallyVariableGenes_adata.h5ad"
)


adata_deconv = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_02_29-02_29_42_PM\ST\Files\ST_deconvoluted_adata.h5ad"
)

adata_deconv.obs
df_patterns = pd.read_excel(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_02_29-02_29_42_PM\ST\Files\ST_Patterns_AEH_2024_02_29-02_29_47_PM.xlsx"
)

df_patterns

cols_to_merge = [0, 1, 2]
df_patterns.set_index("Unnamed: 0", inplace=True)
df_patterns
merged_df = pd.merge(adata_deconv.obs, df_patterns, left_index=True, right_index=True)
# Join in a new dataframe the deconvoluted data, leiden clusters and patterns

merged_df["pattern"] = merged_df[cols_to_merge].idxmax(axis=1)
merged_df


cell_types_from_deconv = adata_deconv.obs.columns[0:6]
merged_df.columns
# Subset the merged_df to only include the cell types and the pattern

# Get the index of column 'hclust_7'
index = list(adata_deconv.obs.columns).index("hclust_7")

# Select all the columns onwards from 'hclust_7'
adata_deconv_cell_types = list(adata_deconv.obs.iloc[:, index + 1 :].columns)
adata_deconv_cell_types

pattern_list = adata_deconv_cell_types.copy()
pattern_list.append("pattern")

merged_df_subset_pattern = merged_df[pattern_list]
merged_df["leiden_clusters"] = merged_df["leiden_clusters"].astype(int)

# run correlation with merged_df
correlation_matrix = merged_df_subset_pattern.corr()
correlation_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.show()

clust_list = adata_deconv_cell_types.copy()
clust_list.append("leiden_clusters")
merged_df_subset_clust = merged_df[clust_list]

# run correlation with merged_df
correlation_matrix_v2 = merged_df_subset_clust.corr()
merged_df_subset_clust.columns
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_v2, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.show()


# SpatialDM
import spatialdm as sdm

adata = sdm.datasets.dataset.melanoma()
adata

sdm.weight_matrix(
    adata, l=1.2, cutoff=0.2, single_cell=False
)  # weight_matrix by rbf kernel
sdm.extract_lr(adata, "human", min_cell=3)  # find overlapping LRs from CellChatDB
sdm.spatialdm_global(
    adata, 1000, specified_ind=None, method="both", nproc=1
)  # global Moran selection
sdm.sig_pairs(
    adata, method="permutation", fdr=True, threshold=0.1
)  # select significant pairs
sdm.spatialdm_local(
    adata, n_perm=1000, method="both", specified_ind=None, nproc=1
)  # local spot selection
sdm.sig_spots(
    adata, method="permutation", fdr=False, threshold=0.1
)  # significant local spots

# visualize global and local pairs
import spatialdm.plottings as spl

sdm.plottings.global_plot(adata)
sdm.plottings.plot_pairs(adata, ["SPP1_CD44"], marker="s")


import scanpy as sc

adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\Keep\PipelineRun_2024_02_23-09_22_35_AM\ST\Files\ST_deconvoluted_adata.h5ad"
)


adata.obs = adata.obs[adata.obsm["deconvolution"].columns]


adata.raw.X[0].todense().shape

adata


# ST_preprocessed_DEG_adata

import scanpy as sc

adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\Keep\PipelineRun_2024_02_23-09_22_35_AM\ST\Files\ST_preprocessed_DEG_adata.h5ad"
)

sc.tl.rank_genes_groups(adata, "leiden_clusters", method="wilcoxon")

sc.pl.rank_genes_groups_violin(adata, n_genes=8)

fnc_plot = sc.pl.violin(
    adata,
    [
        "2_Smooth muscle",
        "AT1",
        "AT2",
        "B cell lineage",
        "EC arterial",
        "EC capillary",
        "EC venous",
        "Fibroblasts",
        "Innate lymphoid cell NK",
        "Macrophages",
        "Mast cells",
        "T cell lineage",
        "Unknown",
        "Lymphatic EC differentiating",
        "Lymphatic EC mature",
    ],
    groupby="leiden_clusters",
)

adata.var["highly_variable"]
sc.pl.dotplot(adata, var_names=["Samd11", "Perm1", "Hes4"], groupby="leiden_clusters")
