# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
import stlearn as st


# Set the timeout to 10 seconds (adjust as needed)
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = (
    "2000"  # Set the timeout to 10 seconds (adjust as needed)
)
os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = (
    "10"  # Set the timeout to 10 seconds (adjust as needed)
)
os.environ["PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT"] = "True"

# Training a model to predict proportions on spatial data using scRNA seq as reference
import scvi

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scipy.sparse import csr_matrix

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


# Loading dataset
adata_st = sc.read_visium(
    path=r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\external",
    count_file="CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_filtered_feature_bc_matrix.h5",
    load_images=True,
    source_image_path=r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\external\spatial",
)
adata_st.var_names_make_unique()
adata_st.var_names = adata_st.var_names.str.lower()
library_names = ["CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma"]

# Quality control
adata_st.var["mito"] = adata_st.var_names.str.startswith("mt-")
adata_st.var.columns
sc.pp.calculate_qc_metrics(
    adata_st, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True
)
adata_st.obs["pct_counts_mito"]
# sc.pl.violin(
#     adata_st,
#     ["pct_counts_mito"],
#     jitter=0.4,
#     rotation=45,
# )


# Visualize. In this case we only have one section per slide (unlike brain for example)
# sc.pl.spatial(adata_st)

# Filtering by mitochondria pct
keep = (adata_st.obs["pct_counts_mito"] < 25) & (
    adata_st.obs["n_genes_by_counts"] > 1000
)

# Keep just those that passed the filtering
adata_st = adata_st[keep, :]

# Top expressed genes
# sc.pl.highest_expr_genes(adata_st, n_top=30)

# We still see MT genes. Let's remove them
# 2.0.3  Filter genes
mt_genes_to_remove = adata_st.var_names.str.startswith("mt-")
keep = np.invert(mt_genes_to_remove)
adata_st = adata_st[:, keep]
print(adata_st.n_obs, adata_st.n_vars)

# Analysis
# save the counts to a separate object for later, we need the normalized counts in raw for DEG dete
adata_st_copy = adata_st.copy()
adata_st_copy.obs
sc.pp.normalize_total(adata_st, inplace=True)
sc.pp.log1p(adata_st)
sc.pp.highly_variable_genes(adata_st, flavor="seurat", n_top_genes=100, inplace=True)
# adata_st.raw = adata_st
adata_st.n_obs  # Number of cells
adata_st.n_vars  # Number of genes
adata_st.var.columns
adata_st = adata_st[:, adata_st.var.highly_variable > 0]
adata_st.var
sc.pp.scale(adata_st)

# Plotting on individual genes
adata_st.var_names
sc.pl.spatial(adata_st, color=["tekt2", "tspan1"])

# 3.0.1 UMAP
sc.pp.neighbors(adata_st, n_neighbors=5)

sc.tl.umap(adata_st)
sc.tl.leiden(adata_st, key_added="clusters")

sc.pl.umap(adata_st, color=["clusters"], palette=sc.pl.palettes.default_20)


clusters_colors = dict(
    zip(
        [str(i) for i in range(len(adata_st.obs.clusters.cat.categories))],
        adata_st.uns["clusters_colors"],
    )
)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

sc.pl.spatial(
    adata_st,
    img_key="hires",
    color="clusters",
    size=1.5,
    palette=[
        v
        for k, v in clusters_colors.items()
        if k in adata_st.obs.clusters.unique().tolist()
    ],
    legend_loc=None,
    show=False,
)
plt.tight_layout()

# 3.0.4 Identification of spatially variable
sc.tl.rank_genes_groups(adata_st, "clusters", method="wilcoxon")

sc.pl.rank_genes_groups_heatmap(adata_st, groups="3", n_genes=10, groupby="clusters")
top_genes = sc.get.rank_genes_groups_df(adata_st, group="2", log2fc_min=1)["names"]
top_genes

# sc.pl.spatial(adata, color=top_genes)

# Integrate with scRNA seq

# TODO: Try connecting with NTNU VPN to see if it still crashes.
sc_adata_raw = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\external\spatial\Lung_facs.h5ad"
)

sc_adata = sc_adata_raw.copy()
# Keep just a few cells to avoid memory errors down the line
# cell_types_to_keep = sc_adata.obs["cell_ontology"].isin(["Goblet", "NK", "T"])
# sc_adata = sc_adata[cell_types_to_keep, :]
# Change index to be the actual gene names
sc_adata.var.index = sc_adata.var.index.str.lower()
# sc_adata.obs.nCount_RNA = sc_adata.obs.nCount_RNA.astype(np.float32)
# sc_adata.var["_index"] = sc_adata.var["_index"].str.lower()
# sc_adata.var["features"] = sc_adata.var["features"].str.lower()
# # sc_adata.var.set_index("_index", inplace=True)
# adata_sc_dictionary = {
#     idx: a for idx, a in zip(sc_adata.var.index, sc_adata.var["features"])
# }
# sc.pl.umap(sc_adata, color="cell_ontology", legend_loc="on data")
# pre-processing
sc.pp.filter_genes(sc_adata, min_counts=200)
sc_adata.layers["counts"] = sc_adata.X.copy()
# normalization for selection of highly variable genes (not for the deconvolution)
sc.pp.normalize_total(sc_adata)
sc.pp.log1p(sc_adata)
sc_adata.raw = sc_adata
sc.pp.highly_variable_genes(
    sc_adata, n_top_genes=250, subset=True, layer="counts", flavor="seurat"
)

# Deconvolution with Stereoscope. Use top DEG genes per cluster - run DEG detection on the scRNA seq data
sc.tl.rank_genes_groups(sc_adata, "cell_ontology_class", method="t-test", n_genes=2000)
# sc.pl.rank_genes_groups_dotplot(
#     sc_adata, n_genes=-4
# )  # The mean expression per group is logarithmized (confirm this)
sc_adata.uns["rank_genes_groups"]
sc.tl.filter_rank_genes_groups(sc_adata, min_fold_change=2)

# genes_sc = sc.get.rank_genes_groups_df(sc_adata, group=None)
# genes_sc["names"] = genes_sc["names"].astype(int)
# genes_sc["names"] = genes_sc["names"].map(adata_sc_dictionary)
# genes_sc
# deg = genes_sc.names.unique().tolist()
# print(len(deg))
# # Replace the numbers with actual gene names and lower case them
# deg = np.intersect1d(deg, adata_st.var.index).tolist()
# print(len(deg))

# genes_st = sc.get.rank_genes_groups_df(adata_st, group=None)
# genes_st
# filter genes to be the same on the spatial data
intersect = np.intersect1d(sc_adata.var_names, adata_st.var_names)
adata_st = adata_st[:, intersect].copy()
sc_adata = sc_adata[:, intersect].copy()

sc.tl.pca(sc_adata, svd_solver="arpack")
sc.pp.neighbors(sc_adata, n_pcs=30, n_neighbors=20)
sc.pl.umap(sc_adata, color=["cell_ontology_class"])

sc_adata.X = sc_adata.layers["counts"].copy()
# Subset to a max of 200 cells per class to be a more fair comparison of cell types
# # target_cells = 200

# # adata_sc2 = [
# #     sc_adata[sc_adata.obs.cell_ontology_class == clust]
# #     for clust in sc_adata.obs.cell_ontology_class.cat.categories
# #     if clust != "nan"
# # ]

# # for dat in adata_sc2:
# #     print(dat)
# #     if dat.n_obs > target_cells:
# #         sc.pp.subsample(dat, n_obs=target_cells)

# # sc_adata = adata_sc2[0].concatenate(*adata_sc2[1:])
# # sc_adata.obs.cell_ontology_class.value_counts()

# # Re-plot

# # sc.pl.umap(
# #     sc_adata,
# #     color=["cell_ontology"],
# #     legend_loc="on data",
# #     palette=sc.pl.palettes.default_20,
# # )


# # sc.pl.rank_genes_groups_dotplot(
# #     sc_adata, n_genes=-4
# # )  # The mean expression per group is logarithmized (confirm this)


# # What you're doing here is basically the normalize_total function I think
# # sc_adata.X = np.nan_to_num(sc_adata.X, copy=False)
# E = sc_adata.X.expm1()
# n = np.sum(E, 1)
# print(np.min(n), np.max(n))
# factor = np.mean(n)
# sc_adata.obs.nCount_RNA = sc_adata.obs.nCount_RNA
# nC = np.array(sc_adata.obs.nCount_RNA)  # true number of counts
# scaleF = nC / factor
# # scaleF = scaleF.reshape(-1, 1)
# # C = np.multiply(E)
# C = csr_matrix(E).multiply(scaleF[:, None])
# C = C.tocsr()
# # C = E * scaleF[:, None]
# C = C.astype("int")
# # C = np.nan_to_num(C, copy=False)
# sc_adata = sc_adata.copy()
# sc_adata.X = C

# # de_genes_from_sc = genes_sc.names.str.lower().unique().tolist()
# # de_genes_from_st = genes_st.names.unique().tolist()
# # common_genes = list(set(de_genes_from_sc) & set(de_genes_from_st))
# sc_adata.var.set_index("features", inplace=True)
# sc_adata.layers["total_count"] = sc_adata.X.copy()
# # sc_adata.layers["total_count"] = np.nan_to_num(sc_adata.layers["total_count"], nan=0)
# sc_adata = sc_adata[:, deg].copy()

# # Update the original CSR matrix with the modified data

# # TODO: perhaps filter scRNA data by reads that only have >= 1 reads?
# # keep_read_above_1 = sc_adata.layers["total_count"].data > 0

sc_adata.layers["counts"].data = sc_adata.layers["counts"].data.astype(np.float32)

# sc_adata.layers["counts"].data = np.maximum(sc_adata.layers["counts"].data, 1)
# # sum(sc_adata.layers["total_count"].data)
RNAStereoscope.setup_anndata(sc_adata, layer="counts", labels_key="cell_ontology_class")

train = False
if train:
    sc_model = RNAStereoscope(sc_adata)
    sc_model.train(max_epochs=100)
    sc_model.history["elbo_train"][10:].plot()
    sc_model.save("scmodel", overwrite=True)
else:
    sc_model = RNAStereoscope.load(
        r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\scmodel",
        sc_adata,
    )

adata_st
st_adata = adata_st.copy()
st_adata.layers["counts"] = st_adata.X.copy()
# st_adata = st_adata[:, common_genes].copy()
# import scanorama

# adatas = [sc_adata, st_adata]
# adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True)
# integrated_sc_st_adata = sc.concat(
#     adatas_cor,
#     label="dataset",
#     keys=["scRNA", "visium"],
#     join="outer",
#     uns_merge="first",
# )


# integrated_sc_st_adata.layers["counts"] = integrated_sc_st_adata.X.copy()

SpatialStereoscope.setup_anndata(st_adata, layer="counts")

if train:
    spatial_model = SpatialStereoscope.from_rna_model(
        st_adata, sc_model, prior_weight="minibatch"
    )
    spatial_model.train(
        max_epochs=5000,
        plan_kwargs={"weight_decay": 0},
        early_stopping=False,
        early_stopping_monitor="elbo_train",
        early_stopping_patience=5,
        early_stopping_min_delta=0.4,
    )
    plt.plot(spatial_model.history["elbo_train"], label="train")
    plt.title("loss over training epochs")
    plt.legend()
    plt.show()
    spatial_model.save("stmodel", overwrite=True)
else:
    spatial_model = SpatialStereoscope.load(
        r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\stmodel",
        st_adata,
    )
    print("Loaded Spatial model from file!")

st_adata.obsm["deconvolution"] = spatial_model.get_proportions()
for ct in st_adata.obsm["deconvolution"].columns:
    print(ct)
    st_adata.obs[ct] = st_adata.obsm["deconvolution"][ct]

# st_adata.obsm["deconvolution"].columns
# st_adata.obs["B cell"]
