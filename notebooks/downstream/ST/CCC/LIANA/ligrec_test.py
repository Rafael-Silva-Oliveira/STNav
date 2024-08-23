# %%
from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
import scanpy as sc
import liana as li
import squidpy as sq
import gc
import decoupler as dc
from mudata import MuData

# %%
adata = sc.read_h5ad(r"/mnt/archive2/RO_src/data/adata_subset3.h5ad")

adata.var_names = adata.var_names.str.upper()
adata.var.index = adata.var.index.str.upper()

# %%
# # Grab the lig-rec pairs
# ligrec_pairs = li.mt.cellphonedb(
#     adata,
#     groupby="cell_type",
#     resource_name="consensus",
#     expr_prop=0.1,
#     verbose=True,
#     use_raw=False,
#     key_added="cpdb_res",
# )

# # Get liana results rom the ligrec_pairs
# liana_results = li.mt.rank_aggregate(
#     adata,
#     groupby="cell_type",
#     resource_name="consensus",
#     use_raw=False,
#     groupby_pairs=adata.uns["cpdb_res"],
#     expr_prop=0.05,
#     verbose=True,
# )
# plot, _ = li.ut.query_bandwidth(
#     coordinates=adata.obsm["spatial"], start=0, end=4500, interval_n=30
# )
# plot

# plot.savefig("./bandwidth.png")

# li.pl.dotplot(
#     adata=adata,
#     colour="magnitude_rank",
#     size="specificity_rank",
#     inverse_size=True,
#     inverse_colour=True,
#     source_labels=["SCLC-P", "SCLC-N", "SCLC-A", "Fibroblast"],
#     target_labels=["B cell", "T cell", "Macrophage", "Tuft"],
#     top_n=100,
#     orderby="magnitude_rank",
#     orderby_ascending=True,
#     figure_size=(8, 7),
# )

######### Spatial Connectivity
# %%
li.ut.spatial_neighbors(
    adata,
    bandwidth=100,
    cutoff=0.1,
    kernel="gaussian",
    spatial_key="liana_spatial_connectivity",
    set_diag=False,
    max_neighbours=4,
)
# %%
li.mt.bivariate(
    adata,
    layer="lognorm_counts",
    connectivity_key="liana_spatial_connectivity",
    resource_name="consensus",  # NOTE: uses HUMAN gene symbols!
    local_name="jaccard",  # Name of the function
    global_name="morans",  # Name global function
    n_perms=20,  # Number of permutations to calculate a p-value
    mask_negatives=False,  # Whether to mask LowLow/NegativeNegative interactions
    add_categories=True,  # Whether to add local categories to the results
    nz_prop=0.01,  # Minimum expr. proportion for ligands/receptors and their subunits
    use_raw=False,
    verbose=True,
)
# %%
adata.write_h5ad(
    filename=r"/mnt/archive2/RO_src/data/adata_subset_liana_jaccard_100_4.h5ad"
)
# adata = sc.read_h5ad(r"/mnt/archive2/RO_src/data/adata_subset_liana_v3.h5ad")

# %%
# Global summaries
lrdata = adata.obsm["local_scores"]

# The average local metric used in the global summary represents coverage
lrdata.var.sort_values("mean", ascending=False).head(10)

# We can also use Moran's R, an extension of Moran's I for spatially variable genes. Among most variable interactions and with the highest global morans R, is an interaction that most likely represents biological relationships, with distinct spatial clustering patterns.
lrdata.var.sort_values("morans", ascending=False).head(10)


# %%
sq.pl.spatial_scatter(
    lrdata,
    color=["VIM^CD44", "TIMP1^CD63", "SPP1^CD44"],
    size=1.4,
    img_alpha=0.5,
    dpi=300,
)

# %% Local summaries
"""
Here, we can distinguish areas in which the interaction between interaction members is positive (high-high) in Red (1), while interactions in which one is high the other is low or negative (High-low) are in Blue (-1). We also see that some interactions are neither, these are predominantly interactions in which both members are lowly-expressed (i.e. low-low); we see those in white (0).

When set to `mask_negatives=False`, we also return interactions that are not between necessarily positive/high magnitude genes ; when set to `mask_negatives=True`, we mask interactions that are negative (low-low) or uncategorized; for both the p-values and the local scores.

"""
sc.pl.spatial(
    lrdata,
    layer="cats",
    color=["VTN^ITGAV_ITGB5", "TIMP1^CD63"],
    size=1.4,
    cmap="coolwarm",
)


# %%
"""
Intercellular Patterns

Having the LR scores, we can use NMF to identify coordinated cell-cell communication signatures.
* W (basis matrix): Each basis vector is a patten of ligand-receptor expression in the dataset. The values in W (factor score) indicate the strenghts of factor in each spot; high values indicate high influence by the associated communication signature, and vice-versa
* H (coefficient matrix): Each row of H is the participation of the corresponding sample in the identified factor. The elements of each basis vector indicate the contribution of different interactions to the pattern (factor).
"""

li.multi.nmf(
    lrdata, n_components=3, inplace=True, random_state=0, max_iter=200, verbose=True
)


# Extract the variable loadings
lr_loadings = li.ut.get_variable_loadings(lrdata, varm_key="NMF_H").set_index("index")

# Extract the factor scores
factor_scores = li.ut.get_factor_scores(lrdata, obsm_key="NMF_W")

nmf = sc.AnnData(
    X=lrdata.obsm["NMF_W"],
    obs=lrdata.obs,
    var=pd.DataFrame(index=lr_loadings.columns),
    uns=lrdata.uns,
    obsm=lrdata.obsm,
)

sc.pl.spatial(nmf, color=[*nmf.var.index, None], size=1.4, ncols=2)
lr_loadings.sort_values("Factor2", ascending=False).head(10)


# %%

# Create a .obsm one-hot encoding of the variables
# If the labels are in a single string, separated by commas, split them into lists
df = pd.DataFrame()
df["Cell Types"] = adata.obs["cell_type"].str.split(", ")
# Explode the list into separate rows
df_exploded = df.explode("Cell Types")
df_binary = pd.get_dummies(df_exploded["Cell Types"])
df_binary = df_binary.groupby(df_exploded.index).sum()

# Combine the binary labels with the original DataFrame (without the original 'Cell Types' column)
df_final = df.drop(columns=["Cell Types"]).join(df_binary)

# %%
adata.obs.drop(columns=list(df_final.columns), inplace=True)

# %%
adata.obsm["compositions"] = df_final.copy()
# let's extract those
comps = li.ut.obsm_to_adata(adata, "compositions")
# check key cell types
sc.pl.spatial(comps, color=["T cell", "B cell", "SCLC-N", "SCLC-P"], size=1.3, ncols=2)


# Get transcription factor resource
net = dc.get_collectri()

# run enrichment
dc.run_ulm(
    mat=adata,
    net=net,
    source="source",
    target="target",
    weight="weight",
    verbose=True,
    use_raw=False,
)
adata.obsm["collectri_ulm_estimate"] = adata.obsm["ulm_estimate"].copy()
adata.obsm["collectri_ulm_pvals"] = adata.obsm["ulm_pvals"].copy()

acts = dc.get_acts(adata, obsm_key="collectri_ulm_estimate")
acts
# %%

sc.pl.spatial(
    acts,
    color=["IRF1", "leiden_clusters"],
    cmap="RdBu_r",
    size=1.5,
    vcenter=0,
    frameon=False,
)
sc.pl.violin(acts, keys="IRF1", groupby="leiden_clusters", rotation=90)
# %%
### Decoupler -Transcription factor activity inference
sc.pl.umap(acts, color=["PAX5", "leiden_clusters"], cmap="RdBu_r", vcenter=0)
sc.pl.violin(acts, keys=["PAX5"], groupby="leiden_clusters", rotation=90)
df = dc.rank_sources_groups(
    acts, groupby="leiden_clusters", reference="rest", method="wilcoxon"
)

df
# %%
n_markers = 3
source_markers = (
    df.groupby("group")
    .head(n_markers)
    .groupby("group")["names"]
    .apply(lambda x: list(x))
    .to_dict()
)
source_markers


# %%
sc.pl.matrixplot(
    acts,
    source_markers,
    "leiden_clusters",
    dendrogram=True,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="RdBu_r",
)

# %%
sc.pl.violin(acts, keys=["EBF1"], groupby="leiden_clusters", rotation=90)

# %%
dc.plot_network(
    net=net,
    n_sources=["PAX5", "EBF1", "RFXAP"],
    n_targets=15,
    node_size=100,
    s_cmap="white",
    t_cmap="white",
    c_pos_w="darkgreen",
    c_neg_w="darkred",
    figsize=(5, 5),
)
# %%
### Decoupler - Functional enrichment of biological terms
progeny = dc.get_progeny(organism="human", top=500)


# %%
# Extract highly-variable TF activities
est = li.ut.obsm_to_adata(adata, "ulm_estimate")
est.var["cv"] = est.X.std(axis=0) / est.X.mean(axis=0)
top_tfs = est.var.sort_values("cv", ascending=False, key=abs).head(5).index

# Create MuData object with TF activities and cell type proportions - In this case, the proportion is actually 1 or 0;
mdata = MuData({"tf": est, "comps": comps})
mdata.obsp = adata.obsp
mdata.uns = adata.uns
mdata.obsm = adata.obsm
from itertools import product

interactions = list(product(comps.var.index, top_tfs))
# %%
len(interactions)
# %%
# Estimate cosine similarity
li.mt.bivariate(
    mdata,
    x_mod="comps",
    y_mod="tf",
    local_name="cosine",
    interactions=interactions,
    mask_negatives=False,
    add_categories=False,
    x_use_raw=False,
    y_use_raw=False,
    nz_prop=0.01,  # Minimum expr. proportion for ligands/receptors and their subunits
    xy_sep="<->",
    x_name="celltype",
    y_name="tf",
)

# %%
######### JUST PLOTTING ########
adata = sc.read_h5ad(
    r"/mnt/archive2/RO_src/data/processed/PipelineRun_2024_08_19-06_22_09_PM/ST/Files/preprocessed_adata.h5ad"
)
# %%
lrdata = adata.obsm["local_scores"]

# %%
sc.set_figure_params(
    dpi=80, dpi_save=300, format="png", frameon=False, transparent=True, figsize=[5, 5]
)
# %%
sc.pl.spatial(lrdata, color=["TIMP1^CD63"], size=1.4, vmax=1, cmap="magma")

# %%
sc.pl.spatial(
    lrdata,
    layer="pvals",
    color=["TIMP1^CD63"],
    size=1.4,
    cmap="magma_r",
)

# %%
sc.pl.spatial(
    lrdata,
    layer="cats",
    color=["TIMP1^CD63"],
    size=1.4,
    cmap="coolwarm",
)
# %%
nmf = sc.read_h5ad(
    r"/mnt/archive2/RO_src/data/processed/PipelineRun_2024_08_19-10_47_37_AM/ST/Files/LIANA_NMF.h5ad"
)
# %%
sc.pl.spatial(nmf, color=[*nmf.var.index, None], size=1.4, ncols=2)


# %%
est = li.ut.obsm_to_adata(adata, "ulm_estimate")
est.var["cv"] = est.X.std(axis=0) / est.X.mean(axis=0)
top_tfs = est.var.sort_values("cv", ascending=False, key=abs).head(5).index

# Create MuData object with TF activities and cell type proportions - In this case, the proportion is actually 1 or 0;
mdata = MuData({"tf": est, "comps": comps})
mdata.obsp = adata.obsp
mdata.uns = adata.uns
mdata.obsm = adata.obsm
from itertools import product

interactions = list(product(comps.var.index, top_tfs))
# %%

sc.pl.spatial(
    mdata.mod["local_scores"],
    color=["Myeloid<->SNAI2", "CM<->HAND1"],
    size=1.4,
    cmap="coolwarm",
    vmax=1,
    vmin=-1,
)

# %%
