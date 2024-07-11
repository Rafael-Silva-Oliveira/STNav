import scanpy as sc
import os
import celltypist
from celltypist import models
from matplotlib import rcParams
import matplotlib.pyplot as plt
from loguru import logger


sc_adata = sc.read_h5ad(r"/mnt/work/RO_src/data/raw/scRNA/SCLC/SCLC_Annotated.h5ad")

sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
healthy_adata = sc_adata[
    sc_adata.obs["histo"].isin(["normal"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]
sclc_adata = sc_adata[
    sc_adata.obs["histo"].isin(["SCLC"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]


logger.info("Training SCLC model")

sc_model_sclc = celltypist.train(
    sclc_adata,
    labels="cell_type_fine",
    n_jobs=-1,
    feature_selection=True,
    use_SGD=False,
    top_genes=500,
)
logger.info("Training healthy model")

sc_model_healthy = celltypist.train(
    healthy_adata,
    labels="cell_type_fine",
    n_jobs=-1,
    feature_selection=True,
    use_SGD=False,
    top_genes=500,
)
logger.info("Saving SCLC model")
sc_model_sclc.write(
    "/mnt/work/RO_src/Pipelines/STAnalysis/STNav/models/model_sclc_lung.pkl"
)
logger.info("Saving healthy model")

sc_model_healthy.write(
    "/mnt/work/RO_src/Pipelines/STAnalysis/STNav/models/model_healthy_lung.pkl"
)


# sc_model_sclc = models.Model.load(
#     model="/mnt/work/RO_src/Pipelines/STAnalysis/STNav/models/model_sclc_lung_balanced.pkl"
# )
# sc_model_healthy = models.Model.load(
#     model="/mnt/work/RO_src/Pipelines/STAnalysis/STNav/models/model_healthy_lung_balanced.pkl"
# )


# ### Do predictions with CellTypist

# adata_b2c = sc.read_h5ad(
#     "/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/B2C/adata_b2c.h5ad"
# )
# adata_st = adata_b2c.copy()
# adata_st.var_names_make_unique()
# adata_st = adata_st[adata_st.obs["bin_count"] > 4]  # min 6 bins'
# adata_st

# need integers for seuratv3 hvgs
# adata_st.X.data = np.round(adata_st.X.data)
# adata_st.raw = adata_st.copy()
# sc.pp.highly_variable_genes(adata_st, n_top_genes=10000, flavor="seurat_v3")
# sc.pp.normalize_total(adata_st, target_sum=1e4)
# sc.pp.log1p(adata_st)


# import squidpy as sq

# # Add spatial connectivities

# sq.gr.spatial_neighbors(adata_st)

# adata_st.X = adata_st.obsp["spatial_connectivities"].dot(adata_st.X)


# predictions_healthy = celltypist.annotate(
#     adata_st, model=sc_model_healthy, majority_voting=False
# )

# predictions_sclc = celltypist.annotate(
#     adata_st, model=sc_model_sclc, majority_voting=False
# )

# combine 2 models for cell annotations
# pred_healthy = predictions_healthy.to_adata()
# pred_healthy.obs["predicted_labels_healthy"] = pred_healthy.obs["predicted_labels"]
# pred_healthy.obs["conf_score_healthy"] = pred_healthy.obs["conf_score"]

# add the colorectal cancer model
# pred_sclc = predictions_sclc.to_adata()
# pred_sclc.obs["predicted_labels_sclc"] = pred_sclc.obs["predicted_labels"]
# pred_sclc.obs["conf_score_sclc"] = pred_sclc.obs["conf_score"]


# remove old annotations
# del cdata.obs["predicted_labels"]
# del cdata.obs["conf_score"]

# cdata = adata_st.copy()
# find the cell that have higher confidence in the crc model
# cdata.obs["higher_in_sclc"] = (
#     pred_healthy.obs["conf_score_healthy"] < pred_sclc.obs["conf_score_sclc"]
# )

# cdata.obs.to_excel("./sclc_conf_score.xlsx")
# eclude the ones that labeled as unknown in crc model
# cdata.obs.loc[pred_sclc.obs["predicted_labels_sclc"] == "Unknown", "higher_in_sclc"] = (
#     False
# )
# cdata.obs["higher_in_sclc"].value_counts()

# create new unified annotations
# cdata.obs["predicted_labels"] = cdata.obs["predicted_labels_healthy"]
# cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("object")
# cdata.obs.loc[cdata.obs["higher_in_sclc"], "predicted_labels"] = cdata.obs.loc[
#     cdata.obs["higher_in_sclc"], "predicted_labels_sclc"
# ]
# cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("category")
# cdata.obs["predicted_labels"]

# cdata.obs["predicted_labels"].value_counts()


# cdata.obs["array_row"] = cdata.obs["array_row"].astype(int)
# cdata.obs["array_col"] = cdata.obs["array_col"].astype(int)
# cdata.obsm["spatial"] = cdata.obsm["spatial"].astype(float)

# import squidpy as sq

# plt.style.use("default")
# sq.pl.spatial_scatter(
#     cdata,
#     color=f"predicted_labels",  # use the column from the whole dataframesize=1.15,
#     dpi=3500,
#     size=1,
#     img_alpha=0.5,
#     shape="circle",
#     save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/b2c_celltypist_balanced_4_bin_conn_adj.png",
#     legend_fontsize=3.5,
# )

# from matplotlib.colors import ListedColormap

# Get unique cell types
# cell_types = cdata.obs["predicted_labels"].unique()
# color_map = ListedColormap(["yellow" for _ in range(len(cell_types))])

# for cell_type in cell_types:
#     Create a subset of cdata for the current cell type
#     cdata_subset = cdata[cdata.obs["predicted_labels"] == cell_type]

#     Plot the subset
#     sq.pl.spatial_scatter(
#         cdata_subset,
#         color="predicted_labels",  # use the column from the subset dataframe
#         size=1,
#         dpi=3500,
#         shape="circle",
#         palette=color_map,
#         img_alpha=0.5,  # adjust this value to make the image lighter
#         save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/b2c_celltypist_{cell_type}_balanced_4_bin_conn_adj.png",
#         legend_fontsize=3.5,  # adjust this value to make the legend smaller
#     )
