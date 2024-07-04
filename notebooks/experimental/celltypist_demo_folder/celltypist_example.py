import scanpy as sc
import celltypist
from celltypist import models
import json
import squidpy as sq
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colorspacious import cspace_converter
import numpy as np
import pandas as pd

# Load ST

# config_json = r"/mnt/work/RO_src/Pipelines/STAnalysis/configs/analysis_cloud.json"
# # Open config json with json.load
# config_total = json.load(open(config_json, "r"))
# config = config_total["ST"]
# adata_st = sc.read_visium(
#     path=config["path"],
#     count_file=config["count_file"],
#     load_images=config["load_images"],
#     source_image_path=config["source_image_path"],
# )

adata_st = sc.read_h5ad(r"/mnt/work/RO_src/data/raw/adata_02micron.h5ad")


adata_st.var_names_make_unique()
sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)


# Load scRNA data

# Train a model with the SCLC dataset

sc_adata = sc.read_h5ad(r"/mnt/work/RO_src/data/raw/scRNA/SCLC/Combined samples.h5ad")
sc_adata.var.set_index("feature_name", inplace=True)
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc_adata.var_names = sc_adata.var_names.str.upper()
sc_adata.var.index = sc_adata.var.index.str.upper()

sc_adata = sc_adata[~sc_adata.obs["cell_type_fine"].isin(["Hepatocyte", "NSCLC"])]
sc_adata = sc_adata[
    sc_adata.obs["histo"].isin(["normal", "SCLC"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]
sc_adata.write_h5ad(r"/mnt/work/RO_src/data/raw/scRNA/SCLC/SCLC_lung.h5ad")

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

# subset_sc_adata = sc_adata[sc_adata.obs["histo"].isin(["SCLC","normal"]) & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]), :]


healthy_adata.obs["cell_type_fine"].value_counts()

sclc_adata.obs["cell_type_fine"].value_counts()


sc_model_healthy = celltypist.train(
    healthy_adata,
    labels="cell_type_fine",
    n_jobs=10,
    feature_selection=True,
    epochs=30000,
)
sc_model_healthy.write(
    "/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/Non_Adjusted/model_healthy_lung.pkl"
)

sc_model_sclc = celltypist.train(
    sclc_adata, labels="cell_type_fine", n_jobs=10, feature_selection=True, epochs=30000
)
sc_model_sclc.write(
    "/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/Non_Adjusted/model_sclc_lung.pkl"
)


sc_model_sclc = models.Model.load(
    model="/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/model_sclc_lung.pkl"
)
sc_model_healthy = models.Model.load(
    model="/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/model_healthy_lung.pkl"
)


sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)

# # Add spatial connectivities4

# sq.gr.spatial_neighbors(adata_st)

# adata_st.obsp["spatial_connectivities"] = csr_matrix(
#     adata_st.obsp["spatial_connectivities"]
# )

# adata_st.X = adata_st.obsp["spatial_connectivities"].dot(adata_st.X)


predictions_healthy = celltypist.annotate(
    adata_st, model=sc_model_healthy, majority_voting=False
)

predictions_sclc = celltypist.annotate(
    adata_st, model=sc_model_sclc, majority_voting=False
)

# combine 2 models for cell annotations
# add the healthy gut model


pred_healthy = predictions_healthy.to_adata()
pred_healthy.obs["predicted_labels_healthy"] = pred_healthy.obs["predicted_labels"]
pred_healthy.obs["conf_score_healthy"] = pred_healthy.obs["conf_score"]

# add the colorectal cancer model
pred_sclc = predictions_sclc.to_adata()
pred_sclc.obs["predicted_labels_crc"] = pred_sclc.obs["predicted_labels"]
pred_sclc.obs["conf_score_crc"] = pred_sclc.obs["conf_score"]

# remove old annotations
# del cdata.obs["predicted_labels"]
# del cdata.obs["conf_score"]

cdata = adata_st.copy()
# find the cell that have higher confidence in the crc model
cdata.obs["higher_in_crc"] = (
    pred_healthy.obs["conf_score_healthy"] < pred_sclc.obs["conf_score_crc"]
)

# eclude the ones that labeled as unknown in crc model
cdata.obs.loc[pred_sclc.obs["predicted_labels_crc"] == "Unknown", "higher_in_crc"] = (
    False
)
cdata.obs["higher_in_crc"].value_counts()

# create new unified annotations
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels_healthy"]
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("object")
cdata.obs.loc[cdata.obs["higher_in_crc"], "predicted_labels"] = cdata.obs.loc[
    cdata.obs["higher_in_crc"], "predicted_labels_crc"
]
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("category")
cdata.obs["predicted_labels"]

cdata.obs["predicted_labels"].value_counts()


cdata.obs["array_row"] = cdata.obs["array_row"].astype(int)
cdata.obs["array_col"] = cdata.obs["array_col"].astype(int)
cdata.obsm["spatial"] = cdata.obsm["spatial"].astype(float)
cdata.obs["in_tissue"] = cdata.obs["in_tissue"].astype(int)

cdata.write_h5ad(
    "/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/combined_adata_nonadjusted.h5ad"
)

cdata = sc.read_h5ad(
    r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/combined_adata.h5ad"
)

cdata.obs["predicted_labels"].value_counts()

# Generate a Glasbey color palette with 25 distinct colors
glasbey = sns.color_palette("tab20", 25)
glasbey_rgb = np.array(glasbey) * 255

# Convert the seaborn color palette to a ListedColormap object
palette = ListedColormap(glasbey_rgb / 255.0)

plt.style.use("default")
sq.pl.spatial_scatter(
    cdata,
    color=f"predicted_labels",  # use the column from the whole dataframe
    size=1,
    palette=palette,
    dpi=1500,
    shape="square",
    save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/cell_typist_sclc_nonadjusted.png",
    legend_fontsize=3.5,  # adjust this value to make the legend smaller
)

palette = ListedColormap(["#FFA500"])  # Hex code for orange color

for cell_type in list(cdata.obs["predicted_labels"].unique()):
    # Create a subset of the data for the current cell type
    subset: sc.AnnData = cdata[cdata.obs[f"predicted_labels"] == cell_type].copy()
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size as needed
    sq.pl.spatial_scatter(
        subset,
        color="predicted_labels",  # use the column from the whole dataframe
        size=1,
        palette=palette,
        dpi=1500,
        shape="square",
        save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/{cell_type}_sclc_nonadjusted.png",
        legend_fontsize=3.5,  # adjust this value to make the legend smaller
    )
