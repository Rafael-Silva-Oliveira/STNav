import os
import tempfile

import gdown
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scvi.external import CellAssign
import json
import numpy as np

scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()

sc_adata = sc.read(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_15-09_33_10_AM/scRNA/Files/raw_adata.h5ad"
)

config_json = r"/mnt/work/RO_src/Pipelines/STNav/configs/analysis_cloud.json"
# Open config json with json.load
config_total = json.load(open(config_json, "r"))
config = config_total["ST"]

adata_st = sc.read_visium(
    path=config["path"],
    count_file=config["count_file"],
    load_images=config["load_images"],
    source_image_path=config["source_image_path"],
)

adata_st.var_names_make_unique()


markers = pd.read_csv(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_21-10_49_01_PM/ST/Files/ST_scMAGS_markers.csv"
)

# Ensure that there are no duplicates
markers = markers.drop_duplicates()

# Pivot the DataFrame
# Pivot the DataFrame
pivot_df = markers.pivot_table(
    index="Markers", columns="CellType", aggfunc=len, fill_value=0
)

pivot_df
# Convert float to int
pivot_df = pivot_df.astype(int)
pivot_df.columns.name = None
pivot_df.reset_index(inplace=True)
pivot_df.rename(columns={"Markers": "Gene"}, inplace=True)
pivot_df.set_index("Gene", inplace=True)


adata_st.obs.index = adata_st.obs.index.astype("str")
adata_st.var.index = adata_st.var.index.astype("str")
adata_st.var_names_make_unique()
adata_st.obs_names_make_unique()
adata_st.var_names = adata_st.var_names.str.upper()
adata_st.var.index = adata_st.var.index.str.upper()

lib_size = adata_st.X.sum(1)
adata_st.obs["size_factor"] = lib_size / np.mean(lib_size)

follicular_bdata = adata_st[:, pivot_df.index].copy()

scvi.external.CellAssign.setup_anndata(follicular_bdata, size_factor_key="size_factor")
follicular_model = CellAssign(follicular_bdata, pivot_df)
follicular_model.train(max_epochs=30)
follicular_model.history["elbo_validation"].plot()

predictions = follicular_model.predict()
predictions.head()


sns.clustermap(predictions, cmap="viridis")
follicular_bdata.obs["cellassign_predictions"] = predictions.idxmax(axis=1).values

sc.pl.umap(
    follicular_bdata,
    color=["celltype", "cellassign_predictions"],
    frameon=False,
    ncols=1,
)
