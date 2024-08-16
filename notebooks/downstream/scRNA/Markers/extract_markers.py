# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from loguru import logger

import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
import torch
from scvi.external import Tangram
from tqdm import tqdm

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import spatialdm.plottings as pl
import squidpy as sq


from loguru import logger
import inspect
import scvi
import scanpy as sc
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI, SCVI
import torch
import celltypist
from celltypist import models
import scanpy as sc
import celltypist
from celltypist import models
import json
from scipy.sparse import csr_matrix
import scmags as sm

sc_adata = sc.read_h5ad(r"/mnt/work/RO_src/data/raw/scRNA/SCLC/SCLC_Annotated_pp.h5ad")


def convert_form_anndata(adata, cell_annotation_col):

    adata.var_names_make_unique()
    exp_data = pd.DataFrame(
        data=adata.X.todense(), columns=adata.var_names, index=adata.obs.index
    ).to_numpy()
    labels = adata.obs[cell_annotation_col].to_numpy()
    gene_names = adata.var_names.to_numpy()

    return exp_data, labels, gene_names


def CellTypist_mapper(sc_adata, config, STNavCorePipeline, gene_col, celltype_col):

    # sc.pp.scale(sc_adata, max_value=10)
    if config["train"]:
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.log1p(sc_adata)

        model = celltypist.train(
            sc_adata,
            labels=config["labels"],
            n_jobs=config["n_jobs"],
            feature_selection=config["feature_selection"],
            epochs=config["epochs"],
            use_SGD=config["use_SGD"],
            mini_batch=config["mini_batch"],
            batch_size=config["batch_size"],
            balance_cell_type=config["balance_cell_type"],
        )
        model.write(f"{config['pre_trained_model_path']}\\celltypist_model.pkl")
    else:
        model = models.Model.load(
            model=f"{config['pre_trained_model_path']}/celltypist_model.pkl",
        )
    num_list = []
    markers = []
    for cell in model.cell_types:
        top_markers = model.extract_top_markers(cell, config["top_genes"])
        num_list += [cell] * len(top_markers)
        markers += list(top_markers)

    # Create DataFrame
    cell_markers_df = pd.DataFrame(
        {
            gene_col: markers,
            celltype_col: num_list,
        }
    )
    cell_markers_df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_CellTypist_markers_{date}.csv",
        index=False,
    )
    cell_type_markers_dict = (
        cell_markers_df.groupby(f"{celltype_col}")
        .apply(lambda x: [marker.upper() for marker in x[gene_col].tolist()])
        .to_dict()
    )
    # Convert dictionary to DataFrame
    df = pd.DataFrame(cell_type_markers_dict)

    # Melt DataFrame to long format
    cell_type_markers_df = df.melt(
        var_name=celltype_col,
        value_name=gene_col,
    )

    cell_type_markers_df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_CellTypist_markers.csv",
        index=False,
    )
    return cell_type_markers_dict, cell_type_markers_df


def SCVI_mapper(sc_adata, config, STNavCorePipeline, gene_col, celltype_col):
    # https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/api_overview.html
    # sc.pp.filter_genes(sc_adata, min_counts=50)
    # sc_adata.layers["counts"] = sc_adata.X.copy()  # preserve counts

    # sc.pp.normalize_total(sc_adata, target_sum=1e4)
    # sc.pp.log1p(sc_adata)
    # sc_adata.raw = sc_adata  # freeze the state in `.raw`

    # sc.pp.highly_variable_genes(
    #     sc_adata,
    #     n_top_genes=10000,
    #     subset=True,
    #     layer=config["layer"],
    #     flavor="seurat_v3",
    # )

    # scVI uses non normalized data so we keep the original data in a separate AnnData object, then the normalization steps are performed (layer = raw_counts)
    sc_adata_cp = sc_adata.copy()
    if config["train"]:
        SCVI.setup_anndata(
            sc_adata_cp,
            layer=config["layer"],
            labels_key=config["labels_key"],
        )

        sc_model = SCVI(sc_adata_cp)
        # sc_model = model(adata_sc)
        sc_model.view_anndata_setup()
        sc_model.train(max_epochs=config["max_epochs"])
        sc_model.save(f"{config['pre_trained_model_path']}", overwrite=True)
    else:
        sc_model = SCVI.load(
            config["pre_trained_model_path"],
            sc_adata_cp,
        )
    latent = sc_model.get_latent_representation()
    sc_adata.obsm["X_scVI"] = latent

    sc_adata.layers["scvi_normalized"] = sc_model.get_normalized_expression(
        library_size=10e4
    )
    sc_adata.var_names_make_unique()

    de_df = sc_model.differential_expression(groupby=config["labels_key"])

    de_df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_scVI_DEG_{date}.csv",
        index=False,
    )

    markers = {}
    cats = sc_adata.obs[config["labels_key"]].cat.categories
    for i, c in enumerate(cats):
        cid = f"{c} vs Rest"
        cell_type_df = de_df.loc[de_df.comparison == cid]

        cell_type_df = cell_type_df[cell_type_df.lfc_mean > 0]

        cell_type_df = cell_type_df[cell_type_df["bayes_factor"] > 3]
        cell_type_df = cell_type_df[cell_type_df["non_zeros_proportion1"] > 0.1]

        markers[c] = cell_type_df.index.tolist()[: config["top_genes"]]

    # Flatten the dictionary
    data = [(val.upper(), key) for key, values in markers.items() for val in values]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=[gene_col, celltype_col])

    sc.tl.dendrogram(sc_adata, groupby=config["labels_key"], use_rep="X_scVI")
    with plt.rc_context():  # Use this to set figure params like size and dpi

        plotting_func = sc.pl.heatmap(
            sc_adata,
            markers,
            groupby=config["labels_key"],
            layer="scvi_normalized",
            standard_scale="var",
            dendrogram=True,
            figsize=(45, 45),
            show=False,
            show_gene_labels=True,
        )
        plt.savefig(
            f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_heatmap_top_markers_SCVI_{date}.png",
            bbox_inches="tight",
        )
        plt.close()

    cell_type_markers_dict = (
        df.groupby(f"{celltype_col}")
        .apply(lambda x: [marker.upper() for marker in x[gene_col].tolist()])
        .to_dict()
    )
    # Convert dictionary to DataFrame
    df = pd.DataFrame(cell_type_markers_dict)

    # Melt DataFrame to long format
    cell_type_markers_df = df.melt(
        var_name=celltype_col,
        value_name=gene_col,
    )

    cell_type_markers_df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_scVI_markers.csv",
        index=False,
    )
    return cell_type_markers_dict, cell_type_markers_df


def scMAGS_mapper(
    sc_adata,
    saving_path,
    annotated_col="cell_type_fine",
    nof_markers=6,
    gene_col="Markers",
    celltype_col="CellType",
):

    exp_data, labels, gene_names = convert_form_anndata(
        sc_adata, cell_annotation_col=annotated_col
    )
    mags = sm.ScMags(data=exp_data, labels=labels, gene_ann=gene_names)
    mags.filter_genes()
    mags.sel_clust_marker(nof_markers=nof_markers)
    df = mags.get_markers()
    cell_type_markers_dict = {
        index.replace("C_", ""): [val.upper() for val in row.tolist()]
        for index, row in df.iterrows()
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(cell_type_markers_dict)

    # Melt DataFrame to long format
    cell_type_markers_df = df.melt(
        var_name=celltype_col,
        value_name=gene_col,
    )

    cell_type_markers_df.to_csv(
        path_or_buf=f"/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/Markers/scMAGS_markers.csv",
        index=False,
    )

    return cell_type_markers_dict, cell_type_markers_df


markers_dict, markers_df = scMAGS_mapper(
    sc_adata=sc_adata,
    saving_path="/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/Markers",
)
