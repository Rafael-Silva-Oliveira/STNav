# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import SpatialDE
import NaiveDE
import squidpy as sq
import json
import torch
from loguru import logger
from GraphST.utils import project_cell_to_spot
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)
import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
import inspect
from GraphST import GraphST
import torch
import mudata
from scvi.external import Tangram
import mudata
from scvi.external import Tangram

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
from STNav.utils.decorators import pass_STNavCore_params
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


def CellTypist_mapper(sc_adata, config, STNavCorePipeline):

    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)
    sc.pp.scale(sc_adata, max_value=10)

    if config["train"]:
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
    else:
        model = models.Model.load(
            model=config["pre_trained_model_path"],
        )
    num_list = []
    markers = []
    for cell in model.cell_types:
        top_markers = model.extract_top_markers(cell, 4)
        num_list += [cell] * len(top_markers)
        markers += list(top_markers)

    # Create DataFrame
    cell_markers_df = pd.DataFrame(
        {
            config["markers_column_name"]: markers,
            config["cell_type_column_name"]: num_list,
        }
    )
    cell_type_markers = (
        cell_markers_df.groupby(f"{config['cell_type_column_name']}")
        .apply(lambda x: x.index.tolist())
        .to_dict()
    )

    cell_markers_df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_CellTypist_Markers_{date}.csv",
        index=False,
    )

    return cell_type_markers


def SCVI_mapper(sc_adata, config, STNavCorePipeline):

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

    if config["train"]:
        SCVI.setup_anndata(
            sc_adata,
            layer=config["layer"],
            labels_key=config["labels_key"],
        )

        sc_model = SCVI(sc_adata)
        # sc_model = model(adata_sc)
        sc_model.view_anndata_setup()
        sc_model.train(max_epochs=config["max_epochs"])
        sc_model.save(f"{config['pre_trained_model_path']}", overwrite=True)
    else:
        sc_model = SCVI.load(
            config["pre_trained_model_path"],
            sc_adata,
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
    data = [
        (val.upper(), key + " marker")
        for key, values in markers.items()
        for val in values
    ]

    # Create a DataFrame
    df = pd.DataFrame(
        data, columns=[config["markers_column_name"], config["cell_type_column_name"]]
    )

    df.to_csv(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_scVI_markers_{date}.csv",
        index=False,
    )

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

        cell_type_markers = (
            df.groupby(f"{config['cell_type_column_name']}")
            .apply(lambda x: x[config["markers_column_name"]].tolist())
            .to_dict()
        )

    return cell_type_markers


class SpatialMarkersMapping:

    def __init__(self, STNavCorePipeline):
        self.STNavCorePipeline = STNavCorePipeline

    def _get_cell_type_markers(self, mapping_config, sc_adata, st_adata):
        config = mapping_config["get_cell_type_markers"]

        load_types = [
            marker_loading_type
            for marker_loading_type, marker_config in config.items()
            if marker_config["usage"]
        ]
        if len(load_types) >= 2:
            raise ValueError(
                logger.error(
                    f"Please, choose only 1 loading format for the gene markers. Current active loading types {load_types = }"
                )
            )
        elif len(load_types) == 0:
            logger.warning(
                f"Please, set at least one loading type for the markers (from csv or from scratch using scRNA models). "
            )
            return None
        load_type = load_types[0]

        if load_type == "from_csv":
            # Get all top genes for each function (into dictionary form)
            cell_markers_df = pd.read_csv(f"{config[load_type]['path']}", index_col=0)

            cell_type_markers = (
                cell_markers_df.groupby(f"{config['cell_type_column_name']}")
                .apply(lambda x: x[config["markers_column_name"]].tolist())
                .to_dict()
            )

        elif load_type == "from_models":
            if config[load_type]["SCVI"]["usage"]:
                cell_type_markers = SCVI_mapper(
                    sc_adata, config[load_type]["SCVI"], self.STNavCorePipeline
                )
            if config[load_type]["CellTypist"]["usage"]:
                cell_type_markers = CellTypist_mapper(
                    sc_adata, config[load_type]["CellTypist"], self.STNavCorePipeline
                )
        st_adata.var.index = st_adata.var.index.str.upper()
        st_adata.var_names = st_adata.var_names.str.upper()

        # Return only the markers that are present in the spatial data
        marker_genes_in_data = dict()
        for ct, markers in cell_type_markers.items():
            markers_found = list()
            for marker in markers:
                if marker in st_adata.var.index:
                    markers_found.append(marker)
            marker_genes_in_data[ct] = markers_found

        return marker_genes_in_data

    def _map_markers_to_spatial_cell_type(self, mapping_config, st_adata, cell_markers):
        config = mapping_config["map_markers_to_spatial_cell_type"]
        df_list = []
        # Create a Bin x Top cell marker gene log normalized
        for cell_type, gene_names in cell_markers.items():
            df = pd.DataFrame(
                index=st_adata.obs.index,
                columns=st_adata.var_names.str.upper(),
                data=st_adata.X.toarray(),
            )
            common_genes = list(set(df.columns) & set(gene_names))
            df_gene_subset = df[common_genes]

            # TODO: add the combination method
            df_gene_subset[cell_type + "_Mean_LogNorm"] = df_gene_subset.mean(axis=1)
            df_list.append(df_gene_subset)

        # Add this information to the adata_sp.obs and plot results
        for cell_type, gene_names in cell_markers.items():
            for df in df_list:
                if cell_type + "_Mean_LogNorm" in df.columns:
                    st_adata.obs[cell_type + "_Mean_LogNorm"] = df[
                        cell_type + "_Mean_LogNorm"
                    ]

        return st_adata

    def _map_to_clusters(self, mapping_config, st_adata, cell_markers):
        config = mapping_config["map_to_clusters"]

        for cell_type in cell_markers.keys():
            # Calculate the 80th percentile
            percentile = st_adata.obs[cell_type + "_Mean_LogNorm"].quantile(
                config["percentile_threshold"]
            )

            # Create a new binary column
            st_adata.obs[cell_type + "_flag"] = (
                st_adata.obs[cell_type + "_Mean_LogNorm"] > percentile
            ).astype(int)

        for cell_type in cell_markers.keys():
            # Create a new column that combines the cell type and the cluster
            st_adata.obs[cell_type + "_cell_type_cluster"] = st_adata.obs.apply(
                lambda row: (
                    cell_type + "_" + str(row[config["cluster_column_name"]])
                    if row[cell_type + "_flag"] == 1
                    else ""
                ),
                axis=1,
            )

        return st_adata

    def run_mapping(self, mapping_config):
        logger.info(
            f"Running mapping based on ranked genes with the group {self.STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        )

        st_adata_to_use = mapping_config["spatial_adata_to_use"]
        st_path = self.STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
        st_adata = sc.read_h5ad(st_path)

        sc_adata_to_use = mapping_config["scRNA_adata_to_use"]
        sc_path = self.STNavCorePipeline.adata_dict["scRNA"][sc_adata_to_use]
        sc_adata = sc.read_h5ad(sc_path)

        # Extract the top cell type markers from the annotated scRNA by training/loading SCVI and/Or CellTypist model OR load the markers from a CSV file
        cell_markers_dict = self._get_cell_type_markers(
            mapping_config=mapping_config, sc_adata=sc_adata, st_adata=st_adata
        )

        # Map the cell type markers to the spatial data (intersect the top markers that were found with the genes present in the spatial data). Apply combination method to get the spatial cell types. TODO: see a way so that this combination method is based on spatial information (e.g. spatially variable genes, etc).
        spatial_cell_type_adata = self._map_markers_to_spatial_cell_type(
            mapping_config, st_adata, cell_markers_dict
        )

        # Map the spatial cell types to the clusters based on top percentile of the combination score
        spatial_cell_type_and_clusters_adata = self._map_to_clusters(
            mapping_config, spatial_cell_type_adata, cell_markers_dict
        )

        save_processed_adata(
            STNavCorePipeline=self.STNavCorePipeline,
            name="mapped_adata",
            adata=spatial_cell_type_and_clusters_adata,
        )
