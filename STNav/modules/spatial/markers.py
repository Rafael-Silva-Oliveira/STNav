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
from scipy.sparse import csr_matrix
import scmags as sm


def convert_form_anndata(adata, cell_annotation_col):

    adata.var_names_make_unique()
    exp_data = pd.DataFrame(
        data=adata.X.todense(), columns=adata.var_names, index=adata.obs.index
    ).to_numpy()
    labels = adata.obs[cell_annotation_col].to_numpy()
    gene_names = adata.var_names.to_numpy()

    return exp_data, labels, gene_names


def CellTypist_mapper(sc_adata, config, STNavCorePipeline, gene_col, celltype_col):

    # sc.pp.normalize_total(sc_adata, target_sum=1e4)
    # sc.pp.log1p(sc_adata)
    # sc.pp.scale(sc_adata, max_value=10)
    # TODO: check if the adata used is already normalized
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


def scMAGS_mapper(sc_adata, config, STNavCorePipeline, gene_col, celltype_col):

    exp_data, labels, gene_names = convert_form_anndata(
        sc_adata, "ann_level_3_transferred_label"
    )
    mags = sm.ScMags(data=exp_data, labels=labels, gene_ann=gene_names)
    mags.filter_genes(nof_sel=config["nof_sel"])
    mags.sel_clust_marker(nof_markers=config["nof_markers"])
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
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_scMAGS_markers.csv",
        index=False,
    )
    return cell_type_markers_dict, cell_type_markers_df


class SpatialMarkersMapping:

    def __init__(self, STNavCorePipeline):
        self.STNavCorePipeline = STNavCorePipeline

    def _get_intersect(self, sc_adata, st_adata):

        st_adata.var.index = st_adata.var.index.str.upper()
        st_adata.var_names = st_adata.var_names.str.upper()

        sc_adata.var.index = sc_adata.var.index.str.upper()
        sc_adata.var_names = sc_adata.var_names.str.upper()

        intersect = np.intersect1d(
            sc_adata.var_names,
            st_adata.var_names,
        )
        sc_adata.X = sc_adata.layers["raw_counts"]
        st_adata.X = st_adata.layers["raw_counts"]

        st_adata_intersect = st_adata[:, intersect]

        sc_adata_intersect = sc_adata[:, intersect]

        logger.info(
            f"N_obs x N_var for ST and scRNA after intersection: \n{st_adata.n_obs} x {st_adata.n_vars} \n {sc_adata.n_obs} x {sc_adata.n_vars}"
        )

        return st_adata_intersect, sc_adata_intersect, intersect

    def _get_cell_type_markers(self, mapping_config, sc_adata):
        config = mapping_config["get_cell_type_markers"]
        logger.info(
            "Extracting top cell type markers from scRNA reference annotated data."
        )

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

            gene_markers_col_name = config["from_models"]["markers_column_name"]
            celltype_col_name = config["from_models"]["cell_type_column_name"]
            # Get all top genes for each function (into dictionary form)
            cell_markers_df = pd.read_csv(f"{config[load_type]['path']}", index_col=0)

            cell_type_markers_dict = (
                cell_markers_df.groupby(f"{celltype_col_name}")
                .apply(lambda x: x[gene_markers_col_name].tolist())
                .to_dict()
            )

        elif load_type == "from_models":
            cell_type_markers_SCVI_dict = None
            cell_type_markers_CellTypist_dict = None
            cell_type_markers_scMAGS_dict = None

            gene_markers_col_name = config["from_models"]["markers_column_name"]
            celltype_col_name = config["from_models"]["cell_type_column_name"]

            if config[load_type]["SCVI"]["usage"]:
                cell_type_markers_SCVI_dict, cell_type_markers_SCVI_df = SCVI_mapper(
                    sc_adata,
                    config[load_type]["SCVI"],
                    self.STNavCorePipeline,
                    gene_markers_col_name,
                    celltype_col_name,
                )

            if config[load_type]["CellTypist"]["usage"]:
                cell_type_markers_CellTypist_dict, cell_type_markers_CellTypist_df = (
                    CellTypist_mapper(
                        sc_adata,
                        config[load_type]["CellTypist"],
                        self.STNavCorePipeline,
                        gene_markers_col_name,
                        celltype_col_name,
                    )
                )

            if config[load_type]["scMAGS"]["usage"]:
                cell_type_markers_scMAGS_dict, cell_type_markers_scMAGS_df = (
                    scMAGS_mapper(
                        sc_adata,
                        config[load_type]["scMAGS"],
                        self.STNavCorePipeline,
                        gene_markers_col_name,
                        celltype_col_name,
                    )
                )

            if (
                cell_type_markers_SCVI_dict is not None
                and cell_type_markers_CellTypist_dict is not None
                and cell_type_markers_scMAGS_dict is not None
            ):
                # Create a new dictionary to hold the merged results
                cell_type_markers_dict = {}

                for key, value in cell_type_markers_SCVI_dict.items():
                    # Copy the values from the first dictionary to the new dictionary
                    cell_type_markers_dict[key] = value[:]

                for key, value in cell_type_markers_CellTypist_dict.items():
                    if key in cell_type_markers_dict:
                        # If key exists in the new dictionary, append the values
                        cell_type_markers_dict[key] += value
                    else:
                        # If key doesn't exist in the new dictionary, create it
                        cell_type_markers_dict[key] = value

                for key, value in cell_type_markers_scMAGS_dict.items():
                    if key in cell_type_markers_dict:
                        # If key exists in the new dictionary, append the values
                        cell_type_markers_dict[key] += value
                    else:
                        # If key doesn't exist in the new dictionary, create it
                        cell_type_markers_dict[key] = value

                # Remove duplicates
                for key, value in cell_type_markers_dict.items():
                    cell_type_markers_dict[key] = list(set(value))

                # Convert dictionary to DataFrame
                df = pd.DataFrame.from_dict(
                    cell_type_markers_dict, orient="index"
                ).transpose()

                # Melt DataFrame to long format
                cell_type_markers_df = df.melt(
                    var_name=celltype_col_name,
                    value_name=gene_markers_col_name,
                )

                cell_type_markers_df.dropna(inplace=True)
                cell_type_markers_df.drop_duplicates(inplace=True)
                cell_type_markers_df.to_csv(
                    f"{self.STNavCorePipeline.saving_path}\\{self.STNavCorePipeline.data_type}\\Files\\{self.STNavCorePipeline.data_type}_merged_markers_{date}.csv",
                    index=False,
                )

            elif cell_type_markers_SCVI_dict is not None:
                cell_type_markers_dict = cell_type_markers_SCVI_dict

            elif cell_type_markers_CellTypist_dict is not None:
                cell_type_markers_dict = cell_type_markers_CellTypist_dict

            elif cell_type_markers_scMAGS_dict is not None:
                cell_type_markers_dict = cell_type_markers_scMAGS_dict

        return cell_type_markers_dict

    def _map_markers_to_spatial_cell_type(
        self, mapping_config, st_adata, cell_markers, intersection
    ):
        logger.info("Mapping markers to spatial cell types.")
        config = mapping_config["map_markers_to_spatial_cell_type"]
        cell_type_col_name = "cell_type"
        decomposition_type = "_Mean_LogNorm_Conn_Adj"

        # Add spatial connectivities
        sq.gr.spatial_neighbors(st_adata)

        # Add the new columns ()
        st_adata.obsp["spatial_connectivities"] = csr_matrix(
            st_adata.obsp["spatial_connectivities"]
        )

        st_adata = st_adata[:, intersection]
        st_adata.X = st_adata.layers["lognorm"]
        st_adata.X = csr_matrix(st_adata.X)

        df = pd.DataFrame.sparse.from_spmatrix(
            st_adata.X,
            index=st_adata.obs.index,
            columns=st_adata.var_names.str.upper(),
        )
        # Create a Bin x Top cell marker gene log normalized
        for cell_type, gene_names in cell_markers.items():
            # Get the common genes
            common_genes = list(set(df.columns) & set(gene_names))

            # Get the subset of df that includes the common genes
            df_gene_subset = df[common_genes]

            # Calculate the mean of df_gene_subset along axis=1 and add it to adata_sp.obs
            st_adata.obs[cell_type + decomposition_type] = df_gene_subset.mean(
                axis=1
            ).astype(float)

            col = cell_type + decomposition_type
            print(col)
            save_path = (
                self.STNavCorePipeline.saving_path
                + "\\Plots\\"
                + cell_type
                + decomposition_type
                + ".png"
            )
            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    st_adata,
                    cmap="magma",
                    color=[col],
                    img_key="hires",
                    size=1.75,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

        logger.info(
            "Saving the spatial plot using the max of the mean lognorms cell type markers."
        )
        # cell_lognorms = [col for col in st_adata.obs.columns if "Mean_LogNorm" in col]
        # st_adata.obs[cell_type_col_name] = st_adata.obs[cell_lognorms].idxmax(axis=1)

        # # Calculate centralities
        # sq.gr.spatial_neighbors(st_adata)
        # sq.gr.centrality_scores(st_adata, cell_type_col_name)
        # sq.pl.centrality_scores(st_adata, cell_type_col_name)
        # from scipy.sparse import csr_matrix

        # # Add the new columns ()
        # st_adata.obsp["spatial_connectivities"] = csr_matrix(
        #     st_adata.obsp["spatial_connectivities"]
        # )
        # st_adata.X = csr_matrix(st_adata.X)
        # # Perform the dot product operation on the sparse matrices
        # st_adata.X = st_adata.obsp["spatial_connectivities"].dot(st_adata.X)

        # Add the final cell type predictions
        cell_lognorms = [col for col in st_adata.obs.columns if "Conn" in col]
        st_adata.obs[cell_type_col_name] = st_adata.obs[cell_lognorms].idxmax(axis=1)
        save_path = (
            self.STNavCorePipeline.saving_path
            + "\\Plots\\"
            + "all_cell_types"
            + "_mean_max_conn_adj"
            + ".png"
        )
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plot_func = sc.pl.spatial(
                st_adata,
                cmap="magma",
                color=cell_type_col_name,
                img_key="hires",
                size=1.75,
                alpha_img=0.5,
                show=False,
            )
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        return st_adata

    def _map_to_clusters(self, mapping_config, st_adata, cell_markers):
        logger.info("Mapping cell types to clusters.")
        config = mapping_config["map_to_clusters"]
        decomposition_type = "_Mean_LogNorm_Conn_Adj"

        for cell_type in tqdm(cell_markers.keys()):
            # Calculate the 80th percentile
            percentile = st_adata.obs[cell_type + decomposition_type].quantile(
                config["percentile_threshold"]
            )

            # Create a new binary column
            st_adata.obs[cell_type + "_flag"] = (
                st_adata.obs[cell_type + decomposition_type] > percentile
            ).astype(int)

            # Create a new column that combines the cell type and the cluster
            st_adata.obs[cell_type + "_cell_type_cluster"] = st_adata.obs.apply(
                lambda row: (
                    cell_type + "_" + str(row[config["cluster_column_name"]])
                    if row[cell_type + "_flag"] == 1
                    else ""
                ),
                axis=1,
            )
            # Set the color of the legend text to black
            plt.rcParams["text.color"] = "black"

            # Set the background color to white
            plt.rcParams["figure.facecolor"] = "white"
            plt.rcParams["axes.facecolor"] = "white"
            col = cell_type + "_cell_type_cluster"
            print(col)

            save_path = (
                self.STNavCorePipeline.saving_path
                + "\\Plots\\"
                + cell_type
                + "_cluster"
                + ".png"
            )
            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    st_adata,
                    cmap="magma",
                    color=[col],
                    img_key="hires",
                    size=1,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

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

        raw_st_adata_subset, raw_sc_adata_subset, intersection = self._get_intersect(
            sc_adata, st_adata
        )

        # Extract the top cell type markers from the annotated scRNA by training/loading SCVI and/Or CellTypist model OR load the markers from a CSV file
        cell_markers_dict = self._get_cell_type_markers(
            mapping_config=mapping_config,
            sc_adata=raw_sc_adata_subset,
        )

        # Map the cell type markers to the spatial data (intersect the top markers that were found with the genes present in the spatial data). Apply combination method to get the spatial cell types.
        spatial_cell_type_adata = self._map_markers_to_spatial_cell_type(
            mapping_config, st_adata, cell_markers_dict, intersection
        )

        # Map the spatial cell types to the clusters based on top percentile of the combination score
        spatial_cell_type_and_clusters_adata = self._map_to_clusters(
            mapping_config, spatial_cell_type_adata, cell_markers_dict
        )
        save_processed_adata(
            STNavCorePipeline=self.STNavCorePipeline,
            name="deconvoluted_adata",
            adata=spatial_cell_type_and_clusters_adata,
        )
        # TODO: add this approach here for the cell type https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_vizgen_mouse_liver.html#assign-cell-types
        del raw_sc_adata_subset
        del raw_st_adata_subset
        del st_adata
        del spatial_cell_type_and_clusters_adata
        del spatial_cell_type_adata
        del sc_adata

        return cell_markers_dict
