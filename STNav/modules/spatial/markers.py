# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
    swap_layer,
)
from tqdm import tqdm

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl
import squidpy as sq


from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
import scanpy as sc
import scanpy as sc
import celltypist
from celltypist import models
import json
from scipy.sparse import csr_matrix
import scmags as sm


class SpatialMarkersMapping:

    def __init__(self, STNavCorePipeline):
        self.STNavCorePipeline = STNavCorePipeline

    def _get_cell_type_markers(self, mapping_config):
        config = mapping_config["get_cell_type_markers"]
        logger.info("Loading CSV file with markers from scRNA reference file.")

        # Load markers from CSV file. Check the markers file on the single_cell module on how to extract such markers
        gene_markers_col_name = config["markers_column_name"]
        celltype_col_name = config["cell_type_column_name"]
        # Get all top genes for each function (into dictionary form)
        cell_markers_df = pd.read_csv(
            filepath_or_buffer=f"{config['path']}", index_col=0
        )

        cell_type_markers_dict = (
            cell_markers_df.groupby(by=f"{celltype_col_name}")
            .apply(func=lambda x: x[gene_markers_col_name].tolist())
            .to_dict()
        )

        return cell_type_markers_dict

    def _map_markers_to_spatial_cell_type(self, mapping_config, st_adata, cell_markers):
        logger.info("Mapping markers to spatial cell types.")
        config = mapping_config["map_markers_to_spatial_cell_type"]
        cell_type_col_name = "cell_type_from_markers"

        # Add spatial connectivities
        # sq.gr.spatial_neighbors(st_adata)

        # st_adata.obsp["spatial_connectivities"] = csr_matrix(
        #     arg1=st_adata.obsp["spatial_connectivities"]
        # )

        # Replace the raw counts with the log normalized counts
        st_adata.X = csr_matrix(arg1=st_adata.layers["lognorm_counts"])

        # st_adata.X = st_adata.obsp["spatial_connectivities"].dot(st_adata.X)

        # from decouplr - https://decoupler-py.readthedocs.io/en/latest/notebooks/spatial.html#
        # st_adata.X = st_adata.obsp["spatial_connectivities"].A.dot(st_adata.X.A)

        # st_adata.X = csr_matrix(st_adata.layers["lognorm"])
        df = pd.DataFrame.sparse.from_spmatrix(
            st_adata.X,
            index=st_adata.obs.index,
            columns=st_adata.var_names.str.upper(),
        )
        for cell_type, markers in cell_markers.items():
            # Get the common genes
            common_genes = list(set(df.columns) & set(markers))

            # Get the subset of df that includes the common genes
            df_gene_subset = df[common_genes]

            # Calculate the mean of df_gene_subset along axis=1
            mean_value = df_gene_subset.mean(axis=1).astype(float)
            col = cell_type + " Markers"

            # Add the mean value to adata_sp.obs
            st_adata.obs[col] = mean_value

            print(col)
            save_path = (
                self.STNavCorePipeline.saving_path
                + "/Plots/"
                + cell_type
                + " Markers"
                + ".png"
            )

            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    adata=st_adata,
                    cmap="magma",
                    color=[col],
                    img_key="hires",
                    size=1.1,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

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
        cell_lognorms = [col for col in st_adata.obs.columns if " Markers" in col]
        st_adata.obs[cell_type_col_name] = (
            st_adata.obs[cell_lognorms].idxmax(axis=1).str.replace(" Markers", "")
        )
        logger.info(
            "Saving the spatial plot using the max of the mean lognorms cell type markers."
        )
        save_path = (
            self.STNavCorePipeline.saving_path
            + "/Plots/"
            + "all_cell_types"
            + "_mean_max_conn_adj"
            + ".png"
        )
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plot_func = sc.pl.spatial(
                adata=st_adata,
                cmap="magma",
                color=cell_type_col_name,
                img_key="hires",
                size=1,
                alpha_img=0.5,
                show=False,
            )
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        return st_adata

    def _map_to_clusters(self, mapping_config, st_adata, cell_markers):
        logger.info("Mapping cell types to clusters.")
        config = mapping_config["map_to_clusters"]

        for cell_type, markers in cell_markers.items():
            col = cell_type + " Markers"

            # Calculate the 80th percentile
            percentile = st_adata.obs[col].quantile(config["percentile_threshold"])

            # Create a new binary column
            st_adata.obs[cell_type + "_flag"] = (st_adata.obs[col] > percentile).astype(
                int
            )

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
                + "/Plots/"
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

        st_adata_to_use = mapping_config["adata_to_use"]
        st_path = self.STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
        st_adata: sc.AnnData = sc.read_h5ad(filename=st_path)

        # load the markers from a CSV file
        cell_markers_dict = self._get_cell_type_markers(
            mapping_config=mapping_config,
        )

        # Map the cell type markers to the spatial data (intersect the top markers that were found with the genes present in the spatial data). Apply combination method to get the spatial cell types.
        spatial_cell_type_adata: sc.AnnData = self._map_markers_to_spatial_cell_type(
            mapping_config=mapping_config,
            st_adata=st_adata,
            cell_markers=cell_markers_dict,
        )
        save_processed_adata(
            STNavCorePipeline=self.STNavCorePipeline,
            name=mapping_config["save_as"],
            adata=spatial_cell_type_adata,
        )
        # Map the spatial cell types to the clusters based on top percentile of the combination score
        # spatial_cell_type_and_clusters_adata: sc.AnnData = self._map_to_clusters(
        #     mapping_config=mapping_config,
        #     st_adata=spatial_cell_type_adata,
        #     cell_markers=cell_markers_dict,
        # )
        # save_processed_adata(
        #     STNavCorePipeline=self.STNavCorePipeline,
        #     name="preprocessed_spatial_cell_type_and_clusters",
        #     adata=spatial_cell_type_and_clusters_adata,
        # )

        # TODO: add this approach here for the cell type https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_vizgen_mouse_liver.html#assign-cell-types
        del st_adata
        del spatial_cell_type_and_clusters_adata
        del spatial_cell_type_adata
        return cell_markers_dict
