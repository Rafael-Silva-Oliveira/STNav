# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
from utils import unnormalize, return_filtered_params
import scvi
import inspect
import cell2location as c2l
import re
import gseapy as gp
from gseapy.plot import gseaplot

# Training a model to predict proportions on spatial data using scRNA seq as reference
import scvi
from loguru import logger

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scipy.sparse import csr_matrix

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


class SpatialTranscriptomics(object):
    def __init__(self, config) -> None:
        self.config = config

    def read_visium(self):
        config = self.config["spatial"]

        # Load Visium dataset
        st_adata_raw = sc.read_visium(
            path=config["path"],
            count_file=config["count_file"],
            load_images=config["load_images"],
            source_image_path=config["source_image_path"],
        )
        st_adata = st_adata_raw.copy()
        logger.info(
            f"Loaded 10X Visium dataset with {st_adata.n_obs} cells and {st_adata.n_vars} genes."
        )

        return st_adata_raw, st_adata

    def quality_control(self, st_adata):
        config = self.config["spatial"]["quality_control"]

        # mitochondrial genes
        st_adata.var["mt"] = st_adata.var_names.str.startswith("mt-")
        # ribosomal genes
        st_adata.var["ribo"] = st_adata.var_names.str.startswith(("rps", "rpl"))
        # hemoglobin genes.
        st_adata.var["hb"] = st_adata.var_names.str.contains(("^hb[^(p)]"))

        if config["calculate_qc_metrics"]["usage"]:
            filtered_params = return_filtered_params(
                config=config["calculate_qc_metrics"], st_adata=st_adata
            )
            sc.pp.calculate_qc_metrics(**filtered_params)

            # Filtering by mitochondria pct
            keep = (
                st_adata.obs["pct_counts_mt"]
                < config["calculate_qc_metrics"]["pct_counts_mt"]
            ) & (
                st_adata.obs["n_genes_by_counts"]
                > config["calculate_qc_metrics"]["n_genes_by_counts"]
            )
            # Keep just those that passed the filtering
            st_adata = st_adata[keep, :]

            # Remove genes that still passed the previous condition
            genes_to_remove_pattern = re.compile(
                "|".join(
                    map(re.escape, config["calculate_qc_metrics"]["params"]["qc_vars"])
                )
            )

            genes_to_remove = st_adata.var_names.str.contains(genes_to_remove_pattern)
            keep = np.invert(genes_to_remove)
            st_adata = st_adata[:, keep]
            print(st_adata.n_obs, st_adata.n_vars)

        return st_adata

    def preprocessing(self, st_adata) -> an.AnnData:
        config = self.config["spatial"]["preprocessing"]

        st_adata.var_names_make_unique()
        try:
            st_adata.var.set_index("_index", inplace=True)
        except Exception as e:
            logger.warning(
                f"Failed to set new index _index. This might've happened because the index of var is already the genes/feature names, so no changes need to be made."
            )
        # st_adata.var.index = st_adata.var.index.str.lower()
        st_adata.var_names = st_adata.var_names.str.lower()
        st_adata.raw.var_names = st_adata.raw.var_names.str.lower()

        # Filter genes by counts
        if config["filter_genes"]["usage"]:
            sc.pp.filter_genes(
                **return_filtered_params(
                    config=config["filter_genes"], st_adata=st_adata
                )
            )
        # Add count layer
        st_adata.layers["counts"] = st_adata.X.copy()

        # Dimensionality reduction
        # Plotting prep and extra info to add to the anndata
        if config["plotting_prep"]["neighbors"]["usage"]:
            sc.pp.neighbors(
                **return_filtered_params(
                    config=config["plotting_prep"]["neighbors"], st_adata=st_adata
                )
            )
        if config["plotting_prep"]["pca"]["usage"]:
            sc.tl.pca(
                **return_filtered_params(
                    config=config["plotting_prep"]["pca"], st_adata=st_adata
                )
            )
        if config["plotting_prep"]["umap"]["usage"]:
            sc.tl.umap(
                **return_filtered_params(
                    config=config["plotting_prep"]["umap"], st_adata=st_adata
                )
            )
        if config["plotting_prep"]["tsne"]["usage"]:
            sc.tl.tsne(
                **return_filtered_params(
                    config=config["plotting_prep"]["tsne"], st_adata=st_adata
                )
            )

        # Clustering
        if config["plotting_prep"]["leiden"]["usage"]:
            sc.tl.leiden(
                **return_filtered_params(
                    config=config["plotting_prep"]["leiden"], st_adata=st_adata
                )
            )

        if config["plotting_prep"]["louvain"]["usage"]:
            sc.tl.louvain(
                **return_filtered_params(
                    config=config["plotting_prep"]["louvain"], st_adata=st_adata
                )
            )

        if config["unnormalize"]["usage"]:
            st_adata = unnormalize(
                st_adata, count_col=config["unnormalize"]["col_name"]
            )
        else:
            # For DGE analysis we would like to run with all genes, but on normalized values, so we will have to revert back to the raw matrix and renormalize.
            # st_adata = st_adata.raw.to_adata()

            # Normalized total to CPM (?)
            sc.pp.normalize_total(st_adata, inplace=True)

            # Log data
            sc.pp.log1p(st_adata)

            # Scale
            # sc.pp.scale(st_adata, inplace=True)

        # Save original X data
        st_adata.layers["original_X"] = st_adata.X.copy()

        st_adata.raw = st_adata

        # Select highly variable genes
        if config["highly_variable_genes"]["usage"]:
            sc.pp.highly_variable_genes(
                **return_filtered_params(
                    config=config["highly_variable_genes"], st_adata=st_adata
                )
            )
            # subset for variable genes in the dataset
            st_adata = st_adata[:, st_adata.var["highly_variable"]]

        # Rank genes groups
        if config["rank_genes_groups"]["usage"]:
            sc.tl.rank_genes_groups(
                **return_filtered_params(
                    config=config["rank_genes_groups"], st_adata=st_adata
                )
            )
            tt = 2
        # Filter rank genes groups
        if config["filter_rank_genes_groups"]["usage"]:
            sc.tl.filter_rank_genes_groups(
                **return_filtered_params(
                    config=config["filter_rank_genes_groups"], st_adata=st_adata
                )
            )

        if config["rank_genes_groups_df"]["usage"]:
            st_genes = sc.get.rank_genes_groups_df(
                **return_filtered_params(
                    config=config["rank_genes_groups_df"], st_adata=st_adata
                )
            )
            st_genes.to_excel("st_genes.xlsx")

        return st_adata

    def train_or_load_model(self, st_adata, sc_model):
        config = self.config["spatial"]

        st_adata_copy = st_adata.copy()

        # Add original counts for stereoscope (unnormalized counts)
        st_adata_copy.X = st_adata_copy.layers["counts"].copy()

        SpatialStereoscope.setup_anndata(
            st_adata_copy,
            layer="counts",
        )

        train = config["model"]["train"]
        if train:
            st_model = SpatialStereoscope.from_rna_model(
                st_adata_copy, sc_model, prior_weight="minibatch"
            )
            training_params = config["model"]["params"]
            valid_arguments = inspect.signature(st_model.train).parameters.keys()
            filtered_params = {
                k: v for k, v in training_params.items() if k in valid_arguments
            }
            st_model.train(**filtered_params)
            plt.plot(st_model.history["elbo_train"], label="train")
            plt.title("loss over training epochs")
            plt.legend()
            plt.show()
            st_model.save("stmodel", overwrite=True)
        else:
            st_model = SpatialStereoscope.load(
                r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\stmodel",
                st_adata_copy,
            )
        return st_model
