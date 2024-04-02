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
from loguru import logger
import re
import gseapy as gp
from gseapy.plot import gseaplot

# Training a model to predict proportions on spatial data using scRNA seq as reference
import scvi
import inspect

# import cell2location as c2l
from utils import unnormalize, return_filtered_params

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scipy.sparse import csr_matrix

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


class scRNA(object):
    def __init__(self, config) -> None:
        self.config = config

    def read_rna(self):
        config = self.config["rna"]

        # Load H5AD scRNA reference dataset
        sc_adata_raw = sc.read_h5ad(config["path"])
        sc_adata = sc_adata_raw.copy()
        logger.info(
            f"Loaded scRNA dataset with {sc_adata.n_obs} cells and {sc_adata.n_vars} genes."
        )

        return sc_adata_raw, sc_adata

    def quality_control(self, sc_adata):
        config = self.config["rna"]["quality_control"]

        # mitochondrial genes
        sc_adata.var["mt"] = sc_adata.var_names.str.startswith("mt-")
        # ribosomal genes
        sc_adata.var["ribo"] = sc_adata.var_names.str.startswith(("rps", "rpl"))
        # hemoglobin genes.
        sc_adata.var["hb"] = sc_adata.var_names.str.contains(("^hb[^(p)]"))

        if config["calculate_qc_metrics"]["usage"]:
            sc.pp.calculate_qc_metrics(
                **return_filtered_params(
                    config=config["calculate_qc_metrics"], sc_adata=sc_adata
                )
            )

            # Filtering by mitochondria pct
            keep = (
                sc_adata.obs["pct_counts_mt"]
                < config["calculate_qc_metrics"]["pct_counts_mt"]
            ) & (
                sc_adata.obs["n_genes_by_counts"]
                > config["calculate_qc_metrics"]["n_genes_by_counts"]
            )
            # Keep just those that passed the filtering
            sc_adata = sc_adata[keep, :]

            # Remove genes that still passed the previous condition
            genes_to_remove_pattern = re.compile(
                "|".join(
                    map(re.escape, config["calculate_qc_metrics"]["params"]["qc_vars"])
                )
            )

            genes_to_remove = sc_adata.var_names.str.contains(genes_to_remove_pattern)
            keep = np.invert(genes_to_remove)
            sc_adata = sc_adata[:, keep]
            print(sc_adata.n_obs, sc_adata.n_vars)

        return sc_adata

    def preprocessing(self, sc_adata) -> an.AnnData:
        config = self.config["rna"]["preprocessing"]

        try:
            sc_adata.var.set_index("features", inplace=True)
            sc_adata.var.drop(columns=["_index"], inplace=True)

        except Exception as e:
            logger.warning(
                f"Failed to set new index. This might've happened because the index of var is already the genes/feature names, so no changes need to be made."
            )

        sc_adata.var_names_make_unique()
        sc_adata.var.index = sc_adata.var.index.str.lower()
        sc_adata.raw.var.index = sc_adata.raw.var.index.str.lower()

        # Save original X data
        sc_adata.layers["original_X"] = sc_adata.X.copy()

        # Filter genes by counts
        if config["filter_genes"]["usage"]:
            sc.pp.filter_genes(
                **return_filtered_params(
                    config=config["filter_genes"], sc_adata=sc_adata
                )
            )

        # Dimensionality reduction
        # Plotting prep and extra info to add to the anndata
        if config["plotting_prep"]["neighbors"]["usage"]:
            sc.pp.neighbors(
                **return_filtered_params(
                    config=config["plotting_prep"]["neighbors"], sc_adata=sc_adata
                )
            )
        if config["plotting_prep"]["pca"]["usage"]:
            sc.tl.pca(
                **return_filtered_params(
                    config=config["plotting_prep"]["pca"], sc_adata=sc_adata
                )
            )
        if config["plotting_prep"]["umap"]["usage"]:
            sc.tl.umap(
                **return_filtered_params(
                    config=config["plotting_prep"]["umap"], sc_adata=sc_adata
                )
            )
        if config["plotting_prep"]["tsne"]["usage"]:
            sc.tl.tsne(
                **return_filtered_params(
                    config=config["plotting_prep"]["tsne"], sc_adata=sc_adata
                )
            )

        # Clustering
        if config["plotting_prep"]["leiden"]["usage"]:
            sc.tl.leiden(
                **return_filtered_params(
                    config=config["plotting_prep"]["leiden"], sc_adata=sc_adata
                )
            )

        if config["plotting_prep"]["louvain"]["usage"]:
            sc.tl.louvain(
                **return_filtered_params(
                    config=config["plotting_prep"]["louvain"], sc_adata=sc_adata
                )
            )

        if config["unnormalize"]["usage"]:
            sc_adata = unnormalize(
                sc_adata, count_col=config["unnormalize"]["col_name"]
            )
        else:
            # For DGE analysis we would like to run with all genes, but on normalized values, so we will have to revert back to the raw matrix and renormalize.
            sc_adata = sc_adata.raw.to_adata()

            # Normalized total to CPM (?)
            sc.pp.normalize_total(sc_adata, inplace=True)

            # Log data
            sc.pp.log1p(sc_adata)

        # Add count layer
        sc_adata.layers["counts"] = sc_adata.X.copy()

        # store normalized counts in the raw slot,
        # we will subset adata.X for variable genes, but want to keep all genes matrix as well.
        sc_adata.raw = sc_adata

        # Select highly variable genes
        if config["highly_variable_genes"]["usage"]:
            sc.pp.highly_variable_genes(
                **return_filtered_params(
                    config=config["highly_variable_genes"], sc_adata=sc_adata
                )
            )
            # subset for variable genes in the dataset
            sc_adata = sc_adata[:, sc_adata.var["highly_variable"]]

        # Rank genes groups
        if config["rank_genes_groups"]["usage"]:
            sc.tl.rank_genes_groups(
                **return_filtered_params(
                    config=config["rank_genes_groups"], sc_adata=sc_adata
                )
            )
            # temp_df = pd.DataFrame(sc_adata.uns["rank_genes_groups"]["names"])
            # temp_df = temp_df.applymap(lambda s: s.lower() if type(s) == str else s)
            # sc_adata.uns["rank_genes_groups"]["names"] = np.rec.fromrecords(
            #     temp_df, names=temp_df.columns.tolist()
            # )

        # Filter rank genes groups
        if config["filter_rank_genes_groups"]["usage"]:
            sc.tl.filter_rank_genes_groups(
                **return_filtered_params(
                    config=config["filter_rank_genes_groups"], sc_adata=sc_adata
                )
            )
        if config["rank_genes_groups_df"]["usage"]:
            sc_genes = sc.get.rank_genes_groups_df(
                **return_filtered_params(
                    config=config["rank_genes_groups_df"], sc_adata=sc_adata
                )
            )
            sc_genes.to_excel("sc_genes.xlsx")

        return sc_adata

    def train_or_load_model(self, sc_adata):
        config = self.config["rna"]

        sc_adata_copy = sc_adata.copy()

        # Add original counts for stereoscope (unnormalized counts)
        sc_adata_copy.X = sc_adata_copy.layers["counts"].copy()

        RNAStereoscope.setup_anndata(
            sc_adata_copy,
            layer="counts",
            labels_key=config["preprocessing"]["rank_genes_groups"]["params"][
                "groupby"
            ],
        )

        train = config["model"]["train"]
        if train:
            sc_model = RNAStereoscope(sc_adata_copy)
            training_params = config["model"]["params"]
            valid_arguments = inspect.signature(sc_model.train).parameters.keys()
            filtered_params = {
                k: v for k, v in training_params.items() if k in valid_arguments
            }
            sc_model.train(**filtered_params)
            sc_model.history["elbo_train"][10:].plot()
            sc_model.save("scmodel", overwrite=True)
        else:
            sc_model = RNAStereoscope.load(
                config["model"]["pre_trained_model_path"],
                sc_adata_copy,
            )

        return sc_model
