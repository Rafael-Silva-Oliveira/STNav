from src.modules.decorators import pass_analysis_pipeline

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
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import torch
import NaiveDE
import SpatialDE
from scipy import sparse
import scarches as sca
import squidpy as sq
from gseapy import GSEA
from GraphST.utils import clustering
from GraphST import GraphST
from typing import Union

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# Training a model to predict proportions on Spatial data using scRNA seq as reference
import scvi
import inspect
import cell2location as c2l
from src.utils.utils import (
    unnormalize,
    return_filtered_params,
    log_adataX,
    ensembleID_to_GeneSym_mapping,
    run_enrichr,
    run_prerank,
    run_gsea,
    SpatialDM_wrapper,
)

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
from scipy.sparse import csr_matrix
from src.utils.utils import fix_write_h5ad, GARD

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
import json
import celltypist
from celltypist import models
from GraphST.utils import project_cell_to_spot
import spatialdm as sdm
import anndata as ad
import SpatialDE


@pass_analysis_pipeline
def SpatiallyVariableGenes(STNavCorePipeline):
    """
    g - The name of the gene
    pval - The P-value for spatial differential expression
    qval - Significance after correcting for multiple testing
    l - A parameter indicating the distance scale a gene changes expression over
    """

    config = STNavCorePipeline.config[STNavCorePipeline.data_type][
        "SpatiallyVariableGenes"
    ]
    logger.info("Obtaining spatially variable genes.")
    for method_name, methods in config.items():
        for config_name, config_params in methods.items():
            if config_params["usage"]:
                adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    config_params["adata_to_use"]
                ].copy()
                current_config_params = config_params["params"]

                logger.info(
                    f"Running {method_name} method with {config_name} configuration \n Configuration parameters: {current_config_params} \n using the following adata {config_params['adata_to_use']}"
                )
                data_type = config_params["data_type"]

                if method_name == "SpatialDE":
                    # https://scanpy-tutorials.readthedocs.io/en/multiomics/analysis-visualization-spatial.html
                    if config_name == "config_1":
                        logger.info(
                            f"	Running method {method_name} with config {config_name}."
                        )
                        counts = pd.DataFrame(
                            adata.X.todense(),
                            columns=adata.var_names,
                            index=adata.obs_names,
                        )
                        coord = pd.DataFrame(
                            adata.obsm["spatial"],
                            columns=[
                                current_config_params["x_coord_name"],
                                current_config_params["y_coord_name"],
                            ],
                            index=adata.obs_names,
                        ).to_numpy(dtype="int")

                        results = SpatialDE.run(coord, counts)

                        results.index = results["g"]

                        # Concat making sure they're concatenated in the correct positions with adata.var
                        adata.var = pd.concat(
                            [adata.var, results.loc[adata.var.index.values, :]],
                            axis=1,
                        )

                    if config_name == "config_2":
                        raw_counts = adata.to_df(layer="raw_counts")
                        # Convert the raw_counts to a DataFrame
                        counts = pd.DataFrame(
                            data=raw_counts.T,
                            index=adata.var.index,  # Assuming 'gene_ids' is the gene identifier
                            columns=adata.obs_names,
                        ).T  # Assuming 'obs_names' are the sample names

                        sample_info = adata.obs[
                            [
                                current_config_params["x_coord_name"],
                                current_config_params["y_coord_name"],
                                current_config_params["counts"],
                            ]
                        ]
                        norm_expr = NaiveDE.stabilize(counts.T).T
                        counts = NaiveDE.regress_out(
                            sample_info, norm_expr.T, "np.log(total_counts)"
                        ).T

                        coord = (
                            sample_info[
                                [
                                    current_config_params["x_coord_name"],
                                    current_config_params["y_coord_name"],
                                ]
                            ]
                            .astype("int")
                            .values
                        )
                        results = SpatialDE.run(coord, counts)
                        results.index = results["g"]

                    logger.info("		Saving spatially variable genes")
                    results.sort_values("qval", inplace=True)

                    with pd.ExcelWriter(
                        f"{STNavCorePipeline.saving_path}\\{data_type}\\Files\\{data_type}_SpatiallyVarGenes_{date}.xlsx"
                    ) as writer:
                        results.to_excel(
                            writer,
                            sheet_name="Spatially Variable Genes",
                            index=True,
                        )

                    results.sort_values("qval", inplace=True)
                    # Need to filter first for significant genes
                    sign_results = results.query("qval < 0.05")
                    logger.info(
                        f"Sign value results:\n\n{sign_results['l'].value_counts()}"
                    )
                    # Automatic expression histology https://github.com/Teichlab/SpatialDE
                    if config_params["AEH"]["usage"]:

                        # Get the value counts
                        val_counts = sign_results["l"].value_counts()

                        # Calculate the average length scale - A parameter indicating the distance scale a gene changes expression over
                        average_length = np.average(
                            val_counts.index, weights=val_counts.values
                        )

                        logger.info(
                            f"Running AEH with the average lenghtscale of {average_length}"
                        )

                        logger.info("Running automatic expression histology.")
                        histology_results, patterns = SpatialDE.aeh.spatial_patterns(
                            coord,
                            counts,
                            sign_results,
                            C=config_params["AEH"]["params"]["C"],
                            l=average_length,
                            verbosity=1,
                        )

                        # Add the results to the adata and save it as SpatiallyVariableGenes adata

                        STNavCorePipeline.save_as("SpatiallyVariableGenes_adata", adata)

                        logger.info("		Saving spatially variable genes with AEH.")

                        with pd.ExcelWriter(
                            f"{STNavCorePipeline.saving_path}\\{data_type}\\Files\\{data_type}_histology_results_AEH_{date}.xlsx"
                        ) as writer:
                            histology_results.to_excel(
                                writer,
                                sheet_name="histology_results AEH",
                                index=True,
                            )

                        with pd.ExcelWriter(
                            f"{STNavCorePipeline.saving_path}\\{data_type}\\Files\\{data_type}_Patterns_AEH_{date}.xlsx"
                        ) as writer:
                            patterns.to_excel(
                                writer,
                                sheet_name="patterns AEH",
                                index=True,
                            )

                        for i in range(3):
                            plt.subplot(1, 3, i + 1)
                            plt.scatter(
                                coord["array_row"],
                                coord["array_col"],
                                c=patterns[i],
                            )
                            plt.axis("equal")
                            plt.title(
                                "Pattern {} - {} genes".format(
                                    i,
                                    histology_results.query("pattern == @i").shape[0],
                                )
                            )
                            plt.colorbar(ticks=[])

                        # for i, g in enumerate(["Dnah7", "Ak9", "Muc4"]):
                        #     plt.subplot(1, 3, i + 1)
                        #     plt.scatter(
                        #         coord["array_row"],
                        #         coord["array_col"],
                        #         c=norm_expr[g],
                        #     )
                        #     plt.title(g)
                        #     plt.axis("equal")

                        #     plt.colorbar(ticks=[])

                        # In regular differential expression analysis, we usually investigate the relation between significance and effect size by so called volcano plots. We don't have the concept of fold change in our case, but we can investigate the fraction of variance explained by spatial variation.

                        plt.yscale("log")
                        plt.scatter(results["FSV"], results["qval"], c="black")
                        plt.axhline(0.05, c="black", lw=1, ls="--")
                        plt.gca().invert_yaxis()
                        plt.xlabel("Fraction spatial variance")
                        plt.ylabel("Adj. P-value")

                        logger.info(
                            "		Saving genes associated with the patterns as json file."
                        )
                        pattern_dict = {}
                        for i in histology_results.sort_values(
                            "pattern"
                        ).pattern.unique():
                            pattern_dict.setdefault(
                                f"pattern_{i}",
                                ", ".join(
                                    histology_results.query("pattern == @i")
                                    .sort_values("membership")["g"]
                                    .tolist()
                                ),
                            )

                        with open(
                            f"{STNavCorePipeline.saving_path}\\{data_type}\\Files\\{data_type}_patterns_genes_{date}.json",
                            "w",
                        ) as outfile:
                            json.dump(pattern_dict, outfile)

                elif method_name == "Squidpy_MoranI":
                    genes = adata[:, adata.var.highly_variable].var_names.values[
                        : config_params["n_genes"]
                    ]
                    sq.gr.spatial_neighbors(adata)

                    config_params["params"].setdefault("genes", genes)
                    # Run spatial autocorrelation morans I
                    sq.gr.spatial_autocorr(
                        **return_filtered_params(config=config_params, adata=adata)
                    )
                    logger.info(f"{adata.uns['moranI'].head(10)}")

                    # Save to excel file
                    with pd.ExcelWriter(
                        f"{STNavCorePipeline.saving_path}\\{data_type}\\Files\\{data_type}_Squidpy_MoranI_{date}.xlsx"
                    ) as writer:
                        adata.uns["moranI"].to_excel(
                            writer,
                            sheet_name="Squidpy_MoranI",
                            index=True,
                        )
                    logger.info(f"Saving adata to adata_dict as '{config_name}_adata'.")

                    STNavCorePipeline.save_as(f"{method_name}_adata", adata)
                    # sq.pl.spatial_scatter(adata, color=["Olfm1", "Plp1", "Itpka", "cluster"])
