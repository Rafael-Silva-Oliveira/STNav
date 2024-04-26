# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import inspect
import os

# Unnormalize data
from datetime import datetime

import anndata as an
import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanorama
import scanpy as sc
import scarches as sca
import seaborn as sns
import spatialdm as sdm
import squidpy as sq
from gseapy.plot import gseaplot
import stlearn as st

# Training a model to predict proportions on spatial data using scRNA seq as reference
from loguru import logger
from scipy import sparse
from scipy.sparse import csr_matrix
from scvi.external import RNAStereoscope, SpatialStereoscope
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

from STNav.utils.helpers import return_filtered_params, unnormalize
from STNav.utils.decorators import logger_wraps

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


def run_plots(plotting_config, adata_dict: dict, directory: str):

    logger.info("Running plots.")
    # Plotting section
    for plot_type, configs in plotting_config.items():
        for config_name, params in configs.items():
            if params["usage"]:
                logger.info(
                    f"Running plot type {plot_type} with the config name {config_name}"
                )

                # Check if adata is being loaded
                try:
                    adata_path = adata_dict[params["data_type"]][params["adata_to_use"]]
                    adata = sc.read_h5ad(adata_path)
                except Exception as e:
                    logger.error(f"Error: {e}. The adata could not be loaded.")
                    continue

                if any(
                    substring in params["func_str"]
                    for substring in ["sc.", "sq.", "st.", "sdm."]
                ):

                    # TODO: add some asserts/logging so that the plots such as rank_genes_groups_stacked, rank_genes_groups_heatmap, etc, will only be plotted if the rank_genes_groups from the upstream config is set to true. Otherwise, it+s not possible to plot these. Add a warning saying that these will be skipped and if we want to add it, we need to set the rank_genes_groups to true;
                    len_func_str = len(params["func_str"].split("."))
                    func_str = params["func_str"]
                    if len_func_str == 3:
                        class_eval_str, module_str, func_str = func_str.split(".")
                        class_eval = eval(class_eval_str)
                        module = getattr(class_eval, module_str)
                        plot_func = getattr(module, func_str)

                    valid_arguments = inspect.signature(plot_func).parameters.keys()
                    filtered_params = {
                        k: v
                        for k, v in params["params"].items()
                        if k in valid_arguments
                    }

                    if "adata_or_result" in filtered_params.keys():
                        filtered_params["adata_or_result"] = eval(
                            filtered_params["adata_or_result"]
                        )
                    if "adata" in filtered_params.keys():
                        filtered_params["adata"] = eval(filtered_params["adata"])

                    if "sample" in filtered_params.keys():
                        filtered_params["sample"] = eval(filtered_params["sample"])

                    # try:
                    #     filtered_params["adata"] = eval(filtered_params["adata"])
                    # except Exception as e:
                    #     filtered_params["adata_or_result"] = eval(
                    #         filtered_params["adata_or_result"]
                    #     )

                    if "cluster_palette" in config_name:
                        # We need to add umap to add the palette colors
                        clusters_names = filtered_params["color"]
                        sc.pl.umap(
                            filtered_params["adata"],
                            color=[clusters_names],
                        )
                        palette_lst = [
                            v
                            for k, v in dict(
                                zip(
                                    [
                                        str(i)
                                        for i in range(
                                            len(
                                                filtered_params["adata"]
                                                .obs[clusters_names]
                                                .cat.categories
                                            )
                                        )
                                    ],
                                    filtered_params["adata"].uns[
                                        clusters_names + "_colors"
                                    ],
                                )
                            ).items()
                            if k
                            in filtered_params["adata"]
                            .obs[clusters_names]
                            .unique()
                            .tolist()
                        ]
                        filtered_params.setdefault("palette", palette_lst)

                    if "embedding" in plot_type:
                        ct_list = adata.obsm["deconvolution"].columns
                        filtered_params.setdefault("color", ct_list)

                    logger.info(
                        f"The following parameters were validated and will be used for the {plot_type = }: {filtered_params} "
                    )
                else:
                    PLOT = CustomPlots()
                    plot_func = getattr(PLOT, plot_type)
                    valid_arguments = inspect.signature(plot_func).parameters.keys()
                    filtered_params = {
                        k: v
                        for k, v in params["params"].items()
                        if k in valid_arguments
                    }
                    try:
                        filtered_params["adata"] = eval(filtered_params["adata"])
                    except Exception as e:
                        filtered_params["adata_or_result"] = eval(
                            filtered_params["adata_or_result"]
                        )
                    save_path = directory + "\\Plots\\" + config_name
                    filtered_params.setdefault("save", save_path)

                    logger.info(
                        f"The following parameters were validated and will be used for the {plot_type} plot: {filtered_params} "
                    )

                save_path = directory + "\\Plots\\" + config_name
                try:
                    with plt.rc_context():  # Use this to set figure params like size and dpi
                        plt.figure(figsize=(20, 20))  # Set the figure size
                        plotting_func = plot_func(**filtered_params)
                        plt.savefig(save_path, bbox_inches="tight")
                        plt.close()

                except Exception as e:
                    logger.error(
                        f"\n\n\n ###################################################################\n\nAn error occurred in the plot {plot_func}.\n {e}"
                    )


class CustomPlots(object):
    def __init__(self) -> None:
        pass

    def proportion_per_cell_plot(self, adata, group, condition, save):
        tmp = pd.crosstab(adata.obs[group], adata.obs[condition], normalize="index")
        tmp.plot.bar(stacked=True).legend(loc="upper right")
        plt.savefig(save)
