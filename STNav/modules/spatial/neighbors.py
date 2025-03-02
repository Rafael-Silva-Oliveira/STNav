# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import json
import torch
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)


# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl
import squidpy as sq


@pass_STNavCore_params
def SpatialNeighbors(STNavCorePipeline):

    # https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_vizgen_mouse_liver.html#assign-cell-types

    step = "SpatialNeighbors"
    config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]
    logger.info("Calcualting Spatial Neighbors scores.")
    for method_name, config_params in config.items():
        if config_params["usage"]:

            if return_from_checkpoint(
                STNavCorePipeline,
                config_params=config_params,
                checkpoint_step=step,
                method_name=method_name,
            ):
                continue
            adata_path = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                config_params["adata_to_use"]
            ]
            adata = sc.read_h5ad(adata_path)

            logger.info(
                f"Running {method_name} method configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
            )

            if method_name == "Squidpy":
                sq.gr.spatial_neighbors(adata, coord_type="generic")
                sq.gr.nhood_enrichment(
                    **return_filtered_params(
                        config=config_params["Squidpy_NHoodEnrichment"], adata=adata
                    )
                )

                sq.gr.co_occurrence(
                    **return_filtered_params(
                        config=config_params["Squidpy_Co_Ocurrence"], adata=adata
                    )
                )

                sq.gr.centrality_scores(
                    **return_filtered_params(
                        config=config_params["Squidpy_Centrality"], adata=adata
                    )
                )

                save_path = (
                    STNavCorePipeline.saving_path + "/Plots/" + "centrality" + ".png"
                )

                with plt.rc_context():  # Use this to set figure params like size and dpi
                    plot_func = sq.pl.centrality_scores(
                        adata,
                        config_params["Squidpy_Centrality"]["params"]["cluster_key"],
                    )
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.close()

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )

                del adata
