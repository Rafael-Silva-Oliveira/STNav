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


# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import spatialdm.plottings as pl
import squidpy as sq


@pass_STNavCore_params
def SpatialNeighbors(STNavCorePipeline):
    step = "SpatialNeighbors"
    config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]
    logger.info("Calcualting Spatial Neighbors scores.")
    for method_name, config_params in config.items():
        if config_params["usage"]:
            adata_path = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                config_params["adata_to_use"]
            ]
            adata = sc.read_h5ad(adata_path)

            if return_from_checkpoint(
                STNavCorePipeline,
                path_to_check=adata_path,
                checkpoint_step=step,
                checkpoint_boolean=STNavCorePipeline.config[
                    STNavCorePipeline.data_type
                ][step][method_name]["checkpoint"]["usage"],
            ):
                logger.info(
                    f"Returning adata from checkpoint '{STNavCorePipeline.config[STNavCorePipeline.data_type][step][method_name]['checkpoint']['pipeline_run']}' with the following adata:\n\n {adata}."
                )
                return adata
            logger.info(
                f"Running {method_name} method configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
            )

            if method_name == "Squidpy_NHoodEnrichment":
                sq.gr.spatial_neighbors(adata)
                sq.gr.nhood_enrichment(
                    **return_filtered_params(config=config_params, adata=adata)
                )

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )
            if method_name == "Squidpy_Co_Ocurrence":
                # TODO: try to save the file with the matrix of co-occurrence probabilities within each spot. Apply this mask over the predictions from deconvolution to adjust based on co-ocurrence (clusters that have high value of co-occurrence will probably have similar cell proportions within each spot)
                sq.gr.co_occurrence(
                    **return_filtered_params(config=config_params, adata=adata)
                )
                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )
