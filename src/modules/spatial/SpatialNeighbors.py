# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import squidpy as sq
from typing import Union

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

from src.utils.helpers import (
    return_filtered_params,
)

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
from src.utils.decorators import pass_STNavCore_params


@pass_STNavCore_params
def SpatialNeighbors(STNavCorePipeline):
    config = STNavCorePipeline.config[STNavCorePipeline.data_type]["SpatialNeighbors"]
    logger.info("Calcualting Spatial Neighbors scores.")
    for method_name, methods in config.items():
        for config_name, config_params in methods.items():
            if config_params["usage"]:
                adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    config_params["adata_to_use"]
                ].copy()
                logger.info(
                    f"Running {method_name} method with {config_name} configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
                )

                if method_name == "Squidpy":
                    if config_name == "NHoodEnrichment":
                        sq.gr.spatial_neighbors(adata)
                        sq.gr.nhood_enrichment(
                            **return_filtered_params(config=config_params, adata=adata)
                        )
                        # STNavCorePipeline.adata_dict[
                        #     config_params["data_type"]
                        # ].setdefault(f"{config_name}_adata", adata.copy())
                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)
                    elif config_name == "Co_Ocurrence":
                        # TODO: try to save the file with the matrix of co-occurrence probabilities within each spot. Apply this mask over the predictions from deconvolution to adjust based on co-ocurrence (clusters that have high value of co-occurrence will probably have similar cell proportions within each spot)
                        sq.gr.co_occurrence(
                            **return_filtered_params(config=config_params, adata=adata)
                        )

                        # STNavCorePipeline.adata_dict[
                        #     config_params["data_type"]
                        # ].setdefault(f"{config_name}_adata", adata.copy())
                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)
