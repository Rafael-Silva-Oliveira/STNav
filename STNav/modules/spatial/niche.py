import monkeybread as mb
import scanpy as sc
from loguru import logger

from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)

from datetime import datetime
import json


# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl
import squidpy as sq
import inspect
import os


# @pass_STNavCore_params
def NicheAnalysis(STNavCorePipeline):
    step = "NicheAnalysis"
    # config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]
    config = STNavCorePipeline["ST"][step]

    logger.info("Performing niche analysis.")
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
                f"Running {method_name} method configuration \n using the following adata {config_params['adata_to_use']}"
            )
            data_type = config_params["data_type"]

            if method_name == "...":
                ...
