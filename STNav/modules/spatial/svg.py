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
def SpatiallyVariableGenes(STNavCorePipeline):
    """
    g - The name of the gene
    pval - The P-value for spatial differential expression
    qval - Significance after correcting for multiple testing
    l - A parameter indicating the distance scale a gene changes expression over
    """
    step = "SpatiallyVariableGenes"
    config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]

    logger.info("Obtaining spatially variable genes.")
    for method_name, config_params in config.items():
        if config_params["usage"]:
            if return_from_checkpoint(
                STNavCorePipeline,
                config_params=config_params,
                checkpoint_step=step,
                method_name=method_name,
            ):
                adata_path = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    config_params["save_as"]
                ]
                adata = sc.read_h5ad(adata_path)
                continue
            adata_path = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                config_params["adata_to_use"]
            ]
            adata = sc.read_h5ad(adata_path)
            logger.info(
                f"Running {method_name} method configuration \n using the following adata {config_params['adata_to_use']}"
            )
            data_type = config_params["data_type"]

            if method_name == "Squidpy_MoranI":
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

                save_processed_adata(
                    STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )
