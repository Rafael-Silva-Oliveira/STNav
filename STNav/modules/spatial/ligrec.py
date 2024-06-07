# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
# import SpatialDE
# import NaiveDE
import squidpy as sq
import json
import torch
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    stLearn_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)
import stlearn as st

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl

@pass_STNavCore_params
def ReceptorLigandAnalysis(STNavCorePipeline):
    step = "ReceptorLigandAnalysis"
    config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]

    logger.info(f"Running Receptor Ligand Analysis for {STNavCorePipeline.data_type}.")

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

            adata.var_names = adata.var_names.str.upper()
            adata.var.index = adata.var.index.str.upper()

            logger.info(
                f"Running {method_name} method\n Configuration parameters:\n\n{config_params} \n ...using the following adata: '{config_params['adata_to_use']}'"
            )

            if method_name == "Squidpy_ligrec":
                res = sq.gr.ligrec(
                    **return_filtered_params(config=config_params, adata=adata)
                )

                with pd.ExcelWriter(
                    f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_LigRec_{date}.xlsx"
                ) as writer:
                    for sheet_name, file in {
                        "LigRec Means": res["means"],
                        "LigRec Pvalues": res["pvalues"],
                        "LigRec Metadata": res["metadata"],
                    }.items():
                        file.to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=True,
                        )

                logger.info(
                    f"Receptor-Ligand Analysis with {method_name} configuration \nCalculated means: \n{res['means'].head()}\n\nCalculated p-values:\n{res['pvalues'].head()}\n\nInteraction metadata: \n{res['metadata'].head()}"
                )

                # TODO: add plots here for each group and save plot individually similar to how I+m doing on the spatial proportions

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )
                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=res,
                    name=f"{method_name}_dictionary",
                )
                del adata
            elif method_name == "stLearn_cci":

                adata_cci = stLearn_wrapper(
                    **return_filtered_params(config=config_params, adata=adata)
                )

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata_cci,
                    name=f"{config_params['save_as']}",
                    fix_write=True,
                )
                del adata_cci
