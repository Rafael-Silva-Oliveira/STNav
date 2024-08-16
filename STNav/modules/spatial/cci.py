# Load packages
from __future__ import annotations
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
    stLearn_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)

import stlearn as st
from typing import Union, Optional
import liana as li
import gc
import decoupler as dc
from mudata import MuData

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl


@pass_STNavCore_params
def CCI(STNavCorePipeline):
    step = "CCI"
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

            if method_name == "LIANA":

                logger.info("Running LIANA+")
                logger.info("Adding spatial neighbors")
                li.ut.spatial_neighbors(
                    **return_filtered_params(
                        config=config_params["spatial_neighbors"], adata=adata
                    ),
                )
                logger.info("Extracting LR from CellPhoneDB")

                adata.X = adata.layers["lognorm_counts"].copy()
                li.mt.cellphonedb(
                    **return_filtered_params(
                        config=config_params["cellphonedb"], adata=adata
                    ),
                )

                # Add the cellphonedb groupby_pairs
                config_params["rank_aggregate"]["groupby_pairs"] = adata.uns[
                    config_params["cellphonedb"]["params"]["key_added"]
                ]
                logger.info("Aggregating LR results into global scores")

                li.mt.rank_aggregate(
                    **return_filtered_params(
                        config=config_params["rank_aggregate"], adata=adata
                    ),
                )
                adata.X = adata.layers["raw_counts"].copy()

                logger.info("Running LIANA+ bivariate CCI analysis")
                li.mt.bivariate(
                    mdata=adata,
                    layer=config_params["bivariate"]["params"]["layer"],
                    connectivity_key=config_params["bivariate"]["params"][
                        "connectivity_key"
                    ],
                    resource_name=config_params["bivariate"]["params"]["resource_name"],
                    local_name=config_params["bivariate"]["params"]["local_name"],
                    global_name=config_params["bivariate"]["params"]["global_name"],
                    n_perms=config_params["bivariate"]["params"]["n_perms"],
                    mask_negatives=config_params["bivariate"]["params"][
                        "mask_negatives"
                    ],
                    add_categories=config_params["bivariate"]["params"][
                        "add_categories"
                    ],
                    nz_prop=config_params["bivariate"]["params"]["nz_prop"],
                    use_raw=config_params["bivariate"]["params"]["use_raw"],
                    verbose=config_params["bivariate"]["params"]["verbose"],
                )

                # Global summaries
                lrdata = adata.obsm["local_scores"]

                # The average local metric used in the global summary represents coverage
                logger.info(
                    "Coverage using global summary results with the 'mean' metric:"
                )
                lrdata.var.sort_values("mean", ascending=False).head(10)

                # We can also use Moran's R, an extension of Moran's I for spatially variable genes. Among most variable interactions and with the highest global morans R, is an interaction that most likely represents biological relationships, with distinct spatial clustering patterns.
                logger.info(
                    "Morans R using global summary results with the 'morans' metric:"
                )
                lrdata.var.sort_values("morans", ascending=False).head(10)

                logger.info("Identifying Intercellular Patterns with NMF")

                """
				Intercellular Patterns

				Having the LR scores, we can use NMF to identify coordinated cell-cell communication signatures.
				* W (basis matrix): Each basis vector is a patten of ligand-receptor expression in the dataset. The values in W (factor score) indicate the strenghts of factor in each spot; high values indicate high influence by the associated communication signature, and vice-versa
				* H (coefficient matrix): Each row of H is the participation of the corresponding sample in the identified factor. The elements of each basis vector indicate the contribution of different interactions to the pattern (factor).
				"""

                li.multi.nmf(
                    **return_filtered_params(config=config_params["nmf"], adata=lrdata),
                )

                # Extract the variable loadings
                logger.info("Extracting and saving the factor loadings from NMF")
                lr_loadings = li.ut.get_variable_loadings(
                    lrdata, varm_key="NMF_H"
                ).set_index("index")

                logger.info("Extracting the factor scores from NMF")
                # Extract the factor scores
                factor_scores = li.ut.get_factor_scores(lrdata, obsm_key="NMF_W")

                # Create a dictionary to store the DataFrames
                dfs = {"lr_loadings": lr_loadings, "factor_scores": factor_scores}

                # Save the DataFrames to an Excel file with multiple tabs
                with pd.ExcelWriter(
                    f"{STNavCorePipeline.saving_path}/ST/Files/liana_nmf.xlsx"
                ) as writer:
                    for sheet_name, df in dfs.items():
                        df.to_excel(
                            excel_writer=writer, sheet_name=sheet_name, index=True
                        )

                nmf = sc.AnnData(
                    X=lrdata.obsm["NMF_W"],
                    obs=lrdata.obs,
                    var=pd.DataFrame(index=lr_loadings.columns),
                    uns=lrdata.uns,
                    obsm=lrdata.obsm,
                )

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=nmf,
                    name=f"LIANA_NMF",
                )
                save_path = STNavCorePipeline.saving_path + "/Plots/" + "nmf" + ".png"

                with plt.rc_context():  # Use this to set figure params like size and dpi
                    plot_func = sq.pl.spatial_scatter(
                        nmf,
                        color=[*nmf.var.index, None],
                        size=1.4,
                        ncols=2,
                        dpi=400,
                    )
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.close()

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata,
                    name=f"{config_params['save_as']}",
                )

                del adata
            elif method_name == "stLearn":

                adata_cci = stLearn_wrapper(
                    **return_filtered_params(
                        config=config_params["stLearn_cci"], adata=adata
                    )
                )

                save_processed_adata(
                    STNavCorePipeline=STNavCorePipeline,
                    adata=adata_cci,
                    name=f"{config_params['save_as']}",
                    fix_write=True,
                )
                del adata_cci
