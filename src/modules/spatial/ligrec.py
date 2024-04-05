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
from src.utils.decorators import pass_STNavCore_params
from src.utils.helpers import return_filtered_params, SpatialDM_wrapper
import stlearn as st

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
def ReceptorLigandAnalysis(STNavCorePipeline):

    config = STNavCorePipeline.config[STNavCorePipeline.data_type][
        "ReceptorLigandAnalysis"
    ]

    logger.info(f"Running Receptor Ligand Analysis for {STNavCorePipeline.data_type}.")

    for method_name, methods in config.items():
        for config_name, config_params in methods.items():
            if config_params["usage"]:
                adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    config_params["adata_to_use"]
                ].copy()

                adata.var_names = adata.var_names.str.upper()
                adata.var.index = adata.var.index.str.upper()

                logger.info(
                    f"Running {method_name} method with {config_name} configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
                )

                if method_name == "Squidpy":
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
                        f"Receptor-Ligand Analysis with {method_name} using {config_name} configuration \nCalculated means: \n{res['means'].head()}\n\nCalculated p-values:\n{res['pvalues'].head()}\n\nInteraction metadata: \n{res['metadata'].head()}"
                    )

                    # TODO: add plots here for each group and save plot individually similar to how I+m doing on the spatial proportions

                    STNavCorePipeline.save_as(f"{config_name}_adata", adata)

                    STNavCorePipeline.save_as(
                        f"{config_name}_dictionary", res, copy=False
                    )

                elif method_name == "SpatialDM":

                    adata = SpatialDM_wrapper(
                        **return_filtered_params(config=config_params, adata=adata)
                    )
                    with pd.ExcelWriter(
                        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\{STNavCorePipeline.data_type}_SpatialDM_LigRec_{date}.xlsx"
                    ) as writer:
                        for sheet_name in [
                            "global_res",
                            "geneInter",
                            "selected_spots",
                        ]:
                            if not adata.uns[sheet_name].empty:
                                adata.uns[sheet_name].to_excel(
                                    writer,
                                    sheet_name=sheet_name,
                                )

                    STNavCorePipeline.save_as(f"{config_name}_adata", adata)

                    # TODO: re-do plots. Add dictionary on the plots for this as well
                    # Filter out sparse interactions with fewer than 3 identified interacting spots. Cluster into 6 patterns.

                    # # visualize global and local pairs
                    #

                    # pl.global_plot(adata, figsize=(6, 5), cmap="RdGy_r", vmin=-1.5, vmax=2)
                    # pl.plot_pairs(adata, ["SPP1_CD44"], marker="s")

                    bin_spots = adata.uns["selected_spots"].astype(int)[
                        adata.uns["local_stat"]["n_spots"] > 2
                    ]
                    logger.info(
                        f"{bin_spots.shape[0]} pairs used for spatial clustering"
                    )

                    if bin_spots.shape[0] != 0:
                        results = SpatialDE.run(
                            adata.obsm["spatial"], bin_spots.transpose()
                        )

                        histology_results, patterns = SpatialDE.aeh.spatial_patterns(
                            adata.obsm["spatial"],
                            bin_spots.transpose(),
                            results,
                            C=3,
                            l=3,
                            verbosity=1,
                        )

                        plt.figure(figsize=(9, 8))
                        for i in range(3):
                            plt.subplot(2, 2, i + 2)
                            plt.scatter(
                                adata.obsm["spatial"][:, 0],
                                adata.obsm["spatial"][:, 1],
                                marker="s",
                                c=patterns[i],
                                s=35,
                            )
                            plt.axis("equal")
                            pl.plt_util(
                                "Pattern {} - {} genes".format(
                                    i,
                                    histology_results.query("pattern == @i").shape[0],
                                )
                            )
                        plt.savefig("mel_DE_clusters.pdf")

                    STNavCorePipeline.save_as(f"{config_name}_adata", adata)

                elif method_name == "stLearn":
                    use_label = "dominant_spot_cell_type"

                    adata = st.convert_scanpy(adata)
                    adata.obs[f"{use_label}"] = (
                        adata.obs[adata.obsm["deconvolution"].columns]
                        .idxmax(axis=1)
                        .astype("category")
                    )
                    if config_name == "stLearn_ligrec":
                        logger.info(
                            f"Running Receptor Ligand Analysis for {STNavCorePipeline.data_type} with connectomeDB2020_lit ligand-receptor pairs."
                        )
                        lrs = st.tl.cci.load_lrs(
                            ["connectomeDB2020_lit"], species="human"
                        )

                        config_params["params"].setdefault("lrs", lrs)
                        st.tl.cci.run(
                            **return_filtered_params(config=config_params, adata=adata),
                        )
                        try:
                            lr_info = adata.uns["lr_summary"]
                        except Exception as e:
                            logger.warning(
                                f"Could not retrieve ligand-receptor interactions: {e}"
                            )

                        logger.info(
                            f"Receptor-Ligand Analysis with {method_name} using {config_name} configuration \nCalculated ligand-receptor interactions: \n\n{lr_info}.\n\nThe following parameters were used:\n\n{config_params} "
                        )

                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)

                    elif config_name == "stLearn_cci":

                        assert config["stLearn"]["stLearn_ligrec"][
                            "usage"
                        ], "stLearn_ligrec must be set to true before running stLearn_cci"

                        logger.infor(
                            f"Predicting significant CCIs.\n\nWith the establishment of significant areas of LR interaction from stLearn_ligrec, we can now determine the significantly interacting cell types using the label '{use_label}'."
                        )
                        config_params["params"].setdefault("use_label", use_label)
                        st.tl.cci.run_cci(
                            **return_filtered_params(config=config_params, adata=adata)
                        )

                        logger.info(
                            f"CCI analysis with {method_name} using {config_name} configuration.\n\nThe following parameters were used:\n\n{config_params}"
                        )

                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)
