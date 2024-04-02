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


@pass_STNavCore_params
def deconvolution(STNavCorePipeline, st_model, model_name):
    logger.info(
        f"Running deconvolution based on ranked genes with the group {STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
    )

    st_adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
        "subset_preprocessed_adata"
    ].copy()

    if model_name == "GraphST":

        adata_sc = STNavCorePipeline.adata_dict["scRNA"][
            "subset_preprocessed_adata"
        ].copy()
        adata_sc_preprocessed = STNavCorePipeline.adata_dict["scRNA"][
            "preprocessed_adata"
        ].copy()

        project_cell_to_spot(st_adata, adata_sc, retain_percent=0.15)

        columns_cell_type_names = list(adata_sc.obs["cell_type"].unique())

        for cell_type in columns_cell_type_names:
            save_path = STNavCorePipeline.saving_path + "\\Plots\\" + cell_type + ".png"
            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    st_adata,
                    cmap="magma",
                    color=cell_type,
                    img_key="hires",
                    size=1.5,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")

        # Return to the original naming convention for plotting purposes
        adata_sc.obs.rename(
            columns={
                "cell_type": f"{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            },
            inplace=True,
        )
        adata_sc_preprocessed.obs.rename(
            columns={
                "cell_type": f"{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            },
            inplace=True,
        )

        STNavCorePipeline.adata_dict["scRNA"][
            "subset_preprocessed_adata"
        ] = adata_sc.copy()
        STNavCorePipeline.adata_dict["scRNA"][
            "preprocessed_adata"
        ] = adata_sc_preprocessed.copy()

        st_adata.obsm["deconvolution"] = st_adata.obs[columns_cell_type_names]

        STNavCorePipeline.save_as("deconvoluted_adata", st_adata)

        st_adata.obs.to_excel(
            f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
            index=False,
        )
        adata_sc.obs.rename(
            columns={
                "cell_type": {
                    STNavCorePipeline.config["scRNA"]["DEG"]["rank_genes_groups"][
                        "params"
                    ]["groupby"]
                }
            },
            inplace=True,
        )
    else:
        # Deconvolution
        st_adata.obsm["deconvolution"] = st_model.get_proportions()
        with torch.no_grad():
            keep_noise = False
            res = torch.nn.functional.softplus(st_model.module.V).cpu().numpy().T
            if not keep_noise:
                res = res[:, :-1]

        column_names = st_model.cell_type_mapping
        st_adata.obsm["deconvolution_unconstr"] = pd.DataFrame(
            data=res,
            columns=column_names,
            index=st_model.adata.obs.index,
        )

        for ct in st_adata.obsm["deconvolution"].columns:
            st_adata.obs[ct] = st_adata.obsm["deconvolution"][ct]

        st_adata.obs[
            f"spatial_{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        ] = st_adata.obs[column_names].idxmax(axis=1)

        STNavCorePipeline.save_as("deconvoluted_adata", st_adata)

        for cell_type in st_adata.obsm["deconvolution"].columns:
            save_path = STNavCorePipeline.saving_path + "\\Plots\\" + cell_type + ".png"
            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    st_adata,
                    cmap="magma",
                    color=cell_type,
                    img_key="hires",
                    size=1.6,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")

        st_adata.obs.to_excel(
            f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
            index=False,
        )
    return st_model


@pass_STNavCore_params
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
