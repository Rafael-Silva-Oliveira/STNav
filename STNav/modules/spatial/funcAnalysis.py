import pandas as pd
import decoupler as dc
import scanpy as sc
import numpy as np
import squidpy as sq
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    return_from_checkpoint,
)


@pass_STNavCore_params
def FunctionalAnalysis(STNavCorePipeline):
    step = "FunctionalAnalysis"
    config = STNavCorePipeline.config[STNavCorePipeline.data_type][step]

    logger.info(f"Running Functional Analysis for {STNavCorePipeline.data_type}.")

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

            if method_name == "decoupleR":
                for database, database_params in config_params["databases"].items():
                    logger.info(
                        f"Running Functional Analysis with decoupleR using {database} database"
                    )

                    if database == "CollecTRI":
                        db: pd.DataFrame = dc.get_collectri(
                            organism="human", split_complexes=False
                        )
                    elif database == "PROGENy":
                        db = dc.get_progeny("human", top=500)
                    else:
                        db = dc.get_resource(name=database)
                        db = db[db["collection"] == "hallmark"]

                        # Remove duplicated entries
                        db = db[~db.duplicated(["geneset", "genesymbol"])]

                        # Rename
                        db.loc[:, "geneset"] = [
                            name.split("HALLMARK_")[1] for name in db["geneset"]
                        ]

                    logger.info(
                        "Running activity inference with Univariate Linear Model (ULM)"
                    )

                    if config_params["databases"][database]["func_str"] == "dc.run_ulm":
                        dc.run_ulm(
                            mat=adata,
                            net=db,
                            source=database_params["params"]["source"],
                            target=database_params["params"]["target"],
                            weight=database_params["params"]["weight"],
                            verbose=database_params["params"]["verbose"],
                            use_raw=database_params["params"]["use_raw"],
                        )
                        adata.obsm[f"{database}_ulm_estimate"] = adata.obsm[
                            "ulm_estimate"
                        ].copy()
                        del adata.obsm["ulm_estimate"]

                        adata.obsm[f"{database}_ulm_pvals"] = adata.obsm[
                            "ulm_pvals"
                        ].copy()
                        del adata.obsm["ulm_pvals"]

                        # Get activity scores and save them to obsm
                        acts = dc.get_acts(adata, obsm_key=f"{database}_ulm_estimate")

                    elif (
                        config_params["databases"][database]["func_str"] == "dc.run_mlm"
                    ):
                        dc.run_mlm(
                            mat=adata,
                            net=db,
                            source=database_params["params"]["source"],
                            target=database_params["params"]["target"],
                            weight=database_params["params"]["weight"],
                            verbose=database_params["params"]["verbose"],
                            use_raw=database_params["params"]["use_raw"],
                        )

                        adata.obsm[f"{database}_mlm_estimate"] = adata.obsm[
                            "mlm_estimate"
                        ].copy()
                        del adata.obsm["mlm_estimate"]

                        adata.obsm[f"{database}_mlm_pvals"] = adata.obsm[
                            "mlm_pvals"
                        ].copy()
                        del adata.obsm["mlm_pvals"]

                        # Get activity scores and save them to obsm
                        acts = dc.get_acts(adata, obsm_key=f"{database}_mlm_estimate")

                    elif (
                        config_params["databases"][database]["func_str"] == "dc.run_ora"
                    ):
                        dc.run_ora(
                            mat=adata,
                            net=db,
                            source=database_params["params"]["source"],
                            target=database_params["params"]["target"],
                            verbose=database_params["params"]["verbose"],
                            use_raw=database_params["params"]["use_raw"],
                        )

                        adata.obsm[f"{database}_ora_estimate"] = adata.obsm[
                            "ora_estimate"
                        ].copy()
                        del adata.obsm["ora_estimate"]

                        adata.obsm[f"{database}_ora_pvals"] = adata.obsm[
                            "ora_pvals"
                        ].copy()
                        del adata.obsm["ora_pvals"]
                        # Get activity scores and save them to obsm
                        acts = dc.get_acts(adata, obsm_key=f"{database}_ora_estimate")

                    adata.obsm[f"{database}_acts"] = acts.copy()

                    logger.info(
                        "Ranking sources groups for cell type and leiden clusters"
                    )

                    df_cell_type = dc.rank_sources_groups(
                        acts,
                        groupby="cell_type",
                        reference="rest",
                        method="wilcoxon",
                    )
                    df_leiden_clusters = dc.rank_sources_groups(
                        acts,
                        groupby="leiden_clusters",
                        reference="rest",
                        method="wilcoxon",
                    )
                    logger.info(
                        f"Saving results of DEG functional analysis on adata.uns as {database}_ranked_groups_acts"
                    )

                    adata.uns[f"{database}_DEG_leiden_clusters_acts"] = (
                        df_leiden_clusters.copy()
                    )
                    adata.uns[f"{database}_DEG_cell_type_acts"] = df_cell_type.copy()

                    # n_markers = 3
                    # source_markers = (
                    #     df.groupby("group")
                    #     .head(n_markers)
                    #     .groupby("group")["names"]
                    #     .apply(lambda x: list(x))
                    #     .to_dict()
                    # )
            tt = 2
