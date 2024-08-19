# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import re
from datetime import datetime
from typing import Union

import anndata as an
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
import squidpy as sq
import decoupler as dc

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

sc.settings.n_jobs >= -1

from STNav.utils.helpers import (
    return_filtered_params,
    run_enrichr,
    run_gsea,
    run_prerank,
    transform_adata,
    save_processed_adata,
    return_from_checkpoint,
    swap_layer,
)

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


class STNavCore(object):
    adata_dict_suffix = "_adata"
    sc_model = None

    def __init__(
        self,
        config: dict,
        saving_path: str,
        data_type: str,
        adata_dict: dict = None,
    ) -> None:
        self.config = config
        self.saving_path: str = saving_path
        self.data_type: str = data_type
        self.adata_dict = adata_dict

    def read_rna(self):
        config = self.config[self.data_type]

        # Load H5AD scRNA reference dataset
        adata: an.AnnData = sc.read_h5ad(filename=config["path"])
        logger.info(
            f"Loaded scRNA dataset with {adata.n_obs} cells and {adata.n_vars} genes."
        )
        adata.var_names = adata.var_names.str.capitalize()
        adata.var.index = adata.var.index.str.capitalize()
        adata.var_names_make_unique()
        debug = True
        if debug:
            min_row = adata.obs["array_row"].min()
            max_row = adata.obs["array_row"].max()
            min_col = adata.obs["array_col"].min()
            max_col = adata.obs["array_col"].max()
            # define a mask to easily pull out this region of the object in the future
            mask = (
                (adata.obs["array_row"] > min_row + (max_row - min_row) * 0.1)
                & (adata.obs["array_row"] < min_row + (max_row - min_row) * 0.15)
                & (adata.obs["array_col"] > min_col + (max_col - min_col) * 0.1)
                & (adata.obs["array_col"] < min_col + (max_col - min_col) * 0.15)
            )

            adata = adata[mask].copy()
            # subset_fraction = 0.3  # Define the fraction of data to keep as subset
            # sc.pp.subsample(data=adata, fraction=subset_fraction)
        logger.info(
            f"Loaded 10X Visium dataset with {adata.n_obs} sequencing spots and {adata.n_vars} genes."
        )
        # As destriping from b2c provides non-integer values, if integers are strictly necessary for a downstream application just round the count matrix.
        adata.X.data = np.round(adata.X.data)
        adata.raw = adata.copy()

        try:
            adata.var.set_index(keys="features", inplace=True)
            adata.var.drop(columns=["_index"], inplace=True)

        except Exception as e:
            logger.warning(
                f"Failed to set new index. This might've happened because the index of var is already the genes/feature names, so no changes need to be made."
            )

        save_processed_adata(
            STNavCorePipeline=self,
            name="raw_adata",
            adata=adata,
        )

        del adata

    def read_visium(self):
        config = self.config[self.data_type]

        # Load Visium dataset
        adata: an.AnnData = sc.read_visium(
            path=config["path"],
            count_file=config["count_file"],
            load_images=config["load_images"],
            source_image_path=config["source_image_path"],
        )

        # If debug is True, select a random subset of the data
        debug = False
        if debug:
            subset_fraction = 0.3  # Define the fraction of data to keep as subset
            sc.pp.subsample(data=adata, fraction=subset_fraction)
        logger.info(
            f"Loaded 10X Visium dataset with {adata.n_obs} sequencing spots and {adata.n_vars} genes."
        )
        adata.var_names = adata.var_names.str.capitalize()
        adata.var.index = adata.var.index.str.capitalize()

        # Saving to adata to raw data simply to make sure that genes are now capitalized. This is to overcome an issue from scanpy.
        adata.raw = adata

        try:
            adata.var.set_index(keys="_index", inplace=True)
        except Exception as e:
            logger.warning(
                f"Failed to set new index _index. This might've happened because the index of var is already the genes/feature names, so no changes need to be made."
            )

        save_processed_adata(STNavCorePipeline=self, name="raw_adata", adata=adata)

        del adata

    def QC(self) -> None:
        step = "quality_control"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata: an.AnnData = sc.read_h5ad(filename=adata_path)

        logger.info("Running quality control.")
        adata_original: an.AnnData = adata.copy()
        # mitochondrial genes
        adata.var["Mt"] = adata.var_names.str.startswith(pat="Mt-")
        # ribosomal genes
        adata.var["Ribo"] = adata.var_names.str.startswith(pat=("Rps", "Rpl"))
        # hemoglobin genes.
        adata.var["Hb"] = adata.var_names.str.contains(
            pat=("^Hb[^(p)]")
        )  # adata.var_names.str.contains('^Hb.*-')

        if config["calculate_qc_metrics"]["usage"]:

            sc.pp.calculate_qc_metrics(
                **return_filtered_params(
                    config=config["calculate_qc_metrics"], adata=adata
                )
            )

            quantile_usage = config["calculate_qc_metrics"]["n_genes_by_counts"][
                "quantile"
            ]["usage"]
            manual_interval_usage = config["calculate_qc_metrics"]["n_genes_by_counts"][
                "manual_interval"
            ]["usage"]

            if quantile_usage == manual_interval_usage:
                raise Exception(
                    "Both quantile and manual interval are set to true when defining n_genes_by_counts. "
                    "Please, set only one to true."
                )

            if quantile_usage:
                upper_lim_n_genes_by_counts = np.quantile(
                    adata.obs.n_genes_by_counts.values,
                    config["calculate_qc_metrics"]["n_genes_by_counts"]["quantile"][
                        "upper_quantile"
                    ],
                )
                lower_lim_n_genes_by_counts = np.quantile(
                    adata.obs.n_genes_by_counts.values,
                    config["calculate_qc_metrics"]["n_genes_by_counts"]["quantile"][
                        "lower_quantile"
                    ],
                )
                logger.info(
                    f"Quantile is set to true. Lower and upper limits for n_genes_by_counts calculated: {upper_lim_n_genes_by_counts = } and {lower_lim_n_genes_by_counts = }"
                )
            else:
                upper_lim_n_genes_by_counts = np.quantile(
                    adata.obs.n_genes_by_counts.values,
                    config["calculate_qc_metrics"]["n_genes_by_counts"][
                        "manual_interval"
                    ]["upper_bound"],
                )
                lower_lim_n_genes_by_counts = np.quantile(
                    adata.obs.n_genes_by_counts.values,
                    config["calculate_qc_metrics"]["n_genes_by_counts"][
                        "manual_interval"
                    ]["lower_bound"],
                )

            adata = adata[
                (adata.obs["n_genes_by_counts"] > lower_lim_n_genes_by_counts)
                & (adata.obs["n_genes_by_counts"] < upper_lim_n_genes_by_counts)
            ]

            adata = adata[
                adata.obs["pct_counts_Mt"]
                < config["calculate_qc_metrics"]["pct_counts_Mt"]
            ]

            # Remove genes that still passed the previous condition
            genes_to_remove_pattern = re.compile(
                "|".join(
                    map(re.escape, config["calculate_qc_metrics"]["params"]["qc_vars"])
                )
            )

            genes_to_remove = adata.var_names.str.contains(pat=genes_to_remove_pattern)
            keep = np.invert(genes_to_remove)
            adata = adata[:, keep]
            print(
                f"{sum(genes_to_remove)} genes removed. Original size was {adata_original.n_obs} cells and {adata_original.n_vars} genes. New size is {adata.n_obs} cells and {adata.n_vars} genes"
            )

        save_processed_adata(
            STNavCorePipeline=self, name=config["save_as"], adata=adata
        )
        del adata

    def preprocessing(self) -> None:
        step = "preprocessing"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata: an.AnnData = sc.read_h5ad(filename=adata_path)

        counts: pd.DataFrame = sc.get.obs_df(
            adata, keys=list(adata.var_names), use_raw=True
        )

        logger.info(f"Retrieving counts matrix:\n{counts}")

        logger.info(
            f"Running preprocessing for {self.data_type} with '{config['adata_to_use']}' adata file."
        )

        # adata.X[0,:] -> this would give all the values/genes for 1 individual cell
        # adata.X[cells, genes]
        # adata.X[0,:].sum() would give the sum of UMI counts for a given cell
        logger.info(f"Current adata.X (raw data/counts) shape: \n {adata.X.shape = }")
        logger.info(
            f"Current adata.raw.X (raw data/counts) shape: \n {adata.raw.X.shape = }"
        )

        logger.info(
            f"adata.var contains the current gene information: \n {adata.var=} \n with the following columns: {adata.var.columns=}"
        )
        logger.info(
            f"adata.obs contains the current bin information: \n {adata.obs=} \n with the following columns: {adata.obs.columns=}"
        )

        # Filter genes by counts
        if config["filter_genes"]["usage"]:
            logger.info(
                f"Applying filtering genes with the following params {config['filter_genes']['params']}."
            )
            sc.pp.filter_genes(
                **return_filtered_params(config=config["filter_genes"], adata=adata)
            )
            logger.info(
                f"	After filtering genes: {adata.n_obs} observations x {adata.n_vars} genes."
            )

        # Filter cells by counts
        if config["filter_cells"]["usage"]:
            logger.info(
                f"Applying filtering cells with the following params {config['filter_cells']['params']} - Getting rid of cells with fewer than min_cells genes."
            )
            sc.pp.filter_cells(
                **return_filtered_params(config=config["filter_cells"], adata=adata)
            )
            logger.info(
                f"	After filtering cells: {adata.n_obs= } observations x {adata.n_vars= } cells. "
            )

        # Save original X data - adata.X would be the raw counts
        if self.data_type == "ST":

            adata_original: an.AnnData = sc.read_h5ad(
                filename=self.adata_dict[self.data_type]["raw_adata"]
            )
            adata_original.obs["status"] = np.where(
                adata_original.obs.index.isin(adata.obs.index), "Passed", "Failed"
            )
            sq.pl.spatial_scatter(
                adata_original,
                color="status",
                shape="square",
                size=1,
                legend_fontsize=20,  # adjust this value to make the legend smaller
            )
            plt.savefig(f"{self.saving_path}/Plots/QC_passed.png", dpi=1000)
            del adata_original

        adata.layers["raw_counts"] = adata.X.copy()
        adata.raw = adata

        # Normalize counts
        if config["normalize"]["usage"]:
            logger.info(
                f"Applying normalization with the following params {config['normalize']['params']}"
            )

            # Create a normalized counts copy of the raw_counts so that this is used to get the normalization
            adata.layers[config["normalize"]["params"]["layer"]] = adata.layers[
                "raw_counts"
            ].copy()

            sc.pp.normalize_total(
                **return_filtered_params(config=config["normalize"], adata=adata)
            )

        if config["log1p"]["usage"]:
            # It requires a positional argument and not just keyword arguments
            # Get the parameters from return_filtered_params
            logger.info(
                f"Applying log1p with the following params {config['log1p']['params']}"
            )
            adata.layers[config["log1p"]["params"]["layer"]] = adata.layers[
                config["normalize"]["params"]["layer"]
            ].copy()

            sc.pp.log1p(adata, layer=f"{config['log1p']['params']['layer']}")

        if config["highly_variable_genes"]["usage"]:
            logger.info(
                f"Selecting highly variable genes with the following params {config['highly_variable_genes']['params']}"
            )

            # assert (
            #     config["log1p"]["usage"] == True
            #     and config["highly_variable_genes"]["params"]["flavor"] == "seurat"
            # ) or (
            #     config["log1p"]["usage"] != True
            #     and config["highly_variable_genes"]["params"]["flavor"] != "seurat"
            # ), "Highly variable genes with log1p applied to the data expects flavor to be seurat. Please, deactivate log1p if you want to use seurat_v3. Expects logarithmized data, except when flavor='seurat_v3', in which count data is expected."

            sc.pp.highly_variable_genes(
                **return_filtered_params(
                    config=config["highly_variable_genes"], adata=adata
                )
            )

        if config["scale"]["usage"]:
            logger.info(
                f"Applying scaling with the following params {config['scale']['params']}"
            )
            # Get the parameters from return_filtered_params
            filtered_params = return_filtered_params(
                config=config["scale"], adata=adata
            )

            # Extract 'X' from the parameters
            X_value = filtered_params.pop("X", None)

            adata.layers[config["scale"]["params"]["layer"]] = adata.layers[
                config["log1p"]["params"]["layer"]
            ].copy()

            # Call log1p function with X as positional argument and the rest as keyword arguments
            sc.pp.scale(
                X_value, **filtered_params
            )  # before scalling the minimum of adata.X.min() would be 0, but after scaling we can now have negative numbers.Raw data wont have these negative values.

        logger.info("Adding extra info for plotting.")

        if config["plotting_prep"]["pca"]["usage"]:
            logger.info(
                f"	Applying pca with the following params {config['plotting_prep']['pca']['params']}"
            )
            # adata.obsm["X_pca"] is the embeddings
            # adata.uns["pca"] is pc variance
            # adata.varm['PCs'] is the loadings
            sc.tl.pca(
                **return_filtered_params(
                    config=config["plotting_prep"]["pca"], adata=adata
                )
            )

        if config["plotting_prep"]["neighbors"]["usage"]:
            logger.info(
                f"	Applying neighbors with the following params {config['plotting_prep']['neighbors']['params']}"
            )
            sc.pp.neighbors(
                **return_filtered_params(
                    config=config["plotting_prep"]["neighbors"], adata=adata
                )
            )
        if config["plotting_prep"]["umap"]["usage"]:
            logger.info(
                f"	Applying umap with the following params {config['plotting_prep']['umap']['params']}"
            )
            sc.tl.umap(
                **return_filtered_params(
                    config=config["plotting_prep"]["umap"], adata=adata
                )
            )
        if config["plotting_prep"]["tsne"]["usage"]:
            logger.info(
                f"	Applying tsne with the following params {config['plotting_prep']['tsne']['params']}"
            )
            sc.tl.tsne(
                **return_filtered_params(
                    config=config["plotting_prep"]["tsne"], adata=adata
                )
            )

        logger.info("Adding extra info for clustering")

        if config["plotting_prep"]["leiden"]["usage"]:
            logger.info(
                f"	Applying leiden with the following params {config['plotting_prep']['leiden']['params']}"
            )
            sc.tl.leiden(
                **return_filtered_params(
                    config=config["plotting_prep"]["leiden"], adata=adata
                )
            )

        if config["plotting_prep"]["dendogram"]["usage"]:
            logger.info(
                f"	Applying dendogram with the following params {config['plotting_prep']['dendogram']['params']}"
            )
            sc.tl.dendrogram(
                adata=adata,
                **return_filtered_params(
                    config=config["plotting_prep"]["dendogram"],
                ),
            )

        save_processed_adata(
            STNavCorePipeline=self, name=config["save_as"], adata=adata
        )
        del adata

    def DEG(self) -> None:

        step = "DEG"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata: an.AnnData = sc.read_h5ad(filename=adata_path)

        logger.info(
            f"Running DEG for {self.data_type} with '{config['adata_to_use']}' adata file."
        )

        # For DGE analysis we would like to run with all genes, on normalized values, so we will have to revert back to the raw matrix. In case you have raw counts in the matrix you also have to renormalize and logtransform. In this case, raw already has the normalized and log data for scRNA

        # Rank genes groups - Differential Expression of Genes (DEG)
        if config["rank_genes_groups"]["usage"]:

            # Add this one just to make sure we have ranked genes on the subset with the highly variable genes as well. Mainly for plotting reasons.
            sc.tl.rank_genes_groups(
                **return_filtered_params(
                    config=config["rank_genes_groups"], adata=adata
                )
            )

            save_processed_adata(
                STNavCorePipeline=self, name=config["save_as"], adata=adata
            )

        # Filter rank genes groups
        if config["filter_rank_genes_groups"]["usage"]:
            sc.tl.filter_rank_genes_groups(
                **return_filtered_params(
                    config=config["filter_rank_genes_groups"], adata=adata
                )
            )
        # Save DEG as dataframe
        if config["rank_genes_groups_df"]["usage"]:
            rank_genes_groups_df = config["rank_genes_groups_df"]["params"]["key"]
            rank_genes_groups = config["rank_genes_groups"]["params"]["key_added"]
            assert (
                config["rank_genes_groups_df"]["params"]["key"]
                == config["rank_genes_groups"]["params"]["key_added"]
            ), f"Key on rank_genes_groups is different than the one used in rank_genes_groups_df. Please, make sure they're the same. {rank_genes_groups_df = } is different than {rank_genes_groups =}"
            ranked_genes_list: pd.DataFrame = sc.get.rank_genes_groups_df(
                **return_filtered_params(
                    config=config["rank_genes_groups_df"], adata=adata
                )
            )

            # Convert all genes to upper case for GSEA
            ranked_genes_list["names"] = ranked_genes_list["names"].str.upper()

            # Short by logfoldchange as preparation for GSEA
            ranked_genes_list.sort_values(
                by=["logfoldchanges"], inplace=True, ascending=False
            )

            # Get the grouping that is being used
            groupby_used = config["rank_genes_groups"]["params"]["groupby"]

            # Save to excel
            ranked_genes_list.to_excel(
                excel_writer=self.saving_path
                + "/"
                + self.data_type
                + "/"
                + "Files"
                + "/"
                + self.data_type
                + f"_DEG_rank_genes_groups_by_{groupby_used}_df.xlsx",
                index=False,
            )

        if not config["rank_genes_groups_df"]["usage"] and config["GSEA"]["usage"]:

            raise ValueError(
                f"To run GSEA you need to turn on the setting on 'rank_genes_groups_df' on. Please, re-run the pipeline with the usage set to 'true' and try again."
            )

        if config["GSEA"]["usage"]:

            config_gsea = config["GSEA"]
            logger.info(
                f"Running GSEA for {self.data_type} with '{config['adata_to_use']}' adata file using a DEG ranked genes list with length {len(ranked_genes_list)} grouped by {config['rank_genes_groups']['params']['groupby']}:\n\n {ranked_genes_list}"
            )

            # Define a list to hold gene sets (manual or API)
            gsea_dataframes: dict = {}

            # Handle API gene sets
            if config_gsea["gene_sets"]["usage"]:
                gene_set_list: list = [
                    ontology
                    for ontology, boolean in config_gsea["gene_sets"]["sets"].items()
                    if boolean
                ]

                logger.info(f"Adding API gene sets.\n {gene_set_list}")

            # Iterate over gene sets
            enrichr_list: list = []
            prerank_list: list = []
            gsea_list: list = []
            set_name: str = "API"

            gene_set_names = gp.get_library_name(organism="human")

            # Setdefault is overriding data... I need to save the "set_name" and pass it as parameter to the gsea_dataframes.setdefault("set_name...", res).
            if config_gsea["stratify_by_group"]:
                if config_gsea["enrichr"]["usage"]:
                    _enrichr_sub: pd.DataFrame = run_enrichr(
                        gene_set_list=gene_set_list,
                        ranked_genes_list=ranked_genes_list,
                        config_gsea=config_gsea,
                        data_type=self.data_type,
                        set_name=set_name,
                        group_bool=True,
                    )
                    enrichr_list.append(_enrichr_sub)

                if config_gsea["prerank"]["usage"]:
                    _prerank_sub: pd.DataFrame = run_prerank(
                        gene_set_list=gene_set_list,
                        ranked_genes_list=ranked_genes_list,
                        config_gsea=config_gsea,
                        data_type=self.data_type,
                        set_name=set_name,
                        group_bool=True,
                        saving_path=self.saving_path,
                    )
                    prerank_list.append(_prerank_sub)

                if config_gsea["gsea"]["usage"]:
                    _gsea_sub: pd.DataFrame = run_gsea(
                        adata=adata,
                        gene_set_list=gene_set_list,
                        ranked_genes_list=ranked_genes_list,
                        config_gsea=config_gsea,
                        data_type=self.data_type,
                        set_name=set_name,
                        group_bool=True,
                        saving_path=self.saving_path,
                    )
                    gsea_list.append(_gsea_sub)

            if config_gsea["enrichr"]["usage"]:
                _enrichr: pd.DataFrame = run_enrichr(
                    gene_set_list=gene_set_list,
                    ranked_genes_list=ranked_genes_list,
                    config_gsea=config_gsea,
                    data_type=self.data_type,
                    set_name=set_name,
                    group_bool=False,
                )
                enrichr_list.append(_enrichr)

            if config_gsea["prerank"]["usage"]:
                _prerank: pd.DataFrame = run_prerank(
                    gene_set_list=gene_set_list,
                    ranked_genes_list=ranked_genes_list,
                    config_gsea=config_gsea,
                    data_type=self.data_type,
                    set_name=set_name,
                    group_bool=False,
                    saving_path=self.saving_path,
                )
                prerank_list.append(_prerank)

            if config_gsea["gsea"]["usage"]:
                _gsea: pd.DataFrame = run_gsea(
                    adata=adata,
                    gene_set_list=gene_set_list,
                    ranked_genes_list=ranked_genes_list,
                    config_gsea=config_gsea,
                    data_type=self.data_type,
                    set_name=set_name,
                    group_bool=False,
                    saving_path=self.saving_path,
                )
                gsea_list.append(_gsea)

            if len(enrichr_list) > 0:
                gsea_dataframes["enrichr"] = pd.concat(objs=enrichr_list).reset_index()
            if len(prerank_list) > 0:
                gsea_dataframes["prerank"] = pd.concat(objs=prerank_list).reset_index()
            if len(gsea_list) > 0:
                gsea_dataframes["gsea"] = pd.concat(objs=gsea_list).reset_index()

            with pd.ExcelWriter(
                path=f"{self.saving_path}/{self.data_type}/Files/{self.data_type}_GSEA_{date}.xlsx"
            ) as writer:
                for sheet_name, gsea_df in gsea_dataframes.items():
                    gsea_df.to_excel(writer, sheet_name=sheet_name, index=False)
        del adata
