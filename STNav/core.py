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
import torch
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

sc.settings.n_jobs >= -1
import inspect

import os
from STNav.utils.helpers import (
    GARD,
    fix_write_h5ad,
    log_adataX,
    return_filtered_params,
    run_enrichr,
    run_gsea,
    run_prerank,
    unnormalize,
    transform_adata,
    save_processed_adata,
    return_from_checkpoint,
    extract_pipeline_run,
)

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

import anndata as ad


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
        self.saving_path = saving_path
        self.data_type = data_type
        self.adata_dict = adata_dict

    def read_rna(self):
        config = self.config[self.data_type]

        # Load H5AD scRNA reference dataset
        adata = sc.read_h5ad(config["path"])
        logger.info(
            f"Loaded scRNA dataset with {adata.n_obs} cells and {adata.n_vars} genes."
        )
        adata.var_names = adata.var_names.str.capitalize()
        adata.var.index = adata.var.index.str.capitalize()

        # Saving to adata to raw data simply to make sure that genes are now capitalized. This is to overcome an issue from scanpy.
        adata.raw = adata

        try:
            adata.var.set_index("features", inplace=True)
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

        return adata

    def read_visium(self):
        config = self.config[self.data_type]

        # Load Visium dataset
        adata = sc.read_visium(
            path=config["path"],
            count_file=config["count_file"],
            load_images=config["load_images"],
            source_image_path=config["source_image_path"],
        )
        logger.info(
            f"Loaded 10X Visium dataset with {adata.n_obs} sequencing spots and {adata.n_vars} genes."
        )
        adata.var_names = adata.var_names.str.capitalize()
        adata.var.index = adata.var.index.str.capitalize()

        # Saving to adata to raw data simply to make sure that genes are now capitalized. This is to overcome an issue from scanpy.
        adata.raw = adata

        try:
            adata.var.set_index("_index", inplace=True)
        except Exception as e:
            logger.warning(
                f"Failed to set new index _index. This might've happened because the index of var is already the genes/feature names, so no changes need to be made."
            )

        save_processed_adata(
            STNavCorePipeline=self,
            name="raw_adata",
            adata=adata,
        )
        return adata

    def QC(self):
        step = "quality_control"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata = sc.read_h5ad(adata_path)

        logger.info("Running quality control.")
        adata_original = adata.copy()
        # mitochondrial genes
        adata.var["Mt"] = adata.var_names.str.startswith("Mt-")
        # ribosomal genes
        adata.var["Ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
        # hemoglobin genes.
        adata.var["Hb"] = adata.var_names.str.contains(
            ("^Hb[^(p)]")
        )  # adata.var_names.str.contains('^Hb.*-')

        # TODO: Perform spatial projection with MT genes. Get DEG genes for MT

        if config["print_mt"]:
            adata_cp = adata.copy()
            sc.pp.neighbors(adata_cp, n_pcs=30, n_neighbors=20)
            sc.tl.leiden(adata_cp, key_added="leiden_clusters")
            sc.tl.rank_genes_groups(
                adata_cp,
                groupby="leiden_clusters",
                method="wilcoxon",
                key_added="wilcoxon",
                n_genes=1000,
                use_raw=False,
            )

            # Get top 10 MT DEG genes
            adata_cp_top_genes = sc.get.rank_genes_groups_df(
                adata_cp, key="wilcoxon", pval_cutoff=0.1, log2fc_min=1, group=None
            )
            adata_cp_top_genes_MT = adata_cp_top_genes.loc[
                adata_cp_top_genes["names"].str.contains("MT-"), :
            ]
            adata_cp_top_genes_MT.sort_values(
                by="logfoldchanges", inplace=True, ascending=False
            )
            top_10_MT_genes = adata_cp_top_genes_MT["names"].tolist()[:8]
            # sc.pl.spatial(adata_cp, img_key="hires", color=top_10_MT_genes)
            # sc.pl.spatial(adata_cp, img_key="hires", color="leiden_clusters")

            # tmp = pd.crosstab(adata.obs["cell_ontology_class"], adata.obs["leiden_clusters"], normalize="index")
            # tmp.plot.bar(stacked=True).legend(loc="upper right")

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

            genes_to_remove = adata.var_names.str.contains(genes_to_remove_pattern)
            keep = np.invert(genes_to_remove)
            adata = adata[:, keep]
            print(
                f"{sum(genes_to_remove)} genes removed. Original size was {adata_original.n_obs} cells and {adata_original.n_vars} genes. New size is {adata.n_obs} cells and {adata.n_vars} genes"
            )
        save_processed_adata(
            STNavCorePipeline=self, name=config["save_as"], adata=adata
        )
        return adata

    def preprocessing(self) -> an.AnnData:
        step = "preprocessing"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata = sc.read_h5ad(adata_path)

        adata.var_names_make_unique()
        logger.info(
            f"Running preprocessing for {self.data_type} with '{config['adata_to_use']}' adata file."
        )

        # Save original X data - adata.X would be the raw counts
        adata.layers["raw_counts"] = adata.X.copy()

        # adata.X[0,:] -> this would give all the values/genes for 1 individual cell
        # adata.X[cells, genes]
        # adata.X[0,:].sum() would give the sum of UMI counts for a given cell
        logger.info(f"Current adata.X (raw data/counts) shape: \n {adata.X.shape = }")
        logger.info(
            f"Current adata.raw.X (raw data/counts) shape: \n {adata.raw.X.shape = }"
        )

        logger.info(
            f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X before any filters or normalization: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
        )

        logger.info(
            f"adata.var contains the current gene information: \n {adata.var=} \n with the following columns: {adata.var.columns=}"
        )
        logger.info(
            f"adata.obs contains the current cell/spot information: \n {adata.obs=} \n with the following columns: {adata.obs.columns=}"
        )

        # Filter genes by counts
        if config["filter_genes"]["usage"]:
            logger.info(
                f"Applying filtering genes with the following params {config['filter_genes']['params']} - Getting rid of genes that are found in fewer than 25 counts."
            )
            sc.pp.filter_genes(
                **return_filtered_params(config=config["filter_genes"], adata=adata)
            )
            logger.info(
                f"	After filtering genes: {adata.n_obs} observations (cells if scRNA, spots if ST) x {adata.n_vars} genes."
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
                f"	After filtering cells: {adata.n_obs= } observations (cells if scRNA, spots if ST) x {adata.n_vars= } cells. Confirm if this is true."
            )

        if config["unnormalize"]["usage"]:
            adata_to_unnormalize = adata.copy()
            adata_unnormalized = unnormalize(
                adata_to_unnormalize, count_col=config["unnormalize"]["col_name"]
            )
            logger.info(
                f"	Saving unnormalized data do layers as 'unnormalized_counts' for data type {self.data_type}"
            )
            adata.layers["unnormalized_counts"] = adata_unnormalized.X.copy()

        # Normalized total to CPM (1e6)
        if config["normalize"]["usage"]:
            logger.info(
                f"Applying normalization with the following params {config['normalize']['params']}"
            )

            logger.info(
                f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X before normalizing: \n 1 - {adata.X[0,:].sum()= } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
            )
            sc.pp.normalize_total(
                **return_filtered_params(config=config["normalize"], adata=adata)
            )

            logger.info(
                f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X after normalizing: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
            )

        adata.layers["norm"] = adata.X

        if config["log1p"]["usage"]:
            # It requires a positional argument and not just keyword arguments
            # Get the parameters from return_filtered_params
            logger.info(
                f"Applying log1p with the following params {config['log1p']['params']}"
            )
            filtered_params = return_filtered_params(
                config=config["log1p"], adata=adata
            )

            # Extract 'X' from the parameters
            X_value = filtered_params.pop("X", None)

            # Call log1p function with X as positional argument and the rest as keyword arguments
            sc.pp.log1p(X_value, **filtered_params)

            logger.info(
                f"\n Applying the log changed the counts from UMI counts to log counts. The sum of log counts from the 3 first examples (cells for scRNA or spots for ST) from adata.X after applying log: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
            )

        adata.layers["lognorm"] = adata.X
        adata.raw = adata

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

            logger.info(
                f"\n After applying highly variable genes, 3 first examples (cells for scRNA or spots for ST) from adata.X after applying highly variable genes: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
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

        if config["plotting_prep"]["louvain"]["usage"]:
            logger.info(
                f"	Applying louvain with the following params {config['plotting_prep']['louvain']['params']}"
            )
            sc.tl.louvain(
                **return_filtered_params(
                    config=config["plotting_prep"]["louvain"], adata=adata
                )
            )

        if config["plotting_prep"]["hclust"]["usage"]:
            logger.info(
                f"	Applying hierarchical clustering with the following params {config['plotting_prep']['hclust']['params']}"
            )
            cluster = AgglomerativeClustering(
                **return_filtered_params(config=config["plotting_prep"]["hclust"])
            )
            assert (
                "X_pca" in adata.obsm
            ), f"There's no X_pca component in adata.obsm {adata=}"
            X_pca = adata.obsm["X_pca"]

            adata.obs[
                "hclust_"
                + str(config["plotting_prep"]["hclust"]["params"]["n_clusters"])
            ] = cluster.fit_predict(X_pca).astype(str)

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
        # save the counts to a separate object for later, we need the normalized counts in raw for DEG dete.Save raw data before preprocessing values and further filtering
        adata.layers["preprocessed_counts"] = adata.X.copy()

        logger.info(f"Current adata.X shape after preprocessing: {adata.X.shape}")
        logger.info(
            f"Current adata.raw.X shape after preprocessing: \n {adata.raw.X.shape = }"
        )

        logger.info(log_adataX(adata=adata, raw=False))

        logger.info(
            log_adataX(adata=adata, layer="preprocessed_counts", raw=True, step=step)
        )

        logger.info(log_adataX(adata=adata, layer="raw_counts", raw=True, step=step))

        logger.info(log_adataX(adata=adata, layer="raw_counts", step=step))

        save_processed_adata(
            STNavCorePipeline=self, name=config["save_as"], adata=adata
        )
        return adata

    def DEG(self):

        step = "DEG"
        config = self.config[self.data_type][step]
        adata_path = self.adata_dict[self.data_type][config["adata_to_use"]]
        adata = sc.read_h5ad(adata_path)

        logger.info(
            f"Running DEG for {self.data_type} with '{config['adata_to_use']}' adata file."
        )

        # TODO: have an assert that verifies that the data is not raw. It has to be lognormalized instead of raw data counts to run DEG
        # assert (), f"Adata for {config['adata_to_use']} seems to be raw counts Use a lognormalized version instead
        adata_for_DEG = adata.raw.to_adata()

        # For DGE analysis we would like to run with all genes, on normalized values, so we will have to revert back to the raw matrix. In case you have raw counts in the matrix you also have to renormalize and logtransform. In this case, raw already has the normalized and log data for scRNA
        if (
            adata_for_DEG.n_vars
            == self.config[self.data_type]["preprocessing"]["highly_variable_genes"][
                "params"
            ]["n_top_genes"]
        ):
            logger.warning(
                f"DEG will be run on {adata_for_DEG.n_vars}, but DEG is expected to run on all lognormalized genes. Make sure the AnnData you're using for DEG has not been subsetted by highly_variable_genes. n_top_genes = {self.config[self.data_type]['preprocessing']['highly_variable_genes']['params']['n_top_genes']}"
            )

        # Rank genes groups - Differential Expression of Genes (DEG)
        if config["rank_genes_groups"]["usage"]:
            sc.tl.rank_genes_groups(
                **return_filtered_params(
                    config=config["rank_genes_groups"], adata=adata_for_DEG
                )
            )

            # Add this one just to make sure we have ranked genes on the subset with the highly variable genes as well. Mainly for plotting reasons.
            sc.tl.rank_genes_groups(
                **return_filtered_params(
                    config=config["rank_genes_groups"], adata=adata
                )
            )

            save_processed_adata(
                STNavCorePipeline=self, name=config["save_as"], adata=adata
            )
            save_processed_adata(
                STNavCorePipeline=self, name="DEG_adata", adata=adata_for_DEG
            )

        # Filter rank genes groups
        if config["filter_rank_genes_groups"]["usage"]:
            sc.tl.filter_rank_genes_groups(
                **return_filtered_params(
                    config=config["filter_rank_genes_groups"], adata=adata_for_DEG
                )
            )
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
            ranked_genes_list = sc.get.rank_genes_groups_df(
                **return_filtered_params(
                    config=config["rank_genes_groups_df"], adata=adata_for_DEG
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
                self.saving_path
                + "\\"
                + self.data_type
                + "\\"
                + "Files"
                + "\\"
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
            gene_set_dict = {}
            gsea_dataframes = {}

            # Handle manual gene sets
            if config_gsea["gene_sets"]["manual_sets"]["usage"]:
                gene_set_dict["manual_sets"] = config_gsea["gene_sets"]["manual_sets"][
                    "sets"
                ]
                logger.info(
                    f"Adding manual gene sets.\n {gene_set_dict['manual_sets']}"
                )

            # Handle API gene sets
            if config_gsea["gene_sets"]["api_sets"]["usage"]:
                gene_set_dict["api_sets"] = [
                    ontology
                    for ontology, boolean in config_gsea["gene_sets"]["api_sets"][
                        "sets"
                    ].items()
                    if boolean
                ]

                logger.info(f"Adding API gene sets.\n {gene_set_dict['api_sets']}")

            # Iterate over gene sets
            enrichr_list = []
            prerank_list = []
            gsea_list = []

            for set_name, gene_set_list_or_dict in gene_set_dict.items():
                gene_set_names = gp.get_library_name(organism="human")
                if set_name == "manual_sets":
                    gene_set_list = {}
                    for gene_set_name, gene_set_dict in gene_set_list_or_dict.items():
                        gene_set_list[gene_set_name] = list(gene_set_dict.keys())
                else:
                    gene_set_list = gene_set_list_or_dict
                # Setdefault is overriding data... I need to save the "set_name" and pass it as parameter to the gsea_dataframes.setdefault("set_name...", res).
                if config_gsea["stratify_by_group"]:
                    if config_gsea["enrichr"]["usage"]:
                        _enrichr_sub = run_enrichr(
                            gene_set_list=gene_set_list,
                            ranked_genes_list=ranked_genes_list,
                            config_gsea=config_gsea,
                            data_type=self.data_type,
                            set_name=set_name,
                            group_bool=True,
                        )
                        enrichr_list.append(_enrichr_sub)

                    if config_gsea["prerank"]["usage"]:
                        _prerank_sub = run_prerank(
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
                        _gsea_sub = run_gsea(
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
                    _enrichr = run_enrichr(
                        gene_set_list=gene_set_list,
                        ranked_genes_list=ranked_genes_list,
                        config_gsea=config_gsea,
                        data_type=self.data_type,
                        set_name=set_name,
                        group_bool=False,
                    )
                    enrichr_list.append(_enrichr)

                if config_gsea["prerank"]["usage"]:
                    _prerank = run_prerank(
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
                    _gsea = run_gsea(
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
                gsea_dataframes["enrichr"] = pd.concat(enrichr_list).reset_index()
            if len(prerank_list) > 0:
                gsea_dataframes["prerank"] = pd.concat(prerank_list).reset_index()
            if len(gsea_list) > 0:
                gsea_dataframes["gsea"] = pd.concat(gsea_list).reset_index()

            if (
                self.data_type == "ST"
            ):  # If its manual set and stratify by group is true...
                df1 = gsea_dataframes["enrichr"][
                    gsea_dataframes["enrichr"]["set_name"] == "manual_sets"
                ][["Genes", "Term", "group"]]

                # TODO: check if AR can be present if we run GARD with all the data instead of the preprocessed adata
                df2 = self.adata_dict[self.data_type]["DEG_adata"].obs

                # First, merge the dataframes on the 'group' column
                merged_df = pd.merge(
                    df2.reset_index(),
                    df1,
                    left_on="leiden_clusters",
                    right_on="group",
                    how="left",
                )

                # Pivot the dataframe to get the desired structure
                result_df = merged_df.pivot(
                    index="index", columns="Term", values="Genes"
                ).fillna(0)
                result_df.index.name = None

                # Remove a nan column that is created by the result of merging.
                result_df.drop(result_df.columns[0], axis=1, inplace=True)

                final_df = pd.merge(df2, result_df, left_index=True, right_index=True)
                final_df.fillna(0, inplace=True)

                GARD_final_df = GARD(
                    final_df, config_gsea["gene_sets"]["manual_sets"]["sets"]
                )

                adata.obs = GARD_final_df

                save_processed_adata(
                    STNavCorePipeline=self,
                    name="preprocessed_adata_GARD",
                    adata=adata,
                )
                GARD_final_df.to_excel(
                    f"{self.saving_path}\\{self.data_type}\\Files\\GARD_score.xlsx"
                )

            with pd.ExcelWriter(
                f"{self.saving_path}\\{self.data_type}\\Files\\{self.data_type}_GSEA_{date}.xlsx"
            ) as writer:
                for sheet_name, gsea_df in gsea_dataframes.items():
                    gsea_df.to_excel(writer, sheet_name=sheet_name, index=False)

        return adata
