# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from datetime import datetime
import re
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


def perform_QC(
    adata,
    min_cells=25,
    min_genes=200,
    log1p=True,
    use_raw=False,
    pct_counts_mt=25,
    qc_vars=["Mt", "Ribo", "Hb"],
):
    adata.var_names_make_unique()

    # Filter genes by counts
    logger.info(f"Applying filtering genes.")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(
        f"	After filtering genes: {adata.n_obs} observations (cells if scRNA, spots if ST) x {adata.n_vars} genes."
    )

    # Filter cells by counts
    logger.info(f"Applying filtering cells.")
    sc.pp.filter_cells(adata, min_genes=min_genes)

    logger.info(
        f"	After filtering cells: {adata.n_obs= } observations x {adata.n_vars= } cells. "
    )
    logger.info("Running quality control.")
    adata_original = adata.copy()
    # mitochondrial genes
    adata.var["Mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["Ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["Hb"] = adata.var_names.str.contains(
        ("^HB[^(p)]")
    )  # adata.var_names.str.contains('^Hb.*-')

    sc.pp.calculate_qc_metrics(
        adata=adata,
        qc_vars=qc_vars,
        log1p=log1p,
        use_raw=use_raw,
        percent_top=[20],
        inplace=True,
    )

    adata = adata[adata.obs["pct_counts_Mt"] < pct_counts_mt]

    # Remove genes that still passed the previous condition
    genes_to_remove_pattern = re.compile("|".join(map(re.escape, qc_vars)))

    genes_to_remove = adata.var_names.str.contains(genes_to_remove_pattern)
    keep = np.invert(genes_to_remove)
    adata = adata[:, keep]
    print(
        f"{sum(genes_to_remove)} genes removed. Original size was {adata_original.n_obs} cells and {adata_original.n_vars} genes. New size is {adata.n_obs} cells and {adata.n_vars} genes"
    )

    return adata
