# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
import gseapy as gp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import torch
import importlib
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import scarches as sca
import anndata as ad
from scipy import sparse
import gdown
import gzip
import shutil
import urllib.request
import squidpy as sq
from datetime import datetime
from gseapy import barplot, dotplot

# Training a model to predict proportions on spatial data using scRNA seq as reference
import scvi
from loguru import logger

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scipy.sparse import csr_matrix
from src.utils.decorators import logger_wraps

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


@logger_wraps()
def unnormalize(adata: an.AnnData, count_col: str):
    logger.info("Performing unnormalization of adata.X count data.")
    E = adata.X.expm1()
    n = np.sum(E, 1)
    print(np.min(n), np.max(n))
    factor = np.mean(n)
    assert count_col in adata.obs.columns, f"{count_col} is not in the adata.obs!"
    nC = np.array(adata.obs[count_col])  # true number of counts
    scaleF = nC / factor
    # scaleF = scaleF.reshape(-1, 1)
    # C = np.multiply(E)
    C = csr_matrix(E).multiply(scaleF[:, None])
    C = C.tocsr()
    # C = E * scaleF[:, None]
    C = C.astype("int")
    # C = np.nan_to_num(C, copy=False)
    adata.X = C

    return adata


def return_gene_intersect_subsets(adata_dict):
    logger.info("Creating gene subsets.")

    sc_adata = adata_dict["scRNA"]["preprocessed_data"]
    sc_adata = adata_dict["ST"]["preprocessed_data"]

    intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    st_adata_subset = st_adata[:, intersect].copy()
    sc_adata_subset = sc_adata[:, intersect].copy()

    return intersect, st_adata_subset, sc_adata_subset


def fix_write_h5ad(adata):

    for cat in adata.obs.columns:
        if isinstance(adata.obs[cat].values, pd.Categorical):
            pass
        elif pd.api.types.is_float_dtype(adata.obs[cat]):
            pass
        else:
            adata.obs[cat] = adata.obs[cat].astype(str)
    # Fixing the known error on _index reserved naming convention. Workaround is as follows:
    adata.__dict__["_raw"].__dict__["_var"] = (
        adata.__dict__["_raw"]
        .__dict__["_var"]
        .rename(columns={"_index": "features"}, inplace=True)
    )
    del adata.raw

    return adata


def limit_to_n_cell_counts(adata: sc.AnnData, cell_class_col: str):
    target_cells = 200

    adata_2 = [
        adata[adata.obs.cell_class_col == clust]
        for clust in adata.obs.cell_class_col.cat.categories
        if clust != "nan"
    ]

    for dat in adata_2:
        print(dat)
        if dat.n_obs > target_cells:
            sc.pp.subsample(dat, n_obs=target_cells)

    adata = adata_2[0].concatenate(*adata_2[1:])
    adata.obs.cell_class_col.value_counts()

    return adata


import inspect


def SpatialDM_wrapper(
    adata,
    l: float,
    cutoff: float,
    single_cell: bool,
    species: str,
    min_cell: int,
    n_perm: int,
    specified_ind_global: list,
    method_global: str,
    specified_ind_local: list,
    method_local: str,
    nproc: int,
    method_sig_pairs: str,
    fdr_sig_pairs: bool,
    threshold_sig_pairs: float,
    method_sig_spots: str,
    fdr_sig_spots: bool,
    threshold_sig_spots,
):
    import spatialdm as sdm

    adata.obs = adata.obs[
        adata.obsm["deconvolution"].columns
    ]  # Subset to just have the deconvoluted columns

    adata.raw = ad.AnnData(adata.layers["raw_counts"])  # Add raw as actual raw

    adata.var.index = adata.var.index.str.upper()  # Convert genes to all upper case

    sdm.weight_matrix(
        adata, l=l, cutoff=cutoff, single_cell=single_cell
    )  # weight_matrix by rbf kernel
    sdm.extract_lr(
        adata, species=species, min_cell=min_cell
    )  # find overlapping LRs from CellChatDB
    sdm.spatialdm_global(
        adata,
        n_perm,
        specified_ind=specified_ind_global,
        method=method_global,
        nproc=nproc,
    )  # global Moran selection
    sdm.sig_pairs(
        adata,
        method=method_sig_pairs,
        fdr=fdr_sig_pairs,
        threshold=threshold_sig_pairs,
    )  # select significant pairs
    sdm.spatialdm_local(
        adata,
        n_perm=n_perm,
        method=method_local,
        specified_ind=specified_ind_local,
        nproc=nproc,
    )  # local spot selection
    sdm.sig_spots(
        adata,
        method=method_sig_spots,
        fdr=fdr_sig_spots,
        threshold=threshold_sig_spots,
    )  # significant local spots

    adata.uns["global_res"].sort_values(by="fdr")

    return adata


def return_filtered_params(config, adata=None):
    # TODO: change it so that we don't need to establish the adata parameter. Automatically add it using setdefault in the if statement down below.

    params = config["params"]
    len_func_str = len(config["func_str"].split("."))
    func_str = config["func_str"]

    # TODO: try to dynamically import each module so there's no need to load all the necessary model in the utils (avoid redundant imports) with importlib
    if len_func_str == 3:
        class_eval_str, module_str, func_str = func_str.split(".")
        class_eval = eval(class_eval_str)
        module = getattr(class_eval, module_str)
        func = getattr(module, func_str)
    if len_func_str == 2:
        class_eval_str, func_str = func_str.split(".")
        class_eval = eval(class_eval_str)
        func = getattr(class_eval, func_str)
    if len_func_str == 1:
        func = eval(func_str)

    valid_arguments = inspect.signature(func).parameters.keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_arguments}

    # If Adata is None, then simply return filtered_params. Else, add the adata, data or other data type to the filtered_params dictionary
    if any(hasattr(adata, atr) for atr in ["X", "obs", "var"]):
        if "adata" in filtered_params.keys():
            filtered_params["adata"] = eval(filtered_params["adata"])
        if "data" in filtered_params.keys():
            filtered_params["data"] = eval(filtered_params["data"])
        if "rnk" in filtered_params.keys():
            filtered_params["rnk"] = eval(filtered_params["rnk"])
        if "X" in filtered_params.keys():
            filtered_params["X"] = eval(filtered_params["X"])
        return filtered_params
    else:
        return filtered_params


def log_adataX(adata, layer: str = None, step: str = None, raw: bool = None):
    if raw:
        return f"\n After {step} on the {layer} layer, the 3 first examples (cells for scRNA or spots for ST): \n 1 - {adata.raw.X[0,:].sum() = } \n 2 - {adata.raw.X[1,:].sum() = } \n 3 - {adata.raw.X[2,:].sum() = }"
    elif raw == False:
        return f"\n After preprocessing with highly variable genes, the 3 first examples (cells for scRNA or spots for ST): \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    elif isinstance(layer, str):
        return f"\n After {step} on the {layer} layer, the 3 first examples (cells for scRNA or spots for ST): \n 1 - {adata.layers[layer][0,:].sum() = } \n 2 - {adata.layers[layer][1,:].sum() = } \n 3 - {adata.layers[layer][2,:].sum() = }"


@logger_wraps()
def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (np.broadcast_to(True, adata.n_obs), (cat.codes, np.arange(adata.n_obs))),
        shape=(len(cat.categories), adata.n_obs),
    )

    return ad.AnnData(
        indicator @ adata.X, var=adata.var, obs=pd.DataFrame(index=cat.categories)
    )


def download_atlas():
    ref_model_dir_prefix = (
        "./HCLA"  # directory in which to store the reference model directory
    )
    surgery_model_dir_prefix = (
        "./HCLA"  # directory in which to store the surgery model directory
    )
    path_reference_emb = (
        "./HCLA/HLCA_emb_and_metadata.h5ad"  # path to reference embedding to be created
    )
    path_query_data = "./HCLA/HLCA_query.h5ad"  # input test query data
    # don't change the following paths:
    ref_model_dir = os.path.join(
        ref_model_dir_prefix, "HLCA_reference_model"
    )  # don't change this
    surgery_model_dir = os.path.join(
        surgery_model_dir_prefix, "surgery_model"
    )  # don't change this

    # url = "https://zenodo.org/record/7599104/files/HLCA_reference_model.zip"
    # output = "HLCA_reference_model.zip"
    # gdown.download(url, output, quiet=False)
    # shutil.unpack_archive("HLCA_reference_model.zip", extract_dir=ref_model_dir_prefix)
    # os.remove(output)

    # url = "https://zenodo.org/record/7599104/files/HLCA_full_v1.1_emb.h5ad"
    # output = path_reference_emb
    # gdown.download(url, output, quiet=False)

    adata_ref = sc.read_h5ad(path_reference_emb)
    adata_query_unprep = sc.read_h5ad(path_query_data)
    adata_query_unprep.X = sparse.csr_matrix(adata_query_unprep.X)
    del adata_query_unprep.obsm
    del adata_query_unprep.varm
    adata_query_unprep.X[:10, :30].toarray()
    ref_model_features = pd.read_csv(
        os.path.join(ref_model_dir, "var_names.csv"), header=None
    )

    adata_query = sca.models.SCANVI.prepare_query_anndata(
        adata=adata_query_unprep, reference_model=ref_model_dir, inplace=False
    )
    surgery_model = sca.models.SCANVI.load_query_data(
        adata_query,
        ref_model_dir,
        freeze_dropout=True,
    )

    surgery_model.registry_["setup_args"]
    adata_query.obs["dataset"] = "Delorey_batch_1"
    adata_query.obs["scanvi_label"] = "unlabeled"
    surgery_model = sca.models.SCANVI.load_query_data(
        adata_query,
        ref_model_dir,
        freeze_dropout=True,
    )

    surgery_model.train(max_epochs=surgery_epochs, **early_stopping_kwargs_surgery)
    surgery_model.save(surgery_model_dir, overwrite=True)

    surgery_model = sca.models.SCANVI.load(
        surgery_model_dir, adata_query
    )  # if already trained


@logger_wraps()
def ensembleID_to_GeneSym_mapping(
    gene_mapping_path, adata_query_unprep, gene_name_column_name: str = "gene_names"
):
    logger.info(
        f"Mapping Ensemble IDs to gene names using the following file: {gene_mapping_path}"
    )
    gene_id_to_gene_name_df = pd.read_csv(gene_mapping_path, index_col=0)
    # if gene names are in .var.index:
    adata_query_unprep.var["gene_names"] = adata_query_unprep.var.index

    n_overlap = (
        adata_query_unprep.var[gene_name_column_name]
        .isin(gene_id_to_gene_name_df.gene_symbol)
        .sum()
    )
    n_genes_model = gene_id_to_gene_name_df.shape[0]
    print(
        f"Number of model input genes detected: {n_overlap} out of {n_genes_model} ({round(n_overlap/n_genes_model*100)}%)"
    )
    adata_query_unprep = adata_query_unprep[
        :,
        adata_query_unprep.var[gene_name_column_name].isin(
            gene_id_to_gene_name_df.gene_symbol
        ),
    ].copy()  # subset your data to genes used in the reference model
    adata_query_unprep.var.index = adata_query_unprep.var[gene_name_column_name].map(
        dict(zip(gene_id_to_gene_name_df.gene_symbol, gene_id_to_gene_name_df.index))
    )  # add gene ids for the gene names, and store in .var.index
    # remove index name to prevent bugs later on
    adata_query_unprep.var.index.name = None
    adata_query_unprep.var["gene_ids"] = adata_query_unprep.var.index
    adata_query_unprep = sum_by(
        adata_query_unprep.transpose(), col="gene_ids"
    ).transpose()

    adata_query_unprep.var = adata_query_unprep.var.join(
        gene_id_to_gene_name_df
    ).rename(columns={"gene_symbol": "gene_names"})

    return adata_query_unprep


def run_enrichr(
    gene_set_list,
    ranked_genes_list,
    config_gsea,
    data_type,
    set_name,
    group_bool,
):

    if group_bool:
        df_list = []
        groups = list(set(ranked_genes_list.group))
        for group in groups:
            logger.info(f"Running stratified enrichr with {set_name} by {group = }.")
            # Filter the glist to have only the current group
            ranked_genes_list_per_group = ranked_genes_list[
                ranked_genes_list["group"] == group
            ]

            # For each group, create a new set of enrichr parameters
            enrichr_params = {}

            # Run Enrichr
            enrichr_params = return_filtered_params(
                config=config_gsea["enrichr"], adata=ranked_genes_list_per_group
            )

            # Save key parameters to this dictionary
            enrichr_params.setdefault("gene_sets", gene_set_list)
            enrichr_params.setdefault(
                "gene_list", ranked_genes_list_per_group.names.tolist()
            )
            try:
                enr_res = gp.enrichr(**enrichr_params)
                enr_df = enr_res.results
                enr_df["set_name"] = set_name
                enr_df["group"] = group
                df_list.append(enr_df)

            except Exception as e:
                logger.error(f"No genes matched - {e}")
    else:
        df_list = []
        logger.info(f"Running enrichr with {set_name} with all genes.")
        # For each group, create a new set of enrichr parameters
        enrichr_params = {}

        # Run Enrichr
        enrichr_params = return_filtered_params(
            config=config_gsea["enrichr"], adata=ranked_genes_list
        )

        # Save key parameters to this dictionary
        enrichr_params.setdefault("gene_sets", gene_set_list)
        enrichr_params.setdefault("gene_list", ranked_genes_list.names.tolist())
        enr_res = gp.enrichr(**enrichr_params)
        enr_df = enr_res.results
        enr_df["set_name"] = set_name
        enr_df["group"] = "All genes"
        df_list.append(enr_df)

    try:
        gp.barplot(enr_res.res2d, title=data_type)
    except Exception as e:
        logger.warning(f"Couldn't apply gp.barplot - {e}")

    df = pd.concat(df_list)
    return df


def run_prerank(
    gene_set_list,
    ranked_genes_list,
    config_gsea,
    data_type,
    set_name,
    group_bool,
    saving_path,
):
    """
    Besides the enrichment using hypergeometric test, we can also perform gene set enrichment analysis (GSEA), which scores ranked genes list (usually based on fold changes) and computes permutation test to check if a particular gene set is more present in the Up-regulated genes, among the DOWN_regulated genes or not differentially regulated.

    We need a table with all DEGs and their log foldchanges. However, many lowly expressed genes will have high foldchanges and just contribue noise, so also filter for expression in enough cells - This is being done in the quality_control method.
    """

    if group_bool:
        df_list = []
        groups = list(set(ranked_genes_list.group))
        for group in groups:
            logger.info(f"Running stratified prerank with {set_name} by {group = }.")
            # Filter the glist to have only the current group
            ranked_genes_list_per_group = ranked_genes_list[
                ranked_genes_list["group"] == group
            ]
            # Create new glist with correct input for gp.prerank
            glist_prerank = ranked_genes_list_per_group[["names", "logfoldchanges"]]
            # Keep just first from duplicates (use only the duplicated ID with highest values of logfoldchanges which have already been sorted)
            glist_prerank.drop_duplicates(subset=["names"], keep="first", inplace=True)
            glist_prerank.set_index("names", inplace=True)

            config_gsea["prerank"]["params"]["gene_sets"] = gene_set_list
            try:
                res = gp.prerank(
                    **return_filtered_params(
                        config=config_gsea["prerank"], adata=glist_prerank
                    )
                )
                res_df = res.res2d
                res_df["set_name"] = set_name
                res_df["group"] = group
                df_list.append(res_df)
                gp.gseaplot(
                    rank_metric=res.ranking,
                    term=res.res2d.Term[0:5],
                    ofname=f"{saving_path}\\{data_type}\\Files\\{data_type}_gseaplot_{group}_{date}.png",
                    **res.results[res.res2d.Term[0]],
                )
            except Exception as e:
                logger.error(f"No genes matched - {e}")
    else:
        logger.info(f"Running prerank with {set_name} with all genes.")
        df_list = []
        # Create new glist with correct input for gp.prerank
        glist_prerank = ranked_genes_list[["names", "logfoldchanges"]]
        # Keep just first from duplicates (use only the duplicated ID with highest values of logfoldchanges which have already been sorted)
        glist_prerank.drop_duplicates(subset=["names"], keep="first", inplace=True)
        glist_prerank.set_index("names", inplace=True)

        config_gsea["prerank"]["params"]["gene_sets"] = gene_set_list
        res = gp.prerank(
            **return_filtered_params(config=config_gsea["prerank"], adata=glist_prerank)
        )
        res_df = res.res2d
        res_df["set_name"] = set_name
        res_df["group"] = "All genes"
        df_list.append(res_df)

        gp.gseaplot(
            rank_metric=res.ranking,
            term=res.res2d.Term[0:5],
            ofname=f"{saving_path}\\{data_type}\\Files\\{data_type}_gseaplot_AllGenes_{date}.png",
            **res.results[res.res2d.Term[0]],
        )

    df = pd.concat(df_list)
    return df


def run_gsea(
    gene_set_list,
    ranked_genes_list,
    config_gsea,
    data_type,
    set_name,
    group_bool,
    saving_path,
):

    if group_bool:
        df_list = []
        groups = list(set(ranked_genes_list.group))
        for group in groups:
            logger.info(f"Running stratified gsea with {set_name} by {group = }.")
            # Filter the glist to have only the current group
            ranked_genes_list_per_group = ranked_genes_list[
                ranked_genes_list["group"] == group
            ]

            config_gsea["gsea"]["params"]["gene_sets"] = gene_set_list
            try:
                res = gp.gsea(
                    **return_filtered_params(
                        config=config_gsea["gsea"], adata=ranked_genes_list_per_group
                    )
                )
                res_df = res.res2d
                res_df["set_name"] = set_name
                res_df["group"] = group
                df_list.append(res_df)
                gp.gseaplot(
                    rank_metric=res.ranking,
                    term=res.res2d.Term[0],
                    ofname=f"{saving_path}\\{data_type}\\Files\\{data_type}_gseaplot_{group}_{date}.png",
                    **res.results[res.res2d.Term[0]],
                )
            except Exception as e:
                logger.error(f"No genes matched - {e}")
    else:
        logger.info(f"Running gsea with {set_name} with all genes.")
        df_list = []

        config_gsea["gsea"]["params"]["gene_sets"] = gene_set_list
        res = gp.gsea(
            **return_filtered_params(
                config=config_gsea["gsea"], adata=ranked_genes_list
            )
        )
        res_df = res.res2d
        res_df["set_name"] = set_name
        res_df["group"] = "All genes"
        df_list.append(res_df)

        gp.gseaplot(
            rank_metric=res.ranking,
            term=res.res2d.Term[0],
            ofname=f"{saving_path}\\{data_type}\\Files\\{data_type}_gseaplot_AllGenes_{date}.png",
            **res.results[res.res2d.Term[0]],
        )

    df = pd.concat(df_list)
    return df


def GARD(df, rank_dictionary):
    import copy

    logger.info(f"Running GARD.")
    df_subset = df[list(rank_dictionary.keys())]
    RSI_list = []
    for index, row in df_subset.iterrows():
        RIS = 0
        temp_rank_dictionary = copy.deepcopy(rank_dictionary)
        for column, genes in row.items():
            if isinstance(
                genes, int
            ):  # If gene list is 0 and not a list of genes then put the whole set with rank as 0 (positive or negative)
                for g_name, rank in temp_rank_dictionary[column].items():
                    temp_rank_dictionary[column][g_name][
                        "rank"
                    ] = 0  # TODO: change to NA instead of integer
            if genes != 0 and column in temp_rank_dictionary:
                for dict_gene in temp_rank_dictionary[column].keys():
                    if dict_gene not in genes:
                        temp_rank_dictionary[column][dict_gene]["rank"] = 0

        RSI = (
            (
                -0.0098009
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["AR"][
                    "rank"
                ]
            )
            + (
                0.0128283
                * temp_rank_dictionary["Positive Radiation Sensitivity (RR)"]["JUN"][
                    "rank"
                ]
            )
            + (
                0.0254552
                * temp_rank_dictionary["Positive Radiation Sensitivity (RR)"]["STAT1"][
                    "rank"
                ]
            )
            + (
                -0.0017589
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["PRKCB"][
                    "rank"
                ]
            )
            + (
                -0.0038171
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["RELA"][
                    "rank"
                ]
            )
            + (
                -0.1070213
                * temp_rank_dictionary["Positive Radiation Sensitivity (RR)"]["ABL1"][
                    "rank"
                ]
            )
            + (
                -0.0002509
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["SUMO1"][
                    "rank"
                ]
            )
            + (
                -0.0092431
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["PAK2"][
                    "rank"
                ]
            )
            + (
                -0.0204469
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["HDAC1"][
                    "rank"
                ]
            )
            + (
                0.0441683
                * temp_rank_dictionary["Negative Radiation Sensitivity (RS)"]["IRF1"][
                    "rank"
                ]
            )
        )
        # df.at[index, "RSI"] = RSI
        RSI_list.append(RSI)

    df["RSI"] = RSI_list

    return df