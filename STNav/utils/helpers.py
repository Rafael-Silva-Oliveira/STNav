# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import inspect
import os

# Unnormalize data
from datetime import datetime

import anndata as an
import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanorama
import scanpy as sc

# import scarches as sca
import seaborn as sns

# import spatialdm as sdm
import squidpy as sq
from gseapy.plot import gseaplot

# import stlearn as st

# Training a model to predict proportions on spatial data using scRNA seq as reference
from loguru import logger
from scipy import sparse
from scipy.sparse import csr_matrix

# from scvi.external import RNAStereoscope, SpatialStereoscope
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
from STNav.utils.decorators import logger_wraps

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

from STNav.utils.decorators import logger_wraps, pass_STNavCore_params


@pass_STNavCore_params
def return_from_checkpoint(
    STNavCorePipeline,
    config_params,
    checkpoint_step: str,
    method_name: str,
):

    checkpoint_boolean = STNavCorePipeline.config[STNavCorePipeline.data_type][
        checkpoint_step
    ][method_name]["checkpoint"]["usage"]

    if checkpoint_boolean:
        path_to_check = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
            config_params["save_as"]
        ]
        logger.info(f"Looking for checkpoint {checkpoint_step}.")

        current_pipeline_run = f"{STNavCorePipeline.saving_path}".split(os.sep)[-1]
        look_for_checkpoint_run = extract_pipeline_run(path_to_check)

        if current_pipeline_run == look_for_checkpoint_run:
            logger.info(f"Checkpoint not found for {checkpoint_step}.")
            return False
        else:
            logger.warning(
                f"Checkpoint found.\n\nThe following step '{checkpoint_step}' is being returned from the following pipeline run: '{look_for_checkpoint_run}."
            )
            return True
    else:
        logger.info(f"Checkpoint is not being used for {checkpoint_step}.")
        return False


def extract_pipeline_run(path):
    # Split the path into a list of directories
    dirs = path.split(os.sep)

    # Find the index of the first directory that starts with "PipelineRun"
    index = next(
        (i for i, dir in enumerate(dirs) if dir.startswith("PipelineRun")), None
    )

    # If a directory starting with "PipelineRun" was found, return it
    if index is not None:
        return dirs[index]

    # If no directory starting with "PipelineRun" was found, return None
    return None


@pass_STNavCore_params
def save_processed_adata(
    STNavCorePipeline,
    name,
    adata,
    fix_write: bool = None,
    data_type=None,
    checkpoint_step: str = None,
):  # noqa: F811
    # Function that saves the adata file being passed as well as adding the path to the adata_dict for later use

    if data_type is not None:
        STNavCorePipeline.data_type = data_type

    adata_name = name

    if isinstance(adata, an.AnnData):

        if checkpoint_step is not None:  # using checkpoint pipeline run
            checkpoint = STNavCorePipeline.config[STNavCorePipeline.data_type][
                checkpoint_step
            ]["checkpoint"]["usage"]
            if checkpoint:
                checkpoint_name = STNavCorePipeline.config[STNavCorePipeline.data_type][
                    checkpoint_step
                ]["checkpoint"]["pipeline_run"]
                # Split the path into a list of directories
                dirs = f"{STNavCorePipeline.saving_path}".split(os.sep)

                # Replace the last directory in the list with the checkpoint_name
                dirs[-1] = checkpoint_name

                # Join the directories back into a path
                new_path = os.sep.join(dirs)
                # Replace the pipeline run name with the checkpoint name

                final_path = (
                    f"{new_path}"
                    + "/"
                    + f"{STNavCorePipeline.data_type}/Files"
                    + "/"
                    + f"{adata_name}.h5ad"
                )

                logger.info(f"Saving checkpoint path to adata_dict: {final_path}.")
                STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    adata_name
                ] = final_path
            else:
                logger.info(
                    f"Saving {STNavCorePipeline.data_type} data type with the name {name}.h5ad file."
                )
                # Saving file after processing
                adata_final = adata.copy()
                try:
                    del adata_final.uns["rank_genes_groups"]

                except Exception as e:
                    pass

                try:
                    del adata_final.uns["rank_genes_groups_filtered"]

                except Exception as e:
                    pass

                final_path = (
                    f"{STNavCorePipeline.saving_path}"
                    + "/"
                    + f"{STNavCorePipeline.data_type}/Files"
                    + "/"
                    + f"{adata_name}.h5ad"
                )
                try:
                    if not isinstance(adata_final.obsm["spatial"], str):
                        adata_final.obsm["spatial"] = adata_final.obsm[
                            "spatial"
                        ].astype(str)
                        adata_final.uns["spatial"] = adata_final.uns["spatial"].astype(
                            str
                        )

                except Exception as e:
                    pass

                if fix_write:
                    try:
                        adata_final = fix_write_h5ad(adata=adata_final)
                    except Exception as e:
                        logger.warning(f"fix_write_h5ad failed {e}")
                    if "lrfeatures" in adata_final.uns.keys():
                        adata_final.uns["lrfeatures"] = adata_final.uns[
                            "lrfeatures"
                        ].astype(float)
                    adata_final.write_h5ad(final_path)
                # Save the path to the adata_dict for later use
                if (
                    adata_name
                    in STNavCorePipeline.adata_dict[STNavCorePipeline.data_type]
                ):
                    logger.warning(
                        f"Warning: {adata_name} path is already in the dictionary. The path will be overwritten."
                    )

                logger.info(
                    f"Saving data to adata_dict as '{adata_name}' for {STNavCorePipeline.data_type} data type."
                )

                STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    adata_name
                ] = final_path
        else:  # saving data from scratch
            logger.info(
                f"Saving {STNavCorePipeline.data_type} data type with the name {name}.h5ad file."
            )
            # Saving file after processing
            adata_final = adata.copy()
            try:
                del adata_final.uns["rank_genes_groups"]

            except Exception as e:
                pass

            try:
                del adata_final.uns["rank_genes_groups_filtered"]

            except Exception as e:
                pass

            final_path = (
                f"{STNavCorePipeline.saving_path}"
                + "/"
                + f"{STNavCorePipeline.data_type}/Files"
                + "/"
                + f"{adata_name}.h5ad"
            )
            try:
                adata_final.obs["in_tissue"] = adata_final.obs["in_tissue"].astype(str)
                adata_final.obs["array_row"] = adata_final.obs["array_row"].astype(str)
                adata_final.obs["array_col"] = adata_final.obs["array_col"].astype(str)
                (str)
                adata_final.obsm["spatial"] = adata_final.obsm["spatial"].astype(float)

            except Exception as e:
                pass

            if fix_write:
                try:
                    adata_final = fix_write_h5ad(adata=adata_final)
                except Exception as e:
                    logger.warning(f"fix_write_h5ad failed {e}")
                if "lrfeatures" in adata_final.uns.keys():
                    adata_final.uns["lrfeatures"] = adata_final.uns[
                        "lrfeatures"
                    ].astype(float)
                adata_final.write_h5ad(final_path)
            else:
                try:
                    adata_final.write_h5ad(filename=final_path)
                except Exception as e:
                    logger.error(f"Exception occurred saving {adata_name} - {e}")

            # Save the path to the adata_dict for later use
            if adata_name in STNavCorePipeline.adata_dict[STNavCorePipeline.data_type]:
                logger.warning(
                    f"Warning: {adata_name} path is already in the dictionary. The path will be overwritten."
                )

            logger.info(
                f"Saving data to adata_dict as '{adata_name}' for {STNavCorePipeline.data_type} data type."
            )

            STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                adata_name
            ] = final_path

    else:
        logger.warning(f"Adata {adata_name} is not an AnnData object.")

        # Save the path to the adata_dict for later use
        if adata_name in STNavCorePipeline.adata_dict[STNavCorePipeline.data_type]:
            logger.warning(
                f"Warning: {adata_name} path is already in the dictionary. The path will be overwritten."
            )

        logger.info(
            f"Saving data to adata_dict as '{adata_name}' for {STNavCorePipeline.data_type} data type."
        )
        STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][adata_name] = adata


def transform_adata(
    adata: an.AnnData, layer_type, column: str, astype, chunk_size: int
):
    assert isinstance(adata, an.AnnData), "adata must be an instance of anndata.AnnData"
    assert column in adata.obsm, f"{column} not found in adata[{layer_type}]"
    assert astype in [
        int,
        str,
        float,
        np.int64,
        np.float64,
    ], "astype must be int, str, float, np.int64, or np.float64"

    for start in range(0, adata.obsm[column].shape[0], chunk_size):
        end = start + chunk_size
        adata.obsm[column][start:end] = adata.obsm[column][start:end].astype(astype)

    return adata


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
    st_adata = adata_dict["ST"]["preprocessed_data"]

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

    adata.raw = an.AnnData(adata.layers["raw_counts"])  # Add raw as actual raw

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


def stLearn_wrapper(
    adata: an.AnnData,
    min_spots: int,
    distance,
    n_pairs: int,
    spot_mixtures: bool,
    sig_spots: bool,
    cell_prop_cutoff: float,
    p_cutoff: float,
    n_perms: int,
    n_cpus: int,
    verbose: bool,
):
    import stlearn as st

    if "norm" not in adata.layers.keys():
        logger.warning(
            f"norm layer not found in adata.layers. Please make sure to run the normalization step before running stLearn LR analysis. See more: https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_CCI.html."
        )

    adata.X = adata.layers["norm"]
    # TODO: check if adata is just norm and not log norm
    use_label = "cell_type"

    adata = st.convert_scanpy(adata)

    # NOTE: following commented code deprecated as we're no longer using deconvolution approaches. Simpler gene marker approach is now being used instead.
    # adata.obs[f"{use_label}"] = (
    #     adata.obs[adata.obsm["deconvolution"].columns].idxmax(axis=1).astype("category")
    # )
    logger.info(
        f"Running Receptor Ligand Analysis with connectomeDB2020_lit ligand-receptor pairs."
    )
    lrs = st.tl.cci.load_lrs(["connectomeDB2020_lit"], species="human")

    st.tl.cci.run(
        adata=adata,
        lrs=lrs,
        min_spots=min_spots,  # Filter out any LR pairs with no scores for less than min_spots
        distance=distance,  # None defaults to spot+immediate neighbours; distance=0 for within-spot mode
        n_pairs=n_pairs,  # Number of random pairs to generate; low as example, recommend ~10,000
        n_cpus=n_cpus,  # Number of CPUs for parallel. If None, detects & use all available.
        verbose=verbose,
    )

    # st.tl.cci.adj_pvals(adata, correct_axis='spot',
    #                 pval_adj_cutoff=0.05, adj_method='fdr_bh')
    try:
        lr_info = adata.uns["lr_summary"]
        logger.info(f"LR summary: {lr_info}")
    except Exception as e:
        logger.warning(f"Could not retrieve ligand-receptor interactions: {e}")

    logger.info(
        f"Predicting significant CCIs.\n\nWith the establishment of significant areas of LR interaction from stLearn_ligrec, we can now determine the significantly interacting cell types using the label '{use_label}'."
    )

    st.tl.cci.run_cci(
        adata=adata,
        use_label=use_label,
        min_spots=min_spots,  # Minimum number of spots for LR to be tested.
        spot_mixtures=spot_mixtures,  # If True will use the label transfer scores,
        # so spots can have multiple cell types if score>cell_prop_cutoff
        sig_spots=sig_spots,  # Only consider neighbourhoods of spots which had significant LR scores.
        cell_prop_cutoff=cell_prop_cutoff,  # Spot considered to have cell type if score>0.2
        n_perms=n_perms,  # Permutations of cell information to get background, recommend ~1000
        verbose=verbose,
    )

    return adata


def return_filtered_params(config, adata=None):
    # TODO: change it so that we don't need to establish the adata parameter. Automatically add it using setdefault in the if statement down below.

    params = config["params"]
    len_func_str = len(config["func_str"].split("."))
    func_str = config["func_str"]

    # TODO: try to dynamically import each module so there's no need to load all the necessary model in the utils (avoid redundant imports) with importlib
    if len_func_str == 4:
        package_name, class_eval_str, module_str, func_str = func_str.split(".")
        package_eval = eval(package_name)
        class_eval = getattr(package_eval, class_eval_str)
        module = getattr(class_eval, module_str)
        func = getattr(module, func_str)

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
def sum_by(adata: an.AnnData, col: str) -> an.AnnData:
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (np.broadcast_to(True, adata.n_obs), (cat.codes, np.arange(adata.n_obs))),
        shape=(len(cat.categories), adata.n_obs),
    )

    return an.AnnData(
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
    """
    Standard GSEA and GSEA preranked give different results because the genes are ranked differently, and they (by default) use different permutation methods. How did you rank your genes for GSEA Preranked? If you used Log2(FC) then to compare results you can use log2_ratio_of_classes  in standard GSEA and the rankings should be generally similar but we recommend using the default signal to noise ratio if you have more than three samples. Signal to noise ratio includes information about both the magnitude of change and the standard deviation of the sample groups which gives an improved result over log2(FC) in our hands.

    GSEA Preranked, because it doesnt have access to the sample level information has to run in gene_set permutation mode for pValue and FDR calculation. Standard GSEA runs in phenotype permutation mode by default but if you have fewer than 7 samples per group we recommend changing this to gene_set permutation mode as well because it is not possible to generate 1000 distinct permutations from smaller experiments.

    If you can run GSEA in standard mode, you should, if you have enough samples to run phenotype permutation you should. If you cant run phenotype permutation, you should run standard GSEA with gene set permutation. If you dont have enough samples to run signal2noise ratio, then its best to use Preranked with your own ranking metric. Log2(FC) isnt really ideal, sign(log2(fc))*-log10(pValue) seems to work better in users hands, or if youre using DESeq2 you could try the Wald Statistic.

    """
    if group_bool:
        df_list = []
        groups = list(set(ranked_genes_list.group))
        for group in groups:
            logger.info(
                f"Running stratified enrichr for {data_type} with {set_name} stratified by {group = }."
            )
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
        logger.info(f"Running enrichr for {data_type} with {set_name} with all genes.")
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
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.figure(figsize=(20, 20))  # Set the figure size
            plotting_func = gp.barplot(enr_res.res2d, title=data_type)
            plt.savefig("./test.png", bbox_inches="tight")
            plt.close()

    except Exception as e:
        logger.warning(f"Couldn't apply gp.barplot - {e}")

    try:
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.figure(figsize=(20, 20))  # Set the figure size
            gp.dotplot(
                enr_res.res2d.head(), figsize=(3, 5), x="UP", size=3, show_ring=True
            )
            plt.savefig("./test_dotplot.png", bbox_inches="tight")
            plt.close()

    except Exception as e:
        logger.warning(f"Couldn't apply gp.dotplot - {e}")

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
            logger.info(
                f"Running stratified prerank for {data_type} with {set_name} by {group = }."
            )
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
                    ofname=f"{saving_path}/{data_type}/Files/{data_type}_gseaplot_{group}_{date}.png",
                    **res.results[res.res2d.Term[0]],
                )
            except Exception as e:
                logger.error(f"No genes matched - {e}")
    else:
        logger.info(f"Running prerank for {data_type} with {set_name} with all genes.")
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
            ofname=f"{saving_path}/{data_type}/Files/{data_type}_gseaplot_AllGenes_{date}.png",
            **res.results[res.res2d.Term[0]],
        )

    df = pd.concat(df_list)
    return df


def run_gsea(
    adata,
    gene_set_list,
    ranked_genes_list,
    config_gsea,
    data_type,
    set_name,
    group_bool,
    saving_path,
):
    # TODO: check this implementation with phenotype https://gseapy.readthedocs.io/en/latest/singlecell_example.html

    if group_bool:
        df_list = []
        groups = list(set(ranked_genes_list.group))
        for group in groups:
            logger.info(
                f"Running stratified gsea for {data_type} with {set_name} by {group = }."
            )
            # Filter the glist to have only the current group
            adata_by_group = adata[adata.obs["group"] == group]

            config_gsea["gsea"]["params"]["gene_sets"] = gene_set_list
            try:
                res = gp.gsea(
                    **return_filtered_params(
                        config=config_gsea["gsea"], adata=adata_by_group.to_df().T
                    )
                )
                res_df = res.res2d
                res_df["set_name"] = set_name
                res_df["group"] = group
                df_list.append(res_df)
                gp.gseaplot(
                    rank_metric=res.ranking,
                    term=res.res2d.Term[0],
                    ofname=f"{saving_path}/{data_type}/Files/{data_type}_gseaplot_{group}_{date}.png",
                    **res.results[res.res2d.Term[0]],
                )
            except Exception as e:
                logger.error(f"No genes matched - {e}")
    else:
        logger.info(f"Running gsea for {data_type} with {set_name} with all genes.")
        df_list = []

        # TODO: add similar above but stratify the adata to the respective group that contains the cls
        if config_gsea["gsea"]["adata_cls_col_name"] is not None:
            config_gsea["gsea"]["params"]["cls"] = adata.obs[
                config_gsea["gsea"]["adata_cls_col_name"]
            ]

        config_gsea["gsea"]["params"]["gene_sets"] = gene_set_list
        res = gp.gsea(
            **return_filtered_params(config=config_gsea["gsea"], adata=adata.to_df().T)
        )
        res_df = res.res2d
        res_df["set_name"] = set_name
        res_df["group"] = "All genes"
        df_list.append(res_df)

        gp.gseaplot(
            rank_metric=res.ranking,
            term=res.res2d.Term[0],
            ofname=f"{saving_path}/{data_type}/Files/{data_type}_gseaplot_AllGenes_{date}.png",
            **res.results[res.res2d.Term[0]],
        )

    df = pd.concat(df_list)
    return df


def is_normalized(adata):
    """
    Check if the data in an AnnData object is normalized or not.

    Parameters:
    adata : anndata.AnnData
        Annotated data matrix.

    Returns:
    bool
        True if the data is likely normalized, False otherwise.
    """
    import numpy as np

    # TODO: fix this, still showing normalized when it is raw

    # Check the maximum, minimum, and mean value in the matrix
    max_value = np.max(adata.X.data)
    min_value = np.min(adata.X.data)
    mean_value = np.mean(adata.X.data)

    # If the maximum value is very large and the values are integers, it's likely that the matrix contains raw counts
    if max_value > 1000 and np.issubdtype(adata.X.dtype, np.integer):
        logger.info("The data is likely raw counts.")
        return False

    # If the values are floating point numbers and the maximum value is small, it's likely that the matrix has been normalized
    if np.issubdtype(adata.X.dtype, np.floating):
        logger.info("The data is likely normalized.")
        return True

    # If none of the above conditions are met, log an info message and return None
    logger.info("The data type does not match typical raw counts or normalized data.")
    return None


def assert_counts(adata, check_for):
    """
    Assert if the data in an AnnData object is normalized or raw counts.

    Parameters:
    adata : anndata.AnnData
        Annotated data matrix.
    check_for : str
        The type of data to check for. Should be either "raw" or "normalized".

    Raises:
    AssertionError
        If the data does not match the expected type.
    """
    import numpy as np

    # Check the maximum, minimum, and mean value in the matrix
    max_value = np.max(adata.X.data)
    min_value = np.min(adata.X.data)
    mean_value = np.mean(adata.X.data)

    if check_for == "raw":
        assert max_value > 1000 and np.issubdtype(
            adata.X.dtype, np.integer
        ), "The data is not raw counts."
    elif check_for == "normalized":
        assert np.issubdtype(adata.X.dtype, np.floating), "The data is not normalized."
    else:
        raise ValueError(
            "Invalid value for check_for. Should be either 'raw' or 'normalized'."
        )


def convert_column(df, column):
    if df[column].dtype == "object":
        try:
            df[column] = df[column].astype(int)
        except ValueError:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                df[column] = df[column].astype(str)
