# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
from loguru import logger
import re
import gseapy as gp
from gseapy.plot import gseaplot
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import torch
import NaiveDE
import SpatialDE
from scipy import sparse
import scarches as sca
import squidpy as sq
from gseapy import GSEA
from GraphST.utils import clustering
from GraphST import GraphST

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# Training a model to predict proportions on Spatial data using scRNA seq as reference
import scvi
import inspect
import cell2location as c2l
from src.utils.helpers import (
    unnormalize,
    return_filtered_params,
    log_adataX,
    ensembleID_to_GeneSym_mapping,
    run_enrichr,
    run_prerank,
    run_gsea,
    SpatialDM_wrapper,
)

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
from scipy.sparse import csr_matrix
from src.utils.helpers import fix_write_h5ad, GARD

# Unnormalize data
import sys

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
import json
import celltypist
from celltypist import models
from GraphST.utils import project_cell_to_spot
import spatialdm as sdm
import anndata as ad


def perform_celltypist(AnalysisPipeline):

    config = AnalysisPipeline.config[AnalysisPipeline.data_type]["celltypist_surgery"]
    path_backbone = config["path_backbone"]
    path_reference = os.path.join(path_backbone, config["path_query_data"])
    # TODO: fix config and add if statements for each type (single, transfer, etc)

    adata = sc.read_h5ad(path_reference)
    adata.var.drop_duplicates(keep="first", subset=["gene_names"], inplace=True)
    adata.var["gene_names"] = adata.var["gene_names"].cat.as_ordered()
    adata.var.set_index("gene_names", inplace=True)

    # Reorder adata.X based on the updated index
    adata = adata[:, adata.var.index]
    adata.X.expm1().sum(axis=1)[:10]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    models.download_models(force_update=True)
    models.models_description()
    model = models.Model.load(model="Human_Lung_Atlas.pkl")
    model.cell_types
    predictions = celltypist.annotate(
        adata,
        model="Human_Lung_Atlas.pkl",
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
    )
    predictions.predicted_labels
    adata = predictions.to_adata()
    adata.obs
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["majority_voting"], legend_loc="on data")

    # pd.crosstab(adata.obs.cell_type, adata.obs.majority_voting).loc[
    #     ["Microglia", "Macro_pDC"]
    # ]
    celltypist.dotplot(
        predictions,
        use_as_reference="cell_type",
        use_as_prediction="predicted_labels",
    )
    celltypist.dotplot(
        predictions,
        use_as_reference="cell_type",
        use_as_prediction="majority_voting",
    )

    predictions = celltypist.annotate(
        adata,
        model="Human_Lung_Atlas.pkl",
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
    )
    adata = predictions.to_adata()
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["cell_type", "majority_voting"], legend_loc="on data")
    adata = predictions.to_adata(insert_prob=True)
    adata.obs[["cell_type", "Plasma cells"]]

    return adata
    # if config["train"]:
    #     logger.info(
    #         f"Performing surgery (training the model) on the reference model by training with the query dataset with the following params: \n {config['surgery_params']} \n NOTE: During surgery, only those parts of the model are trained that affect how your query is embedded; the reference embedding cannot change. In that way, the embedding of your query data is partly based on pre-learned patterns in the reference, and partly based on the query data itself"
    #     )
    # # Train using custom dataset
    # adata_2000 = sc.read('celltypist_demo_folder/demo_2000_cells.h5ad', backup_url = 'https://celltypist.cog.sanger.ac.uk/Notebook_demo_data/demo_2000_cells.h5ad')
    # adata_500 = sc.read('celltypist_demo_folder/demo_500_cells.h5ad', backup_url = 'https://celltypist.cog.sanger.ac.uk/Notebook_demo_data/demo_500_cells.h5ad')
    # new_model = celltypist.train(adata_2000, labels = 'cell_type', n_jobs = 10, feature_selection = True)
    # new_model.write('./model_from_immune2000.pkl')
    # new_model = models.Model.load('./model_from_immune2000.pkl')
    # predictions = celltypist.annotate(adata_500, model = './model_from_immune2000.pkl', majority_voting = True, mode = 'prob match', p_thres = 0.5)
    # adata = predictions.to_adata(insert_prob = True)
    # sc.tl.umap(adata)
    # sc.pl.umap(adata, color = ['cell_type', 'majority_voting'], legend_loc = 'on data')

    # # Examine expression of cell type-driving genes
    # model = models.Model.load(model = 'celltypist_demo_folder/model_from_immune2000.pkl')
    # model.cell_types
    # top_3_genes = model.extract_top_markers("Macrophages", 3)
    # top_3_genes
    # sc.pl.violin(adata_2000, top_3_genes, groupby = 'cell_type', rotation = 90)
    # sc.pl.violin(adata_500, top_3_genes, groupby = 'majority_voting', rotation = 90)


def perform_scArches_surgery(AnalysisPipeline):
    # https://docs.scarches.org/en/latest/hlca_map_classify.html#Visualization-of-the-query-alone,-using-reference-based-embedding-and-including-original-gene-expression-values
    # https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/scarches_scvi_tools.html using scvi-tools
    config = AnalysisPipeline.config[AnalysisPipeline.data_type]["scArches_surgery"]

    # TODO: save dataframe with the ann_levels depth (ann_level_1, etc) so we can open an excel file and see which data we have
    # Read paths
    path_backbone = config["path_backbone"]
    path_reference = os.path.join(path_backbone, config["path_reference"])
    path_query_data = os.path.join(path_backbone, config["path_query_data"])
    ref_model_features = os.path.join(path_backbone, config["ref_model_features"])
    ref_model_dir = os.path.join(path_backbone, config["ref_model_dir"])
    surgery_model_dir = os.path.join(path_backbone, config["surgery_model_dir"])
    path_celltypes = os.path.join(path_backbone, "HLCA_celltypes_ordered.csv")

    # Load reference dataset
    adata_ref = sc.read_h5ad(path_reference)
    # Load query dataset
    adata_query_unprep = sc.read_h5ad(path_query_data)
    logger.info(
        f"Running scArches surgery with the following info: \n Query dataset: {adata_query_unprep} \n Reference dataset: {adata_ref}."
    )

    # If your query feature naming (ensembl IDs or gene symbols) does not match the reference model feature naming, apply this function
    if config["ensembleID_to_GeneSym_mapping"]["usage"]:
        gene_mapping_path = os.path.join(path_backbone, config["gene_mapping_path"])
        adata_query_unprep = ensembleID_to_GeneSym_mapping(
            gene_mapping_path=gene_mapping_path,
            adata_query_unprep=adata_query_unprep,
        )
    # Start prepping query data so that it includes the right genes (depends on the genes used in the reference model, missing genes are padded with zeros).
    adata_query_unprep.X = sparse.csr_matrix(adata_query_unprep.X)

    # Remove obsm and varm to prevent errors downstream
    try:
        del adata_query_unprep.obsm
        del adata_query_unprep.varm
    except Exception as e:
        logger.info(f"Exception occurred - {e}")

    logger.info(
        f"Checking if raw count data is present in the matrix: \n {adata_query_unprep.X[:10, :30].toarray()}"
    )

    logger.info(f"Reading reference model features: \n {ref_model_features}")
    ref_model_features = pd.read_csv(ref_model_features, header=None)

    # Prepare query data for scArches:
    adata_query = sca.models.SCANVI.prepare_query_anndata(
        adata=adata_query_unprep, reference_model=ref_model_dir, inplace=False
    )
    logger.info(
        f"Query data after preparing query anndata: \n {adata_query} \n {sca.models.SCANVI.prepare_query_anndata(adata=adata_query_unprep,reference_model=ref_model_dir, inplace=False)}"
    )

    logger.info(
        f"Loading reference model on which we will perform surgery (i.e. set relevant query variables)"
    )
    # Load reference model and set relevant query variables:
    surgery_model = sca.models.SCANVI.load_query_data(
        adata_query,
        ref_model_dir,
        freeze_dropout=True,
    )

    logger.info(
        f"Surgery model registry: {surgery_model.registry_['setup_args'] = } \n Three key arguments used for building the reference model that should also be used to prep scArches surgery: \n 1. batch_key: Used to specify from which batch our query dataset comes from. \n 2. labels_key: As the reference has a scANVI reference model, it used cell type labels as input for the training. These cell types labels were sotred in a column named scanvi_label. Setting this to unlabeled. \n 3. unlabeled_category: This variable specifies how cells without label were named for this specific model."
    )
    adata_query.obs["dataset"] = config["adata_query_batch"]
    adata_query.obs["scanvi_label"] = "unlabeled"

    logger.info(f"Reloading surgery model, now with the variables for adata_query set.")
    surgery_model = sca.models.SCANVI.load_query_data(
        adata_query,
        ref_model_dir,
        freeze_dropout=True,
    )

    # Perform surgery on reference model by training with the query dataset
    early_stopping_kwargs_surgery = config["surgery_params"].copy()
    del early_stopping_kwargs_surgery["epochs"]

    if config["train"]:
        logger.info(
            f"Performing surgery (training the model) on the reference model by training with the query dataset with the following params: \n {config['surgery_params']} \n NOTE: During surgery, only those parts of the model are trained that affect how your query is embedded; the reference embedding cannot change. In that way, the embedding of your query data is partly based on pre-learned patterns in the reference, and partly based on the query data itself"
        )
        surgery_model.train(
            max_epochs=config["surgery_params"]["epochs"],
            **early_stopping_kwargs_surgery,
        )
        surgery_model.save(surgery_model_dir, overwrite=True)
    else:
        logger.info(
            f"Loading the surgery model: {surgery_model_dir} against the query dataset."
        )
        surgery_model = sca.models.SCANVI.load(
            surgery_model_dir, adata_query
        )  # if already trained

    # Obtain query latent embedding
    logger.info(
        f"Obtaining query latent embedding. Now that we have the updated model, we can calculate low-dimensinal representation or 'embedding' of our query data which is in the same space as our HLCA reference. The latent embedding will be stored in a new anndata under .X"
    )
    adata_query_latent = sc.AnnData(
        surgery_model.get_latent_representation(adata_query)
    )
    # Copy over .obs metadata from our query data
    adata_query_latent.obs = adata_query.obs.loc[adata_query.obs.index, :]

    # Combine reference and query embedding into one joint embedding for further processing
    logger.info(
        f"Combining reference and query embedding into one joint embedding. \n NOTE: if you expect non-unique barcodes (.obs index), set index_unique to e.g. '_'. This will add a suffix to our barcodes to ensure we can keep apart reference and query barcodes and batch_key to the obs column that you want to use as a barcode suffix (e.g. ref_or_query)"
    )
    adata_query_latent.obs["ref_or_query"] = "query"
    adata_ref.obs["ref_or_query"] = "ref"

    combined_emb = sc.concat(
        (adata_ref, adata_query_latent), index_unique=None, join="outer"
    )  # index_unique="_", batch_key="ref_or_query")

    logger.info(f"Establishing data types.")
    for cat in combined_emb.obs.columns:
        if isinstance(combined_emb.obs[cat].values, pd.Categorical):
            pass
        elif pd.api.types.is_float_dtype(combined_emb.obs[cat]):
            pass
        else:
            print(
                f"	Setting obs column {cat} (not categorical neither float) to strings to prevent writing error."
            )
            combined_emb.obs[cat] = combined_emb.obs[cat].astype(str)

    logger.info(f"Performing label transfering. ")
    cts_ordered = pd.read_csv(path_celltypes, index_col=0).rename(
        columns={f"Level_{lev}": f"labtransf_ann_level_{lev}" for lev in range(1, 6)}
    )
    logger.info(
        f"Adding annotations for all available labels. They will be stored in adata_ref.obs unde labtransf_ann_level_."
    )
    adata_ref.obs = adata_ref.obs.join(cts_ordered, on="ann_finest_level")
    columns_to_check = [
        "labtransf_ann_level_1",
        "labtransf_ann_level_2",
        "labtransf_ann_level_3",
        "labtransf_ann_level_4",
        "labtransf_ann_level_5",
    ]
    adata_ref = adata_ref[~adata_ref.obs[columns_to_check].isnull().all(axis=1)]
    logger.info(f"adata_ref columns: \n {adata_ref.obs.columns} \n {adata_ref}")
    logger.info(f"Preparing KNN transformer for label transfering")
    knn_transformer = sca.utils.knn.weighted_knn_trainer(
        train_adata=adata_ref,
        train_adata_emb="X",  # location of our joint embedding
        n_neighbors=50,
    )

    logger.info(
        f"Transfering labels for the levels of labels in the reference (e.g. ann_level_1 to ann_level_5)."
    )
    labels, uncert = sca.utils.knn.weighted_knn_transfer(
        query_adata=adata_query_latent,
        query_adata_emb="X",  # location of our embedding, query_adata.X in this case
        label_keys="labtransf_ann_level_",  # (start of) obs column name(s) for which to transfer labels
        knn_model=knn_transformer,
        ref_adata_obs=adata_ref.obs,
    )
    labels.rename(
        columns={
            f"labtransf_ann_level_{lev}": f"ann_level_{lev}_transferred_label_unfiltered"
            for lev in range(1, 6)
        },
        inplace=True,
    )
    uncert.rename(
        columns={
            f"labtransf_ann_level_{lev}": f"ann_level_{lev}_transfer_uncert"
            for lev in range(1, 6)
        },
        inplace=True,
    )
    combined_emb.obs = combined_emb.obs.join(labels)
    combined_emb.obs = combined_emb.obs.join(uncert)

    # copy over labels from reference adata
    for cat in [f"labtransf_ann_level_{lev}" for lev in range(1, 6)]:
        combined_emb.obs.loc[adata_ref.obs.index, cat] = adata_ref.obs[cat]

    uncertainty_threshold = config["label_transfer"]["uncertainty_threshold"]
    logger.info(
        f"Applying uncertainty threshold of {uncertainty_threshold} and setting labels transferred with uncertainty greater than {uncertainty_threshold} to 'Unknown'."
    )

    for lev in range(1, 6):
        combined_emb.obs[f"ann_level_{lev}_transferred_label"] = combined_emb.obs[
            f"ann_level_{lev}_transferred_label_unfiltered"
        ].mask(
            combined_emb.obs[f"ann_level_{lev}_transfer_uncert"]
            > uncertainty_threshold,
            "Unknown",
        )
    logger.info(
        f"Percentage of unknown per level, with uncertainty_threshold={uncertainty_threshold}:"
    )
    for level in range(1, 6):
        try:
            logger.info(
                f"Level {level}: {np.round(sum(combined_emb.obs[f'ann_level_{level}_transferred_label'] =='Unknown')/adata_query.n_obs*100,2)}%"
            )
        except Exception as e:
            logger.error(e)

    adata_query_final = (
        adata_query_unprep.copy()
    )  # copy the original query adata, including gene counts
    adata_query_final.obsm["X_scarches_emb"] = adata_query_latent[
        adata_query_final.obs.index, :
    ].X  # copy over scArches/reference-based embedding

    # If original query_adata has gene IDs instead of gene symbols as var.index, switch that here for easier gene querying.
    # if config["ensembleID_to_GeneSym_mapping"]["usage"]:
    logger.info(
        f"Setting gene symbols instead of gene IDs as index for easier querying."
    )
    adata_query_final.var["gene_ids"] = adata_query_final.var.index
    adata_query_final.var.index = adata_query_final.var.gene_names
    adata_query_final.var.index.name = None

    logger.info(f"Copying over label transfer columns")
    for col in combined_emb.obs.columns:
        if col.startswith("ann_level") and "transfer" in col:
            adata_query_final.obs[col] = combined_emb.obs.loc[
                adata_query_final.obs.index, col
            ]

    adata_query_final.var_names = adata_query_final.var_names.str.capitalize()
    adata_query_final.var.index = adata_query_final.var.index.str.capitalize()
    adata_query_final.raw = adata_query_final

    logger.info(f"Saving final adata_query_final as 'raw_adata' to adata_dict.")

    # self.adata_dict[self.data_type].setdefault("raw_adata", adata_query_final)

    return adata_query_final
