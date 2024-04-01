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
from src.utils.utils import (
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
from src.utils.utils import fix_write_h5ad, GARD

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


# Other names: STNavigator, STHub,
class STNav(object):
    adata_dict_suffix = "_adata"

    def __init__(
        self, config: dict, saving_path: str, data_type: str, adata_dict: dict = None
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

        self.adata_dict[self.data_type].setdefault("raw_adata", adata)

        return adata

    def perform_celltypist(self):

        config = self.config[self.data_type]["celltypist_surgery"]
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

    def perform_scArches_surgery(self):
        # https://docs.scarches.org/en/latest/hlca_map_classify.html#Visualization-of-the-query-alone,-using-reference-based-embedding-and-including-original-gene-expression-values
        # https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/scarches_scvi_tools.html using scvi-tools
        # TODO: add celltypist implementation of this method (use their models and approach as a 2nd option to merge reference data with our query scRNA )
        config = self.config[self.data_type]["scArches_surgery"]

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

        logger.info(
            f"Reloading surgery model, now with the variables for adata_query set."
        )
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
            columns={
                f"Level_{lev}": f"labtransf_ann_level_{lev}" for lev in range(1, 6)
            }
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

        logger.info(
            f"Saving final adata_query_final as 'raw_adata' to adata_dict."
        )  # TODO: save raw adata as a new file in the data processed, etc so theres no need to run scArches again
        self.adata_dict[self.data_type].setdefault("raw_adata", adata_query_final)

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
        self.adata_dict[self.data_type].setdefault("raw_adata", adata)

        logger.info(f"Saving adata to adata_dict as 'raw_adata'.")

        return adata

    def QC(self):
        config = self.config[self.data_type]["quality_control"]
        adata = self.adata_dict[self.data_type]["raw_adata"].copy()

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
        logger.info(f"Saving adata to adata_dict as 'QCed_adata'.")
        self.adata_dict[self.data_type].setdefault("QCed_adata", adata.copy())

        return adata

    def preprocessing(self) -> an.AnnData:
        config = self.config[self.data_type]["preprocessing"]
        adata = self.adata_dict[self.data_type][config["adata_to_use"]].copy()

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

        adata.raw = adata

        if config["highly_variable_genes"]["usage"]:
            logger.info(
                f"Selecting highly variable genes with the following params {config['highly_variable_genes']['params']}"
            )

            assert (
                config["log1p"]["usage"] == True
                and config["highly_variable_genes"]["params"]["flavor"] == "seurat"
            ) or (
                config["log1p"]["usage"] != True
                and config["highly_variable_genes"]["params"]["flavor"] != "seurat"
            ), "Highly variable genes with log1p applied to the data expects flavor to be seurat. Please, deactivate log1p if you want to use seurat_v3. Expects logarithmized data, except when flavor='seurat_v3', in which count data is expected."

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

        # TODO: Put this in a new class method called plotting_preprocessing
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

        # save the counts to a separate object for later, we need the normalized counts in raw for DEG dete.Save raw data before preprocessing values and further filtering
        adata.layers["preprocessed_counts"] = adata.X.copy()

        logger.info(f"Current adata.X shape after preprocessing: {adata.X.shape}")
        logger.info(
            f"Current adata.raw.X shape after preprocessing: \n {adata.raw.X.shape = }"
        )

        logger.info(log_adataX(adata=adata, raw=False))

        logger.info(
            log_adataX(
                adata=adata, layer="preprocessed_counts", raw=True, step="preprocessing"
            )
        )

        logger.info(
            log_adataX(adata=adata, layer="raw_counts", raw=True, step="preprocessing")
        )

        logger.info(log_adataX(adata=adata, layer="raw_counts", step="preprocessing"))

        logger.info(f"Saving adata to adata_dict as 'preprocessed_adata'.")
        # TODO: add a method to give the name through the config ("save as..") so that it gives the names automatically.Add a warning everytime a name is already in the dictionary to warn the user that the results will be overwritten.
        self.adata_dict[self.data_type].setdefault("preprocessed_adata", adata.copy())

        return adata

    def DEG(self):
        config = self.config[self.data_type]["DEG"]
        adata = self.adata_dict[self.data_type][config["adata_to_use"]].copy()

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

            self.adata_dict[self.data_type].setdefault(
                "DEG_adata", adata_for_DEG.copy()
            )
            self.adata_dict[self.data_type].setdefault(
                "preprocessed_DEG_adata", adata.copy()
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
                logger.info("Adding manual gene sets.")

            # Handle API gene sets
            if config_gsea["gene_sets"]["api_sets"]["usage"]:
                gene_set_dict["api_sets"] = [
                    ontology
                    for ontology, boolean in config_gsea["gene_sets"]["api_sets"][
                        "sets"
                    ].items()
                    if boolean
                ]

                logger.info("Adding API gene sets.")

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

                self.adata_dict[self.data_type][
                    "preprocessed_adata_GARD"
                ] = adata.copy()

                GARD_final_df.to_excel(
                    f"{self.saving_path}\\{self.data_type}\\Files\\GARD_score.xlsx"
                )

            with pd.ExcelWriter(
                f"{self.saving_path}\\{self.data_type}\\Files\\{self.data_type}_GSEA_{date}.xlsx"
            ) as writer:
                for sheet_name, gsea_df in gsea_dataframes.items():
                    gsea_df.to_excel(writer, sheet_name=sheet_name, index=False)

        return adata

    def train_or_load_sc_deconvolution_model(self):
        config = self.config[self.data_type]

        model_types = [
            model_name
            for model_name, model_config in config["model"]["model_type"].items()
            if model_config["usage"]
        ]
        if len(model_types) >= 2:
            raise ValueError(
                logger.error(
                    f"Please, choose only 1 model to use. Current active models {model_types = }"
                )
            )
        elif len(model_types) == 0:
            logger.warning(
                f"Returning no model as no models were set to True for training or loading. "
            )
            return None
        model_name = model_types[0]

        adata = self.adata_dict[self.data_type][
            config["model"]["model_type"][model_name]["adata_to_use"]
        ].copy()
        model = eval(model_name)
        # TODO: add assertion that checks if selected layer is normalized or unnormalized counts [0,15,0,23] instead of [0,6.2123,0,8.2123] etc
        model.setup_anndata(
            adata,
            layer=config["model"]["model_type"][model_name]["layer"],
            labels_key=config["DEG"]["rank_genes_groups"]["params"]["groupby"],
        )

        train = config["model"]["model_type"][model_name]["train"]

        if train:
            logger.info(
                f"Training the {model_name} model for deconvolution with '{config['model']['adata_to_use']}' adata file using the layer {config['model']['layer']} and the following parameters {config['model']['params']}."
            )
            sc_model = model(adata)
            logger.info(sc_model.view_anndata_setup())
            training_params = config["model"]["model_type"][model_name]["params"]
            valid_arguments = inspect.signature(sc_model.train).parameters.keys()
            filtered_params = {
                k: v for k, v in training_params.items() if k in valid_arguments
            }
            sc_model.train(**filtered_params)
            sc_model.history["elbo_train"][10:].plot()
            sc_model.save("scmodel", overwrite=True)
        else:
            logger.info(
                f"Loading the pre-trained {model_name} model for deconvolution."
            )
            sc_model = model.load(
                config["model"]["pre_trained_model_path"],
                adata,
            )

        return sc_model

    def train_or_load_st_deconvolution_model(self, sc_model):
        config = self.config[self.data_type]

        model_types = [
            model_name
            for model_name, model_config in config["model"]["model_type"].items()
            if model_config["usage"]
        ]
        if len(model_types) >= 2:
            raise ValueError(
                logger.error(
                    f"Please, choose only 1 model to use. Current active models {model_types = }"
                )
            )

        model_name = model_types[0]

        train = config["model"]["model_type"][model_name]["train"]
        if model_name == "GraphST" and not train:
            raise ValueError(
                logger.error(
                    f"Mode name is {model_name}, but training is set to {train}. When using GraphST, please make sure training is set to True."
                )
            )
        adata = self.adata_dict[self.data_type][
            config["model"]["model_type"][model_name]["adata_to_use"]
        ].copy()

        if train:
            if model_name == "GraphST":
                device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
                adata_sc = self.adata_dict["scRNA"][
                    config["model"]["model_type"][model_name]["adata_to_use"]
                ].copy()

                GraphST.get_feature(adata)

                # Change to cell_type as GraphST only accepts cell_type ...
                adata_sc.obs.rename(
                    columns={
                        f"{self.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}": "cell_type"
                    },
                    inplace=True,
                )

                adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)

                st_model = GraphST.GraphST(
                    adata,
                    adata_sc,
                    epochs=config["model"]["model_type"][model_name]["params"][
                        "epochs"
                    ],
                    random_seed=config["model"]["model_type"][model_name]["params"][
                        "random_seed"
                    ],
                    device=device,
                    deconvolution=config["model"]["model_type"][model_name]["params"][
                        "deconvolution"
                    ],
                )

                adata, adata_sc = st_model.train_map()

                self.adata_dict[self.data_type]["preprocessed_adata"] = adata.copy()

                self.adata_dict["scRNA"]["preprocessed_adata"] = adata_sc.copy()

            if model_name != "GraphST":

                model = eval(model_name)
                logger.info(
                    model.setup_anndata(
                        adata,
                        layer=config["model"]["model_type"][model_name]["layer"],
                    )
                )

                logger.info(
                    f"Training the {model_name} model for deconvolution with '{config['model']['adata_to_use']}' adata file adata file using the layer {config['model']['layer']} and the following parameters {config['model']['params']}."
                )
                st_model = model.from_rna_model(adata, sc_model)
                st_model.view_anndata_setup()
                training_params = config["model"]["model_type"][model_name]["params"]
                valid_arguments = inspect.signature(st_model.train).parameters.keys()
                filtered_params = {
                    k: v for k, v in training_params.items() if k in valid_arguments
                }
                st_model.train(**filtered_params)
                plt.plot(st_model.history["elbo_train"], label="train")
                plt.title("loss over training epochs")
                plt.legend()
                plt.show()
                st_model.save("stmodel", overwrite=True)
        else:
            if model_name != "GraphST":
                logger.info(
                    f"Loading the pre-trained {model_name} model for deconvolution."
                )
                st_model = model.load(
                    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\stmodel",
                    adata,
                )
        return st_model, model_name

    def deconvolution(self, st_model, model_name):
        logger.info(
            f"Running deconvolution based on ranked genes with the group {self.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        )

        st_adata = self.adata_dict[self.data_type]["subset_preprocessed_adata"].copy()

        if model_name == "GraphST":

            adata_sc = self.adata_dict["scRNA"]["subset_preprocessed_adata"].copy()
            adata_sc_preprocessed = self.adata_dict["scRNA"][
                "preprocessed_adata"
            ].copy()

            project_cell_to_spot(st_adata, adata_sc, retain_percent=0.15)

            columns_cell_type_names = list(adata_sc.obs["cell_type"].unique())

            for cell_type in columns_cell_type_names:
                save_path = self.saving_path + "\\Plots\\" + cell_type + ".png"
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
                    "cell_type": f"{self.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
                },
                inplace=True,
            )
            adata_sc_preprocessed.obs.rename(
                columns={
                    "cell_type": f"{self.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
                },
                inplace=True,
            )

            self.adata_dict["scRNA"]["subset_preprocessed_adata"] = adata_sc.copy()
            self.adata_dict["scRNA"][
                "preprocessed_adata"
            ] = adata_sc_preprocessed.copy()

            st_adata.obsm["deconvolution"] = st_adata.obs[columns_cell_type_names]

            logger.info(f"Saving adata to adata_dict as 'deconvoluted_adata'.")
            self.adata_dict[self.data_type].setdefault("deconvoluted_adata", st_adata)
            st_adata.obs.to_excel(
                f"{self.saving_path}\\{self.data_type}\\Files\\Deconvoluted_{date}.xlsx",
                index=False,
            )
            adata_sc.obs.rename(
                columns={
                    "cell_type": {
                        self.config["scRNA"]["DEG"]["rank_genes_groups"]["params"][
                            "groupby"
                        ]
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
                f"spatial_{self.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            ] = st_adata.obs[column_names].idxmax(axis=1)

            logger.info(f"Saving adata to adata_dict as 'deconvoluted_adata'.")
            self.adata_dict[self.data_type].setdefault("deconvoluted_adata", st_adata)

            for cell_type in st_adata.obsm["deconvolution"].columns:
                save_path = self.saving_path + "\\Plots\\" + cell_type + ".png"
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
                f"{self.saving_path}\\{self.data_type}\\Files\\Deconvoluted_{date}.xlsx",
                index=False,
            )
        return st_model

    def SpatiallyVariableGenes(self):
        """
        g - The name of the gene
        pval - The P-value for spatial differential expression
        qval - Significance after correcting for multiple testing
        l - A parameter indicating the distance scale a gene changes expression over
        """
        import SpatialDE

        config = self.config[self.data_type]["SpatiallyVariableGenes"]
        logger.info("Obtaining spatially variable genes.")
        for method_name, methods in config.items():
            for config_name, config_params in methods.items():
                if config_params["usage"]:
                    adata = self.adata_dict[self.data_type][
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
                            f"{self.saving_path}\\{data_type}\\Files\\{data_type}_SpatiallyVarGenes_{date}.xlsx"
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
                            histology_results, patterns = (
                                SpatialDE.aeh.spatial_patterns(
                                    coord,
                                    counts,
                                    sign_results,
                                    C=config_params["AEH"]["params"]["C"],
                                    l=average_length,
                                    verbosity=1,
                                )
                            )

                            # Add the results to the adata and save it as SpatiallyVariableGenes adata

                            self.adata_dict[self.data_type].setdefault(
                                f"SpatiallyVariableGenes_adata", adata.copy()
                            )

                            logger.info("		Saving spatially variable genes with AEH.")

                            with pd.ExcelWriter(
                                f"{self.saving_path}\\{data_type}\\Files\\{data_type}_histology_results_AEH_{date}.xlsx"
                            ) as writer:
                                histology_results.to_excel(
                                    writer,
                                    sheet_name="histology_results AEH",
                                    index=True,
                                )

                            with pd.ExcelWriter(
                                f"{self.saving_path}\\{data_type}\\Files\\{data_type}_Patterns_AEH_{date}.xlsx"
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
                                        histology_results.query("pattern == @i").shape[
                                            0
                                        ],
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
                                f"{self.saving_path}\\{data_type}\\Files\\{data_type}_patterns_genes_{date}.json",
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
                            f"{self.saving_path}\\{data_type}\\Files\\{data_type}_Squidpy_MoranI_{date}.xlsx"
                        ) as writer:
                            adata.uns["moranI"].to_excel(
                                writer,
                                sheet_name="Squidpy_MoranI",
                                index=True,
                            )
                        logger.info(
                            f"Saving adata to adata_dict as '{config_name}_adata'."
                        )
                        self.adata_dict[self.data_type].setdefault(
                            f"{method_name}_adata", adata.copy()
                        )

                        # sq.pl.spatial_scatter(adata, color=["Olfm1", "Plp1", "Itpka", "cluster"])

    def ReceptorLigandAnalysis(self):
        import squidpy as sq
        import spatialdm.plottings as pl

        config = self.config[self.data_type]["ReceptorLigandAnalysis"]

        logger.info(f"Running Receptor Ligand Analysis for {self.data_type}.")

        for method_name, methods in config.items():
            for config_name, config_params in methods.items():
                if config_params["usage"]:
                    adata = self.adata_dict[self.data_type][
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
                            f"{self.saving_path}\\{self.data_type}\\Files\\{self.data_type}_LigRec_{date}.xlsx"
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

                        logger.info(
                            f"Saving adata to adata_dict as '{config_name}_adata'."
                        )
                        self.adata_dict[self.data_type].setdefault(
                            f"{config_name}_adata", adata.copy()
                        )
                        # TODO: check why it isnt saving and not displaying on the plots adata dict
                        logger.info(
                            f"Saving res to adata_dict as '{config_name}_dictionary'."
                        )
                        self.adata_dict[self.data_type].setdefault(
                            f"{config_name}_dictionary", res
                        )
                    elif method_name == "SpatialDM":

                        adata = SpatialDM_wrapper(
                            **return_filtered_params(config=config_params, adata=adata)
                        )
                        with pd.ExcelWriter(
                            f"{self.saving_path}\\{self.data_type}\\Files\\{self.data_type}_SpatialDM_LigRec_{date}.xlsx"
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

                        logger.info(
                            f"Saving adata to adata_dict as '{config_name}_adata'."
                        )
                        self.adata_dict[self.data_type].setdefault(
                            f"{config_name}_adata", adata.copy()
                        )
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

                            histology_results, patterns = (
                                SpatialDE.aeh.spatial_patterns(
                                    adata.obsm["spatial"],
                                    bin_spots.transpose(),
                                    results,
                                    C=3,
                                    l=3,
                                    verbosity=1,
                                )
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
                                        histology_results.query("pattern == @i").shape[
                                            0
                                        ],
                                    )
                                )
                            plt.savefig("mel_DE_clusters.pdf")

                        logger.info(
                            f"Saving adata to adata_dict as '{config_name}_adata'."
                        )
                        self.adata_dict[self.data_type].setdefault(
                            f"{config_name}_adata", adata.copy()
                        )

    def SpatialNeighbors(self):
        config = self.config[self.data_type]["SpatialNeighbors"]
        logger.info("Calcualting Spatial Neighbors scores.")
        for method_name, methods in config.items():
            for config_name, config_params in methods.items():
                if config_params["usage"]:
                    adata = self.adata_dict[self.data_type][
                        config_params["adata_to_use"]
                    ].copy()
                    logger.info(
                        f"Running {method_name} method with {config_name} configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
                    )

                    if method_name == "Squidpy":
                        if config_name == "NHoodEnrichment":
                            sq.gr.spatial_neighbors(adata)
                            sq.gr.nhood_enrichment(
                                **return_filtered_params(
                                    config=config_params, adata=adata
                                )
                            )
                            self.adata_dict[config_params["data_type"]].setdefault(
                                f"{config_name}_adata", adata.copy()
                            )
                        elif config_name == "Co_Ocurrence":
                            # TODO: try to save the file with the matrix of co-occurrence probabilities within each spot. Apply this mask over the predictions from deconvolution to adjust based on co-ocurrence (clusters that have high value of co-occurrence will probably have similar cell proportions within each spot)
                            sq.gr.co_occurrence(
                                **return_filtered_params(
                                    config=config_params, adata=adata
                                )
                            )

                            self.adata_dict[config_params["data_type"]].setdefault(
                                f"{config_name}_adata", adata.copy()
                            )

    def save_processed_adata(self, fix_write: bool = None):
        logger.info(
            f"Saving {self.data_type}.h5ad file.\nPlease note that if you have several configurations defined for plotting, this might change the saved settings in the .h5ad files (i.e. latest settings from the latest plotting configs will be used)."
        )
        for adata_name, adata in self.adata_dict[self.data_type].items():
            if isinstance(adata, ad.AnnData):
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

                if fix_write:
                    try:
                        adata_final = fix_write_h5ad(adata=adata_final)
                    except Exception as e:
                        logger.warning(f"fix_write_h5ad failed {e}")
                    adata_final.write_h5ad(
                        self.saving_path
                        + "\\"
                        + f"{self.data_type}\\Files"
                        + "\\"
                        + f"{self.data_type}_{adata_name}.h5ad"
                    )
                else:
                    try:
                        adata_final.write_h5ad(
                            self.saving_path
                            + "\\"
                            + f"{self.data_type}\\Files"
                            + "\\"
                            + f"{self.data_type}_{adata_name}.h5ad"
                        )
                    except Exception as e:
                        logger.error(f"Exception occurred saving {adata_name} - {e}")
            else:
                logger.warning(f"Adata {adata_name} is not an AnnData object.")
