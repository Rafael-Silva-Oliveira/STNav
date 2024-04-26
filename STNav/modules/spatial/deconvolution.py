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
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)
import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
import inspect
from GraphST import GraphST
import torch
import mudata
from scvi.external import Tangram
import mudata
from scvi.external import Tangram

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import spatialdm.plottings as pl
import squidpy as sq


from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
import inspect
import scvi
import scanpy as sc
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
import torch
import celltypist
from celltypist import models


class Deconvolution:

    def __init__(self, STNavCorePipeline):
        self.STNavCorePipeline = STNavCorePipeline

    def train_or_load_deconv_model(self):

        config = self.STNavCorePipeline.config["ST"]["DeconvolutionModels"]

        model_types = [
            model_type
            for model_type, model_config in config.items()
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
        model_type = model_types[0]

        if model_type in ["VariationalInference", "Stereoscope"]:
            st_adata_deconvoluted = ProbabilisticDeconv(
                self.STNavCorePipeline, model_type, config
            )

        elif model_type == "Tangram":
            st_adata_deconvoluted = TangramDeconv(
                STNavCorePipeline=self.STNavCorePipeline,
                model_type=model_type,
                config=config,
            )

        elif model_type == "CellTypist":
            st_adata_deconvoluted = CellTypistDeconv(
                self.STNavCorePipeline, model_type, config
            )

        save_processed_adata(
            STNavCorePipeline=self.STNavCorePipeline,
            name="deconvoluted_adata",
            adata=st_adata_deconvoluted,
        )

    def run_deconvolution(self):
        logger.info(
            f"Running deconvolution based on ranked genes with the group {self.STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        )
        self.train_or_load_deconv_model()


def CellTypistDeconv(STNavCorePipeline, model_type, config):

    model_config = config[model_type]
    train = model_config["Train"]["train"]

    sc_adata_to_use = model_config["Train"]["adata_to_use"]
    sc_path = STNavCorePipeline.adata_dict["scRNA"][sc_adata_to_use]
    sc_adata = sc.read_h5ad(sc_path)

    st_adata_to_use = model_config["Annotate"]["adata_to_use"]
    st_path = STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
    st_adata = sc.read_h5ad(st_path)

    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)
    sc.pp.normalize_total(st_adata, target_sum=1e4)
    sc.pp.log1p(st_adata)

    sc_adata.var_names = sc_adata.var_names.str.upper()
    st_adata.var_names = st_adata.var_names.str.upper()

    if train:
        sc_model = celltypist.train(
            X=sc_adata,
            epochs=model_config["Train"]["params"]["epochs"],
            labels=model_config["Train"]["params"]["labels"],
            n_jobs=model_config["Train"]["params"]["n_jobs"],
            feature_selection=model_config["Train"]["params"]["feature_selection"],
            use_SGD=model_config["Train"]["params"]["use_SGD"],
            mini_batch=model_config["Train"]["params"]["mini_batch"],
            batch_number=model_config["Train"]["params"]["batch_number"],
            batch_size=model_config["Train"]["params"]["batch_size"],
            balance_cell_type=model_config["Train"]["params"]["balance_cell_type"],
        )
        sc_model.write(
            f"{model_config['Train']['pre_trained_model_path']}\\celltypist.pkl"
        )
    else:
        sc_model = models.Model.load(
            f"{model_config['Train']['pre_trained_model_path']}\\celltypist.pkl"
        )

    annotation_config = model_config["Annotate"]["params"]

    predictions = celltypist.annotate(
        st_adata,
        model=sc_model,
        majority_voting=annotation_config["majority_voting"],
        mode=annotation_config["mode"],
        p_thres=annotation_config["p_thres"],
        min_prop=annotation_config["min_prop"],
    )

    adata_st = predictions.to_adata().copy()

    # Convert 'predicted_labels' into dummy variables
    dummies = pd.get_dummies(adata_st.obs["predicted_labels"])

    # Convert True/False to 1/0
    dummies = dummies.astype(int)

    # Add the dummy variables to the original DataFrame
    adata_st.obs = pd.concat([adata_st.obs, dummies], axis=1)

    for cell_type in list(adata_st.obs["predicted_labels"]):
        if cell_type != "Unassigned":
            save_path = (
                STNavCorePipeline.saving_path
                + "\\Plots\\"
                + cell_type
                + model_type
                + ".png"
            )
            with plt.rc_context():  # Use this to set figure params like size and dpi
                plot_func = sc.pl.spatial(
                    adata_st,
                    cmap="magma",
                    color=cell_type,
                    img_key="hires",
                    size=1.5,
                    alpha_img=0.5,
                    show=False,
                )
                plt.savefig(save_path, bbox_inches="tight")
                save_path_dotplot = (
                    STNavCorePipeline.saving_path
                    + "\\Plots\\"
                    + cell_type
                    + model_type
                    + " dotplot"
                    + ".png"
                )
                dp_func = celltypist.dotplot(
                    predictions,
                    use_as_reference="predicted_labels",
                    use_as_prediction="majority_voting",
                    show=False,
                )
                plt.savefig(save_path_dotplot, bbox_inches="tight")

    return adata_st


def ProbabilisticDeconv(STNavCorePipeline, model_type, config):

    model_config = config[model_type]

    if model_type == "VariationalInference":
        sc_model_name = "CondSCVI"
        st_model_name = "DestVI"
    elif model_type == "Stereoscope":
        sc_model_name = "RNAStereoscope"
        st_model_name = "SpatialStereoscope"

    train_sc = model_config[sc_model_name]["train"]
    train_st = model_config[st_model_name]["train"]

    sc_adata_to_use = model_config[sc_model_name]["adata_to_use"]
    sc_path = STNavCorePipeline.adata_dict["scRNA"][sc_adata_to_use]
    sc_adata = sc.read_h5ad(sc_path)

    st_adata_to_use = model_config[sc_model_name]["adata_to_use"]
    st_path = STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
    st_adata = sc.read_h5ad(st_path)

    if train_sc:
        model = eval(sc_model_name)
        model.setup_anndata(
            sc_adata,
            layer=model_config[sc_model_name]["layer"],
            labels_key=model_config[sc_model_name]["labels_key"],
        )
        logger.info(
            f"Training the {model_type} model for deconvolution with '{model_config[sc_model_name]['adata_to_use']}' adata file using the layer {model_config[sc_model_name]['layer']} and the following parameters {model_config[sc_model_name]['params']}."
        )
        sc_model = model(sc_adata)
        logger.info(sc_model.view_anndata_setup())
        training_params = model_config[sc_model_name]["params"]
        valid_arguments = inspect.signature(sc_model.train).parameters.keys()
        filtered_params = {
            k: v for k, v in training_params.items() if k in valid_arguments
        }
        sc_model.train(**filtered_params)
        sc_model.history["elbo_train"][10:].plot()
        sc_model.save(r".\\STNav\\models\\scmodel", overwrite=True)
    else:
        model = eval(sc_model_name)
        logger.info(f"Loading the pre-trained {model_type} model for deconvolution.")
        sc_model = model.load(
            model_config[sc_model_name]["pre_trained_model_path"],
            sc_adata,
        )

    if train_st:
        model = eval(st_model_name)
        logger.info(
            model.setup_anndata(
                st_adata,
                layer=model_config[sc_model_name]["layer"],
            )
        )

        logger.info(
            f"Training the {model_type} model for deconvolution with '{model_config[st_model_name]['adata_to_use']}' adata file using the layer {model_config[st_model_name]['layer']} and the following parameters {model_config[st_model_name]['params']}."
        )
        st_model = model.from_rna_model(st_adata, sc_model)
        st_model.view_anndata_setup()
        training_params = model_config[st_model_name]["params"]
        valid_arguments = inspect.signature(st_model.train).parameters.keys()
        filtered_params = {
            k: v for k, v in training_params.items() if k in valid_arguments
        }
        st_model.train(**filtered_params)
        plt.plot(st_model.history["elbo_train"], label="train")
        plt.title("loss over training epochs")
        plt.legend()
        plt.show()
        st_model.save(r".\\STNav\\models\\stmodel", overwrite=True)
    else:
        model = eval(st_model_name)
        logger.info(f"Loading the pre-trained {model_type} model for deconvolution.")
        st_model = model.load(
            model_config[st_model_name]["pre_trained_model_path"],
            st_adata,
        )

    # Perform deconvolution on the dataset that has genes common between scRNA and ST datasets
    adata_st = sc.read_h5ad(
        STNavCorePipeline.adata_dict["ST"]["subset_preprocessed_adata"]
    )
    # Deconvolution
    adata_st.obsm["deconvolution"] = st_model.get_proportions()
    with torch.no_grad():
        keep_noise = False
        res = torch.nn.functional.softplus(st_model.module.V).cpu().numpy().T
        if not keep_noise:
            res = res[:, :-1]

    column_names = st_model.cell_type_mapping
    adata_st.obsm["deconvolution_unconstr"] = pd.DataFrame(
        data=res,
        columns=column_names,
        index=st_model.adata.obs.index,
    )

    for ct in adata_st.obsm["deconvolution"].columns:
        adata_st.obs[ct] = adata_st.obsm["deconvolution"][ct]

    adata_st.obs[
        f"spatial_{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
    ] = adata_st.obs[column_names].idxmax(axis=1)

    save_processed_adata(
        STNavCorePipeline=STNavCorePipeline,
        name="deconvoluted_adata",
        adata=adata_st,
        fix_write=True,
    )
    for cell_type in adata_st.obsm["deconvolution"].columns:
        save_path = (
            STNavCorePipeline.saving_path
            + "\\Plots\\"
            + cell_type
            + f"_{model_type}"
            + ".png"
        )
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plot_func = sc.pl.spatial(
                adata_st,
                cmap="magma",
                color=cell_type,
                img_key="hires",
                size=1.6,
                alpha_img=0.5,
                show=False,
            )
            plt.savefig(save_path, bbox_inches="tight")

    adata_st.obs.to_excel(
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
        index=False,
    )
    return adata_st


def TangramDeconv(STNavCorePipeline, model_type, config):

    model_config = config[model_type]
    train = model_config["train"]

    # Load scRNA reference data
    sc_adata_to_use = model_config["adata_sc_to_use"]
    sc_path = STNavCorePipeline.adata_dict["scRNA"][sc_adata_to_use]
    adata_sc = sc.read_h5ad(sc_path)

    # Load ST data to infer using the reference scRNA
    st_adata_to_use = model_config["adata_st_to_use"]
    st_path = STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
    adata_st = sc.read_h5ad(st_path)

    if train:
        logger.info(
            f"Training {model_type} model with the following config:\n\n{model_config}"
        )

        adata_st.var_names_make_unique()
        adata_st.obs_names_make_unique()

        mdata = mudata.MuData(
            {
                "sp": adata_st,
                "sc": adata_sc,
            }
        )

        img = sq.im.ImageContainer(model_config["image_path"])

        sq.im.process(img=img, layer="image", method="smooth")
        sq.im.segment(
            img=img,
            layer="image_smooth",
            method="watershed",
            channel=0,
        )
        # define image layer to use for segmentation
        features_kwargs = {
            "segmentation": {
                "label_layer": "segmented_watershed",
                "props": ["label", "centroid"],
                "channels": [1, 2],
            }
        }
        # calculate segmentation features
        sq.im.calculate_image_features(
            adata_st,
            img,
            layer="image",
            key_added="image_features",
            features_kwargs=features_kwargs,
            features="segmentation",
            mask_circle=True,
        )

        adata_st.obs["cell_count"] = adata_st.obsm["image_features"][
            "segmentation_label"
        ]
        sq.pl.spatial_scatter(
            adata_st,
            color=["leiden_clusters", "cell_count"],
            frameon=False,
            wspace=0.01,
        )

        sc.tl.rank_genes_groups(
            adata_sc, groupby="ann_level_3_transferred_label", use_raw=False
        )

        # Find genes for mapping
        markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[
            0:100, :
        ]
        genes_sc = np.unique(markers_df.melt().value.values)
        genes_st = adata_st.var_names.values
        genes = list(set(genes_sc).intersection(set(genes_st)))
        len(genes)

        # Add training objects to mudata
        target_count = adata_st.obs.cell_count.sum()
        adata_st.obs["density_prior"] = (
            np.asarray(adata_st.obs.cell_count) / target_count
        )
        rna_count_per_spot = np.asarray(adata_st.X.sum(axis=1)).squeeze()
        adata_st.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(
            rna_count_per_spot
        )
        adata_st.obs["uniform_density"] = (
            np.ones(adata_st.X.shape[0]) / adata_st.X.shape[0]
        )
        mdata.mod["sp_train"] = mdata.mod["sp"][:, genes].copy()
        mdata.mod["sc_train"] = mdata.mod["sc"][:, genes].copy()
        mdata.update()

        # Run tangram

        Tangram.setup_mudata(
            mdata,
            density_prior_key="density_prior",
            modalities={
                "density_prior_key": "sp_train",
                "sc_layer": "sc_train",
                "sp_layer": "sp_train",
            },
        )
        model = Tangram(mdata, constrained=True, target_count=target_count)
        model.train(max_epochs=5000)
        mapper = model.get_mapper_matrix()
        mdata.mod["sc"].obsm["tangram_mapper"] = mapper
        labels = mdata.mod["sc"].obs["ann_level_3_transferred_label"]
        mdata.mod["sp"].obsm["tangram_ct_pred"] = model.project_cell_annotations(
            mdata.mod["sc"], mdata.mod["sp"], mapper, labels
        )
        mdata.mod["sp_sc_projection"] = model.project_genes(
            mdata.mod["sc"], mdata.mod["sp"], mapper
        )
        adata_st.obs = adata_st.obs.join(adata_st.obsm["tangram_ct_pred"])
    else:
        model = Tangram.load(...)

    sq.pl.spatial_scatter(
        adata_st,
        color=[
            "B cell lineage",
            "AT2",
            "AT1",
            "Macrophages",
            "Mast cells",
            "T cell lineage",
            "EC venous",
            "Fibroblasts",
            "EC arterial",
        ],
        wspace=0.1,
        ncols=1,
        size=10,
    )
    st_adata_deconvoluted = adata_st
    return st_adata_deconvoluted
