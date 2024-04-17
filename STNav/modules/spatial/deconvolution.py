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


class Deconvolution:

    def __init__(self, STNavCorePipeline):
        self.STNavCorePipeline = STNavCorePipeline

    def train_or_load_sc_deconvolution_model(self):
        config = self.STNavCorePipeline.config["scRNA"]

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

        adata = sc.read_h5ad(
            self.STNavCorePipeline.adata_dict["scRNA"][
                config["model"]["model_type"][model_name]["adata_to_use"]
            ]
        )

        model = eval(model_name)
        model.setup_anndata(
            adata,
            layer=config["model"]["model_type"][model_name]["layer"],
            labels_key=config["DEG"]["rank_genes_groups"]["params"]["groupby"],
        )

        train = config["model"]["model_type"][model_name]["train"]

        if model_name in ["RNAStereoscope", "CondSCVI"]:
            sc_model = siceVI(
                self.STNavCorePipeline, adata, model_name, config, train, model
            )
            return sc_model
        else:
            return None

    def train_or_load_st_deconvolution_model(self, sc_model):
        config = self.STNavCorePipeline.config["ST"]

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

        if model_name == "Tangram":
            sc_adata_to_use = self.STNavCorePipeline.config["ST"]["model"][
                "model_type"
            ][model_name]["adata_sc_to_use"]
            sc_path = self.STNavCorePipeline.adata_dict["scRNA"][sc_adata_to_use]
            sc_adata = sc.read_h5ad(sc_path)
            st_adata_to_use = self.STNavCorePipeline.config["ST"]["model"][
                "model_type"
            ][model_name]["adata_st_to_use"]
            st_path = self.STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
            st_adata = sc.read_h5ad(st_path)

            st_model = tangram_model(
                STNavCorePipeline=self.STNavCorePipeline,
                adata_sc=sc_adata,
                adata_st=st_adata,
                model_name=model_name,
                config=config,
                train=train,
            )
        elif model_name in ["SpatialStereoscope", "DestVI"]:
            st_adata_to_use = config["model"]["model_type"][model_name]["adata_to_use"]
            st_path = self.STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
            adata = sc.read_h5ad(st_path)
            st_model = spatVI(
                self.STNavCorePipeline,
                adata,
                model_name,
                config,
                train,
                sc_model=sc_model,
            )

            adata_st_subset = sc.read_h5ad(
                self.STNavCorePipeline.adata_dict["ST"]["subset_preprocessed_adata"]
            )
            # Deconvolution
            adata_st_subset.obsm["deconvolution"] = st_model.get_proportions()
            with torch.no_grad():
                keep_noise = False
                res = torch.nn.functional.softplus(st_model.module.V).cpu().numpy().T
                if not keep_noise:
                    res = res[:, :-1]

            column_names = st_model.cell_type_mapping
            adata_st_subset.obsm["deconvolution_unconstr"] = pd.DataFrame(
                data=res,
                columns=column_names,
                index=st_model.adata.obs.index,
            )

            for ct in adata_st_subset.obsm["deconvolution"].columns:
                adata_st_subset.obs[ct] = adata_st_subset.obsm["deconvolution"][ct]

            adata_st_subset.obs[
                f"spatial_{self.STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            ] = adata_st_subset.obs[column_names].idxmax(axis=1)

            save_processed_adata(
                STNavCorePipeline=self.STNavCorePipeline,
                name="deconvoluted_adata",
                adata=adata_st_subset,
            )
            for cell_type in adata_st_subset.obsm["deconvolution"].columns:
                save_path = (
                    self.STNavCorePipeline.saving_path
                    + "\\Plots\\"
                    + cell_type
                    + ".png"
                )
                with plt.rc_context():  # Use this to set figure params like size and dpi
                    plot_func = sc.pl.spatial(
                        adata_st_subset,
                        cmap="magma",
                        color=cell_type,
                        img_key="hires",
                        size=1.6,
                        alpha_img=0.5,
                        show=False,
                    )
                    plt.savefig(save_path, bbox_inches="tight")

            adata_st_subset.obs.to_excel(
                f"{self.STNavCorePipeline.saving_path}\\{self.STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
                index=False,
            )
        return st_model, adata_st_subset

    def run_deconvolution(self):
        logger.info(
            f"Running deconvolution based on ranked genes with the group {self.STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        )
        sc_model = self.train_or_load_sc_deconvolution_model()
        st_model, st_adata = self.train_or_load_st_deconvolution_model(
            sc_model=sc_model
        )


def siceVI(STNavCorePipeline, adata, model_name, config, train, model):
    if train:
        logger.info(
            f"Training the {model_name} model for deconvolution with '{config['model']['model_type'][model_name]['adata_to_use']}' adata file using the layer {config['model']['model_type'][model_name]['layer']} and the following parameters {config['model']['model_type'][model_name]['params']}."
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
        sc_model.save(r".\\STNav\\models\\scmodel", overwrite=True)
    else:
        logger.info(f"Loading the pre-trained {model_name} model for deconvolution.")
        sc_model = model.load(
            config["model"]["pre_trained_model_path"],
            adata,
        )


def spatVI(STNavCorePipeline, adata, model_name, config, train, sc_model=None):
    if train:
        model = eval(model_name)
        logger.info(
            model.setup_anndata(
                adata,
                layer=config["model"]["model_type"][model_name]["layer"],
            )
        )

        logger.info(
            f"Training the {model_name} model for deconvolution with '{config['model']['model_type'][model_name]['adata_to_use']}' adata file using the layer {config['model']['model_type'][model_name]['layer']} and the following parameters {config['model']['model_type'][model_name]['params']}."
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
        st_model.save(r".\\STNav\\models\\stmodel", overwrite=True)
        return st_model
    else:
        model = eval(model_name)
        logger.info(f"Loading the pre-trained {model_name} model for deconvolution.")
        st_model = model.load(
            config["model"]["pre_trained_model_path"],
            adata,
        )
        return st_model


def tangram_model(STNavCorePipeline, adata_sc, adata_st, model_name, config, train):
    config = config["model"]["model_type"][model_name]
    if train:
        logger.info(
            f"Training {model_name} model with the following config:\n\n{config}"
        )

        adata_st.var_names_make_unique()
        adata_st.obs_names_make_unique()

        mdata = mudata.MuData(
            {
                "sp": adata_st,
                "sc": adata_sc,
            }
        )

        img = sq.im.ImageContainer(config["image_path"])

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
