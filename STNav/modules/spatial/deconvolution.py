# Load packages
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import json
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)
import inspect

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import squidpy as sq

from STNav.utils.decorators import pass_STNavCore_params
import scanpy as sc
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

        if model_type == "CellTypist":
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
