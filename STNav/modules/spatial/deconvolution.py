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

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import spatialdm.plottings as pl
import squidpy as sq


@pass_STNavCore_params
def deconvolution(STNavCorePipeline, st_model, model_name):
    logger.info(
        f"Running deconvolution based on ranked genes with the group {STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
    )

    st_adata = sc.read_h5ad(
        STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
            "subset_preprocessed_adata"
        ]
    )

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
        f"spatial_{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
    ] = st_adata.obs[column_names].idxmax(axis=1)

    save_processed_adata(
        STNavCorePipeline=STNavCorePipeline,
        name="deconvoluted_adata",
        adata=st_adata,
    )
    for cell_type in st_adata.obsm["deconvolution"].columns:
        save_path = STNavCorePipeline.saving_path + "\\Plots\\" + cell_type + ".png"
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
        f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
        index=False,
    )
    return st_model


@pass_STNavCore_params
def train_or_load_st_deconvolution_model(STNavCorePipeline):
    config = STNavCorePipeline.config[STNavCorePipeline.data_type]

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
    adata = sc.read_h5ad(
        STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
            config["model"]["model_type"][model_name]["adata_to_use"]
        ]
    )

    if train:
        if model_name == "GraphST":
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            adata_sc = STNavCorePipeline.adata_dict["scRNA"][
                config["model"]["model_type"][model_name]["adata_to_use"]
            ]

            GraphST.get_feature(adata)

            # Change to cell_type as GraphST only accepts cell_type ...
            adata_sc.obs.rename(
                columns={
                    f"{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}": "cell_type"
                },
                inplace=True,
            )

            adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)

            st_model = GraphST.GraphST(
                adata,
                adata_sc,
                epochs=config["model"]["model_type"][model_name]["params"]["epochs"],
                random_seed=config["model"]["model_type"][model_name]["params"][
                    "random_seed"
                ],
                device=device,
                deconvolution=config["model"]["model_type"][model_name]["params"][
                    "deconvolution"
                ],
            )

            adata, adata_sc = st_model.train_map()

            STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                "preprocessed_adata"
            ] = adata.copy()

            STNavCorePipeline.adata_dict["scRNA"][
                "preprocessed_adata"
            ] = adata_sc.copy()

        if model_name != "GraphST":

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
            st_model = model.from_rna_model(adata, self.sc_model)
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
            model = eval(model_name)
            logger.info(
                f"Loading the pre-trained {model_name} model for deconvolution."
            )
            st_model = model.load(
                config["model"]["pre_trained_model_path"],
                adata,
            )
    STNavCorePipeline.st_model = st_model

    return st_model, model_name
