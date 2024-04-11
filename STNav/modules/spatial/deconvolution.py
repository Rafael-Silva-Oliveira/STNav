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
from STNav.utils.helpers import return_filtered_params, SpatialDM_wrapper
import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI

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

    st_adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
        "subset_preprocessed_adata"
    ].copy()

    if model_name == "GraphST":

        adata_sc = STNavCorePipeline.adata_dict["scRNA"][
            "subset_preprocessed_adata"
        ].copy()
        adata_sc_preprocessed = STNavCorePipeline.adata_dict["scRNA"][
            "preprocessed_adata"
        ].copy()

        project_cell_to_spot(st_adata, adata_sc, retain_percent=0.15)

        columns_cell_type_names = list(adata_sc.obs["cell_type"].unique())

        for cell_type in columns_cell_type_names:
            save_path = STNavCorePipeline.saving_path + "\\Plots\\" + cell_type + ".png"
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
                "cell_type": f"{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            },
            inplace=True,
        )
        adata_sc_preprocessed.obs.rename(
            columns={
                "cell_type": f"{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
            },
            inplace=True,
        )

        STNavCorePipeline.adata_dict["scRNA"][
            "subset_preprocessed_adata"
        ] = adata_sc.copy()
        STNavCorePipeline.adata_dict["scRNA"][
            "preprocessed_adata"
        ] = adata_sc_preprocessed.copy()

        st_adata.obsm["deconvolution"] = st_adata.obs[columns_cell_type_names]

        STNavCorePipeline.save_as("deconvoluted_adata", st_adata)

        st_adata.obs.to_excel(
            f"{STNavCorePipeline.saving_path}\\{STNavCorePipeline.data_type}\\Files\\Deconvoluted_{date}.xlsx",
            index=False,
        )
        adata_sc.obs.rename(
            columns={
                "cell_type": {
                    STNavCorePipeline.config["scRNA"]["DEG"]["rank_genes_groups"][
                        "params"
                    ]["groupby"]
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
            f"spatial_{STNavCorePipeline.config['scRNA']['DEG']['rank_genes_groups']['params']['groupby']}"
        ] = st_adata.obs[column_names].idxmax(axis=1)

        STNavCorePipeline.save_as("deconvoluted_adata", st_adata)

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
