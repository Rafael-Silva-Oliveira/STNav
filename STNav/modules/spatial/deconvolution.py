# Load packages
import warnings
from datetime import datetime
from celltypist.classifier import AnnotationResult
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq
from loguru import logger
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    save_processed_adata,
    return_from_checkpoint,
    swap_layer,
)

# Set scanpy parameters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# Ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import squidpy as sq

import scanpy as sc
import celltypist
from celltypist import models


class Deconvolution:

    def __init__(self, STNavCorePipeline) -> None:
        self.STNavCorePipeline = STNavCorePipeline

    def train_or_load_deconv_model(self) -> None:

        config = self.STNavCorePipeline.config["ST"]["DeconvolutionModels"]

        model_types: list = [
            model_type
            for model_type, model_config in config.items()
            if isinstance(model_config, dict) and model_config["usage"]
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
            st_adata_deconvoluted: sc.AnnData = CellTypistDeconv(
                STNavCorePipeline=self.STNavCorePipeline,
                model_type=model_type,
                config=config,
            )

        save_processed_adata(
            STNavCorePipeline=self.STNavCorePipeline,
            name=config["save_as"],
            adata=st_adata_deconvoluted,
        )


def CellTypistDeconv(STNavCorePipeline, model_type, config) -> sc.AnnData:

    model_config = config[model_type]
    st_adata_to_use = model_config["Annotate"]["adata_to_use"]
    st_path = STNavCorePipeline.adata_dict["ST"][st_adata_to_use]
    st_adata: sc.AnnData = sc.read_h5ad(filename=st_path)

    # Preprocess the data
    # Re-add the raw counts to the X and raw
    st_adata.X = st_adata.layers["raw_counts"].copy()

    # Normalize for CellTypist prediction
    st_adata_to_annotate: sc.AnnData = st_adata.copy()
    sc.pp.normalize_total(adata=st_adata_to_annotate, target_sum=1e4)
    sc.pp.log1p(st_adata_to_annotate)
    st_adata_to_annotate.var_names = st_adata_to_annotate.var_names.str.upper()

    # Create a dictionary to store the models and their predictions
    model_predictions: dict = {}

    for model_info in model_config["Annotate"]["models"]:
        if model_info["usage"]:
            # Load the model
            model_path: str = model_info["model_path"]
            model = models.Model.load(model=model_path)

            # Get model-specific parameters
            annotation_params = model_info["params"]

            # Predict with the model
            predictions: AnnotationResult = celltypist.annotate(
                st_adata_to_annotate,
                model=model,
                majority_voting=annotation_params["majority_voting"],
                mode=annotation_params["mode"],
                p_thres=annotation_params["p_thres"],
                min_prop=annotation_params["min_prop"],
            )

            # Store the predictions
            adata_pred: sc.AnnData = predictions.to_adata().copy()
            model_name: str = model_info["model_name"]
            model_predictions[model_name] = adata_pred

    # Combine the predictions
    for model_name, adata_pred in model_predictions.items():
        st_adata.obs[f"predicted_labels_{model_name}"] = adata_pred.obs[
            "predicted_labels"
        ]
        st_adata.obs[f"conf_score_{model_name}"] = adata_pred.obs["conf_score"]

    # Determine the highest confidence score for each cell
    model_names = list(model_predictions.keys())
    conf_score_columns: list[str] = [f"conf_score_{name}" for name in model_names]

    highest_conf_model = st_adata.obs[conf_score_columns].idxmax(axis=1)

    highest_conf_labels = highest_conf_model.map(
        arg=lambda x: x.replace("conf_score_", "")
    )
    highest_conf_labels = st_adata.obs[
        [f"predicted_labels_{name}" for name in model_names]
    ].values[
        range(len(highest_conf_model)),
        highest_conf_labels.map(arg=lambda x: model_names.index(x)),
    ]
    # Create the final annotation
    st_adata.obs["cell_type"] = highest_conf_labels

    # Convert 'predicted_labels' into dummy variables
    dummies: pd.DataFrame = pd.get_dummies(data=st_adata.obs["cell_type"])

    # Convert True/False to 1/0
    dummies = dummies.astype(dtype=int)

    # Add the dummy variables to the original DataFrame
    st_adata.obs = pd.concat(objs=[st_adata.obs, dummies], axis=1)

    # for cell_type in list(st_adata.obs["cell_type"]):
    #     if cell_type != "Unassigned":
    #         save_path = (
    #             STNavCorePipeline.saving_path
    #             + "/Plots/"
    #             + cell_type
    #             + "_"
    #             + model_type
    #             + ".png"
    #         )
    #         with plt.rc_context():  # Use this to set figure params like size and dpi
    #             plot_func = sc.pl.spatial(
    #                 st_adata,
    #                 cmap="magma",
    #                 color=cell_type,
    #                 img_key="hires",
    #                 size=1,
    #                 alpha_img=0.5,
    #                 show=False,
    #             )
    #             plt.savefig(save_path, bbox_inches="tight", dpi=750)

    # if annotation_params["majority_voting"]:
    #     save_path_dotplot = (
    #         STNavCorePipeline.saving_path
    #         + "/Plots/"
    #         + cell_type
    #         + "_"
    #         + model_type
    #         + " dotplot"
    #         + ".png"
    #     )
    #     dp_func = celltypist.dotplot(
    #         predictions=predictions,
    #         use_as_reference="predicted_labels",
    #         use_as_prediction="majority_voting",
    #         show=False,
    #     )
    #     plt.savefig(save_path_dotplot, bbox_inches="tight")

    return st_adata
