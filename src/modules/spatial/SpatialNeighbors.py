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
from typing import Union

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
from src.modules.decorators import pass_analysis_pipeline


@pass_analysis_pipeline
def SpatialNeighbors(STNavCorePipeline):
    config = STNavCorePipeline.config[STNavCorePipeline.data_type]["SpatialNeighbors"]
    logger.info("Calcualting Spatial Neighbors scores.")
    for method_name, methods in config.items():
        for config_name, config_params in methods.items():
            if config_params["usage"]:
                adata = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
                    config_params["adata_to_use"]
                ].copy()
                logger.info(
                    f"Running {method_name} method with {config_name} configuration \n Configuration parameters: {config_params} \n using the following adata {config_params['adata_to_use']}"
                )

                if method_name == "Squidpy":
                    if config_name == "NHoodEnrichment":
                        sq.gr.spatial_neighbors(adata)
                        sq.gr.nhood_enrichment(
                            **return_filtered_params(config=config_params, adata=adata)
                        )
                        # STNavCorePipeline.adata_dict[
                        #     config_params["data_type"]
                        # ].setdefault(f"{config_name}_adata", adata.copy())
                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)
                    elif config_name == "Co_Ocurrence":
                        # TODO: try to save the file with the matrix of co-occurrence probabilities within each spot. Apply this mask over the predictions from deconvolution to adjust based on co-ocurrence (clusters that have high value of co-occurrence will probably have similar cell proportions within each spot)
                        sq.gr.co_occurrence(
                            **return_filtered_params(config=config_params, adata=adata)
                        )

                        # STNavCorePipeline.adata_dict[
                        #     config_params["data_type"]
                        # ].setdefault(f"{config_name}_adata", adata.copy())
                        STNavCorePipeline.save_as(f"{config_name}_adata", adata)
