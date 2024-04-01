# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import json
from datetime import datetime
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
import torch
import scvi
import squidpy as sq

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

# from scvi.data import register_tensor_from_anndata
from scvi.external import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI
from scipy.sparse import csr_matrix
from loguru import logger
import inspect
import gseapy as gp
from gseapy.plot import gseaplot

import sys

sc.set_figure_params(facecolor="white", figsize=(20, 20))
sc.settings.verbosity = 3

# from sc_adata import scRNA
# from st_adata import SpatialTranscriptomics
from src.STNav import STNav
from src.modules.plots import run_plots

username = os.path.expanduser("~")


class Orchestrator(object):
    _spatial_pipeline = STNav
    SCRNA = "scRNA"
    ST = "ST"

    def __init__(self, analysis_config, plotting_config) -> None:
        self.analysis_config = self._update_config(analysis_config)
        self.plotting_config = self._update_config(plotting_config)

    def _update_config(self, config):
        config["saving_path"] = username + config.get("saving_path", "")
        return config

    def run_analysis(self, saving_dir: str) -> an.AnnData:
        adata_dict = self.initialize_adata_dict()
        for data_type, data_type_dict in self.analysis_config.items():
            if data_type == "saving_path":
                continue
            self.run_pipeline_for_data_type(
                data_type, data_type_dict, adata_dict, saving_dir
            )

        return (
            adata_dict,
            self.sc_model,
            self.st_model,
        )

    def initialize_adata_dict(self):
        adata_dict = {}
        for data_type in self.analysis_config.keys():
            if data_type != "saving_path":
                adata_dict.setdefault(data_type, {})
        return adata_dict

    def run_pipeline_for_data_type(
        self, data_type, data_type_dict, adata_dict, saving_dir
    ):
        AnalysisPipeline = self._spatial_pipeline(
            config=self.analysis_config,
            saving_path=saving_dir,
            data_type=data_type,
            adata_dict=adata_dict,
        )
        logger.info(f"Running Analysis for {data_type}")

        self.run_analysis_steps(data_type, data_type_dict, AnalysisPipeline)

    def run_analysis_steps(self, data_type, data_type_dict, AnalysisPipeline):
        if data_type == self.SCRNA:
            self.sc_model = self.run_scrna_analysis_steps(
                data_type_dict, AnalysisPipeline
            )
        if data_type == self.ST:
            self.run_st_analysis_steps(AnalysisPipeline)

    def run_scrna_analysis_steps(self, data_type_dict, AnalysisPipeline):
        self.perform_surgery_if_needed(data_type_dict, AnalysisPipeline)
        AnalysisPipeline.QC()
        AnalysisPipeline.preprocessing()
        AnalysisPipeline.DEG()
        self.sc_model = AnalysisPipeline.train_or_load_sc_deconvolution_model()
        AnalysisPipeline.save_processed_adata(fix_write=True)

    def perform_surgery_if_needed(self, data_type_dict, AnalysisPipeline):
        if data_type_dict["scArches_surgery"]["usage"]:
            AnalysisPipeline.perform_scArches_surgery()
        elif data_type_dict["celltypist_surgery"]["usage"]:
            AnalysisPipeline.perform_celltypist()
        else:
            AnalysisPipeline.read_rna()

    def run_st_analysis_steps(self, AnalysisPipeline):
        AnalysisPipeline.read_visium()
        AnalysisPipeline.QC()
        AnalysisPipeline.preprocessing()
        AnalysisPipeline.DEG()
        self.st_model, model_name = (
            AnalysisPipeline.train_or_load_st_deconvolution_model(self.sc_model)
        )
        self.apply_subset_and_log(AnalysisPipeline)
        AnalysisPipeline.deconvolution(self.st_model, model_name)
        AnalysisPipeline.SpatiallyVariableGenes()
        AnalysisPipeline.SpatialNeighbors()
        AnalysisPipeline.ReceptorLigandAnalysis()
        AnalysisPipeline.save_processed_adata()

    def apply_subset_and_log(self, AnalysisPipeline):
        logger.warning(
            "Subset is now being applied to common genes between Spatial data and single cell data. Plots might not show the exact raw data because of this."
        )
        intersect = self.get_intersect(AnalysisPipeline)
        self.apply_subset(AnalysisPipeline, intersect)
        self.log_subset_info(AnalysisPipeline)

    def get_intersect(self, AnalysisPipeline):
        return np.intersect1d(
            AnalysisPipeline.adata_dict[self.SCRNA]["preprocessed_adata"].var_names,
            AnalysisPipeline.adata_dict[self.ST]["preprocessed_adata"].var_names,
        )

    def apply_subset(self, AnalysisPipeline, intersect):
        AnalysisPipeline.adata_dict[self.ST]["subset_preprocessed_adata"] = (
            AnalysisPipeline.adata_dict[self.ST]["preprocessed_adata"][
                :, intersect
            ].copy()
        )
        AnalysisPipeline.adata_dict[self.SCRNA]["subset_preprocessed_adata"] = (
            AnalysisPipeline.adata_dict[self.SCRNA]["preprocessed_adata"][
                :, intersect
            ].copy()
        )

    def log_subset_info(self, AnalysisPipeline):
        logger.info(
            f"N_obs x N_var for ST and scRNA after intersection: \n{AnalysisPipeline.adata_dict[self.ST]['subset_preprocessed_adata'].n_obs} x {AnalysisPipeline.adata_dict[self.ST]['subset_preprocessed_adata'].n_vars} \n {AnalysisPipeline.adata_dict[self.SCRNA]['subset_preprocessed_adata'].n_obs} x {AnalysisPipeline.adata_dict[self.SCRNA]['subset_preprocessed_adata'].n_vars}"
        )

    def run_plots(
        self,
        adata_dict: dict,
        saving_dir: str,
    ) -> None:

        logger.info(f"Running plotting functions.")

        # Run plots
        run_plots(
            plotting_config=self.plotting_config["plots"],
            adata_dict=adata_dict,
            directory=saving_dir,
        )
