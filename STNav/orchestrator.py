# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from datetime import datetime

import anndata as an
import numpy as np
import scanpy as sc

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

from loguru import logger

sc.set_figure_params(facecolor="white", figsize=(20, 20))
sc.settings.verbosity = 3

from STNav import STNavCore
from STNav.modules.pl import run_plots
from STNav.modules.sc import (
    perform_celltypist,
    perform_scArches_surgery,
    train_or_load_sc_deconvolution_model,
)
from STNav.modules.st import (
    ReceptorLigandAnalysis,
    SpatiallyVariableGenes,
    SpatialNeighbors,
    deconvolution,
    train_or_load_st_deconvolution_model,
)

username = os.path.expanduser("~")


class Orchestrator(object):

    STNavCore_cls = STNavCore
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
        STNavCorePipeline = self.STNavCore_cls(
            config=self.analysis_config,
            saving_path=saving_dir,
            data_type=data_type,
            adata_dict=adata_dict,
        )
        logger.info(f"Running Analysis for {data_type}")

        self.run_analysis_steps(data_type, data_type_dict, STNavCorePipeline)

    def run_analysis_steps(self, data_type, data_type_dict, STNavCorePipeline):
        if data_type == self.SCRNA:
            self.run_scrna_analysis_steps(data_type_dict, STNavCorePipeline)
        if data_type == self.ST:
            self.run_st_analysis_steps(STNavCorePipeline)

    def run_scrna_analysis_steps(self, data_type_dict, STNavCorePipeline):
        self.perform_surgery_if_needed(data_type_dict, STNavCorePipeline)
        STNavCorePipeline.QC()
        STNavCorePipeline.preprocessing()
        STNavCorePipeline.DEG()
        self.sc_model = train_or_load_sc_deconvolution_model(STNavCorePipeline)

    def perform_surgery_if_needed(self, data_type_dict, STNavCorePipeline):
        if data_type_dict["cell_annotation"]["scArches_surgery"]["usage"]:
            adata_raw = perform_scArches_surgery(STNavCorePipeline)
            STNavCorePipeline.adata_dict[STNavCorePipeline.data_type].setdefault(
                "raw_adata", adata_raw
            )
        elif data_type_dict["cell_annotation"]["celltypist_surgery"]["usage"]:
            adata_raw = perform_celltypist(STNavCorePipeline)
            STNavCorePipeline.adata_dict[STNavCorePipeline.data_type].setdefault(
                "raw_adata", adata_raw
            )
        else:
            STNavCorePipeline.read_rna()

    def run_st_analysis_steps(self, STNavCorePipeline):
        STNavCorePipeline.read_visium()
        STNavCorePipeline.QC()
        STNavCorePipeline.preprocessing()
        STNavCorePipeline.DEG()
        self.st_model, model_name = train_or_load_st_deconvolution_model(
            STNavCorePipeline=STNavCorePipeline
        )
        self.apply_subset_and_log(STNavCorePipeline)

        # External modules to the core pipeline
        deconvolution(STNavCorePipeline, st_model=self.st_model, model_name=model_name)
        SpatiallyVariableGenes(STNavCorePipeline)
        SpatialNeighbors(STNavCorePipeline)
        ReceptorLigandAnalysis(STNavCorePipeline)

    def apply_subset_and_log(self, STNavCorePipeline):
        logger.warning(
            "Subset is now being applied to common genes between Spatial data and single cell data. Plots might not show the exact raw data because of this."
        )
        intersect = self.get_intersect(STNavCorePipeline)
        self.apply_subset(STNavCorePipeline, intersect)
        self.log_subset_info(STNavCorePipeline)

    def get_intersect(self, STNavCorePipeline):
        return np.intersect1d(
            STNavCorePipeline.adata_dict[self.SCRNA]["preprocessed_adata"].var_names,
            STNavCorePipeline.adata_dict[self.ST]["preprocessed_adata"].var_names,
        )

    def apply_subset(self, STNavCorePipeline, intersect):
        STNavCorePipeline.adata_dict[self.ST]["subset_preprocessed_adata"] = (
            STNavCorePipeline.adata_dict[self.ST]["preprocessed_adata"][
                :, intersect
            ].copy()
        )
        STNavCorePipeline.adata_dict[self.SCRNA]["subset_preprocessed_adata"] = (
            STNavCorePipeline.adata_dict[self.SCRNA]["preprocessed_adata"][
                :, intersect
            ].copy()
        )

    def log_subset_info(self, STNavCorePipeline):
        logger.info(
            f"N_obs x N_var for ST and scRNA after intersection: \n{STNavCorePipeline.adata_dict[self.ST]['subset_preprocessed_adata'].n_obs} x {STNavCorePipeline.adata_dict[self.ST]['subset_preprocessed_adata'].n_vars} \n {STNavCorePipeline.adata_dict[self.SCRNA]['subset_preprocessed_adata'].n_obs} x {STNavCorePipeline.adata_dict[self.SCRNA]['subset_preprocessed_adata'].n_vars}"
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
