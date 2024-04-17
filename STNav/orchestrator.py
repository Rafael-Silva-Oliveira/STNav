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
)
from STNav.modules.st import (
    ReceptorLigandAnalysis,
    SpatiallyVariableGenes,
    SpatialNeighbors,
    Deconvolution,
)
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
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

        return adata_dict

    def initialize_adata_dict(self):
        def process_checkpoint(params, data_type):
            if "checkpoint" in params and params["checkpoint"]["usage"]:
                pipeline_run = params["checkpoint"]["pipeline_run"]
                adata_name = params["save_as"]

                checkpoint_path = (
                    f"{self.analysis_config['saving_path']}"
                    + "\\"
                    + pipeline_run
                    + "\\"
                    + f"{data_type}\\Files"
                    + "\\"
                    + f"{adata_name}.h5ad"
                )

                if os.path.exists(checkpoint_path):
                    adata_dict[data_type].update({adata_name: checkpoint_path})
                    logger.info(
                        f"Successfully created checkpoint on the {data_type} data type path for\n\n'{adata_name}' under the path:\n\n'{checkpoint_path}'."
                    )
                else:
                    logger.warning(
                        f"File '{checkpoint_path}' does not exist in the directory."
                    )

        adata_dict = {}
        for data_type, steps in self.analysis_config.items():
            if data_type != "saving_path":
                adata_dict.setdefault(data_type, {})
                for step, params in steps.items():
                    try:
                        if step in [
                            "SpatialNeighbors",
                            "SpatiallyVariableGenes",
                            "ReceptorLigandAnalysis",
                        ]:
                            for method_name, config_params in params.items():
                                process_checkpoint(config_params, data_type)
                        process_checkpoint(params, data_type)
                    except Exception as e:
                        logger.error(f"Error on {step}: {e}")
                        continue

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
        self.apply_subset_and_log(STNavCorePipeline)

        DECONV = Deconvolution(STNavCorePipeline)
        DECONV.run_deconvolution()
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

        sc_adata = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.SCRNA]["preprocessed_adata"]
        )
        st_adata = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.ST]["preprocessed_adata"]
        )

        return np.intersect1d(
            sc_adata.var_names,
            st_adata.var_names,
        )

    def apply_subset(self, STNavCorePipeline, intersect):
        st_adata_intersect = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.ST]["preprocessed_adata"]
        )[:, intersect]

        sc_adata_intersect = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.SCRNA]["preprocessed_adata"]
        )[:, intersect]

        save_processed_adata(
            STNavCorePipeline,
            name="subset_preprocessed_adata",
            adata=sc_adata_intersect,
            data_type=self.SCRNA,
        )
        save_processed_adata(
            STNavCorePipeline,
            name="subset_preprocessed_adata",
            adata=st_adata_intersect,
            data_type=self.ST,
        )

    def log_subset_info(self, STNavCorePipeline):

        st_adata = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.ST]["subset_preprocessed_adata"]
        )
        sc_adata = sc.read_h5ad(
            STNavCorePipeline.adata_dict[self.SCRNA]["subset_preprocessed_adata"]
        )

        logger.info(
            f"N_obs x N_var for ST and scRNA after intersection: \n{st_adata.n_obs} x {st_adata.n_vars} \n {sc_adata.n_obs} x {sc_adata.n_vars}"
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
