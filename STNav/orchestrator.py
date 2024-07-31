# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from datetime import datetime

import anndata as an
import scanpy as sc

date: str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

from loguru import logger

sc.set_figure_params(facecolor="white", figsize=(20, 20))
sc.settings.verbosity = 3

from STNav import STNavCore
from STNav.modules.pl import run_plots
from STNav.modules.st import (
    ReceptorLigandAnalysis,
    SpatiallyVariableGenes,
    SpatialNeighbors,
    SpatialMarkersMapping,
    Deconvolution,
)

username: str = os.path.expanduser(path="~")


class Orchestrator(object):

    STNavCore_cls = STNavCore
    cell_markers_dict = None
    deconv_results = None

    def __init__(self, analysis_config, saving_path, plotting_config) -> None:
        self.analysis_config = analysis_config
        self.saving_path = saving_path
        self.plotting_config = plotting_config

    def run_analysis(self, saving_dir: str) -> an.AnnData:
        adata_dict = self.initialize_adata_dict()
        for data_type, data_type_dict in self.analysis_config.items():
            STNavCorePipeline = self.STNavCore_cls(
                config=self.analysis_config,
                saving_path=saving_dir,
                data_type=data_type,
                adata_dict=adata_dict,
            )

            self.run_st_analysis_steps(
                STNavCorePipeline=STNavCorePipeline,
            )

        return adata_dict

    def initialize_adata_dict(self):
        def process_checkpoint(params, data_type):
            if "checkpoint" in params and params["checkpoint"]["usage"]:
                pipeline_run = params["checkpoint"]["pipeline_run"]
                adata_name = params["save_as"]

                checkpoint_path = (
                    f"{self.saving_path}"
                    + "/"
                    + pipeline_run
                    + "/"
                    + f"{data_type}/Files"
                    + "/"
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
            adata_dict.setdefault(data_type, {})
            for step, params in steps.items():
                try:
                    if step in [
                        "SpatialNeighbors",
                        "SpatiallyVariableGenes",
                        "ReceptorLigandAnalysis",
                        "NicheAnalysis",
                    ]:
                        for method_name, config_params in params.items():
                            process_checkpoint(config_params, data_type)
                    process_checkpoint(params, data_type)
                except Exception as e:
                    logger.error(f"Error on {step}: {e}")
                    continue

        return adata_dict

    def run_st_analysis_steps(self, STNavCorePipeline):
        STNavCorePipeline.read_rna()
        STNavCorePipeline.QC()
        STNavCorePipeline.preprocessing()
        STNavCorePipeline.DEG()

        if STNavCorePipeline.config["ST"]["SpatialMarkersMapping"]["usage"]:
            mapping_config = STNavCorePipeline.config["ST"]["SpatialMarkersMapping"]
            MAPPING = SpatialMarkersMapping(STNavCorePipeline=STNavCorePipeline)
            self.cell_markers_dict = MAPPING.run_mapping(mapping_config=mapping_config)

        if STNavCorePipeline.config["ST"]["DeconvolutionModels"]["usage"]:
            DECONV = Deconvolution(STNavCorePipeline=STNavCorePipeline)
            self.deconv_results = DECONV.train_or_load_deconv_model()

        SpatiallyVariableGenes(STNavCorePipeline=STNavCorePipeline)
        SpatialNeighbors(STNavCorePipeline=STNavCorePipeline)
        ReceptorLigandAnalysis(STNavCorePipeline=STNavCorePipeline)

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
            cell_markers_dict=self.cell_markers_dict,
        )
