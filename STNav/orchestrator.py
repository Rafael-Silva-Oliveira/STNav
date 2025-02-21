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
    CCI,
    SpatiallyVariableGenes,
    SpatialNeighbors,
    SpatialMarkersMapping,
    Deconvolution,
    FunctionalAnalysis,
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
        """
        Initializes the adata_dict attribute with the paths to the saved AnnData objects.

        This function iterates over the steps and parameters in the analysis_config attribute and checks if a checkpoint
        is specified for each step. If a checkpoint is specified, it verifies if the checkpoint file exists and adds
        the path to the adata_dict attribute.

        Returns:
            dict: The initialized adata_dict attribute.
        """

        def process_checkpoint(params, data_type):
            """
            Processes the checkpoint for a given step and data type.

            Args:
                params (dict): The parameters for the step.
                data_type (str): The data type.
            """
            # Check if a checkpoint is specified for the step
            if "checkpoint" in params and params["checkpoint"]["usage"]:
                pipeline_run = params["checkpoint"]["pipeline_run"]
                adata_name = params["save_as"]

                # Construct the checkpoint path
                checkpoint_path = (
                    f"{self.saving_path}"
                    + "/"
                    + pipeline_run
                    + "/"
                    + f"{data_type}/Files"
                    + "/"
                    + f"{adata_name}.h5ad"
                )

                # Check if the checkpoint file exists
                if os.path.exists(checkpoint_path):
                    # Add the checkpoint path to the adata_dict attribute
                    adata_dict[data_type].update({adata_name: checkpoint_path})
                    logger.info(
                        f"Successfully created checkpoint on the {data_type} data type path for\n\n'{adata_name}' under the path:\n\n'{checkpoint_path}'."
                    )
                else:
                    logger.warning(
                        f"File '{checkpoint_path}' does not exist in the directory."
                    )

        # Initialize the adata_dict attribute
        adata_dict = {}

        # Iterate over the steps and parameters in the analysis_config attribute
        for data_type, steps in self.analysis_config.items():
            adata_dict.setdefault(data_type, {})
            for step, params in steps.items():
                try:
                    # Check if the step is one of the specified steps with checkpoints
                    if step in [
                        "SpatialNeighbors",
                        "SpatiallyVariableGenes",
                        "ReceptorLigandAnalysis",
                        "NicheAnalysis",
                    ]:
                        # Iterate over the method names and parameters in the step
                        for method_name, config_params in params.items():
                            process_checkpoint(config_params, data_type)

                    # Process the checkpoint for the step
                    process_checkpoint(params, data_type)
                except Exception as e:
                    # Log any errors that occur during processing
                    logger.error(f"Error on {step}: {e}")
                    continue

        return adata_dict

    def run_st_analysis_steps(self, STNavCorePipeline):
        """
        Runs the ST analysis steps on the given STNavCorePipeline object.

        Args:
            STNavCorePipeline (STNav.STNavCore): The STNavCorePipeline object to run the ST analysis steps on.
        """

        # Read RNA data
        STNavCorePipeline.read_rna()

        # Perform QC
        STNavCorePipeline.QC()

        # Preprocess the data
        STNavCorePipeline.preprocessing()

        # Perform DEG analysis
        STNavCorePipeline.DEG()

        # Check if SpatialMarkersMapping is enabled in the config
        if STNavCorePipeline.config["ST"]["SpatialMarkersMapping"]["usage"]:
            # Get the mapping config
            mapping_config = STNavCorePipeline.config["ST"]["SpatialMarkersMapping"]

            # Create a SpatialMarkersMapping object and run the mapping
            MAPPING = SpatialMarkersMapping(STNavCorePipeline=STNavCorePipeline)
            self.cell_markers_dict = MAPPING.run_mapping(mapping_config=mapping_config)

        # Check if DeconvolutionModels is enabled in the config
        if STNavCorePipeline.config["ST"]["DeconvolutionModels"]["usage"]:
            # Create a Deconvolution object and train or load the deconvolution model
            DECONV = Deconvolution(STNavCorePipeline=STNavCorePipeline)
            self.deconv_results = DECONV.train_or_load_deconv_model()

        # Perform SpatiallyVariableGenes analysis
        SpatiallyVariableGenes(STNavCorePipeline=STNavCorePipeline)

        # Perform SpatialNeighbors analysis
        SpatialNeighbors(STNavCorePipeline=STNavCorePipeline)

        # Perform ReceptorLigandAnalysis analysis
        CCI(STNavCorePipeline=STNavCorePipeline)

        # Pefrom function analysis
        FunctionalAnalysis(STNavCorePipeline=STNavCorePipeline)

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
