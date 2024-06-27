import monkeybread as mb
import scanpy as sc
from loguru import logger
import os
import sys

# Add the project root to the sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from STNav.utils.decorators import pass_STNavCore_params
from STNav.utils.helpers import (
    return_filtered_params,
    SpatialDM_wrapper,
    save_processed_adata,
    return_from_checkpoint,
)

from datetime import datetime
import json


# Get current date
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# import spatialdm.plottings as pl
import squidpy as sq

mb_params = json.load(
    open(file=r"/mnt/work/RO_src/Pipelines/STAnalysis/configs/analysis_cloud.json")
)


@pass_STNavCore_params
def NicheAnalysis(STNavCorePipeline):
    step = "NicheAnalysis"
    config = STNavCorePipeline["ST"][step]

    logger.info("Performing niche analysis.")
    for method_name, config_params in config.items():
        if config_params["usage"]:

            # TODO: If config[save_as] in adata_dict.keys() then we already know the step has been checkpointed
            if return_from_checkpoint(
                STNavCorePipeline,
                config_params=config_params,
                checkpoint_step=step,
                method_name=method_name,
            ):
                continue
            adata_path = r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_26-08_35_14_AM/ST/Files/deconvoluted_adata.h5ad"
            # adata_path = STNavCorePipeline.adata_dict[STNavCorePipeline.data_type][
            #     config_params["adata_to_use"]
            # ]
            adata = sc.read_h5ad(adata_path)
            logger.info(
                f"Running {method_name} method configuration \n using the following adata {config_params['adata_to_use']}"
            )
            data_type = config_params["data_type"]

            if method_name == "Monkeybread":
                adata_neighbors = mb.calc.cellular_niches(
                    **return_filtered_params(
                        config=config_params["Niches"], adata=adata
                    )
                )


NicheAnalysis(STNavCorePipeline=mb_params)
