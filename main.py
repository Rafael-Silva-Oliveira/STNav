# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import json
import os
from datetime import datetime

import scanpy as sc
from loguru import logger

from configs.constants import ANALYSIS_CONFIG, PLOTTING_CONFIG
from src.orchestrator import Orchestrator

sc.set_figure_params(facecolor="white", figsize=(20, 20))
sc.settings.verbosity = 3

username = os.path.expanduser("~")
date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


def main(ANALYSIS_CONFIG: str, PLOTTING_CONFIG: str):
    analysis_json = json.load(open(ANALYSIS_CONFIG))
    plotting_json = json.load(open(PLOTTING_CONFIG))

    ORCHESTRATOR = Orchestrator(
        analysis_config=analysis_json, plotting_config=plotting_json
    )
    directory = analysis_json["saving_path"] + "\\" + f"PipelineRun_{date}"

    subdirs = {
        Orchestrator.SCRNA: ["Files", "Model"],
        Orchestrator.ST: ["Files", "Model"],
        "Plots": [],
        "log": [],
    }

    # Check if the directory already exists
    if not os.path.exists(directory):
        for main_dir, sub_dir_list in subdirs.items():
            if sub_dir_list:
                for subdir in sub_dir_list:
                    # TODO: add a flag that adds the model sub-folder if the train model is set to true. Else, don't add that model sub-folder.
                    curr_dir = directory + "\\" + main_dir + "\\" + subdir
                    # Create the directory
                    os.makedirs(curr_dir)
                    print(f"Directory created successfully - {curr_dir=}")

            else:
                curr_dir = directory + "\\" + main_dir
                # Create the directory
                os.makedirs(curr_dir)
                print(f"Directory created successfully - {curr_dir=}")
    else:
        print("Directory already exists or is already populated with files!")

    # Has to be after creating the dir otherwise it will print directory already exists.
    logger.add(f"{directory}\\log\\loguru.log")
    logger.info(f"Directory where outputs will be saved: {directory}")
    (
        adata_dict,
        sc_model,
        st_model,
    ) = ORCHESTRATOR.run_analysis(saving_dir=directory)

    ORCHESTRATOR.run_plots(
        saving_dir=directory,
        adata_dict=adata_dict,
    )
    logger.info(f"Pipeline run on {date} sucessfully completed.")


if __name__ == "__main__":
    main(
        ANALYSIS_CONFIG=ANALYSIS_CONFIG,
        PLOTTING_CONFIG=PLOTTING_CONFIG,
    )
