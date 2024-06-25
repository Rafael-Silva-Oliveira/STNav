# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import json
import os
from datetime import datetime
import sys

import scanpy as sc
from loguru import logger
import subprocess

from configs.constants import (
    ANALYSIS_CONFIG,
    SAVING_PATH,
    PLOTTING_CONFIG,
)
from STNav import Orchestrator

sc.set_figure_params(facecolor="white", figsize=(20, 20))
sc.settings.verbosity = 3

username: str = os.path.expanduser(path="~")
date: str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


def main(
    ANALYSIS_CONFIG: dict,
    PLOTTING_CONFIG: dict,
    SAVING_PATH: dict,
) -> None:

    assert not (
        SAVING_PATH["cloud"]["usage"] and SAVING_PATH["local"]["usage"]
    ), "Only one of 'cloud' or 'local' should be set to True"

    run_loc = "cloud" if SAVING_PATH["cloud"]["usage"] else "local"

    analysis_json = json.load(open(file=ANALYSIS_CONFIG[run_loc]))
    plotting_json = json.load(open(file=PLOTTING_CONFIG[run_loc]))
    saving_path = SAVING_PATH[run_loc]["path"]

    ORCHESTRATOR = Orchestrator(
        analysis_config=analysis_json,
        saving_path=saving_path,
        plotting_config=plotting_json,
    )
    directory = saving_path + "/" + f"PipelineRun_{date}"

    subdirs = {
        Orchestrator.SCRNA: ["Files"],
        Orchestrator.ST: ["Files"],
        "Plots": [],
        "log": [],
        "dependencies": [],
        "configs": [],
    }
    # TODO: save yaml file, as well as the config json files used in the pipeline with the versions of packages, etc
    # Check if the directory already exists
    if not os.path.exists(directory):
        for main_dir, sub_dir_list in subdirs.items():
            if sub_dir_list:
                for subdir in sub_dir_list:
                    curr_dir = directory + "/" + main_dir + "/" + subdir
                    # Create the directory
                    os.makedirs(curr_dir)
                    print(f"Directory created successfully - {curr_dir=}")

            else:
                curr_dir = directory + "/" + main_dir
                # Create the directory
                os.makedirs(curr_dir)
                print(f"Directory created successfully - {curr_dir=}")
    else:
        print("Directory already exists or is already populated with files!")

    # Has to be after creating the dir otherwise it will print directory already exists.
    with open(f"{directory}/dependencies/requirements.txt", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    # with open(f"{directory}/dependencies/requirements_pipreqs.txt", "r") as f:
    #     script_dir = os.path.dirname(os.path.realpath(__file__))
    #     subprocess.run(["pipreqs", script_dir, "--force"])
    with open(f"{directory}/configs/analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_json, f, ensure_ascii=False, indent=4)
    with open(f"{directory}/configs/plotting.json", "w", encoding="utf-8") as f:
        json.dump(plotting_json, f, ensure_ascii=False, indent=4)

    logger.add(f"{directory}/log/loguru.log")
    logger.info(f"Directory where outputs will be saved: {directory}")
    adata_dict = ORCHESTRATOR.run_analysis(saving_dir=directory)

    ORCHESTRATOR.run_plots(
        saving_dir=directory,
        adata_dict=adata_dict,
    )
    logger.info(f"Pipeline run on {date} sucessfully completed.")


if __name__ == "__main__":
    main(
        ANALYSIS_CONFIG=ANALYSIS_CONFIG,
        SAVING_PATH=SAVING_PATH,
        PLOTTING_CONFIG=PLOTTING_CONFIG,
    )
