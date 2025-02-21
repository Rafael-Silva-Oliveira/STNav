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
import shutil

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

    analysis_json = json.load(fp=open(file=ANALYSIS_CONFIG[run_loc]))
    plotting_json = json.load(fp=open(file=PLOTTING_CONFIG[run_loc]))
    saving_path = SAVING_PATH[run_loc]["path"]

    ORCHESTRATOR = Orchestrator(
        analysis_config=analysis_json,
        saving_path=saving_path,
        plotting_config=plotting_json,
    )
    directory = saving_path + "/" + f"PipelineRun_{date}"

    subdirs = {
        "ST": ["Files"],
        "Plots": ["Paths"],
        "log": [],
        "dependencies": [],
        "configs": [],
    }

    # Check if the directory already exists
    if not os.path.exists(path=directory):
        for main_dir, sub_dir_list in subdirs.items():
            if sub_dir_list:
                for subdir in sub_dir_list:
                    curr_dir: str = directory + "/" + main_dir + "/" + subdir
                    # Create the directory
                    os.makedirs(name=curr_dir)
                    print(f"Directory created successfully - {curr_dir=}")

            else:
                curr_dir: str = directory + "/" + main_dir
                # Create the directory
                os.makedirs(name=curr_dir)
                print(f"Directory created successfully - {curr_dir=}")
    else:
        print("Directory already exists or is already populated with files!")

    # Has to be after creating the dir otherwise it will print directory already exists.
    with open(file=f"{directory}/dependencies/requirements.txt", mode="w") as f:
        subprocess.run(args=[sys.executable, "-m", "pip", "freeze"], stdout=f)
    with open(
        file=f"{directory}/configs/analysis.json", mode="w", encoding="utf-8"
    ) as f:
        json.dump(obj=analysis_json, fp=f, ensure_ascii=False, indent=4)
    with open(
        file=f"{directory}/configs/plotting.json", mode="w", encoding="utf-8"
    ) as f:
        json.dump(obj=plotting_json, fp=f, ensure_ascii=False, indent=4)

    logger.add(sink=f"{directory}/log/loguru.log")
    logger.info(f"Directory where outputs will be saved: {directory}")

    # Save adata_dict as json so it can be loaded directly to the run_plots without having to run the analysis
    if (
        plotting_json["run_just_plots"]["usage"]
        and plotting_json["run_just_plots"]["run_path"] != ""
    ):
        logger.warning(
            f"The setting 'run_just_plots' is set to True. Using the adata_dict from the path {plotting_json['run_just_plots']['run_path']}.\n\nNOTE: This will ONLY run the plotting config and not the analysis config and it will overwrite any previous plots in the pipeline. If you want to run the analysis from scratch, set 'run_just_plots' to 'false'."
        )
        adata_dict = json.load(open(file=plotting_json["run_just_plots"]["run_path"]))
        with open(
            file=f"{directory}/Plots/Paths/adata_dict.json", mode="w", encoding="utf-8"
        ) as f:
            json.dump(obj=adata_dict, fp=f, ensure_ascii=False, indent=4)

        # Copy each one of the adata from the adata_dict to the new path with the plots
        for data_type, paths in adata_dict.items():
            for file_name, old_path in paths.items():
                new_path: str = f"{directory}/{data_type}/Files/{file_name}.h5ad"
                shutil.copy(src=old_path, dst=new_path)
                print(
                    f"\n##############\nFile copied successfully from... \n{old_path=}\nto...\n{new_path=}"
                )
        ORCHESTRATOR.run_plots(
            saving_dir=directory,
            adata_dict=adata_dict,
        )
        logger.info(f"Plots from {directory} sucessfully generated/updated.")

    else:
        adata_dict: sc.AnnData = ORCHESTRATOR.run_analysis(saving_dir=directory)
        with open(
            file=f"{directory}/Plots/Paths/adata_dict.json", mode="w", encoding="utf-8"
        ) as f:
            json.dump(obj=adata_dict, fp=f, ensure_ascii=False, indent=4)

        # ORCHESTRATOR.run_plots(
        #     saving_dir=directory,
        #     adata_dict=adata_dict,
        # )

        logger.info(f"Pipeline run on {date} sucessfully completed.")


if __name__ == "__main__":
    main(
        ANALYSIS_CONFIG=ANALYSIS_CONFIG,
        SAVING_PATH=SAVING_PATH,
        PLOTTING_CONFIG=PLOTTING_CONFIG,
    )
