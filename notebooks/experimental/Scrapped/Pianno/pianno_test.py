import pianno as po


data_path = "/mnt/work/RO_src/data/raw/VisiumHD/square_008um"
count_file = "filtered_feature_bc_matrix.h5"
adata = po.CreatePiannoObject(
    data_path=data_path, count_file=count_file, min_spots_prop=0.01
)

adata = po.SAVER(adata, layer_key="DenoisedX")

adata = po.CreateMaskImage(adata, scale_factor=1)

import csv
from collections import defaultdict

Patterndict = defaultdict(list)

with open(
    "/mnt/work/RO_src/data/processed/PipelineRun_2024_06_21-03_05_24_PM/ST/Files/ST_scMAGS_markers.csv",
    "r",
) as file:
    reader = csv.DictReader(file)
    for row in reader:
        Patterndict[row["CellType"]].append(row["Markers"])


config_path = "/mnt/work/RO_src/Pipelines/STNav/notebooks/experimental/Pianno"
# Convert defaultdict back to regular dict
Patterndict = dict(Patterndict)
adata = po.AutoPatternRecognition(
    adata,
    Patterndict=Patterndict,
    config_path=config_path,
    param_tuning=True,
    max_experiment_duration="10m",
)
import json

# Print the optimal parameters saved in the previous step.
with open(join(config_path, "best_params.json"), "r") as f:
    best_params_dict = json.load(f)
for key in best_params_dict:
    best_params = best_params_dict[key]
best_params

import scanpy as sc
import matplotlib as mpl

Patterndict = po.ProposedPatterndict(adata, top_n=10)
# Visualization of candidate marker genes
for k, v in Patterndict.items():
    print(k)
    print(v)
    with mpl.rc_context({"axes.facecolor": "black", "figure.figsize": [4.5, 5]}):
        sc.pl.spatial(
            adata,  # cmap='magma',
            layer="DenoisedX",
            color=v,
            ncols=5,
            size=5,
            spot_size=25,
            vmin=0,
            vmax="p99",
        )


adata = po.AnnotationImprovement(adata)
