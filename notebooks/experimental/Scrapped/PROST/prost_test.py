import os

ENVpath = "/home/rafaed/miniconda3/envs/PROST_ENV"
os.environ["R_HOME"] = f"{ENVpath}/lib/R"
os.environ["R_USER"] = f"{ENVpath}/lib/python3.7/site-packages/rpy2"
import pandas as pd
import numpy as np
import scanpy as sc
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import PROST

PROST.__version__


# Set seed
SEED = 818
PROST.setup_seed(SEED)


# Set directory (If you want to use additional data, please change the file path)
rootdir = "/mnt/work/RO_src/data/raw/VisiumHD"
output_dir = "/mnt/work/RO_src/Pipelines/STNav/notebooks/experimental/PROST"
spatial_dir = "/mnt/work/RO_src/data/raw/VisiumHD/square_008um/spatial"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Read data from input_dir
adata = sc.read_visium(
    path="/mnt/work/RO_src/data/raw/VisiumHD/square_008um/",
    count_file="filtered_feature_bc_matrix.h5",
)
adata.var_names_make_unique()
adata


# Calculate PI
adata = PROST.prepare_for_PI(adata, platform="visium")
adata = PROST.cal_PI(adata, platform="visium")

# Calculate spatial autocorrelation statistics and do hypothesis test
PROST.spatial_autocorrelation(adata, k=10, permutations=None)

# Save PI result
adata.write_h5ad(output_dir + "/PI_result.h5")


# Draw SVGs detected by PI
PROST.plot_gene(
    adata, platform="visium", size=2, sorted_by="PI", top_n=25, save_path=output_dir
)


n_clusters = 5
adata = sc.read(output_dir + "/PI_result.h5")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata = PROST.feature_selection(adata, by="prost", n_top_genes=3000)
PROST.run_PNN(
    adata,
    platform="visium",
    key_added="PROST",
    init="mclust",
    n_clusters=n_clusters,
    lap_filter=2,
    lr=0.1,
    SEED=SEED,
    max_epochs=500,
    tol=5e-3,
    post_processing=True,
    pp_run_times=3,
    cuda=False,
)
adata.write_h5ad(output_dir + "/PNN_result.h5")
clustering = adata.obs["clustering"]
clustering.to_csv(output_dir + "/clusters.csv", header=False)
pp_clustering = adata.obs["pp_clustering"]
pp_clustering.to_csv(output_dir + "/pp_clusters.csv", header=False)
embedding = adata.obsm["PROST"]
np.savetxt(output_dir + "/embedding.txt", embedding)
