import os
import random
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler

from nichecompass.models import NicheCompass
from nichecompass.utils import (
    add_gps_from_gp_dict_to_adata,
    compute_communication_gp_network,
    visualize_communication_gp_network,
    create_new_color_dict,
    extract_gp_dict_from_mebocost_es_interactions,
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_omnipath_lr_interactions,
    filter_and_combine_gp_dict_gps,
    generate_enriched_gp_info_plots,
)
from anndata import AnnData

adata2: AnnData = sc.read_h5ad(
    filename=r"/mnt/archive2/RO_src/data/processed/PipelineRun_2024_08_06-02_12_24_PM/ST/Files/preprocessed_adata.h5ad"
)
mask = (
    (adata2.obs["array_row"] > 25)
    & (adata2.obs["array_row"] < 975)
    & (adata2.obs["array_col"] > 50)
    & (adata2.obs["array_col"] < 1950)
)

adata = adata2[mask].copy()
# adata = sc.read_h5ad(r"/mnt/archive2/RO_src/data/adata_subset3.h5ad")
# Convert to np.float32 dtype
import numpy as np

adata.layers["raw_counts"] = adata.layers["raw_counts"].astype(np.float32)
### Dataset ###
dataset = "Visium"
species = "human"
spatial_key = "spatial"
n_neighbors = 4

### Model ###
# AnnData Keys
counts_key = "raw_counts"
adj_key = "spatial_connectivities"
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

# Architecture
conv_layer_encoder = "gcnconv"  # change to "gatv2conv" if enough compute and memory
active_gp_thresh_ratio = 0.01

# Trainer
n_epochs = 10
n_epochs_all_gps = 10
lr = 0.001
lambda_edge_recon = 500000.0
lambda_gene_expr_recon = 300.0
lambda_l1_masked = 0.0  # increase if gene selection desired
lambda_l1_addon = 100.0
edge_batch_size = 1024  # increase if more memory available
n_sampled_neighbors = 4
use_cuda_if_available = True

### Analysis ###
cell_type_key = "cell_type"
latent_leiden_resolution = 0.4
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 0.2
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"
# Get time of notebook execution for timestamping saved artifacts
now = datetime.now()
current_timestamp = now.strftime("%d%m%Y_%H%M%S")
# Define paths
ga_data_folder_path = "../../../data/gene_annotations"
gp_data_folder_path = "../../../data/gene_programs"
so_data_folder_path = "../../../data/spatial_omics"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
collectri_tf_network_file_path = (
    f"{gp_data_folder_path}/collectri_tf_network_{species}.csv"
)
nichenet_lr_network_file_path = (
    f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
)
nichenet_ligand_target_matrix_file_path = (
    f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"
)
mebocost_enzyme_sensor_interactions_folder_path = (
    f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
)
gene_orthologs_mapping_file_path = (
    f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
)
artifacts_folder_path = f"../../../artifacts"
model_folder_path = f"{artifacts_folder_path}/single_sample/{current_timestamp}/model"
figure_folder_path = (
    f"{artifacts_folder_path}/single_sample/{current_timestamp}/figures"
)
os.makedirs(model_folder_path, exist_ok=True)
os.makedirs(figure_folder_path, exist_ok=True)
os.makedirs(so_data_folder_path, exist_ok=True)
os.makedirs(gp_data_folder_path, exist_ok=True)
os.makedirs(ga_data_folder_path, exist_ok=True)
# Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    species=species,
    min_curation_effort=0,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=omnipath_lr_network_file_path,
    plot_gp_gene_count_distributions=True,
    gp_gene_count_distributions_save_path=f"{figure_folder_path}"
    "/omnipath_gp_gene_count_distributions.svg",
)
# Display example OmniPath GP
omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]
print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")
# Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)
mebocost_gp_dict = extract_gp_dict_from_mebocost_es_interactions(
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,
    species=species,
    plot_gp_gene_count_distributions=True,
)
# Display example MEBOCOST GP
mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]
print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")
# Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)
nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,
    version="v2",
    keep_target_genes_ratio=1.0,
    max_n_target_genes_per_gp=250,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=nichenet_lr_network_file_path,
    ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True,
)
# Display example NicheNet GP
nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]
print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")
# Add GPs into one combined dictionary for model training
combined_gp_dict = dict(omnipath_gp_dict)
combined_gp_dict.update(mebocost_gp_dict)
combined_gp_dict.update(nichenet_gp_dict)
# Filter and combine GPs to avoid overlaps
combined_new_gp_dict = filter_and_combine_gp_dict_gps(
    gp_dict=combined_gp_dict,
    gp_filter_mode="subset",
    combine_overlap_gps=True,
    overlap_thresh_source_genes=0.9,
    overlap_thresh_target_genes=0.9,
    overlap_thresh_genes=0.9,
)

print(
    "Number of gene programs before filtering and combining: "
    f"{len(combined_gp_dict)}."
)
print(
    f"Number of gene programs after filtering and combining: "
    f"{len(combined_new_gp_dict)}."
)
# Compute spatial neighborhood
sq.gr.spatial_neighbors(
    adata, coord_type="generic", spatial_key=spatial_key, n_neighs=n_neighbors
)

# Make adjacency matrix symmetric
adata.obsp[adj_key] = adata.obsp[adj_key].maximum(adata.obsp[adj_key].T)
# Add the GP dictionary as binary masks to the adata
add_gps_from_gp_dict_to_adata(
    gp_dict=combined_new_gp_dict,
    adata=adata,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=1,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None,
)
cell_type_colors = create_new_color_dict(adata=adata, cat_key=cell_type_key)
print(f"Number of nodes (observations): {adata.layers['raw_counts'].shape[0]}")
print(f"Number of node features (genes): {adata.layers['raw_counts'].shape[1]}")

# Visualize cell-level annotated data in physical space
sc.pl.spatial(adata, color=cell_type_key, palette=cell_type_colors, spot_size=spot_size)
adata.layers["counts"] = adata.layers["raw_counts"].copy()
# Initialize model
model = NicheCompass(
    adata,
    counts_key=counts_key,
    adj_key=adj_key,
    gp_names_key=gp_names_key,
    active_gp_names_key=active_gp_names_key,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    latent_key=latent_key,
    conv_layer_encoder=conv_layer_encoder,
    active_gp_thresh_ratio=active_gp_thresh_ratio,
)
# Train model
model.train(
    n_epochs=n_epochs,
    n_epochs_all_gps=n_epochs_all_gps,
    lr=lr,
    lambda_edge_recon=lambda_edge_recon,
    lambda_gene_expr_recon=lambda_gene_expr_recon,
    lambda_l1_masked=lambda_l1_masked,
    edge_batch_size=edge_batch_size,
    n_sampled_neighbors=n_sampled_neighbors,
    use_cuda_if_available=use_cuda_if_available,
    verbose=False,
)
# Save trained model
model.save(
    dir_path=model_folder_path,
    overwrite=True,
    save_adata=True,
    adata_file_name="adata2.h5ad",
)
