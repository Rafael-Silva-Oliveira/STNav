import scanpy as sc
import celltypist
from celltypist import models
import json
import squidpy as sq
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colorspacious import cspace_converter
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from tqdm import tqdm
# Map probe sequences to each bin
import gzip
import json
import io
import tarfile
import pandas as pd
import scipy.io as sio
import numpy as np
import scanpy as sc
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colorspacious import cspace_converter
from scipy.sparse import csr_matrix
from celltypist import models
from difflib import SequenceMatcher


adata_st = sc.read_h5ad(r"/mnt/work/RO_src/STAnalysis/notebooks/experimental/B2C/adata_b2c.h5ad")
adata_st


# Path to the Visium dataset directory
#visium_path = "/mnt/archive2/RO_src/data/raw/VisiumHD/square_008um"

# Read the Visium dataset
#adata_st = sc.read_visium(
#    path=visium_path,
#    count_file='filtered_feature_bc_matrix.h5',
#    load_images=True  # Set to False if you don't need the images or they're not available
#)
adata_st.var_names_make_unique()
sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)

# Load scRNA data

# Train a model with the SCLC dataset

sc_adata = sc.read_h5ad(r"/mnt/archive2/RO_src/data/raw/scRNA/SCLC/Combined samples.h5ad")
sc_adata.var.set_index("feature_name", inplace=True)
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc_adata.var_names = sc_adata.var_names.str.upper()
sc_adata.var.index = sc_adata.var.index.str.upper()

sc_adata = sc_adata[~sc_adata.obs["cell_type_fine"].isin(["Hepatocyte", "NSCLC"])]
sc_adata = sc_adata[
    sc_adata.obs["histo"].isin(["normal", "SCLC"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]
sc_adata.write_h5ad(r"/mnt/archive2/RO_src/data/raw/scRNA/SCLC/SCLC_lung.h5ad")

healthy_adata = sc_adata[
    sc_adata.obs["histo"].isin(["normal"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]

sclc_adata = sc_adata[
    sc_adata.obs["histo"].isin(["SCLC"])
    & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
    :,
]

# subset_sc_adata = sc_adata[sc_adata.obs["histo"].isin(["SCLC","normal"]) & sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]), :]


healthy_adata.obs["cell_type_fine"].value_counts()

sclc_adata.obs["cell_type_fine"].value_counts()


sc_model_healthy = celltypist.train(
    healthy_adata,
    labels="cell_type_fine",
    n_jobs=10,
    feature_selection=True,
    epochs=30000,
)
sc_model_healthy.write(
    "/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/celltypist_models/model_healthy_lung.pkl"
)

sc_model_sclc = celltypist.train(
    sclc_adata, labels="cell_type_fine", n_jobs=10, feature_selection=True, epochs=30000
)
sc_model_sclc.write(
    "/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/celltypist_models/model_sclc_lung.pkl"
)


sc_model_sclc = models.Model.load(
    model="/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/celltypist_models/model_sclc_lung.pkl"
)
sc_model_healthy = models.Model.load(
    model="/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA/celltypist_models/model_healthy_lung.pkl"
)


# # Add spatial connectivities4

# sq.gr.spatial_neighbors(adata_st)

# adata_st.obsp["spatial_connectivities"] = csr_matrix(
#     adata_st.obsp["spatial_connectivities"]
# )

# adata_st.X = adata_st.obsp["spatial_connectivities"].dot(adata_st.X)


predictions_healthy = celltypist.annotate(
    adata_st, model=sc_model_healthy, majority_voting=False
)

predictions_sclc = celltypist.annotate(
    adata_st, model=sc_model_sclc, majority_voting=False
)

# combine 2 models for cell annotations
# add the healthy gut model

pred_healthy = predictions_healthy.to_adata()
pred_healthy.obs["predicted_labels_healthy"] = pred_healthy.obs["predicted_labels"]
pred_healthy.obs["conf_score_healthy"] = pred_healthy.obs["conf_score"]

# add the colorectal cancer model
pred_sclc = predictions_sclc.to_adata()
pred_sclc.obs["predicted_labels_crc"] = pred_sclc.obs["predicted_labels"]
pred_sclc.obs["conf_score_crc"] = pred_sclc.obs["conf_score"]

# remove old annotations
# del cdata.obs["predicted_labels"]
# del cdata.obs["conf_score"]

cdata = adata_st.copy()
# find the cell that have higher confidence in the crc model
cdata.obs["higher_in_crc"] = (
    pred_healthy.obs["conf_score_healthy"] < pred_sclc.obs["conf_score_crc"]
)

# eclude the ones that labeled as unknown in crc model
cdata.obs.loc[pred_sclc.obs["predicted_labels_crc"] == "Unknown", "higher_in_crc"] = (
    False
)
cdata.obs["higher_in_crc"].value_counts()

# create new unified annotations
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels_healthy"]
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("object")
cdata.obs.loc[cdata.obs["higher_in_crc"], "predicted_labels"] = cdata.obs.loc[
    cdata.obs["higher_in_crc"], "predicted_labels_crc"
]
cdata.obs["predicted_labels"] = cdata.obs["predicted_labels"].astype("category")
cdata.obs["predicted_labels"]

cdata.obs["predicted_labels"].value_counts()


cdata.obs["array_row"] = cdata.obs["array_row"].astype(int)
cdata.obs["array_col"] = cdata.obs["array_col"].astype(int)
cdata.obsm["spatial"] = cdata.obsm["spatial"].astype(float)
cdata.obs["in_tissue"] = cdata.obs["in_tissue"].astype(int)

cdata.write_h5ad(
    "/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/combined_adata_b2c_nonadjusted.h5ad"
)

cdata = sc.read_h5ad(
    r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/combined_adata.h5ad"
)

cdata.obs["predicted_labels"].value_counts()

# Generate a Glasbey color palette with 25 distinct colors
glasbey = sns.color_palette("tab20", 25)
glasbey_rgb = np.array(glasbey) * 255

# Convert the seaborn color palette to a ListedColormap object
palette = ListedColormap(glasbey_rgb / 255.0)

plt.style.use("default")
sq.pl.spatial_scatter(
    cdata,
    color=f"predicted_labels",  # use the column from the whole dataframe
    size=1,
    palette=palette,
    dpi=1500,
    shape="square",
    save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/cell_typist_sclc_nonadjusted.png",
    legend_fontsize=3.5,  # adjust this value to make the legend smaller
)

palette = ListedColormap(["#FFA500"])  # Hex code for orange color

for cell_type in list(cdata.obs["predicted_labels"].unique()):
    # Create a subset of the data for the current cell type
    subset: sc.AnnData = cdata[cdata.obs[f"predicted_labels"] == cell_type].copy()
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size as needed
    sq.pl.spatial_scatter(
        subset,
        color="predicted_labels",  # use the column from the whole dataframe
        size=1,
        palette=palette,
        dpi=1500,
        shape="square",
        save=f"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/experimental/celltypist_demo_folder/{cell_type}_sclc_nonadjusted.png",
        legend_fontsize=3.5,  # adjust this value to make the legend smaller
    )

from Bio import pairwise2
import pandas as pd
import numpy as np

# Load clone and probe data
def load_clone_data(clone_data_path: str) -> pd.DataFrame:
    """
    Load the combined_clones_data.tsv into a pandas DataFrame.
    """
    clone_df = pd.read_csv(clone_data_path, sep='\t')
    return clone_df

def extract_clone_sequences(clone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clone sequences and relevant columns for further analysis.
    """
    return clone_df[['clone_type', 'targetSequences']]

# Paths to your files
clone_data_path = '/mnt/work/RO_src/Projects/THORA/ImmuneReservoireAnalysis/results_rna/combined_clones_data.tsv'
probe_set_path = '/mnt/archive2/RO_src/data/raw/VisiumHD/Visium_HD_Human_Lung_Cancer_probe_set.csv'

# Load data
clone_df = load_clone_data(clone_data_path)
extracted_clones = extract_clone_sequences(clone_df)
bulk_sequences = extracted_clones["targetSequences"].tolist()
probe_set = pd.read_csv(probe_set_path, comment='#')

# Define a similarity threshold for matching
similarity_threshold = 0.8

# Function to align sequences and check for similarity
def is_sequence_match(probe_seq, bulk_sequences, threshold):
    """
    Checks if the probe sequence matches any of the bulk RNA-seq sequences.
    Returns True if there is a match.
    """
    for bulk_seq in bulk_sequences:
        alignments = pairwise2.align.globalxx(probe_seq, bulk_seq)
        score = max(align.score for align in alignments) / min(len(probe_seq), len(bulk_seq))
        if score >= threshold:
            return True
    return False

# Initialize a dictionary to store match information
probe_matches = {}

# Check each probe sequence for matches
for idx, row in probe_set.iterrows():
    gene_id = row['gene_id']
    probe_seq = row['probe_seq']
    
    # Check if this probe sequence matches any bulk RNA-seq sequence
    match_found = is_sequence_match(probe_seq, bulk_sequences, similarity_threshold)
    
    # Store match result
    if gene_id not in probe_matches:
        probe_matches[gene_id] = []
    probe_matches[gene_id].append({
        "probe_seq": probe_seq,
        "match_found": match_found
    })

# Create a DataFrame for better visualization of match results
match_results = []
for gene_id, probes in probe_matches.items():
    for probe in probes:
        match_results.append({
            "gene_id": gene_id,
            "probe_seq": probe["probe_seq"],
            "match_found": probe["match_found"]
        })

match_df = pd.DataFrame(match_results)

# Display the DataFrame with matching results
print(match_df)

# To analyze the results further or visualize the matched genes, you could
# aggregate the data by gene and visualize which genes have matching probes.
# For instance, you could check how many probes per gene have a match.
matched_genes = match_df.groupby("gene_id")["match_found"].any()
print(matched_genes)