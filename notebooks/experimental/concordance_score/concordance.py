import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import cohen_kappa_score
import squidpy as sq

# Merge the three methods on common indices
OT_pred = sc.read_h5ad(r"/mnt/work/workbench/rafaed/work/RO_src/OT_annotated.h5ad")
LGBM_pred = sc.read_h5ad(
    r"/mnt/work/workbench/rafaed/work/RO_src/Pipelines/STNav/notebooks/experimental/SpatialLGBM/predicted_LGBM.h5ad"
)
scMAGS_pred = sc.read_h5ad(
    r"/mnt/work/workbench/rafaed/work/RO_src/data/processed/PipelineRun_2024_06_20-09_51_35_AM/ST/Files/deconvoluted_adata.h5ad"
)
scMAGS_pred.obs["cell_type"] = scMAGS_pred.obs["cell_type"].str.replace(
    "_Mean_LogNorm_Conn_Adj_scMAGS", "", regex=False
)

# Merge all three methods on their common indices:
file_paths = [
    r"/mnt/work/workbench/rafaed/work/RO_src/OT_annotated.h5ad",
    r"/mnt/work/workbench/rafaed/work/RO_src/Pipelines/STNav/notebooks/experimental/SpatialLGBM/predicted_LGBM.h5ad",
    r"/mnt/work/workbench/rafaed/work/RO_src/data/processed/PipelineRun_2024_06_20-09_51_35_AM/ST/Files/deconvoluted_adata.h5ad",
]

prediction_columns = ["cell_type_OT", "predicted_cell_type_non_conformal", "cell_type"]
dfs = []
for file_path, pred_col in zip(file_paths, prediction_columns):
    adata = sc.read_h5ad(file_path)

    if pred_col == "cell_type":

        adata.obs["cell_type"] = adata.obs["cell_type"].str.replace(
            "_Mean_LogNorm_Conn_Adj_scMAGS", "", regex=False
        )
    df = pd.DataFrame(adata.obs[pred_col], columns=[pred_col])
    dfs.append(df)

# Merge all DataFrames on their common indices
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.join(df, how="inner")

# Display the merged DataFrame
print(merged_df.head())

merged_df["cell_type"] = merged_df["cell_type"].str.replace(y
    "_Mean_LogNorm_Conn_Adj_scMAGS", "", regex=False
)
merged_df
import matplotlib.pyplot as plt
import squidpy as sq

cell_types = merged_df["cell_type_OT"].unique()

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=1000)

# Loop over the unique cell types
for cell_type in cell_types:
    # Create a scatter plot for each prediction
    for ax, adata, column in zip(
        axs, [OT_pred, LGBM_pred, scMAGS_pred], prediction_columns
    ):
        # Create a mask for the current cell type
        mask = adata.obs[column] == cell_type

        # Plot the cells of the current type
        # sc.pl.spatial(adata=adata[mask], color=column, ax=ax)

        sq.pl.spatial_scatter(
            adata[mask],
            color=column,
            size=0.9,
            ax=ax,
            alpha_img=0.5,
            legend_fontsize=3.5,  # adjust this value to make the legend smaller
        )
        # Set the title of the subplot
        ax.set_title(f"{column} - {cell_type}")

    # Save the figure
    plt.savefig(f"{cell_type}_scatter.png", dpi=1000)

    # Clear the figure for the next cell type
    plt.clf()

# Convert the categorical column to a string column
LGBM_pred.obs["predicted_cell_type_non_conformal"] = LGBM_pred.obs[
    "predicted_cell_type_non_conformal"
].astype(str)


sq.pl.spatial_scatter(
    OT_pred,
    color="cell_type_OT",
    size=0.9,
    alpha_img=0.5,
    legend_fontsize=3.5,  # adjust this value to make the legend smaller
)
plt.savefig(f"scatter.png", dpi=1000)

# List of prediction column names
# Read the data and extract predictions


# # Simulating the DataFrame from the given image data
# data = {
#     "prediction_OT": [
#         "T cell",
#         "Fibroblasts",
#         "AT1",
#         "AT1",
#         "B cell",
#         "T cell",
#         "T cell",
#         "Fibroblasts",
#     ],
#     "prediction_scMAGS": [
#         "T cell",
#         "Fibroblasts",
#         "Fibroblasts",
#         "AT1",
#         "B cell",
#         "AT1",
#         "T cell",
#         "Fibroblasts",
#     ],
#     "prediction_LGBM": [
#         "T cell",
#         "Fibroblasts",
#         "T cell",
#         "AT1",
#         "B cell",
#         "T cell",
#         "B cell",
#         "Fibroblasts",
#     ],
# }

# df = pd.DataFrame(data)


# Define methods and cell types
methods = ["cell_type_OT", "predicted_cell_type_non_conformal", "cell_type"]
cell_types = merged_df["cell_type_OT"].unique()

cell_types
# Identify all unique cell types across all prediction methods
unique_cell_types = pd.unique(merged_df.values.ravel("K"))
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

# Pairwise cohens kappa concordance score
# Plot concordance matrix for each cell type
for cell_type in unique_cell_types:
    # Initialize a concordance matrix for the current cell type
    concordance_matrix = pd.DataFrame(index=methods, columns=methods)

    # Calculate pairwise concordance scores for the current cell type
    for method1 in methods:
        for method2 in methods:
            # Filter for current cell type
            method1_filtered = merged_df[method1] == cell_type
            method2_filtered = merged_df[method2] == cell_type
            # Calculate concordance score
            intersection = sum(method1_filtered & method2_filtered)
            union = sum(method1_filtered | method2_filtered)

            if union == 0:
                jaccard_index = 0
            else:
                jaccard_index = intersection / union

            concordance_matrix.loc[method1, method2] = jaccard_index

    # Convert to float
    concordance_matrix = concordance_matrix.astype(float)

    # Plotting the concordance matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(concordance_matrix, annot=True, cmap="viridis", cbar=True)
    plt.title(f"Concordance Score Matrix for {cell_type}")
    plt.xlabel("Methods")
    plt.ylabel("Methods")
    plt.savefig(f"{cell_type} concordance")
    plt.show()
