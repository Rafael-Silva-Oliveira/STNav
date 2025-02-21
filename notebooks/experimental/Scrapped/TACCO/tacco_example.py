import tacco as tc
import scanpy as sc
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt

adata = sc.read_h5ad(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_12-06_07_40_PM/ST/Files/raw_adata.h5ad"
)
reference = sc.read_h5ad(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_12-06_07_40_PM/scRNA/Files/raw_adata.h5ad"
)
reference.obs.columns
adata.var_names_make_unique()
reference.var_names_make_unique()


# Create a cell_type x gene markers
markers = pd.read_csv(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_07_02-10_56_44_AM/ST/Files/ST_scMAGS_markers.csv"
)

cell_type_markers_dict = markers.groupby("CellType")["Markers"].apply(list).to_dict()
cell_type_markers_dict

all_marker_genes = list(
    set([marker for markers in cell_type_markers_dict.values() for marker in markers])
)
all_marker_genes

reference_profiles = pd.DataFrame(
    data=adata.X.toarray(), columns=adata.var.index, index=adata.obs.index
)

reference_profiles


mthd = "OT"
tc.tl.annotate(
    adata,
    reference,
    bisections=0,
    bisection_divisor=3,
    max_annotation=1,
    annotation_key="ann_level_3_transferred_label",
    result_key=f"ann_{mthd}",
    method=mthd,
    remove_constant_genes=True,
    remove_zero_cells=True,
    remove_mito=True,
)


adata.obs[f"cell_type_{mthd}"] = adata.obsm[f"ann_{mthd}"].idxmax(axis=1)

adata.obs[f"cell_type_{mthd}"].value_counts()
adata.write_h5ad(f"annotated_adata_{mthd}.h5ad")

plt.style.use("default")
sq.pl.spatial_scatter(
    adata,
    color=f"cell_type_{mthd}",  # use the column from the whole dataframe
    size=1,
    dpi=1500,
    shape="square",
    save=f"cell_type_{mthd}.png",
    legend_fontsize=3.5,  # adjust this value to make the legend smaller
)
plt.savefig(f"{mthd}_scatter.png", dpi=1000)  # save the plot as a PNG file

unique_cell_types = adata.obs[f"cell_type_{mthd}"].unique()

for cell_type in unique_cell_types:
    # Create a subset of the data for the current cell type
    subset: sc.AnnData = adata[adata.obs[f"cell_type_{mthd}"] == cell_type].copy()
    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size as needed

    # Create the plot for the current cell type from the subset
    sc.pl.spatial(
        subset,
        color=f"cell_type_{mthd}",  # use the column from the subset
        size=0.9,
        alpha_img=0.5,
        library_id="Visium_HD_Human_Lung_Cancer",
        ax=axs[0],  # plot on the first subplot
        show=False,  # do not show the plot yet
    )

    # Create the plot for the current cell type from the whole dataframe
    sc.pl.spatial(
        adata,
        color=f"cell_type_{mthd}",  # use the column from the whole dataframe
        size=1.5,
        alpha_img=0.5,
        library_id="Visium_HD_Human_Lung_Cancer",
        ax=axs[1],  # plot on the second subplot
        show=False,  # do not show the plot yet
    )

    # Save the figure
    plt.savefig(f"{cell_type}_scatter.png", dpi=1000)  # save the plot as a PNG file
    plt.show()


# Concordancy score
