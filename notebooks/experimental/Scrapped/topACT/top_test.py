import pandas as pd

from scipy import io

from topact.countdata import CountMatrix
from topact.classifier import SVCClassifier, train_from_countmatrix
from topact import spatial
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse
from loguru import logger
from tqdm import tqdm
import anndata as ad

import scanpy as sc

sc_adata = sc.read_h5ad(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_30-08_30_51_PM/scRNA/Files/raw_adata.h5ad"
)

sc_adata.obs["ann_level_3_transferred_label"].value_counts()


def readfile(filename) -> list[str]:
    with open(file=filename) as f:
        return [line.rstrip() for line in f]


adata_st: sc.AnnData = sc.read_h5ad(
    filename=r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_15-09_33_10_AM/ST/Files/raw_adata.h5ad"
)
# adata_st_sub = sc.pp.subsample(adata_st, fraction=0.05, copy=True)
# adata_st_sub
sc.pp.filter_genes(adata_st, min_counts=50)
sc.pp.filter_cells(adata_st, min_genes=3)
sc.pp.highly_variable_genes(adata_st, n_top_genes=5000, flavor="seurat_v3", subset=True)

adata_sc: sc.AnnData = sc.read_h5ad(
    filename=r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_15-09_33_10_AM/scRNA/Files/raw_adata.h5ad"
)


def annData_converter(
    sc_adata: ad.AnnData,
    st_adata: ad.AnnData,
    sc_annotation: str = "ann_level_3_transferred_label",
    load_from_parquet: bool = False,
):
    """
    Converts single-cell and spatial transcriptomics data into a format suitable for further analysis.

    This function makes gene names unique, extracts count matrix, gene names, and labels from single-cell data,
    creates a wide format count matrix from spatial data, stacks each individual gene in long format, and
    concatenates all stacked genes into a single long format dataframe.

    Args:
        sc_adata (anndata.AnnData): The single-cell RNA-seq data in an AnnData object.
        st_adata (anndata.AnnData): The spatial transcriptomics data in an AnnData object.
        sc_annotation (str, optional): The annotation level to use from the single-cell data.
            Defaults to "ann_level_3_transferred_label".

    Returns:
        mtx (scipy.sparse.coo_matrix): The count matrix from the single-cell data.
        genes (list): The list of gene names.
        labels (list): The list of labels.
        counts_stacked (pandas.DataFrame): The concatenated dataframe of all stacked genes in long format.
    """

    logger.info("Making gene names unique for both scRNA and ST dataset.")
    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()

    logger.info(f"Shape of st_adata is {st_adata}")
    logger.info(f"Shape of sc_adata is {sc_adata}")

    # Extract the count matrix from the single-cell data
    mtx = scipy.sparse.coo_matrix(sc_adata.X).astype(dtype="int64")

    # Extract the gene names
    genes: list = sc_adata.var_names.tolist()

    # Extract the labels
    labels: list = sc_adata.obs[sc_annotation].tolist()

    if load_from_parquet:
        counts_stacked: pd.DataFrame = pd.read_parquet(path="counts_stacked.parquet")
    else:
        # Extract the columns from spatial_adata.obs that contain the x,y coordinates (row = X, col = Y)
        spatial_data = st_adata.obs[["array_row", "array_col"]]

        logger.info(
            "Creating a wide format count matrix from spatial data with gene as columns and cell ID as rows."
        )
        # Create the DataFrame in wide format
        counts = pd.DataFrame.sparse.from_spmatrix(
            st_adata.X,
            columns=st_adata.var_names,
            index=spatial_data.index,
        )

        # Create an empty list to save each stacked gene in long format
        stacked_columns = []

        logger.info("Stacking each individual gene in long format")
        for column in tqdm(counts.columns):
            stacked_column = (
                counts[column]
                .sparse.to_dense()
                .replace(0, np.nan)
                .dropna()
                .rename("counts")
                .to_frame()
            )
            stacked_column["gene"] = column

            merged_stacked_column = (
                stacked_column.merge(spatial_data, left_index=True, right_index=True)
                .rename(columns={"array_row": "x", "array_col": "y"})
                .reindex(columns=["gene", "x", "y", "counts"])
            )

            # Making sure that x and y are integers
            merged_stacked_column["x"] = merged_stacked_column["x"].astype(int)
            merged_stacked_column["y"] = merged_stacked_column["y"].astype(int)

            # Concate the current gene to the list of stacked columns
            stacked_columns.append(merged_stacked_column)

        logger.info(
            "Concatenating all stacked genes into a single long format dataframe"
        )

        # Append all stacked genes into a single dataframe with columns gene, x,y, counts
        counts_stacked: pd.DataFrame = pd.concat(stacked_columns)

        # Order x and y
        counts_stacked = counts_stacked.sort_values(by=["x", "y"])

        counts_stacked.to_parquet(path="counts_stacked_small.parquet")

    logger.info(
        f"Spatial dataframe of size {counts_stacked.shape} created.\n{counts_stacked}"
    )

    return mtx, genes, labels, counts_stacked


if __name__ == "__main__":
    # Convert AnnData to necessary formats
    mtx, genes, labels, spatial_data = annData_converter(
        sc_adata=adata_sc, st_adata=adata_st, load_from_parquet=False
    )
    spatial_data["x"] = spatial_data["x"].astype(int)
    spatial_data["y"] = spatial_data["y"].astype(int)

    # Create TopACT object
    sc_obj = CountMatrix(matrix=mtx, genes=genes)
    sc_obj.add_metadata(header="celltype", values=labels)

    # Train local classifier
    clf = SVCClassifier()
    train_from_countmatrix(classifier=clf, countmatrix=sc_obj, label="celltype")

    # Passing in genes will automatically filter out genes that are not in the
    # single cell reference
    logger.info("Finished training cls")
    sd: spatial.CountGrid = spatial.CountGrid.from_coord_table(
        table=spatial_data, genes=genes, count_col="counts", gene_col="gene"
    )

    # Classify
    logger.info("Classifying spatial data")
    sd.classify_parallel(
        classifier=clf,
        min_scale=3,
        max_scale=9,
        num_proc=8,
        outfile="outfile.npy",
    )

    confidence_mtx = np.load(file="outfile.npy")

    logger.info("Extracting annotations and image")
    annotations = spatial.extract_image(confidence_matrix=confidence_mtx, threshold=0.5)

    np.savetxt(fname="demo-output-final.txt", X=annotations, fmt="%1.f")
    logger.info(f"Annotations: {annotations}\n confidence matrix: {confidence_mtx}")
    plt.imshow(X=annotations, interpolation="None")
    plt.savefig("demo-output-final.png")
