# Load packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from datetime import datetime
import scanpy as sc
from loguru import logger

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
sc.settings.n_jobs >= -1
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


def perform_preprocessing(
    adata,
    target_sum=1e4,
    subset_hvg=True,
    min_mean_hvg=0.0125,
    max_mean_hvg=2,
    min_disp_hvg=0.25,
    flavor_hvg="seurat",
    n_top_genes=2000,
    zero_center=True,
    n_pcs=50,
    n_neighbors=15,
    resolution=1.0,
):

    logger.info(f"Running preprocessing.")

    # adata.X[0,:] -> this would give all the values/genes for 1 individual cell
    # adata.X[cells, genes]
    # adata.X[0,:].sum() would give the sum of UMI counts for a given cell
    logger.info(f"Current adata.X (raw data/counts) shape: \n {adata.X.shape = }")
    logger.info(
        f"Current adata.raw.X (raw data/counts) shape: \n {adata.raw.X.shape = }"
    )

    logger.info(
        f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X before any filters or normalization: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    )

    logger.info(
        f"adata.var contains the current gene information: \n {adata.var=} \n with the following columns: {adata.var.columns=}"
    )
    logger.info(
        f"adata.obs contains the current cell/spot information: \n {adata.obs=} \n with the following columns: {adata.obs.columns=}"
    )

    adata.layers["raw_counts"] = adata.X.copy()

    logger.info(f"Applying normalization.")

    logger.info(
        f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X before normalizing: \n 1 - {adata.X[0,:].sum()= } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    )
    sc.pp.normalize_total(adata, target_sum=target_sum)

    logger.info(
        f"\n The sum of UMIs from 3 first examples (cells scRNA or spots for ST) from adata.X after normalizing: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    )
    adata.layers["norm"] = adata.X

    # It requires a positional argument and not just keyword arguments
    # Get the parameters from return_filtered_params
    logger.info(f"Applying log1p")
    sc.pp.log1p(adata)

    logger.info(
        f"\n Applying the log changed the counts from UMI counts to log counts. The sum of log counts from the 3 first examples (cells for scRNA or spots for ST) from adata.X after applying log: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    )
    adata.layers["lognorm"] = adata.X

    logger.info(
        f"Examples from saved 'lognorm' layer: {adata.layers['lognorm'][0,10:80].toarray()}"
    )
    adata.raw = adata

    logger.info(f"Selecting highly variable genes")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        min_mean=min_mean_hvg,
        max_mean=max_mean_hvg,
        min_disp=min_disp_hvg,
        flavor=flavor_hvg,
        inplace=True,
        subset=subset_hvg,
    )

    logger.info(
        f"\n After applying highly variable genes, 3 first examples (cells for scRNA or spots for ST) from adata.X after applying highly variable genes: \n 1 - {adata.X[0,:].sum() = } \n 2 - {adata.X[1,:].sum() = } \n 3 - {adata.X[2,:].sum() = }"
    )

    logger.info(f"Applying scaling")
    sc.pp.scale(adata, zero_center=zero_center)

    logger.info("Adding extra info for plotting")

    logger.info(f"	Applying pca")
    # adata.obsm["X_pca"] is the embeddings
    # adata.uns["pca"] is pc variance
    # adata.varm['PCs'] is the loadings
    sc.tl.pca(adata, n_comps=n_pcs)

    logger.info(f"	Applying neighbors")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    logger.info(f"	Applying umap")
    sc.tl.umap(adata)

    logger.info("Adding extra info for clustering")

    logger.info(f"	Applying leiden")
    sc.tl.leiden(adata, resolution=resolution)

    assert "X_pca" in adata.obsm, f"There's no X_pca component in adata.obsm {adata=}"
    X_pca = adata.obsm["X_pca"]

    logger.info(f"	Applying dendogram")
    sc.tl.dendrogram(adata, groupby="leiden")

    # save the counts to a separate object for later, we need the normalized counts in raw for DEG dete.Save raw data before preprocessing values and further filtering
    adata.layers["preprocessed_counts"] = adata.X.copy()

    logger.info(f"Current adata.X shape after preprocessing: {adata.X.shape}")
    logger.info(
        f"Current adata.raw.X shape after preprocessing: \n {adata.raw.X.shape = }"
    )

    return adata
