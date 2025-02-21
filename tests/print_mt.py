import scanpy as sc

if config["print_mt"]:
    adata_cp = adata.copy()
    sc.pp.neighbors(adata_cp, n_pcs=30, n_neighbors=20)
    sc.tl.leiden(adata_cp, key_added="leiden_clusters")
    sc.tl.rank_genes_groups(
        adata_cp,
        groupby="leiden_clusters",
        method="wilcoxon",
        key_added="wilcoxon",
        n_genes=1000,
        use_raw=False,
    )

    # Get top 10 MT DEG genes
    adata_cp_top_genes = sc.get.rank_genes_groups_df(
        adata_cp, key="wilcoxon", pval_cutoff=0.1, log2fc_min=1, group=None
    )
    adata_cp_top_genes_MT = adata_cp_top_genes.loc[
        adata_cp_top_genes["names"].str.contains("MT-"), :
    ]
    adata_cp_top_genes_MT.sort_values(
        by="logfoldchanges", inplace=True, ascending=False
    )
    top_10_MT_genes = adata_cp_top_genes_MT["names"].tolist()[:8]
    # sc.pl.spatial(adata_cp, img_key="hires", color=top_10_MT_genes)
    # sc.pl.spatial(adata_cp, img_key="hires", color="leiden_clusters")

    # tmp = pd.crosstab(adata.obs["cell_ontology_class"], adata.obs["leiden_clusters"], normalize="index")
    # tmp.plot.bar(stacked=True).legend(loc="upper right")
