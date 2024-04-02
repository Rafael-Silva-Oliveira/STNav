import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=Warning)
import scanorama

# verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80)


def cell_type_pred_scanorama(): ...


def cell_type_pred_ingest(): ...


def cell_type_pred_GSA(): ...


def main(): ...


path_file = r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_01_29-09_15_29_AM\scRNA\Files\scRNA_adata.h5ad"
adata = sc.read_h5ad(path_file)
adata
adata.uns["log1p"]["base"] = None
adata.shape
adata.raw.shape
adata.obs["leiden_clusters"].value_counts()

sc.pl.umap(adata, color=["leiden_clusters"])

adata_ref = sc.datasets.pbmc3k_processed()

adata_ref.obs["sample"] = "pbmc3k"

print(adata_ref.shape)
adata_ref.obs

adata.var_names = adata.var_names.str.upper()

adata_ref.var_names

print(adata_ref.shape[1])
print(adata.shape[1])
var_names = adata_ref.var_names.intersection(adata.var_names)
print(len(var_names))

adata_ref = adata_ref[:, var_names]
adata = adata[:, var_names]

adata.var_names

sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)
sc.pl.umap(adata_ref, color="louvain")


sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden_clusters")


### Scanorama ##################################################################
import scanorama

# subset the individual dataset to the same variable genes as in MNN-correct.
alldata = dict()
alldata["ctrl"] = adata
alldata["ref"] = adata_ref

alldata
# convert to list of AnnData objects
adatas = list(alldata.values())
adatas

# run scanorama.integrate
scanorama.integrate_scanpy(adatas, dimred=50)

# add in sample info
adata_ref.obs["sample"] = "pbmc3k"

# create a merged scanpy object and add in the scanorama
adata_merged = alldata["ctrl"].concatenate(
    alldata["ref"], batch_key="sample", batch_categories=["ctrl", "pbmc3k"]
)

adata_merged
embedding = np.concatenate([ad.obsm["X_scanorama"] for ad in adatas], axis=0)
adata_merged.obsm["Scanorama"] = embedding
# run  umap.
sc.pp.neighbors(adata_merged, n_pcs=50, use_rep="Scanorama")
sc.tl.umap(adata_merged)
sc.pl.umap(adata_merged, color=["sample", "louvain"])
from sklearn.metrics.pairwise import cosine_distances

distances = 1 - cosine_distances(
    adata_merged[adata_merged.obs["sample"] == "pbmc3k"].obsm["Scanorama"],
    adata_merged[adata_merged.obs["sample"] == "ctrl"].obsm["Scanorama"],
)


def label_transfer(dist, labels, index):
    lab = pd.get_dummies(labels)
    class_prob = lab.to_numpy().T @ dist
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    # convert to df
    cp_df = pd.DataFrame(class_prob, columns=lab.columns)
    cp_df.index = index
    # classify as max score
    m = cp_df.idxmax(axis=1)

    return m


class_def = label_transfer(distances, adata_ref.obs.louvain, adata.obs.index)

# add to obs section of the original object
adata.obs["predicted"] = class_def

sc.pl.umap(adata, color=["predicted", "cell_ontology_class"])
adata.obs[["predicted", "cell_ontology_class"]]


# add to merged object.
adata_merged.obs["predicted"] = pd.concat(
    [class_def, adata_ref.obs["louvain"]], axis=0
).tolist()

adata_merged
sc.pl.umap(adata_merged, color=["sample", "louvain", "predicted"])
# plot only ctrl cells.
sc.pl.umap(adata_merged[adata_merged.obs["sample"] == "ctrl"], color="predicted")

sc.pl.umap(adata, color=["louvain", "predicted"], wspace=0.5)
tmp = pd.crosstab(
    adata.obs["leiden_clusters"], adata.obs["predicted"], normalize="index"
)
tmp.plot.bar(stacked=True).legend(bbox_to_anchor=(1.8, 1), loc="upper right")
plt.show()

### Ingest ###############################################################
adata
# adata.obs
adata_ref.obs
sc.tl.ingest(adata, adata_ref, obs="louvain")
sc.pl.umap(adata, color=["louvain", "cell_ontology_class"], wspace=0.5)
tmp = pd.crosstab(
    adata.obs["cell_ontology_class"], adata.obs["louvain"], normalize="index"
)
tmp.plot.bar(stacked=True).legend(bbox_to_anchor=(1.8, 1), loc="upper right")
plt.show()

# #### GSEA ##################################################################
# import os

# path_file = "data/human_cell_markers.txt"
# import urllib

# path_data = "https://export.uppmax.uu.se/naiss2023-23-3/workshops/workshop-scrnaseq"

# if not os.path.exists(path_file):
#     urllib.request.urlretrieve(
#         os.path.join(path_data, "human_cell_markers.txt"), path_file
#     )

path_file = r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\raw\human_cell_markers.txt"

df = pd.read_table(path_file)
df

print(df.shape)

# Filter for number of genes per celltype
df["nG"] = df.geneSymbol.str.split(",").str.len()

df
df["geneSymbol"]
df = df[df["nG"] > 5]
df = df[df["nG"] < 100]
d = df[df["cancerType"] == "Normal"]
print(df.shape)
df = df[df["tissueType"] == "Lung"]
df.index = df.cellName
gene_dict = df.geneSymbol.str.split(",").to_dict()
df
gene_dict
# run differential expression per cluster
sc.tl.rank_genes_groups(
    adata, "leiden_clusters", method="wilcoxon", key_added="wilcoxon"
)
# do gene set overlap to the groups in the gene list and top 300 DEGs.
import gseapy

gsea_res = dict()
pred = dict()
gene_dict

gene_dict = {key: [gene.upper() for gene in gene_dict[key]] for key in gene_dict}
gene_dict

gene_dict
adata.obs["leiden_clusters"]

adata_ref.obs
glist_upper

adata.obs
for cl in adata.obs["leiden_clusters"].cat.categories.tolist():
    print(cl)
    glist = (
        sc.get.rank_genes_groups_df(adata, group=cl, key="wilcoxon")["names"]
        .squeeze()
        .str.strip()
        .tolist()
    )
    glist_upper = [g.upper() for g in glist]
    enr_res = gseapy.enrichr(
        gene_list=glist_upper[:300],
        organism="Human",
        gene_sets=gene_dict,
        background=adata.raw.shape[1],
        cutoff=1,
    )
    if enr_res.results.shape[0] == 0:
        pred[cl] = "Unass"
    else:
        enr_res.results.sort_values(by="P-value", axis=0, ascending=True, inplace=True)
        print(enr_res.results.head(2))
        gsea_res[cl] = enr_res
        pred[cl] = enr_res.results["Term"][0]

gsea_res["2"].results

enr_res.results
# 0
#    Gene_set                   Term Overlap   P-value  Adjusted P-value  \
# 0  gs_ind_0  Cancer stem-like cell     1/6  0.088981          0.226652
# 6  gs_ind_0             Macrophage     1/6  0.088981          0.226652

#    Odds Ratio  Combined Score  Genes
# 0   14.996147       36.280703  ANPEP
# 6   14.996147       36.280703   AIF1
# 1
#    Gene_set                    Term Overlap   P-value  Adjusted P-value  \
# 2  gs_ind_0  Effector memory T cell     1/7  0.103024          0.180292
# 4  gs_ind_0                Monocyte     1/7  0.103024          0.180292

#    Odds Ratio  Combined Score Genes
# 2   12.995993       29.537229  IL7R
# 4   12.995993       29.537229  CD52
# 2
#    Gene_set                      Term Overlap   P-value  Adjusted P-value  \
# 6  gs_ind_0                  Monocyte     1/7  0.103024          0.244332
# 7  gs_ind_0  Parietal progenitor cell     1/7  0.103024          0.244332

#    Odds Ratio  Combined Score  Genes
# 6   12.995993       29.537229   CD52
# 7   12.995993       29.537229  ANXA1
# 3
#    Gene_set                    Term Overlap   P-value  Adjusted P-value  \
# 6  gs_ind_0  Effector memory T cell     1/7  0.103024          0.226084
# 8  gs_ind_0            Naive T cell     1/7  0.103024          0.226084

#    Odds Ratio  Combined Score Genes
# 6   12.995993       29.537229  IL7R
# 8   12.995993       29.537229  CCR7
# 4
#    Gene_set      Term Overlap   P-value  Adjusted P-value  Odds Ratio  \
# 0  gs_ind_0    B cell     1/6  0.088981          0.116851   14.996147
# 4  gs_ind_0  Monocyte     1/7  0.103024          0.116851   12.995993

#    Combined Score Genes
# 0       36.280703  CD19
# 4       29.537229  CD52
# 5
#     Gene_set                             Term Overlap   P-value  \
# 11  gs_ind_0  Myeloid-derived suppressor cell     1/6  0.088981
# 2   gs_ind_0                   Dendritic cell     1/7  0.103024

#     Adjusted P-value  Odds Ratio  Combined Score  Genes
# 11          0.183109   14.996147       36.280703  ITGAM
# 2           0.183109   12.995993       29.537229  ITGAM
# 6
#    Gene_set                           Term Overlap   P-value  \
# 0  gs_ind_0          Cancer stem-like cell     1/6  0.088981
# 4  gs_ind_0  Induced pluripotent stem cell     1/6  0.088981

#    Adjusted P-value  Odds Ratio  Combined Score  Genes
# 0          0.164838   14.996147       36.280703  ANPEP
# 4          0.164838   14.996147       36.280703  ITGA6
# 7
#    Gene_set      Term Overlap   P-value  Adjusted P-value  Odds Ratio  \
# 0  gs_ind_0    B cell     1/6  0.088981          0.140221   14.996147
# 5  gs_ind_0  Monocyte     1/7  0.103024          0.140221   12.995993

#    Combined Score Genes
# 0       36.280703  CD19
# 5       29.537229  CD52
# 8
#    Gene_set                             Term Overlap   P-value  \
# 2  gs_ind_0                       Macrophage     1/6  0.088981
# 3  gs_ind_0  Monocyte derived dendritic cell     1/8  0.116851

#    Adjusted P-value  Odds Ratio  Combined Score  Genes
# 2          0.233702   14.996147       36.280703   AIF1
# 3          0.233702   11.466464       24.616832  ITGAX
# 9
#    Gene_set                      Term Overlap   P-value  Adjusted P-value  \
# 3  gs_ind_0  PROM1Low progenitor cell     1/7  0.103024          0.309489
# 1  gs_ind_0             M2 macrophage    1/12  0.170068          0.309489

#    Odds Ratio  Combined Score  Genes
# 3   12.995993       29.537229  ALCAM
# 1    7.795593       13.810336  CD163
# prediction per cluster
pred

{
    "0": "Cancer stem-like cell",
    "1": "CD4+ T cell",
    "2": "CD8+ T cell",
    "3": "Activated T cell",
    "4": "B cell",
    "5": "CD16+ dendritic cell",
    "6": "Cancer stem-like cell",
    "7": "B cell",
    "8": "Circulating fetal cell",
    "9": "Circulating fetal cell",
}
prediction = [pred[x] for x in adata.obs["leiden_clusters"]]
adata.obs["GS_overlap_pred"] = prediction

sc.pl.umap(adata, color="GS_overlap_pred")
