# %% Test
# Standard library imports
import os
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pyparsing import Diagnostics
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import FitFailedWarning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score
import matplotlib.pyplot as plt
from scipy import stats
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

# Initialize and run the analysis
from matplotlib.colors import rgb2hex
from pydeseq2 import preprocessing, dds, ds
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import seaborn as sns

# Standard library imports
import itertools
import os
import random
import shutil
from itertools import combinations, cycle

# Third-party library imports
import anndata
import gseapy as gp
from gseapy import prerank
from gseapy.plot import dotplot, gseaplot
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, rgb2hex, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import pandas as pd
from pydeseq2 import preprocessing, dds, ds
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scipy
from scipy import stats
from scipy.cluster import hierarchy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress, median_abs_deviation
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    make_scorer,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import plot_tree
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from lifelines import CoxPHFitter

# Set global rcParams for consistent formatting
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.labelcolor": "black",
        "legend.fontsize": 12,
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    }
)
# Third-party imports
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import stats
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from matplotlib.colors import rgb2hex
from matplotlib.gridspec import GridSpec


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats
import itertools
from matplotlib.colors import rgb2hex
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Matplotlib settings
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.labelcolor": "black",
        "legend.fontsize": 12,
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    }
)

####################### Functions ######################
def revert_from_conversion(adata):
    conversion_info = adata.uns.get("conversion_info", {})

    for key, original_type in conversion_info.items():
        df_name, col = key.split(":")
        df = getattr(adata, df_name)

        if "datetime" in original_type.lower():
            df[col] = pd.to_datetime(df[col])
        elif "timedelta" in original_type.lower():
            df[col] = pd.to_timedelta(df[col])
        elif original_type == "category":
            df[col] = df[col].astype("category")
        elif "int" in original_type.lower():
            df[col] = df[col].astype("Int64")  # Use nullable integer type
        elif "float" in original_type.lower():
            df[col] = df[col].astype("float64")
        elif "bool" in original_type.lower():
            df[col] = df[col].astype("boolean")
        # Other types will remain as they are

    return adata

####################### Load Data ######################
km_data_ttp = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_ttp.csv"
)
km_data_os = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_os.csv"
)

# /mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_cp_full_preprocessed.h5ad
# /mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_nmf_cp_81_samp.h5ad
import anndata as ad
adata_nmf = ad.read_h5ad(
    r"/mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_cp_full_preprocessed.h5ad"
)

km_data_ttp.rename(columns={"tend": "tend_ttp"}, inplace=True)
km_data_os.rename(columns={"tend": "tend_os"}, inplace=True)
km_data_ttp.drop(
    columns=[
        "ERCC1",
        "ERCC2",
        "ERCC5",
        "BRCA1",
        "TUBB3",
        "STK11",
        "HIF1A",
        "status",
        "Unnamed: 0",
        "status",
    ],
    inplace=True,
)

km_data_os.drop(
    columns=[
        "ERCC1",
        "ERCC2",
        "ERCC5",
        "BRCA1",
        "TUBB3",
        "STK11",
        "HIF1A",
        "status",
        "Unnamed: 0",
        "status",
    ],
    inplace=True,
)
common_cols = km_data_ttp.columns.intersection(km_data_os.columns)
merged_df = pd.merge(km_data_ttp, km_data_os, on=list(common_cols))

log_norm_counts = pd.DataFrame(adata_nmf.layers["lognormalized_counts"], adata_nmf.obs["ID_Sample"], adata_nmf.var_names)

merged_data = pd.merge(merged_df, log_norm_counts, on="ID_Sample")
merged_df.set_index("ID_Sample", inplace=True)
merged_data.set_index("ID_Sample", inplace=True)


adata_nmf = revert_from_conversion(adata_nmf)
adata_nmf_cp = adata_nmf.copy()
adata_nmf_cp.obs.set_index("ID_Sample", inplace=True)

# Get the common indices
common_idx = adata_nmf_cp.obs.index.isin(merged_data.index)

# Subset adata_nmf to only keep samples that are in km_data_os
adata_nmf_cp = adata_nmf_cp[common_idx].copy()

# Now you can safely assign the new observation dataframe
adata_nmf_cp.obs = merged_data
adata_nmf_cp.obs.rename(columns={"status_os": "status-os"}, inplace=True)
adata_nmf_cp.obs["status-os"] = adata_nmf_cp.obs["status-os"].map(
    {0: "Alive", 1: "Dead"}
)
adata_nmf_cp.obs.rename(columns={"status_ttp": "status-ttp"}, inplace=True)
adata_nmf_cp.obs["status-ttp"] = adata_nmf_cp.obs["status-ttp"].map(
    {0: "Did not progress", 1: "Progressed"}
)

# %%
# Re-normalize with the 77 samples
# Delete existing layers

import numpy as np
import scipy.sparse as sp

def tpm_normalize(adata):
    """
    Normalize raw counts to TPM using gene lengths from adata.var['length']
    """
    # Get gene lengths
    gene_lengths = adata.var['length'].values
    
    # Get raw counts
    if sp.issparse(adata.layers["raw_counts"]):
        counts = adata.layers["raw_counts"].toarray()
    else:
        counts = adata.layers["raw_counts"].copy()
    
    # Step 1: Normalize by gene length to get RPK (reads per kilobase)
    # Convert gene lengths to kb
    gene_lengths_kb = np.asarray(gene_lengths, dtype=np.float64).reshape(-1, 1) / 1000

    print("Counts shape:", counts.shape)
    print("Gene lengths shape:", gene_lengths_kb.shape)

    print(gene_lengths_kb.shape)
    # Divide counts by gene length
    rpk = counts.T / gene_lengths_kb  # Now (23260, 77) / (23260, 1) = (23260, 77)    
    # Step 2: Calculate scaling factor for each sample
    scaling_factors = rpk.sum(axis=0) / 1e6
    
    # Step 3: Divide RPK values by scaling factors to get TPM
    tpm = rpk / scaling_factors[np.newaxis, :]
    
    return tpm.T

# Calculate TPM values
tpm_matrix = tpm_normalize(adata_nmf_cp)

# Store in AnnData layers
adata_nmf_cp.layers["TPM"] = tpm_matrix

# Optionally set as default representation
adata_nmf_cp.X = adata_nmf_cp.layers["TPM"].copy()

# Verify that each sample sums to ~1 million
column_sums = tpm_matrix.T.sum(axis=0)
print("TPM column sums (should be close to 1,000,000):", column_sums[:5])  # Show first 5 samples

# %% -------------------------------------------------------
# 3.b) PyWGCNA
# -------------------------------------------------------

import PyWGCNA
adata_nmf_cp.obs = adata_nmf_cp.obs.drop(columns=adata_nmf_cp.var_names)

expr_TPM = pd.DataFrame(adata_nmf_cp.layers["TPM"], index=adata_nmf_cp.obs_names, columns=adata_nmf_cp.var_names)

expr_TPM.to_csv(r"/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/expressionList.csv")
geneExp = '/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/expressionList.csv'
wgcna = PyWGCNA.WGCNA(name='SCLC', 
                              species='Human', 
                              geneExpPath=geneExp, 
                              outputPath='/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA',
                              save=True)
wgcna.geneExpr.to_df().head(5)

# %%
sampleInfo = pd.DataFrame(data=adata_nmf_cp.obs, index=adata_nmf_cp.obs.index)
sampleInfo.rename(columns={'status-ttp': 'ttp', 'status-os': 'os', 'SCLC_Subtype_de_novo': 'subtype'}, inplace=True)

# Replace "_" with ""
sampleInfo["subtype"] = sampleInfo["subtype"].str.replace("_", "")
sampleInfo["ttp"] = sampleInfo["ttp"].str.replace("Did not progress", "DidNotProgress")
sampleInfo["radio_arm"] = sampleInfo["radio_arm"].str.replace(" ","")
sampleInfo["Organ"] = sampleInfo["Organ"].str.replace(" ","")

# sampleInfo["ttp"] = sampleInfo["ttp"].astype(str)
# sampleInfo["os"] = sampleInfo["os"].astype(str)
# sampleInfo["subtype"] = sampleInfo["subtype"].astype(str)

sampleInfo.to_csv('/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/sampleInfo.csv', index=True)
wgcna.updateSampleInfo(path='/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/sampleInfo.csv', sep=',')

# Subset pyWGCNA with just the metadata we want to analyze and give the colors
# wgcna.datExpr.obs = wgcna.datExpr.obs[['radio_arm', 'status-ttp', 'status-os', "sex","brain_metastasis","ps_bin","SCLC_Subtype_de_novo"]]
wgcna.datExpr.obs = wgcna.datExpr.obs[['ttp', 'os',"brain_metastasis","radio_arm","Organ","subtype"]]
# add color for metadata
# wgcna.setMetadataColor('sex', {0: 'green',
#                                        1: 'yellow'})
wgcna.setMetadataColor('radio_arm', {'45Gy': 'darkviolet',
                                            '60Gy': 'deeppink'})
wgcna.setMetadataColor('ttp', {'DidNotProgress': 'thistle',
                                            'Progressed': 'plum'})

wgcna.setMetadataColor('os', {'Alive': 'violet',
                                            'Dead': 'purple'})
wgcna.setMetadataColor('brain_metastasis', {0: 'lightblue',
                                                   1: 'darkblue'})
wgcna.setMetadataColor('Organ', {"Lung": 'lightcoral',
                                                   "LymphNode": 'red'})

# wgcna.setMetadataColor('ps_bin', {0: 'lightcoral',
#                                           1: 'red'})

wgcna.setMetadataColor('subtype', {'A': 'gold',
                                                       'InNE': 'dodgerblue',
                                                       'N': 'limegreen',
                                                       'INE': 'darkorange'})
# wgcna.setMetadataColor('age_at_sample', plt.get_cmap('viridis'))
# wgcna.setMetadataColor('RSI', plt.get_cmap('coolwarm'))

# %%


geneList = pd.DataFrame(data=adata_nmf_cp.var[["ENSG","gene_name","gene_biotype"]])

geneList.to_csv('/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/geneList.csv', index=False)


wgcna.updateGeneInfo(geneInfo=geneList, path ='/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNA/geneList.csv' )

# %%
# Manually updated gene_biotype to the respective gene
wgcna.datExpr.var['gene_biotype'] = wgcna.datExpr.var.index.map(adata_nmf_cp.var['gene_biotype'])

wgcna.datExpr.var['ENSG'] = wgcna.datExpr.var.index.map(adata_nmf_cp.var['ENSG'])

wgcna.datExpr.var['gene_name'] = wgcna.datExpr.var.index.map(adata_nmf_cp.var['gene_name'])

# %%
wgcna.preprocess()
wgcna.findModules()

# %%
# After loading your data but before calling analyseWGCNA()
# Convert categorical columns in wgcna.datExpr.var to string type
# if 'gene_name' in wgcna.datExpr.var.columns and hasattr(wgcna.datExpr.var.gene_name, 'cat'):
#     wgcna.datExpr.var['gene_name'] = wgcna.datExpr.var.gene_name.astype(str)

# # Similarly, if moduleColors is categorical
# if 'moduleColors' in wgcna.datExpr.var.columns and hasattr(wgcna.datExpr.var.moduleColors, 'cat'):
#     wgcna.datExpr.var['moduleColors'] = wgcna.datExpr.var.moduleColors.astype(str)
# %%
wgcna.saveWGCNA()

# %%
plt.figure(figsize=(10, 10))  

wgcna.analyseWGCNA()

# %%
wgcna = PyWGCNA.readWGCNA("/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/WGCNASCLC.p")


for module in wgcna.modules:
    ...

# %%
wgcna.top_n_hub_genes(moduleName="coral", n=10)


# %% GSEA - GO functions
wgcna.figureType = "png"
gene_set_library: List[str] = ["GO_Biological_Process_2021", "GO_Cellular_Component_2021", "GO_Molecular_Function_2021"]
wgcna.functional_enrichment_analysis(type="GO",
                                             moduleName="coral",
                                             sets=gene_set_library,
                                             p_value=0.05,
                                             file_name="GO_coral_2021")

wgcna.functional_enrichment_analysis(type="GO",
                                             moduleName="darksalmon",
                                             sets=gene_set_library,
                                             p_value=0.05)
# %% GSEA- KEGG
wgcna.figureType = "png"
gene_set_library = ["KEGG_2019_Human"]
wgcna.functional_enrichment_analysis(type="KEGG",
                                             moduleName="coral",
                                             sets=gene_set_library,
                                             p_value=0.05)

# %% GSEA - REACTOME
wgcna.figureType = "pdf"
wgcna.functional_enrichment_analysis(type="REACTOME",
                                             moduleName="coral",
                                             p_value=0.05)

# %% Network Analysis
wgcna.CalculateSignedKME(exprWeights=None, MEWeights=None)
wgcna.CoexpressionModulePlot(modules=["coral"], numGenes=10, numConnections=100, minTOM=0)
modules = wgcna.datExpr.var.moduleColors.unique().tolist()
wgcna.CoexpressionModulePlot(modules=modules, numGenes=100, numConnections=1000, minTOM=0, file_name="all")

# %%
filters = {"gene_biotype": ["protein_coding"]}
wgcna.CoexpressionModulePlot(modules=["coral"], filters=filters, file_name="darkgray_protein_coding")

# %% Protein-Protein Interaction (PPI)
genes = wgcna.datExpr.var[wgcna.datExpr.var.moduleColors == "coral"]
genes = genes.index.astype(str).tolist()
PPI = wgcna.request_PPI(genes=genes, species=9606)
PPI

geneList = PPI.gene1[:20].unique().tolist()
wgcna.PPI_network(species=9606, geneList=geneList)
