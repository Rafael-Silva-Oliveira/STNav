import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read_h5ad(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_02_19-10_32_06_AM\ST\Files\ST_preprocessed_adata.h5ad"
)
gsea = pd.read_excel(
    r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\DAIgnostics\dAIgnostics\data\processed\PipelineRun_2024_02_14-01_14_20_PM\ST\Files\ST_GSEA_2024_02_14-01_14_23_PM.xlsx"
)


adata.obs[["leiden_clusters","Negative Radiation Sensitivity", "Positive Radiation Sensitivity"]] #goal is to then add the genes that were ranked higher in each spot associated with the radiation score. 
