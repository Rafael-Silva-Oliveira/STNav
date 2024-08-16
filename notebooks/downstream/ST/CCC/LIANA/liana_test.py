import pandas as pd
import scanpy as sc
import decoupler as dc
import liana as li
import scanpy as sc
import squidpy as sq
import celltypist
from celltypist import models

adata = sc.read_h5ad(r"/mnt/archive2/RO_src/data/adata_subset3.h5ad")
lrdata = adata.obsm["local_scores"]
