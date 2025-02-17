# %% [markdown]
# # Biomarker Tester
# Python script to test existing biomarkers

# %% Imports
# Standard library imports
import os
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %% Read Data
data = pd.read_excel(r"/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/chemo_immuno_gene_set.xlsx")

