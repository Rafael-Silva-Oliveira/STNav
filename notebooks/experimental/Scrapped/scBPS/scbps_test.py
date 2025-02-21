import scbsp
import scanpy as sc

adata = sc.read_h5ad(
    r"/mnt/work/RO_src/data/processed/PipelineRun_2024_06_21-03_05_24_PM/ST/Files/raw_adata.h5ad"
)
adata.obs["array_row"] = adata.obs["array_row"].astype(int)
adata.obs["array_col"] = adata.obs["array_col"].astype(int)
adata
# Load your data into these variables
# Load your data into these variables
input_sp_mat = adata.obs[
    ["array_row", "array_col"]
].values  # Replace 'x' and 'y' with your actual spatial coordinates column names
input_exp_mat_raw = adata.X

# Set the optional parameters
d1 = 10
d2 = 20

# Compute p-values
p_values = scbsp.granp(input_sp_mat, input_exp_mat_raw, d1, d2)
