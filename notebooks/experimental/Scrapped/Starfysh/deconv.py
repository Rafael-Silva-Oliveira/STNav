import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rcParams
import seaborn as sns

sns.set_style("white")
from starfysh import AA, utils, plot_utils, post_analysis
from starfysh import starfysh as sf_model
import pandas as pd
import json
import scanpy as sc

# # Load the JSON data
# with open(
#     r"/mnt/work/RO_src/Pipelines/STNav/notebooks/experimental/gene_marker/best_results.json"
# ) as f:
#     data = json.load(f)

# # Create a dictionary to hold the data
# df_data = {}

# # Iterate over the data
# for cell_type, cell_data in data.items():
#     # Add the cell type to the dictionary with the markers as the values
#     df_data[cell_type] = pd.Series(cell_data["best_markers"])

# # Create the DataFrame
# df = pd.DataFrame(data=df_data)

# # Print the DataFrame
# print(df)
# df.to_csv(
#     r"/mnt/work/RO_src/data/raw/signature.csv",
#     index=False,
# )

# Specify data paths
data_path = (
    "/mnt/work/RO_src/data/raw/"  # adjust this line based on your data folder path
)
sample_id = "HD_sample"
sig_name = "signature.csv"

# Specify the file path
file_path = r"/mnt/work/RO_src/data/raw/VisiumHD/square_008um/spatial/tissue_positions_list.parquet"

# Read the parquet file
df = pd.read_parquet(file_path)

# Specify the output file path
output_file_path = (
    r"/mnt/work/RO_src/data/raw/VisiumHD/square_008um/spatial/tissue_positions_list.csv"
)

df.to_csv(output_file_path, index=False)

# Load expression counts and signature gene sets
adata, adata_normed = utils.load_adata(
    data_folder=data_path,
    sample_id=sample_id,  # sample id
    n_genes=2000,  # number of highly variable genes to keep
)

gene_sig = pd.read_csv(os.path.join(data_path, sig_name))
gene_sig = utils.filter_gene_sig(gene_sig, adata.to_df())
gene_sig.head()


# Load spatial information
img_metadata = utils.preprocess_img(
    data_path, sample_id, adata_index=adata.obs.index, hchannel=False
)
img, map_info, scalefactor = (
    img_metadata["img"],
    img_metadata["map_info"],
    img_metadata["scalefactor"],
)
umap_df = utils.get_umap(adata, display=True)
plt.figure(figsize=(6, 6), dpi=200)
plt.imshow(img)
plt.show()
map_info.head()

# Parameters for training
visium_args = utils.VisiumArguments(
    adata,
    adata_normed,
    gene_sig,
    img_metadata,
    n_anchors=60,
    window_size=3,
    sample_id=sample_id,
)


adata, adata_normed = visium_args.get_adata()
anchors_df = visium_args.get_anchors()
plot_utils.plot_spatial_feature(
    adata, map_info, visium_args.log_lib, label="log library size"
)
plot_utils.plot_spatial_gene(adata, map_info, gene_name="IL7R")
plot_utils.plot_anchor_spots(
    umap_df, visium_args.pure_spots, visium_args.sig_mean, bbox_x=2
)
n_repeats = 3
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, loss = utils.run_starfysh(
    visium_args,
    n_repeats=n_repeats,
    epochs=epochs,
    # poe=True,
    device=device,
)
adata, adata_normed = visium_args.get_adata()
inference_outputs, generative_outputs = sf_model.model_eval(
    model,
    adata,
    visium_args,
    # poe=True,
    device=device,
)
n_cell_types = gene_sig.shape[1]
idx = np.random.randint(0, n_cell_types)
post_analysis.gene_mean_vs_inferred_prop(inference_outputs, visium_args, idx=idx)
plot_utils.pl_spatial_inf_feature(adata, feature="ql_m", cmap="Blues")
plot_utils.pl_spatial_inf_feature(
    adata,
    feature="qc_m",
    # To display for specific cell types:
    # factor = cell type or factor = [cell type1, ...]
    vmax=0.1,
)
plot_utils.pl_spatial_inf_feature(
    adata,
    feature="qz_m",
    # To display for specific cell types:
    # factor = Cell type or factor = [Cell type1, ...]
    factor=["Basal", "LumA", "MBC", "Normal epithelial"],
    spot_size=3,
    vmax=0.2,
)
pred_exprs = sf_model.model_ct_exp(model, adata, visium_args, device=device)
sample_gene = "IL7R"
sample_cell_type = "Tem"

plot_utils.pl_spatial_inf_gene(adata, factor=sample_cell_type, feature=sample_gene)
# Specify output directory
outdir = "."
if not os.path.exists(outdir):
    os.mkdir(outdir)

# save the model
torch.save(model.state_dict(), os.path.join(outdir, "starfysh_model.pt"))

# save `adata` object with inferred parameters
adata.write(os.path.join(outdir, "st.h5ad"))
