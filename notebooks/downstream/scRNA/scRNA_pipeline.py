import sys

from scvi.model._scanvi import SCANVI

sys.path.append("/mnt/work/RO_src/STAnalysis")
from STNav.modules.sc import perform_QC
from STNav.modules.sc import perform_doublet_removal
from STNav.modules.sc import perform_preprocessing
import anndata as ad

# Load packages
import warnings
import scvi
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope
from scvi.model import CondSCVI, DestVI

warnings.simplefilter(action="ignore", category=FutureWarning)
import os
from datetime import datetime
import re
import numpy as np
import pandas as pd
import scanpy as sc
import scarches as sca
from loguru import logger
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

save_dir = tempfile.TemporaryDirectory()
scvi.settings.logging_dir = save_dir.name


date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3
import celltypist
from celltypist import models
from scvi.autotune import ModelTuner
from ray import tune
from scvi import autotune

################# 1. Load scRNA adata ##############################

sc_adata: sc.AnnData = sc.read_h5ad(
	r"/mnt/work/RO_src/data/raw/scRNA/SCLC/SCLC_Annotated_pp.h5ad"
)

sclc_adata = sc_adata[
	sc_adata.obs["histo"].isin(["SCLC"])
	& sc_adata.obs["tissue"].isin(["lung", "pleural effusion"]),
	:,
]

healthy_adata: sc.AnnData = sc_adata[
	sc_adata.obs["histo"].isin(values=["normal"])
	& sc_adata.obs["tissue"].isin(values=["lung", "pleural effusion"]),
	:,
]

adata_to_annotate = sc.read_h5ad(
	r"/mnt/work/RO_src/data/processed/Keep/PipelineRun_2024_02_08-11_55_23_AM - Copy/scRNA/Files/scRNA_raw_adata_adata.h5ad"
)
adata_to_annotate.layers["raw_counts"] = adata_to_annotate.X
adata_to_annotate.var_names = adata_to_annotate.var_names.str.upper()
adata_to_annotate.var.index = adata_to_annotate.var.index.str.upper()

adata_to_annotate.var_names_make_unique()
sclc_adata.var_names_make_unique()
healthy_adata.var_names_make_unique()

adata_to_annotate.obs_names_make_unique()
sclc_adata.obs_names_make_unique()
healthy_adata.obs_names_make_unique()

adata_to_annotate.var_names = adata_to_annotate.var_names.str.upper()
adata_to_annotate.var.index = adata_to_annotate.var.index.str.upper()
adata_to_annotate.var.drop(columns=["genome", "gene_names"], inplace=True)

sclc_adata.var.rename(
	columns={"feature_biotype": "feature_types", "feature_reference": "gene_ids"},
	inplace=True,
)
healthy_adata.var.rename(
	columns={"feature_biotype": "feature_types", "feature_reference": "gene_ids"},
	inplace=True,
)
healthy_adata.var.drop(columns=["feature_is_filtered", "feature_length"], inplace=True)
sclc_adata.var.drop(columns=["feature_is_filtered", "feature_length"], inplace=True)

del sclc_adata.obsm
del sclc_adata.uns

del healthy_adata.obsm
del healthy_adata.uns

del adata_to_annotate.obsm

# Create a batch so we can identify which one is the one we want to keep the predictions from. In this case SCLC is our query dataset we want to transfer predictions to
adata_to_annotate.obs["CellType"] = "Unknown"
adata_to_annotate.obs["Batch"] = "Lung"
adata_to_annotate.obs["Sample"] = adata_to_annotate.obs.index
sclc_adata.obs["CellType"] = sclc_adata.obs["cell_type_fine"]
sclc_adata.obs["Batch"] = "SCLC"
sclc_adata.obs["Sample"] = sclc_adata.obs["donor_id"]
healthy_adata.obs["CellType"] = healthy_adata.obs["cell_type_fine"]
healthy_adata.obs["Batch"] = "Healthy"
healthy_adata.obs["Sample"] = healthy_adata.obs["donor_id"]

################# 2. Perform QC ##############################

sc_adata_qc = perform_QC(
	adata=sc_adata,
	min_cells=25,
	min_genes=200,
	log1p=True,
	use_raw=False,
	pct_counts_mt=25,
	qc_vars=["Mt", "Ribo", "Hb"],
)


##############################
sc.pp.filter_genes(adata_to_annotate, min_cells=25)
min_genes=200
log1p=True
use_raw=False
pct_counts_mt=25
qc_vars=["Mt", "Ribo", "Hb"]

# mitochondrial genes
adata_to_annotate.var["Mt"] = adata_to_annotate.var_names.str.startswith("MT-")
# ribosomal genes
adata_to_annotate.var["Ribo"] = adata_to_annotate.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes.
adata_to_annotate.var["Hb"] = adata_to_annotate.var_names.str.contains(
	("^HB[^(p)]")
)  # adata.var_names.str.contains('^Hb.*-')

sc.pp.calculate_qc_metrics(
	adata=adata_to_annotate,
	qc_vars=qc_vars,
	log1p=log1p,
	use_raw=use_raw,
	percent_top=[20],
	inplace=True,
)
adata_to_annotate = perform_QC(
	adata=adata_to_annotate,
	min_cells=25,
	min_genes=200,
	log1p=True,
	use_raw=False,
	pct_counts_mt=25,
	qc_vars=["Mt", "Ribo", "Hb"],
)


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
		f"adata.var contains the current gene information: \n {adata.var=} \n with the following columns: {adata.var.columns=}"
	)
	logger.info(
		f"adata.obs contains the current cell/spot information: \n {adata.obs=} \n with the following columns: {adata.obs.columns=}"
	)

	adata.layers["raw_counts"] = adata.X.copy()

	logger.info(f"Applying normalization.")
	adata.layers["normalized_counts"] = adata.layers["raw_counts"].copy()

	sc.pp.normalize_total(adata=adata, layer="normalized_counts")

	# It requires a positional argument and not just keyword arguments
	# Get the parameters from return_filtered_params
	logger.info(f"Applying log1p")
	adata.layers["lognorm_counts"] = adata.layers["normalized_counts"].copy()

	sc.pp.log1p(adata, layer="lognorm_counts")

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

	logger.info(f"Applying scaling")
	adata.layers["scaled_lognorm_counts"] = adata.layers["lognorm_counts"].copy()

	sc.pp.scale(adata, zero_center=zero_center, layer="scaled_lognorm_counts")

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

	logger.info(f"Current adata.X shape after preprocessing: {adata.X.shape}")
	logger.info(
		f"Current adata.raw.X shape after preprocessing: \n {adata.raw.X.shape = }"
	)

	return adata


# ################# 3. Perform preprocessing ##############################
sc_adata_pp = perform_preprocessing(
	adata=sc_adata_qc.copy(),
	target_sum=None,
	subset_hvg=False,
	min_mean_hvg=0.0125,
	max_mean_hvg=2,
	min_disp_hvg=0.25,
	flavor_hvg="seurat_v3",
	n_top_genes=5000,
	zero_center=True,
	n_pcs=50,
	n_neighbors=20,
	resolution=1.0,
)

sc_adata_pp.write_h5ad(r"/mnt/work/RO_src/data/raw/scRNA/SCLC/SCLC_Annotated_pp.h5ad")

################# 4. Refine scRNA annotated cell types ##############################
################# 4.1 CellTypist ##############################

# Train and save CellTypist model
def train_celltypist_model(adatas_info: dict) -> celltypist.Model:
	trained_models = []
	for adata_name, adata_params in adatas_info.items():
		adata_cp = adata_params["adata"].copy()
		sc.pp.filter_genes(data=adata_cp, min_cells=10)
		sc.pp.normalize_total(adata=adata_cp, target_sum=1e4)
		sc.pp.log1p(adata_cp)

		model: celltypist.Model = celltypist.train(
			adata_cp,
			labels=adata_params["labels"],
			n_jobs=adata_params["n_jobs"],
			feature_selection=adata_params["feature_selection"],
			use_SGD=adata_params["use_SGD"],
			top_genes=adata_params["top_genes"],
		)
		model.write(
			file=f"/mnt/work/RO_src/Pipelines/STAnalysis/STNav/models/model_{adata_name}.pkl"
		)
		trained_models.append(model)

	return trained_models


adatas_info = {
	"healthy_lung_v2": {
		"adata": healthy_adata,
		"labels": "cell_type_fine",
		"n_jobs": -1,
		"feature_selection": True,
		"use_SGD": False,
		"top_genes": 500,
	},
	"sclc_lung_v2": {
		"adata": sclc_adata,
		"labels": "cell_type_fine",
		"n_jobs": -1,
		"feature_selection": True,
		"use_SGD": False,
		"top_genes": 500,
	},
}

trained_models: list[celltypist.Model] = train_celltypist_model(adatas_info=adatas_info)

# Load trained models from CellTypist
sc_model_healthy = models.Model.load(
	model="/mnt/work/RO_src/STAnalysis/STNav/models/model_healthy_lung_v2.pkl"
)
sc_model_sclc = models.Model.load(
	model="/mnt/work/RO_src/STAnalysis/STNav/models/model_sclc_lung_v2.pkl"
)

################ 4.1.1 CellTypist predictions using the ref models ##############################
def predict_cells(adata, model_dict):
	sc.pp.filter_genes(adata, min_cells=10)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)

	adata.X = adata.X.toarray()

	for model_name, model in model_dict.items():
		predictions = celltypist.annotate(adata, model=model, majority_voting=False)
		predictions_adata = predictions.to_adata()
		adata.obs[f"{model_name}_labels"] = predictions_adata.obs.loc[
			adata.obs.index, "predicted_labels"
		]
		adata.obs[f"{model_name}_score"] = predictions_adata.obs.loc[
			adata.obs.index, "conf_score"
		]

	return adata.obs

def predict_cells_and_subset(adatas_to_annotate, model_dict):
	predictions_ct = [
		predict_cells(adata=ad.copy(), model_dict=model_dict)
		for ad in adatas_to_annotate
	]

	predictions_sub_ct = pd.concat(predictions_ct, axis=1)[
		[f"{model_name}_labels" for model_name in model_dict] + [f"{model_name}_score" for model_name in model_dict]
	]

	return predictions_sub_ct

# Load all the models
model_dict = {
	"healthy_lung": models.Model.load(model="/mnt/work/RO_src/STAnalysis/STNav/models/model_healthy_lung.pkl"),
	"sclc_lung": models.Model.load(model="/mnt/work/RO_src/STAnalysis/STNav/models/model_sclc_lung.pkl")
}

predictions_sub_ct = predict_cells_and_subset(adatas_to_annotate=[adata_to_annotate], model_dict=model_dict)
predictions_sub_ct
############# USING SCVI MODEL ###################
# Merge the predictions_sub_ct with the adata_to_annotate on the indices with the cell typist results
adata_to_annotate.obs = adata_to_annotate.obs.merge(predictions_sub_ct, left_index=True, right_index=True)

concatenated_adatas = sc.concat((adata_to_annotate, sclc_adata, healthy_adata))

sc.pp.highly_variable_genes(
	adata=concatenated_adatas, flavor="seurat_v3", n_top_genes=2500, batch_key="Batch", subset=True
)

################# 4.2.1 scVI ##############################

class TransferLabelSCVI:

	def __init__(self, concatenated_adatas, vae = None, lvae = None, vae_save_path = None, lvae_save_path = None):
		self.concatenated_adatas = concatenated_adatas
		self.vae = vae
		self.vae_save_path = vae_save_path
		self.lvae = lvae
		self.lvae_save_path = lvae_save_path

	def train_or_load_semi_supervised(self, batch_key = "Batch", categorical_covariate_keys = ["Sample"]):

		if self.vae is None:
			scvi.model.SCVI.setup_anndata(
				adata=self.concatenated_adatas, batch_key=batch_key, categorical_covariate_keys=categorical_covariate_keys
			)
			self.vae = scvi.model.SCVI(adata=self.concatenated_adatas)
			self.vae.train()
			self.vae.save(dir_path=self.vae_save_path, overwrite=True, save_anndata=True)

		elif isinstance(self.vae, str):
			self.vae = scvi.model.SCVI.load(self.vae)
		
		return self.vae

	def train_or_load_supervised(self, max_epochs = 20, n_samples_per_label = 100, unlabeled_category="Unknown", labels_key="CellType"):

		if self.lvae is None:
			self.lvae: SCANVI = scvi.model.SCANVI.from_scvi_model(
				scvi_model=self.vae, adata=self.concatenated_adatas, unlabeled_category=unlabeled_category, labels_key=labels_key
			)
			self.lvae.train(max_epochs=max_epochs, n_samples_per_label=n_samples_per_label)
			self.lvae.save(self.lvae_save_path, overwrite=True, save_anndata=True)
			
		elif isinstance(self.lvae, str):
			self.lvae = scvi.model.SCANVI.load(self.lvae)
		
		return self.lvae
	
	def predict(self, save_annotated_adata_path:str = None, batch_key="Lung"):

		# Add the predicted cells
		self.concatenated_adatas.obs["predicted"] = self.lvae.predict(adata=self.concatenated_adatas)
		# Get the actual predicted cell. If we want to have the probabilities instead, we don't do max(axis=1)
		# This will add the transfer score. In this case, the higher transfer score will be the predicted cell type for our query
		self.concatenated_adatas.obs["transfer_score"] = self.lvae.predict(soft=True).max(axis=1)

		# Subset just to get the predictions from the unnotated adata batch we care about
		adata_subset: sc.AnnData = self.concatenated_adatas[self.concatenated_adatas.obs.Batch == batch_key]

		return adata_subset

TFSCVI = TransferLabelSCVI(concatenated_adatas=concatenated_adatas, lvae_save_path = r"/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA",vae_save_path = r"/mnt/work/RO_src/STAnalysis/notebooks/downstream/scRNA")
vae = TFSCVI.train_or_load_semi_supervised()
lvae = TFSCVI.train_or_load_supervised()
concatenated_adatas_subset = TFSCVI.predict(save_annotated_adata_path=r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/downstream/scRNA/adata_annotated.h5ad")

# Merge the scVI transfer predictions to the adata query we want to annotate
adata_to_annotate.obs = adata_to_annotate.obs.merge(
	right=concatenated_adatas_subset.obs[["predicted", "transfer_score"]],
	left_index=True,
	right_index=True,
)

adata_to_annotate.write_h5ad(r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/downstream/scRNA/adata_annotated.h5ad")

# Semi-supervised approach for label transfering
scvi.model.SCVI.setup_anndata(
	adata=concatenated_adatas, batch_key="Batch", categorical_covariate_keys=["Sample"]
)
vae = scvi.model.SCVI(concatenated_adatas)
vae.train()
vae.save(dir_path=r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/downstream/scRNA/model")

# # Create supervised model to transfer the labels
# lvae = scvi.model.SCANVI.from_scvi_model(
# 	scvi_model=vae, adata=concatenated_adatas, unlabeled_category="Unknown", labels_key="CellType"
# )
# # n_samples_per_label is important as it will subsample so that the query and our reference have the same number of cells per label
# lvae.train(max_epochs=20, n_samples_per_label=100)
# lvae.save(r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/downstream/scRNA/lvae_model")

# # Add the predicted cells
# concatenated_adatas.obs["predicted"] = lvae.predict(adata=concatenated_adatas)
# # Get the actual predicted cell. If we want to have the probabilities instead, we don't do max(axis=1)
# # This will add the transfer score. In this case, the higher transfer score will be the predicted cell type for our query
# concatenated_adatas.obs["transfer_score"] = lvae.predict(soft=True).max(axis=1)

# # Subset just to get the predictions from the unnotated adata batch we care about
# concatenated_adatas_sclc: sc.AnnData = concatenated_adatas[concatenated_adatas.obs.Batch == "Lung"]

# # Merge the scVI transfer predictions to the adata query we want to annotate
# adata_to_annotate.obs = adata_to_annotate.obs.merge(
# 	right=concatenated_adatas_sclc.obs[["predicted", "transfer_score"]],
# 	left_index=True,
# 	right_index=True,
# )

# adata_to_annotate.write_h5ad(r"/mnt/work/RO_src/Pipelines/STAnalysis/notebooks/downstream/scRNA/adata_annotated.h5ad")

###### Fine tune the model
import ray
from scvi.autotune import ModelTuner
from ray import tune

model_cls = scvi.model.SCVI
model_cls.setup_anndata(
	adata_to_annotate,
	categorical_covariate_keys=["Sample"],
	continuous_covariate_keys=["pct_counts_Mt", "pct_counts_Ribo"],
)

tuner = ModelTuner(model_cls=model_cls)
tuner.info()

# Define search space of hyperparameters

search_space = {
	"n_hidden": tune.choice([92, 128, 196, 256]),
	"n_latent": tune.choice([10, 20, 30, 40, 50]),
	"n_layers": tune.choice([1, 2, 3]),
	"lr": tune.loguniform(lower=1e-6, upper=1e-2),
	"gene_likelihood": tune.choice(["nb", "zinb"]),
}

# Create tuner
results = tuner.fit(
	adata=adata_to_annotate,
	metric="validation_loss",
	search_space=search_space,
	num_samples=100,
	max_epochs=20,

)

# Find the best result
best_vl = 10000
best_i = 0
for i, res in enumerate(results.results):
	vl = res.metrics["validation_loss"]

	if vl < best_vl:
		best_vl = vl
		best_i = i

results.results[best_i]
print(results.model_kwargs)

# Setup model with the best results
scvi.model.SCVI.setup_anndata(
	adata=adata_to_annotate,
	categorical_covariate_keys=["Sample"],
	continuous_covariate_keys=["pct_counts_Mt", "pct_counts_Ribo"],
)

model = scvi.model.SCVI(
	adata_to_annotate, n_hidden=..., n_latent=..., n_layers=..., gene_likelihood=...
)
kwargs = {"lr": 0.003}

# Re-train model with the optimal parameters
model.train(max_epochs=200, early_stopping=True, plan_kwargs=kwargs)
model.save(r".")

# model = scvi.model.SCVI.load(r".", adata_to_annotate)

y = model.history["reconstruction_loss_validation"][
	"reconstruction_loss_validation"
].min()

# Plot training curve

plt.plot(
	model.history["reconstruction_loss_train"]["reconstruction_loss_train"],
	label="train",
)
plt.plot(
	model.history["reconstruction_loss_validation"]["reconstruction_loss_validation"],
	label="validation",
)

plt.axhline(y, c="k")

plt.legend()
plt.show()

# Extract the embeddings from the trained model. We'll use this to find neighbors and do the clustering
adata_to_annotate.obsm["X_scVI"] = model.get_latent_representation()

sc.pp.neighbors(adata_to_annotate, use_rep="X_scVI")
sc.tl.leiden(adata_to_annotate, resolution=3, key_added="overcluster")
sc.tl.umap(adata_to_annotate)

# Keep raw counts for DEG, etc
adata_to_annotate.layers["counts"] = adata_to_annotate.X.copy()

# Normalize counts
sc.pp.normalize_total(adata=adata_to_annotate)
sc.pp.log1p(adata_to_annotate)

adata_to_annotate.write_h5ad("./temp.h5ad")

############ Do overclustering

# Add from SCLC model
adata_to_annotate.obs["sclc_cancer_labels_major"] = adata_to_annotate.obs.groupby(
	"overcluster"
)["sclc_cancer_labels"]
sc.pl.umap(adata_to_annotate, color=["sclc_cancer_labels_major"], s=5)

# Add from scVI transfer
adata_to_annotate.obs["predicted_major"] = adata_to_annotate.obs.groupby("overcluster")[
	"predicted"
]
sc.pl.umap(adata_to_annotate, color=["predicted_major"], s=5)

# Add from healthy
adata_to_annotate.obs["sclc_normal_labels_major"] = adata_to_annotate.obs.groupby(
	"overcluster"
)["sclc_normal_labels"]
sc.pl.umap(adata_to_annotate, color=["sclc_normal_labels_major"], s=5)

# Look at confidence scores

sc.pl.umap(
	adata_to_annotate,
	color=["transfer_score", "sclc_cancer_score", "sclc_normal_score"],
)

# ################# 4.3.1 Manual annotation ##############################
sc.pl.umap(adata_to_annotate, color=["overcluster"], legend_loc="on data", s=5)

# Plot all the diagnosis (remission, or low trt with high trt, etc)
# np.random.seed(1)
# ri = np.random.permutation(list(range(adata_to_annotate.shape[0])))
# sc.pl.umap(adata_to_annotate[ri,:], color=["diagnosis"], vmin=.5, size=2)


# Score markers for SCLC - Manually look for SCLC markers and then score them against our 3-4 integrated SCLC samples from fresh sample
# TODO: test this on the RSI score
sclc_markers = []
sc.tl.score_genes(adata_to_annotate, sclc_markers, score_name="SCLC_score")

# Check overlap with our data
sc.pl.umap(adata_to_annotate, color=["some", "markers"], s=5)

# Or we can just plot the SCLC score and see which section has it highest
sc.pl.umap(adata_to_annotate, color=["SCLC_score"], s=5)

# We can plot the median values of these scores, and in a barplot we can have an idea of which clusters has a higher SCLC score, and thus confirm the cells that are classified as SCLC
sclc_scores = (
	adata_to_annotate.obs[["overcluster", "SCLC_score"]].groupby("overcluster").median()
)

plt.figure()
sns.barplot(sclc_scores, y="SCLC_score", x=sclc_scores.index)

# Create an isolated dataframe with the labels and the scores from the CellTypist and scVI models

scores = (
	adata_to_annotate.obs[["transfer_score", "sclc_cancer_score", "sclc_normal_score"]]
	.groupby("overcluster")
	.agg(lambda x: x.mode())
)
labels = (
	adata_to_annotate.obs[["predicted", "sclc_cancer_labels", "sclc_normal_labels"]]
	.groupby("overcluster")
	.agg(lambda x: x.mean())
)

mapping_res = labels.merge(right=scores, left_index=True, right_index=True)

# We can now use our SCLC score + transfered labels from the annotated reference + manually looking for marker genes for each cell type, to confirm and/or change the cell types for each cluster

for x in range(len(adata_to_annotate.obs.overcluster.unique())):
	print(f'"{x}":"",')

over2cell = {}
# Now we can use PanglaoDB, CellTypist markers, etc to look for markers associated with a given cell type, or manually look in references for markers

# First, find marker genes for each cluster
sc.tl.rank_genes_groups(adata_to_annotate, groupby="overcluster")
marks = sc.get.rank_genes_groups_df(adata_to_annotate, group=None)

# Plot using umap for the markers, and see if they overlap with the clusters and cell types on the UMAP
sc.pl.umap(adata_to_annotate, color=["marker"], legend_loc="on data", s=5)

# ... and compare with the marks dataframe, to see if the markers on the umap above, correspond to the clusters found in the rank_genes_groups
marks[marks.names.isin(["markers"])].sort_values(
	"logfoldchanges", ascending=False
).head()

# ... we can further compare with our SCLC score to see if the cell type we're evaluating has a high and positive SCLC score (if its SCLC) or not (if its other cell types such as immune, etc). If the markers identify clusters associated with non SCLC and the SCLC score is negative, then we have strong indications that is in fact not a SCLC cell for that cluster


# Plotting trick to make UMAP easily visible, grey
ax = sc.pl.umap(adata_to_annotate, palette="lightgrey", show=False)
sc.pl.umap(adata_to_annotate[adata_to_annotate.obs.overcluster=="1"], color="overcluster", ax=ax, legend_loc=None, palette="k")

# Finally, map the dictionary to the cell type
adata_to_annotate.obs["final_cell_type"] = adata.obs.overcluster.map(over2cell)
sc.pl.umap(adata_to_annotate color = ["final_cell_type"], s=2, legend_loc = "on data")
# ################# 4.3 Annotation using reference atlas and query dataset ##############################
# ################# 4.3.1 scArches ##############################


import decoupler as dc
dc.show_resources()
