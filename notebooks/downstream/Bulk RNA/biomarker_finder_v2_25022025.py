# %% Test
# Standard library imports
import os
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import set_config
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
import decoupler as dc
import gseapy as gp
from gseapy import prerank
from gseapy.plot import dotplot, gseaplot
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, rgb2hex, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from pydeseq2 import preprocessing, dds, ds
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc
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
from sympy import degree
from tqdm import tqdm
import umap

# Local imports
from cnmf import cNMF
from catboost import CatBoostClassifier
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
import scanpy as sc
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
import scanpy as sc
from scipy import stats
import itertools
import adjustText
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

def plot_gene_expression_heatmap(
    adata,
    genes_to_plot,
    gene_categories,
    group_column,
    layer="raw_counts",
    figsize=(20, 8),
    vmin=-2,
    vmax=2,
    cmap_colors=None,
):
    """
    Plot a gene expression heatmap with gene categories and group annotations.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    genes_to_plot : list
        List of genes to plot
    gene_categories : dict
        Dictionary mapping category names to lists of genes
    group_column : str
        Column in adata.obs for grouping cells
    layer : str, default='raw_counts'
        Layer in adata to use for expression values. Options: 'raw_counts', 'normalized_counts', 'lognormalized_counts'
    figsize : tuple, default=(20, 8)
        Figure size
    vmin, vmax : float, default=-2, 2
        Color scale limits
    cmap_colors : list, optional
        List of colors for the heatmap colormap. If None, uses default colors
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # Filter genes that exist in the data
    genes_to_plot = [gene for gene in genes_to_plot if gene in adata.var_names]
    print(f"Number of genes to plot: {len(genes_to_plot)}")

    # Get expression data from the specified layer
    if layer not in adata.layers and layer != "raw_counts":
        raise ValueError(
            f"Layer {layer} not found in AnnData object. Available layers: {list(adata.layers.keys())}"
        )

    if layer == "raw_counts":
        adata.X = adata.layers[layer].copy()
        data = adata.X
    else:
        data = adata.layers[layer]

    # Calculate z-scores
    adata.layers["z_score"] = (data - data.mean(axis=0)) / data.std(axis=0)

    # Calculate average expression for each group
    group_avg_expr = pd.DataFrame(
        adata.layers["z_score"][
            :, [adata.var_names.get_loc(gene) for gene in genes_to_plot]
        ],
        columns=genes_to_plot,
        index=adata.obs_names,
    )
    group_avg_expr[group_column] = adata.obs[group_column]
    group_avg_expr = group_avg_expr.groupby(group_column).mean()

    # Sort cells within each group
    new_order = []
    for group in adata.obs[group_column].cat.categories:
        group_cells = adata.obs_names[adata.obs[group_column] == group]
        group_expr = group_avg_expr.loc[group]
        correlations = group_cells.map(
            lambda cell: np.corrcoef(
                adata[cell, genes_to_plot].layers["z_score"][0], group_expr
            )[0, 1]
        )
        sorted_cells = group_cells[correlations.argsort()[::-1]]
        new_order.extend(sorted_cells)

    # Reorder the AnnData object
    adata_ordered = adata[new_order].copy()

    # Calculate dimensions for the plot
    gene_groups = list(gene_categories.keys())
    widths = []
    genes_by_group = []

    for category in gene_groups:
        genes_in_cat = [
            gene for gene in genes_to_plot if gene in gene_categories[category]
        ]
        widths.append(len(genes_in_cat))
        genes_by_group.append(genes_in_cat)

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(gene_groups), width_ratios=widths)

    # Create custom colormap for heatmap
    if cmap_colors is None:
        cmap_colors = [
            "#00204c",
            "#213d6c",
            "#555b6c",
            "#7b7b7b",
            "#a59c74",
            "#ebc174",
            "#fad77b",
        ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors, N=100)

    # Get group boundaries
    unique_groups = adata_ordered.obs[group_column].unique()
    n_groups = len(unique_groups)

    # Generate colors for groups using seaborn color palette
    group_colors = sns.color_palette("husl", n_colors=n_groups)

    group_sizes = [
        sum(adata_ordered.obs[group_column] == group) for group in unique_groups
    ]
    group_boundaries = np.cumsum([0] + group_sizes)
    group_centers = [
        (start + end) / 2
        for start, end in zip(group_boundaries[:-1], group_boundaries[1:])
    ]

    # Plot each gene group
    for idx, (category, genes) in enumerate(zip(gene_groups, genes_by_group)):
        if not genes:  # Skip empty categories
            continue

        ax = plt.subplot(gs[idx])

        # Get data for these genes
        data = adata_ordered.layers["z_score"][
            :, [adata_ordered.var_names.get_loc(gene) for gene in genes]
        ]

        # Create heatmap
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Set title and labels
        ax.set_title(category, pad=10)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=45, ha="right")

        # Add group annotations for the first subplot
        if idx == 0:
            ax.set_yticks(group_centers)
            ax.set_yticklabels(
                adata_ordered.obs[group_column].cat.categories
            )  # Changed this line
            ax.set_ylabel(group_column)

            # Add colored bars for groups
            for i, (start, end) in enumerate(
                zip(group_boundaries[:-1], group_boundaries[1:])
            ):
                ax.add_patch(
                    plt.Rectangle(
                        (-0.5, start - 0.5),
                        0.15,
                        end - start,
                        facecolor=group_colors[i],
                        clip_on=False,
                    )
                )
        else:
            ax.set_yticks([])

        # Add horizontal lines between groups
        for boundary in group_boundaries[1:-1]:
            ax.axhline(y=boundary - 0.5, color="white", linewidth=2)

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, len(genes), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, axis="x")

        # Add borders
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(True)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Z-score", rotation=270, labelpad=15)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    return fig

class DEGAnalysis:
    def __init__(
        self,
        adata,
        design_factor,
        layer="raw_counts",
        output_dir="deg_analysis_results",
    ):
        self.original_adata = adata
        self.design_factor = design_factor
        self.layer = layer
        self.adata = None
        self.dds = None
        self.results = {}
        self.colors = self._generate_colors()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_colors(self):
        unique_groups = self.original_adata.obs[self.design_factor].unique()
        n_colors = len(unique_groups)
        color_map = plt.cm.get_cmap("tab20")
        colors = {
            group: rgb2hex(color_map(i / n_colors))
            for i, group in enumerate(unique_groups)
        }
        return colors

    def prepare_data(self):
        self.adata = self.original_adata.copy()
        if self.layer in self.adata.layers:
            self.adata.X = self.adata.layers[self.layer].copy()
        elif self.layer != "X":
            raise ValueError(f"Layer '{self.layer}' not found in the AnnData object.")
        min_val = self.adata.X.min()
        if min_val < 0:
            self.adata.X -= min_val

    def create_dds(self):
        if self.adata is None:
            self.prepare_data()
        self.dds = DeseqDataSet(
            adata=self.adata,
            design_factors=self.design_factor,
            refit_cooks=True,
        )
        self.dds.deseq2()

    def run_comparisons(self):
        subtypes = sorted(self.adata.obs[self.design_factor].unique().tolist())
        n_subtypes = len(subtypes)

        # One-vs-One comparisons
        for i in range(n_subtypes):
            for j in range(i + 1, n_subtypes):
                self._run_comparison(subtypes[i], subtypes[j])

        # One-vs-Rest comparisons
        for subtype in subtypes:
            self._run_comparison(subtype, "rest", one_vs_rest=True)

    def _run_comparison(self, group1, group2, one_vs_rest=False):
        comparison_name = f"{group1}_vs_{group2}"
        if one_vs_rest:
            temp_counts = pd.DataFrame(
                self.adata.X, index=self.adata.obs_names, columns=self.adata.var_names
            )
            temp_metadata = pd.DataFrame(
                {self.design_factor: self.adata.obs[self.design_factor]}
            )
            temp_metadata[self.design_factor] = np.where(
                temp_metadata[self.design_factor] == group1, group1, "rest"
            )

            temp_dds = DeseqDataSet(
                counts=temp_counts,
                metadata=temp_metadata,
                design_factors=self.design_factor,
                refit_cooks=True,
            )
            temp_dds.deseq2()
            res = DeseqStats(temp_dds, contrast=[self.design_factor, group1, "rest"])
            res.summary()

            res.results_df["Group"] = np.where(
                res.results_df["log2FoldChange"] < 0, "Up in Others", f"Up in {group1}"
            )
        else:
            res = DeseqStats(self.dds, contrast=[self.design_factor, group1, group2])
            res.summary()

            res.results_df["Group"] = np.where(
                res.results_df["log2FoldChange"] < 0,
                f"Up in {group2}",
                f"Up in {group1}",
            )
        res.results_df["Log2FC_pval"] = res.results_df["log2FoldChange"] * -np.log10(
            res.results_df["pvalue"]
        )
        res.results_df = res.results_df.sort_values("Log2FC_pval", ascending=False)

        self.results[comparison_name] = res

    def get_results(self):
        """
        Get all DEG analysis results.

        Returns:
        --------
        dict
            Dictionary containing all DEG analysis results
        """
        return self.results

    def create_volcano_grid(self, highlight_dict=None):
        """
        Create a grid of volcano plots with highlighted genes from a dictionary.
        """
        # Get unique group names
        group_names = sorted(
            set([k.split("_vs_")[0] for k in self.results.keys() if "_vs_" in k])
        )
        n_groups = len(group_names)

        # Increase figure size for better readability
        fig = plt.figure(figsize=(5 * n_groups, 5 * n_groups))
        gs = GridSpec(n_groups, n_groups)

        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names):
                if i < j:  # Upper triangle
                    ax = fig.add_subplot(gs[i, j])
                    comparison = f"{group1}_vs_{group2}"
                    if comparison in self.results:
                        self._plot_volcano(
                            ax,
                            self.results[comparison].results_df,
                            f"{group1} vs. {group2}",
                            self.colors[group1],
                            group1,
                            group2,
                            highlight_dict=highlight_dict,
                        )
                elif i == j:  # Diagonal
                    ax = fig.add_subplot(gs[i, i])
                    comparison = f"{group1}_vs_rest"
                    if comparison in self.results:
                        self._plot_volcano(
                            ax,
                            self.results[comparison].results_df,
                            f"{group1} vs. Others",
                            self.colors[group1],
                            group1,
                            "rest",
                            highlight_dict=highlight_dict,
                        )

        # Add legend for highlighted genes
        if highlight_dict:
            colors = sns.color_palette("husl", len(highlight_dict))
            legend_elements = []
            for (group, genes), color in zip(highlight_dict.items(), colors):
                # Add marker for the group
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        label=f"{group}: {', '.join(genes)}",
                        markersize=8,
                    )
                )

            # Place legend outside the plot
            fig.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title="Gene Groups",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "volcano_grid.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_volcano(
        self, ax, results, title, color, group1, group2, highlight_dict=None, top_n=10
    ):
        """
        Plot a single volcano plot with highlighted genes from different groups.

        Parameters:
            ...
            top_n: int, optional (default=10)
                Number of top genes to label for both up and down regulated conditions
        """
        # Plot all points
        ax.scatter(
            results["log2FoldChange"],
            -np.log10(results["pvalue"]),
            alpha=0.6,
            s=3,
            color="lightgray",
        )

        # Add threshold lines
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5)

        texts = []  # Collection of text elements for adjustText

        # Add direction annotation
        ax.text(
            0.02,
            0.98,
            f"Positive log2FC: Upregulated in {group1}\nNegative log2FC: Upregulated in {group2}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Find top N up and down regulated genes
        if top_n > 0:
            # Calculate a score combining fold change and p-value
            results["score"] = abs(results["log2FoldChange"]) * (
                -np.log10(results["pvalue"])
            )

            # Get top upregulated genes (positive fold change)
            up_genes = results[results["log2FoldChange"] > 0].nlargest(top_n, "score")
            # Get top downregulated genes (negative fold change)
            down_genes = results[results["log2FoldChange"] < 0].nlargest(top_n, "score")

            # Plot top genes
            for genes, is_up in [(up_genes, True), (down_genes, False)]:
                color = "darkred" if is_up else "darkblue"
                for gene_name, gene_data in genes.iterrows():
                    text = ax.text(
                        gene_data["log2FoldChange"],
                        -np.log10(gene_data["pvalue"]),
                        gene_name,
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                        bbox=dict(
                            facecolor="white", alpha=0.8, edgecolor="none", pad=0.5
                        ),
                    )
                    texts.append(text)

                    # Add point
                    ax.scatter(
                        gene_data["log2FoldChange"],
                        -np.log10(gene_data["pvalue"]),
                        color=color,
                        s=30,
                        zorder=5,
                    )

        # Highlight genes from dictionary
        if highlight_dict:
            colors = sns.color_palette("husl", len(highlight_dict))
            for (group, genes), highlight_color in zip(highlight_dict.items(), colors):
                for gene in genes:
                    if gene in results.index:
                        gene_data = results.loc[gene]
                        # Add text with background
                        text = ax.text(
                            gene_data["log2FoldChange"],
                            -np.log10(gene_data["pvalue"]),
                            gene,
                            fontsize=8,
                            color=highlight_color,
                            fontweight="bold",
                            bbox=dict(
                                facecolor="white", alpha=0.8, edgecolor="none", pad=0.5
                            ),
                        )
                        texts.append(text)

                        # Add point
                        ax.scatter(
                            gene_data["log2FoldChange"],
                            -np.log10(gene_data["pvalue"]),
                            color=highlight_color,
                            s=30,
                            zorder=5,
                        )

        # Adjust text positions to prevent overlap with more space
        if texts:
            adjustText.adjust_text(
                texts,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
                expand_points=(2.0, 2.0),
                force_points=(2.0, 2.0),
                force_text=(2.0, 2.0),
                lim=500,
            )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("log2(Fold Change)", fontsize=10)
        ax.set_ylabel("-log10(p-value)", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, which="both", ls="-", alpha=0.2)

    def create_clustermaps(
        self, marker_list=None, padj_threshold=0.05, log2fc_threshold=1
    ):
        """
        Create clustermaps for each comparison.

        Parameters:
        -----------
        marker_list : list, optional
            List of marker genes to include in the plot
        padj_threshold : float
            Adjusted p-value threshold for significant genes
        log2fc_threshold : float
            Log2 fold change threshold for significant genes
        """
        for comparison_name, res in self.results.items():
            self._create_clustermap(
                res.results_df,
                comparison_name,
                marker_list,
                padj_threshold,
                log2fc_threshold,
            )

    def _create_clustermap(
        self,
        results_df,
        comparison_name,
        marker_list=None,
        padj_threshold=0.05,
        log2fc_threshold=1,
    ):
        """
        Create a single clustermap for a comparison.
        """
        cluster1, cluster2 = comparison_name.split("_vs_")
        significant_genes = results_df[
            (results_df["padj"] < padj_threshold)
            & (abs(results_df["log2FoldChange"]) > log2fc_threshold)
        ].index

        if marker_list is not None:
            marker_genes = [gene for gene in marker_list if gene in significant_genes]
            if len(marker_genes) < 2:
                print(
                    f"Warning: Less than 2 genes from the marker list are present in the significant genes for {comparison_name}."
                )
                print("Using all significant genes instead.")
                genes_to_plot = significant_genes
            else:
                genes_to_plot = marker_genes
        else:
            genes_to_plot = significant_genes

        if len(genes_to_plot) < 2:
            print(
                f"Not enough genes to plot for {comparison_name}. Skipping this comparison."
            )
            return

        if cluster2 == "rest":
            dds_sub = self.dds[:, genes_to_plot]
        else:
            dds_sub = self.dds[
                self.dds.obs[self.design_factor].isin([cluster1, cluster2]),
                genes_to_plot,
            ]

        if dds_sub.shape[1] < 2 or dds_sub.shape[0] < 2:
            print(
                f"Not enough data to plot for {comparison_name}. Skipping this comparison."
            )
            print(
                f"Number of genes: {dds_sub.shape[1]}, Number of samples: {dds_sub.shape[0]}"
            )
            return

        # Sort samples by design factor
        dds_sub = dds_sub[dds_sub.obs[self.design_factor].argsort()]

        # Create expression matrix
        if "lognormalized_counts" in dds_sub.layers:
            expr_matrix = pd.DataFrame(
                dds_sub.layers["lognormalized_counts"].T,
                index=dds_sub.var_names,
                columns=dds_sub.obs_names,
            )
        else:
            sc.pp.normalize_total(dds_sub)
            sc.pp.log1p(dds_sub)
            expr_matrix = pd.DataFrame(
                dds_sub.X.T, index=dds_sub.var_names, columns=dds_sub.obs_names
            )

        # Create the clustermap
        plt.figure(figsize=(30, 15 + 0.2 * len(genes_to_plot)))
        col_colors = dds_sub.obs[self.design_factor].map(self.colors)

        try:
            # Calculate z-scores
            expr_matrix_zscore = (expr_matrix - expr_matrix.mean()) / expr_matrix.std()

            # Set vmin and vmax for consistent color scaling
            vmin = -3
            vmax = 3

            g = sns.clustermap(
                expr_matrix_zscore,
                cmap="RdBu_r",
                col_cluster=False,
                row_cluster=True,
                col_colors=col_colors,
                dendrogram_ratio=(0.1, 0.3),
                vmin=vmin,
                vmax=vmax,
            )

            # Adjust labels
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center", va="top")
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

            # Add legend
            unique_groups = (
                [cluster1, cluster2] if cluster2 != "rest" else [cluster1, "rest"]
            )
            for label in unique_groups:
                g.ax_col_dendrogram.bar(
                    0, 0, color=self.colors[label], label=label, linewidth=0
                )
            g.ax_col_dendrogram.legend(title=self.design_factor, loc="center", ncol=2)

            plt.suptitle(f"Clustermap for {comparison_name}", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, f"clustermap_{comparison_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"Error creating clustermap for {comparison_name}: {str(e)}")
            plt.close()

    def save_results(self):
        """
        Save all DEG analysis results to CSV files.
        """
        for comparison, res in self.results.items():
            output_file = os.path.join(self.output_dir, f"deg_results_{comparison}.csv")
            res.results_df.to_csv(output_file)
            print(f"Saved results for {comparison} to {output_file}")

    def create_boxplots(
        self, genes_to_plot, test="ttest", figsize=(20, 5), save_path=None
    ):
        """
        Create boxplots for specified genes.

        Parameters:
        -----------
        genes_to_plot : list or str
            Gene or list of genes to plot
        test : str
            Statistical test to use ('ttest' or 'mannwhitney')
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if isinstance(genes_to_plot, str):
            genes_to_plot = [genes_to_plot]

        # Prepare data
        valid_genes = [gene for gene in genes_to_plot if gene in self.adata.var_names]
        if not valid_genes:
            print("No valid genes found in the dataset.")
            return

        # Create normalized data if not already present
        adata_subset = self.adata[:, valid_genes].copy()
        if "lognormalized_counts" not in adata_subset.layers:
            sc.pp.normalize_total(adata_subset)
            sc.pp.log1p(adata_subset)
            adata_subset.layers["lognormalized_counts"] = adata_subset.X.copy()

        n_genes = len(valid_genes)
        fig, axes = plt.subplots(1, n_genes, figsize=figsize)
        if n_genes == 1:
            axes = [axes]

        for i, gene in enumerate(valid_genes):
            ax = axes[i]

            # Prepare data for plotting
            data = pd.DataFrame(
                {
                    "expression": adata_subset[:, gene]
                    .layers["lognormalized_counts"]
                    .flatten(),
                    self.design_factor: adata_subset.obs[self.design_factor],
                }
            )

            # Create boxplot
            clusters = sorted(data[self.design_factor].unique())
            cluster_to_pos = {cluster: idx for idx, cluster in enumerate(clusters)}

            sns.boxplot(
                data=data,
                x=self.design_factor,
                y="expression",
                ax=ax,
                order=clusters,
                palette=self.colors,
            )
            sns.stripplot(
                data=data,
                x=self.design_factor,
                y="expression",
                color="black",
                size=2,
                alpha=0.4,
                ax=ax,
                order=clusters,
            )

            # Add statistical comparisons
            comparisons = list(itertools.combinations(clusters, 2))
            max_bars = len(comparisons)

            plot_top = ax.get_ylim()[1]
            bar_height = plot_top * 0.05
            spacing = plot_top * 0.1

            for idx, (c1, c2) in enumerate(comparisons):
                data1 = data[data[self.design_factor] == c1]["expression"]
                data2 = data[data[self.design_factor] == c2]["expression"]

                if test == "ttest":
                    _, p_value = stats.ttest_ind(data1, data2)
                else:  # mannwhitney
                    _, p_value = stats.mannwhitneyu(data1, data2)

                # Add significance bars
                y_pos = plot_top + spacing + (bar_height + spacing) * idx
                x1, x2 = cluster_to_pos[c1], cluster_to_pos[c2]
                ax.plot(
                    [x1, x1, x2, x2],
                    [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos],
                    lw=1.5,
                    c="black",
                )

                # Add significance stars
                significance = self._get_stars(p_value)
                ax.text(
                    (x1 + x2) / 2,
                    y_pos + bar_height,
                    significance,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Customize plot
            ax.set_title(f"{gene}", fontsize=14)
            ax.set_xlabel(self.design_factor.capitalize(), fontsize=12)
            ax.set_ylabel("log2(CPM+1)" if i == 0 else "", fontsize=12)
            ax.set_ylim(0, plot_top + (bar_height + spacing) * (max_bars + 1))
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _get_stars(self, p_value):
        """Convert p-value to star notation."""
        if p_value > 0.05:
            return "ns"
        elif p_value > 0.01:
            return "*"
        elif p_value > 0.001:
            return "**"
        else:
            return "***"

def get_top_genes(
    source_paths: Dict[str, str],
    n_genes: int = 10,
    mode: str = "top",
    sort_by: str = "stat",
) -> Dict[str, List[str]]:
    """
    Get top or bottom genes from CSV files containing differential expression results.

    Args:
        source_paths: Dictionary where keys are comparison names and values are paths to CSV files
        n_genes: Number of genes to return
        mode: Either 'top' or 'bottom' to get the highest or lowest ranked genes
        sort_by: Column to sort by. Can be 'stat', 'log2FoldChange', 'padj', or any other valid column name

    Returns:
        Dictionary containing selected genes for each comparison
    """
    if mode not in ["top", "bottom"]:
        raise ValueError("mode must be either 'top' or 'bottom'")

    deg_results = {}

    for comparison, file_path in source_paths.items():
        # Read CSV file
        results = pd.read_csv(file_path)

        # Verify sort column exists
        if sort_by not in results.columns:
            raise ValueError(
                f"Column '{sort_by}' not found in results. Available columns: {list(results.columns)}"
            )

        # Sort by specified column
        ascending = mode == "bottom"  # True if mode is 'bottom', False if 'top'
        results = results.sort_values(sort_by, ascending=ascending)

        # Print genes info
        print(f"\n{comparison} {mode.capitalize()} Genes (sorted by {sort_by}):")
        print(
            results[["gene_identifier", sort_by, "log2FoldChange", "padj"]].head(
                n_genes
            )
        )

        # Store gene identifiers
        deg_results[comparison] = results["gene_identifier"].head(n_genes).tolist()

    return deg_results

def calculate_risk_scores(
    adata, coef_df, layer="lognormalized_counts", endpoint="os", lower_bound=30, upper_bound=71, cutpoint_method="max_rank"
):
    """
    Calculate risk scores and find optimal cutpoint for stratification using sksurv.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and metadata
    coef_df : pd.DataFrame
        DataFrame containing gene coefficients from Cox model
    layer : str, optional
        Layer in AnnData to use for expression values, by default "lognormalized_counts"
    endpoint : str, optional
        Endpoint to use for survival analysis ('os' or 'ttp'), by default "os"

    Returns
    -------
    pd.DataFrame
        DataFrame containing risk scores and groupings
    """
    # Validate endpoint
    if endpoint not in ["os", "ttp"]:
        raise ValueError("endpoint must be either 'os' or 'ttp'")

    print(f"Running endpoint {endpoint}")
    # Get the relevant survival information based on endpoint
    status_col = f"status-{endpoint}"
    time_col = f"tend_{endpoint}"

    if status_col not in adata.obs.columns or time_col not in adata.obs.columns:
        raise ValueError(f"Missing required columns: {status_col} and/or {time_col}")

    # Create expression matrix for selected genes
    selected_genes = coef_df.index.tolist()
    gene_mask = adata.var_names.isin(selected_genes)
    if not any(gene_mask):
        raise ValueError("None of the genes in coef_df found in adata")

    # Create expression DataFrame
    expr = pd.DataFrame(
        adata.layers[layer][:, gene_mask],
        index=adata.obs.index,
        columns=adata.var_names[gene_mask],
    )

    print(
        f"Expression data being used to calculate the risk scores:\n {expr} using {layer}"
    )

    # Calculate risk scores
    risk_scores = np.zeros(len(expr))
    coefficients = {}

    # Build and print the formula
    formula_parts = []
    for gene in selected_genes:
        if gene in expr.columns:
            coef = coef_df.loc[gene, "coefficient"]
            coefficients[gene] = coef
            risk_scores += expr[gene] * coef
            if abs(coef) > 1e-10:  # Filter out coefficients very close to zero
                formula_parts.append(f"({coef:.4f} × {gene})")

    if formula_parts:
        print(f"\nRisk score formula for {endpoint.upper()}:")
        print("Risk Score = " + " + ".join(formula_parts))
        print(
            "\nWhere positive coefficients indicate higher risk and negative coefficients indicate lower risk"
        )
    else:
        print(f"\nNo non-zero coefficients found for {endpoint.upper()}")

    # Create risk scores DataFrame with metadata
    risk_scores_df = pd.DataFrame(
        {
            "risk_score": risk_scores,
            "time": adata.obs[time_col],
            "status": adata.obs[status_col].map(
                {False: 0, True: 1}
                if endpoint == "os"
                else {False: 0, True: 1}
            ),
        },
        index=adata.obs.index,
    )

    print(f"Risk scores data:\n {risk_scores_df}")

    # Add expression values for selected genes
    risk_scores_df = pd.concat([risk_scores_df, expr], axis=1)

    if cutpoint_method == "max_rank":
        # Find optimal cutpoint using log-rank test
        cutpoints = np.percentile(risk_scores, np.arange(lower_bound, upper_bound, 1))
        max_statistic = 0
        optimal_cutpoint = None

        for cutpoint in cutpoints:
            groups = risk_scores > cutpoint

            # Create structured array for survival data
            survival_data = np.zeros(
                len(risk_scores_df), dtype=[("status", bool), ("time", float)]
            )
            survival_data["status"] = risk_scores_df["status"].values.astype(bool)
            survival_data["time"] = risk_scores_df["time"].values

            # Calculate log-rank test
            chisq, pvalue = compare_survival(survival_data, groups)

            if chisq > max_statistic:
                max_statistic = chisq
                optimal_cutpoint = cutpoint

    elif cutpoint_method == "median":
        optimal_cutpoint = risk_scores_df["risk_score"].median()

    elif cutpoint_method == "youden":
        # Calculate the ROC curve and select the threshold with the maximum Youden index
        from sklearn.metrics import roc_curve
        print(risk_scores_df)
        # Note: risk_scores_df["status"] should be binary (0/1)
        fpr, tpr, thresholds = roc_curve(risk_scores_df["status"], risk_scores_df["risk_score"])
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_cutpoint = thresholds[optimal_idx]
        print(f"Optimal cutpoint (Youden index): {optimal_cutpoint}")

    else:
        print(f"Cutpoint method {cutpoint_method} not supported. ")


    # Assign risk groups - higher scores = higher risk
    risk_scores_df["risk_group"] = (
        risk_scores_df["risk_score"] > optimal_cutpoint
    ).map({True: "high", False: "low"})

    # Sort by risk score for visualization
    risk_scores_df = risk_scores_df.sort_values("risk_score")
    risk_scores_df["rank"] = range(len(risk_scores_df))

    print(f"\nMedian time until event by risk scores group (high vs low):")
    for group in risk_scores_df["risk_group"].unique():
        group_df = risk_scores_df[risk_scores_df["risk_group"] == group]
        print(f"{group}:")
        print(f"Median survival time: {group_df['time'].median():.1f} days")

        # Calculate event statistics
        n_total = len(group_df)
        n_events = group_df["status"].sum()
        n_censored = n_total - n_events
        print(f"Events: {n_events}/{n_total} ({n_events/n_total*100:.1f}%)")
        print(f"Censored: {n_censored}/{n_total} ({n_censored/n_total*100:.1f}%)\n")

    print("Statistics by subtype and group")
    risk_scores_df_with_subtype = pd.concat(
        [risk_scores_df, adata.obs[["SCLC_Subtype_de_novo"]]], axis=1
    )
    summary = (
        risk_scores_df_with_subtype.groupby(["risk_group", "SCLC_Subtype_de_novo"])
        .agg({"time": "mean", "status": ["count", "sum"]})
        .rename(columns={"sum": "events", "count": "total"})
    )

    # Now status=1 means event occurred, so sum gives us events
    # And status=0 means censored, so we can calculate censored as total - events
    summary["status", "censored"] = (
        summary["status", "total"] - summary["status", "events"]
    )

    print(summary)

    # Create the combined risk profile plot
    plot_risk_profile(risk_scores_df, adata, coefficients, optimal_cutpoint, layer)

    # Calculate HR using Cox model
    X = (risk_scores_df["risk_group"] == "high").values.reshape(-1, 1)
    y = np.zeros(len(X), dtype=[("status", bool), ("time", float)])
    y["status"] = risk_scores_df["status"].astype(bool)
    y["time"] = risk_scores_df["time"]

    print(
        f"Cox model data:\n Risk scores df:\n {risk_scores_df} \nX: \n{X}, \nY:\n {y} \n Cutoff point:\n {optimal_cutpoint}"
    )
    cph = CoxPHSurvivalAnalysis()
    cph.fit(X, y)
    hr = np.exp(cph.coef_[0])
    from lifelines.statistics import logrank_test

    high_risk = risk_scores_df[risk_scores_df["risk_group"] == "high"]
    low_risk = risk_scores_df[risk_scores_df["risk_group"] == "low"]

    # Perform Log-rank test
    log_rank_result = logrank_test(
        high_risk["time"], low_risk["time"], 
        event_observed_A=high_risk["status"], 
        event_observed_B=low_risk["status"]
    )

    # Get the p-value
    log_rank_pval = log_rank_result.p_value

    # Plot Kaplan-Meier curves with statistics
    plt.figure(figsize=(10, 6))

    for group, color in zip(["high", "low"], ["red", "blue"]):
        mask = risk_scores_df["risk_group"] == group
        if sum(mask) > 0:
            time = risk_scores_df.loc[mask, "time"]
            status = risk_scores_df.loc[mask, "status"].astype(bool)

            time_km, survival_prob = kaplan_meier_estimator(status, time)

            plt.step(
                time_km,
                survival_prob,
                where="post",
                label=f"{group.capitalize()} risk (n={sum(mask)})",
                color=color,
            )

    # Add HR and p-value to the plot
    stats_text = f"HR = {hr:.2f}\n" f"Log-rank P = {log_rank_pval:.2e}"
    plt.text(
        0.05,
        0.15,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(f"Kaplan-Meier Curves by Risk Group ({endpoint.upper()})")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print statistics
    print("\nRisk group sizes:")
    print(risk_scores_df["risk_group"].value_counts())
    print(f"\nOptimal cutpoint: {optimal_cutpoint:.4f}")
    print(f"Log-rank test p-value: {log_rank_pval:.2e}")
    print(f"Hazard Ratio: {hr:.2f}")

    return risk_scores_df

def plot_risk_profile(
    risk_scores_df, adata, gene_coefficients, optimal_cutpoint, layer="raw_counts", cutpoint_method="max_rank"
):
    """
    Create a combined plot showing risk scores, survival status, and gene expression profiles.

    Parameters
    ----------
    risk_scores_df : pd.DataFrame
        DataFrame containing risk scores and patient information
    adata : AnnData
        AnnData object containing expression data
    gene_coefficients : dict
        Dictionary mapping gene names to their coefficients
    optimal_cutpoint : float
        Optimal cutpoint for risk stratification
    layer : str, optional
        Layer in AnnData to use for expression values, by default "raw_counts"
    """
    # Create figure with subplots
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [1, 1, 2]}, sharex=True
    )
    plt.subplots_adjust(hspace=0.05)

    # 1. Risk score distribution
    axes[0].scatter(
        range(len(risk_scores_df)),
        risk_scores_df["risk_score"],
        c=["red" if x == "high" else "blue" for x in risk_scores_df["risk_group"]],
        s=30,
    )
    axes[0].axvline(
        x=sum(risk_scores_df["risk_group"] == "low"), color="red", linestyle="--"
    )
    axes[0].axhline(y=optimal_cutpoint, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Risk score")
    axes[0].grid(True)

    # 2. Survival status
    survival_data = pd.DataFrame(
        {
            "time": risk_scores_df["time"],
            "status": risk_scores_df["status"],
        }
    )

    print(f"Survival data:\n {survival_data}")
    scatter = axes[1].scatter(
        range(len(survival_data)),
        survival_data["time"],
        c=["red" if s else "blue" for s in survival_data["status"]],
        s=30,
    )
    axes[1].set_ylabel("Time (days)")
    axes[1].grid(True)

    # Add legend for survival status
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            label="Event",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            label="Censored",
            markersize=8,
        ),
    ]
    axes[1].legend(handles=legend_elements, loc="right")

    # 3. Gene expression heatmap
    genes = list(gene_coefficients.keys())

    # Filter out genes with zero coefficients
    non_zero_genes = [gene for gene in genes if abs(gene_coefficients[gene]) > 1e-10]

    # Get expression data from specified layer
    gene_mask = adata.var_names.isin(non_zero_genes)
    expr_data = pd.DataFrame(
        adata.layers[layer][:, gene_mask],
        index=adata.obs.index,
        columns=adata.var_names[gene_mask],
    )

    # Z-score normalization using numpy operations
    expr_matrix = expr_data.values
    expr_data_scaled = (expr_matrix - expr_matrix.mean(axis=0)) / expr_matrix.std(
        axis=0
    )
    expr_data_scaled = pd.DataFrame(
        expr_data_scaled, columns=non_zero_genes, index=expr_data.index
    )

    # Sort genes by absolute coefficient value
    gene_coef_pairs = [(gene, abs(gene_coefficients[gene])) for gene in non_zero_genes]
    sorted_genes = [
        gene for gene, _ in sorted(gene_coef_pairs, key=lambda x: x[1], reverse=True)
    ]

    # Create heatmap with z-scaled data
    sns.heatmap(
        expr_data_scaled.loc[risk_scores_df.index, sorted_genes].T,
        cmap="RdBu_r",
        center=0,
        ax=axes[2],
        cbar_kws={"label": "Z-score"},
    )

    # Add a vertical line to separate low and high risk groups
    axes[2].axvline(
        x=sum(risk_scores_df["risk_group"] == "low"),
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    # Add labels for risk groups
    axes[2].text(
        sum(risk_scores_df["risk_group"] == "low") / 2,
        -0.5,
        "Low Risk",
        ha="center",
        va="top",
    )
    axes[2].text(
        sum(risk_scores_df["risk_group"] == "low")
        + sum(risk_scores_df["risk_group"] == "high") / 2,
        -0.5,
        "High Risk",
        ha="center",
        va="top",
    )

    axes[2].set_xlabel("Patients (ranked by risk score)")
    axes[2].set_ylabel("Genes")

    # Add title
    plt.suptitle(
        f"Risk Score Profile using the {cutpoint_method} method to find the optimal cutoff point",
        y=1.02,
    )

    # Adjust layout
    plt.tight_layout()
    plt.show()


####################### Load Data ######################
km_data_ttp = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_ttp.csv"
)
km_data_os = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_os.csv"
)

# /mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_cp_full_preprocessed.h5ad
# /mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_nmf_cp_81_samp.h5ad
adata_nmf = sc.read_h5ad(
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
adata_nmf_cp.X = adata_nmf_cp.layers["raw_counts"].copy()
for layer in list(adata_nmf_cp.layers.keys()):
    if layer != "raw_counts":
        del adata_nmf_cp.layers[layer]
adata_nmf_cp.layers["raw_counts"] = adata_nmf_cp.X.copy()
adata_nmf_cp.layers["normalized_counts"] = adata_nmf_cp.X.copy()
adata_nmf_cp.layers["lognormalized_counts"] = adata_nmf_cp.X.copy()
sc.pp.normalize_total(adata_nmf_cp, target_sum=1e6,layer="normalized_counts") 
sc.pp.log1p(adata_nmf_cp, layer="lognormalized_counts")

# %%
# ####################### Run PyDESeq2 ######################
# deg_os = DEGAnalysis(
#     adata_nmf_cp,
#     design_factor="status-os",
#     layer="raw_counts",
#     output_dir="./deg_analysis_os",
# )
# deg_os.create_dds()
# deg_os.run_comparisons()
# deg_os.save_results()
# deg_os.create_volcano_grid()
# results_deg_os = deg_os.get_results()
# os_top_genes = get_top_genes(
#     source_paths={
#         "Alive vs Dead": "/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/deg_analysis_os/deg_results_Alive_vs_Dead.csv"
#     },
#     n_genes=100,
#     mode="bottom",
#     sort_by="stat",
# )

# deg_ttp = DEGAnalysis(
#     adata_nmf_cp,
#     design_factor="status-ttp",
#     layer="raw_counts",
#     output_dir="./deg_analysis_ttp",
# )
# deg_ttp.create_dds()
# deg_ttp.run_comparisons()
# deg_ttp.save_results()
# deg_ttp.create_volcano_grid()
# results_deg_ttp = deg_ttp.get_results()
# ttp_top_genes = get_top_genes(
#     source_paths={
#         "Did not progress vs Progressed": "/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/deg_analysis_ttp/deg_results_Did not progress_vs_Progressed.csv"
#     },
#     n_genes=100,
#     mode="bottom",
#     sort_by="stat",
# )

###################### Test DEG ##############################

# status_map = {
#     "os": {"Dead": True, "Alive": False},
#     "ttp": {"Progressed": True, "Did not progress": False},
# }
# endpoint = "os"
# merged_data: pd.DataFrame = adata_nmf_cp.obs.copy()

# # 0. Get OS and TTP DEG genes on the training data.
# merged_data[f"status-{endpoint}"] = merged_data[f"status-{endpoint}"].map(status_map[endpoint])

# X = merged_data.copy()
# status_bool = merged_data[f"status-{endpoint}"]
# time = merged_data[f"tend_{endpoint}"]

# y = merged_data[f"status-{endpoint}"]

# y_structured = np.zeros(
#     len(time), dtype=[("status", bool), ("time", float)]
# )
# y_structured["status"] = status_bool
# y_structured["time"] = time

# # Split the data
# X_train, X_test, y_train_idx, y_test_idx = train_test_split(
#     X,
#     np.arange(len(y)), #y_structured
#     test_size=0.3,
#     random_state=42,
#     stratify=status_bool,
# )
# # Create structured arrays for train and test sets
# y_train = y[y_train_idx]
# y_test = y[y_test_idx]


# deg = DEGAnalysis(
#     X_train,
#     design_factor=f"status-{endpoint}",
#     layer="raw_counts",
#     output_dir=f"./deg_analysis_biomarker_{endpoint}",
# )
# deg.create_dds()
# deg.run_comparisons()
# deg.save_results()
# results_deg = deg.get_results()

# # 1. Center the data (subtract the mean to the lognorm expression data - subtract the mean expression value of each gene across all samples). Expression matrix has shape (n_genes, n_samples) and the values are the log of the CPM normalized counts. 
# expr_log = pd.DataFrame(
#     adata_nmf_cp.layers["lognormalized_counts"], index=adata_nmf_cp.obs.index, columns=adata_nmf_cp.var.index
# ).T
# centered_expr = expr_log - expr_log.mean(axis=0)

# # Fit the Cox model
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from statsmodels.stats.multitest import multipletests

# # Define range of top/bottom genes to test
# num_iterations = 100  # Number of resampling runs
# top_genes = 500
# all_iterations = []  # Store detailed iteration results
# summary_results = []  # Store summarized mean/std results

# # Step 1: Extract top/bottom genes from the DEG results (fixed from training set)
# full_results = results_deg["Alive_vs_Dead"].results_df.sort_values("stat", ascending=False)
# top_genes_full = full_results.head(top_genes).index.tolist()
# bottom_genes_full = full_results.tail(top_genes).index.tolist()

# # Step 2: Iterate over different numbers of genes
# for n_genes in range(5,top_genes,5):
#     hr_train_list = []
#     hr_test_list = []
#     pval_train_list = []
#     pval_test_list = []
#     gene_sets = []  # Store different selected gene sets
    
#     for i in range(num_iterations):
#         # Select n_genes from the top and bottom DEG results
#         selected_genes = top_genes_full[:n_genes] + bottom_genes_full[:n_genes]
#         gene_sets.append(selected_genes)

#         print(f"Running with top genes: {top_genes_full[:n_genes]} and bottom genes: {bottom_genes_full[:n_genes]}")

#         # Split data (randomly resampling without redoing DEG)
#         X_train, X_test, y_train_idx, y_test_idx = train_test_split(
#             expr, np.arange(len(y)), test_size=0.2, random_state=i, stratify=status_bool
#         ) 

#         # Compute gene scores for train and test set
#         filtered_expr_train = centered_expr.loc[selected_genes, X_train.index]
#         adata_nmf_cp.obs.loc[X_train.index, "gene_score"] = filtered_expr_train.sum(axis=0)

#         filtered_expr_test = centered_expr.loc[selected_genes, X_test.index]
#         adata_nmf_cp.obs.loc[X_test.index, "gene_score"] = filtered_expr_test.sum(axis=0)

#         # Fit Cox PH on training data
#         cox_data_train = pd.DataFrame({
#             'time': y_structured['time'][y_train_idx],
#             'status': y_structured['status'][y_train_idx],
#             'gene_score': adata_nmf_cp.obs.loc[X_train.index, "gene_score"]
#         })
#         cph_train = CoxPHFitter()
#         cph_train.fit(cox_data_train, duration_col='time', event_col='status')

#         # Fit Cox PH on test data
#         cox_data_test = pd.DataFrame({
#             'time': y_structured['time'][y_test_idx],
#             'status': y_structured['status'][y_test_idx],
#             'gene_score': adata_nmf_cp.obs.loc[X_test.index, "gene_score"]
#         })
#         cph_test = CoxPHFitter()
#         cph_test.fit(cox_data_test, duration_col='time', event_col='status')

#         # Store HR and p-values
#         hr_train = np.exp(cph_train.params_["gene_score"])
#         hr_test = np.exp(cph_test.params_["gene_score"])
#         pval_train = cph_train.summary.loc["gene_score", "p"]
#         pval_test = cph_test.summary.loc["gene_score", "p"]

#         hr_train_list.append(hr_train)
#         hr_test_list.append(hr_test)
#         pval_train_list.append(pval_train)
#         pval_test_list.append(pval_test)

#         # Store all iteration results
#         all_iterations.append({
#             "iteration": i,
#             "n_genes": n_genes,
#             "genes": selected_genes,
#             "HR_train": hr_train,
#             "HR_test": hr_test,
#             "pval_train": pval_train,
#             "pval_test": pval_test
#         })

#     # Apply FDR correction to p-values
#     fdr_train_corrected = multipletests(pval_train_list, method='fdr_bh')[1]  
#     fdr_test_corrected = multipletests(pval_test_list, method='fdr_bh')[1]

#     # Store summarized mean/std results
#     summary_results.append({
#         "n_genes": n_genes,
#         "HR_train_mean": np.mean(hr_train_list),
#         "HR_train_std": np.std(hr_train_list),
#         "pval_train_mean": np.mean(pval_train_list),
#         "pval_train_std": np.std(pval_train_list),
#         "fdr_train_mean": np.mean(fdr_train_corrected),
#         "fdr_train_std": np.std(fdr_train_corrected),
#         "HR_test_mean": np.mean(hr_test_list),
#         "HR_test_std": np.std(hr_test_list),
#         "pval_test_mean": np.mean(pval_test_list),
#         "pval_test_std": np.std(pval_test_list),
#         "fdr_test_mean": np.mean(fdr_test_corrected),
#         "fdr_test_std": np.std(fdr_test_corrected),
#     })

# # Convert results to DataFrames
# df_iterations = pd.DataFrame(all_iterations)  # All iterations
# df_summary = pd.DataFrame(summary_results)  # Mean/std summary

# # Find the best performing gene set (lowest test p-value)
# best_result = df_summary.loc[df_summary["pval_test_mean"].idxmin()]
# best_gene_number = best_result["n_genes"]
# best_genes = df_iterations.loc[df_iterations["n_genes"] == best_gene_number, "genes"].values[0]

# print("Best Number of Genes:", best_gene_number)
# print("Best Gene Set:", best_genes)
# print(best_result)


##################### Univariate testing of 23k genes + ElasticNet of the significant ones to get the risk score with max rank optimization ################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
# Define status mapping for overall survival (OS)
status_map = {
    "os": {"Dead": True, "Alive": False},
    "ttp": {"Progressed": True, "Did not progress": False},
}
endpoint = "os"

# Assuming adata_nmf_cp is your AnnData object with gene expression and clinical data
merged_data = adata_nmf_cp.obs.copy()

# Map status to boolean values
merged_data[f"status-{endpoint}"] = merged_data[f"status-{endpoint}"].map(status_map[endpoint])

# Prepare feature matrix X (gene expression) and survival data
X = merged_data[adata_nmf_cp.var_names].copy()  # Gene expression data
time = merged_data[f"tend_{endpoint}"]
status = merged_data[f"status-{endpoint}"]

# Create structured array for survival data (required by sksurv)
y_structured = Surv.from_arrays(event=status, time=time)

# Step 1: Split data into training (70%) and testing (30%) sets ONCE
X_train, X_test, y_train, y_test = train_test_split(
    X, y_structured, test_size=0.3, random_state=42, stratify=status
)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Step 2: Perform univariate Cox analysis on the training set to select significant genes (p < 0.05) 
significant_genes = []
cox_model = CoxPHFitter()

for gene in tqdm(X_train.columns, desc="Univariate Cox Analysis"):
    data = pd.DataFrame({
        "time": y_train["time"],
        "status": y_train["event"],
        gene: X_train[gene]
    })
    try:
        cox_model.fit(data, duration_col="time", event_col="status")
        if cox_model.summary.loc[gene, "p"] < 0.05:
            significant_genes.append(gene)
    except Exception:
        continue  # Skip genes that cause fitting errors

print(f"Significant genes from univariate Cox: {len(significant_genes)} selected")
print(significant_genes)
significant_genes = ['KLHL13', 'ICA1', 'USH1C', 'USH2A', 'MTMR1', 'SPI1', 'ING3', 'MKNK1', 'TNS1', 'PCDHA6', 'SLCO1A2', 'MECOM', 'MTIF2', 'CEACAM6', 'KCNH4', 'ESR1', 'FMO2', 'GGT1', 'MCAT', 'COCH', 'PLCB4', 'LAMA1', 'TMC5', 'CCN4', 'ADAP1', 'RNF32', 'RARRES2', 'CORO2A', 'CPEB3', 'CCL2', 'PMP22', 'CRACD', 'DDX25', 'CCDC34', 'SLC15A3', 'WASF1', 'PEX5L', 'STAT1', 'TFAP2E', 'PROX1', 'NRP2', 'FLVCR2', 'PLS1', 'DEPDC7', 'ACVR2A', 'GLIPR2', 'IL9R', 'NKX2-2', 'CD68', 'BST2', 'DOCK2', 'CD63', 'KCNH3', 'MRPL47', 'TUBB2B', 'DDX60', 'MYOF', 'PARP9', 'NDST4', 'RASGEF1B', 'LARP1B', 'CXCL9', 'PPP3CA', 'VAMP1', 'NLRC5', 'MISP3', 'PGGHG', 'MOB3C', 'HMCN1', 'HES6', 'NKD2', 'GABRB2', 'VWDE', 'GPR174', 'OGT', 'UTP23', 'NTMT1', 'SLC5A12', 'KAT14', 'TMEM18', 'NRSN1', 'SCOC', 'BMP6', 'DCK', 'KCNJ6', 'DRC1', 'DYNC1I1', 'ACE', 'JAML', 'NLRX1', 'FCRL3', 'HK3', 'CELF5', 'NPHS1', 'TAMALIN', 'ELAVL4', 'SLC29A4', 'KDM1B', 'CYBB', 'INPPL1', 'PKNOX2', 'CACNB2', 'ARL5B', 'SVOP', 'TPP1', 'PRKCB', 'NYAP1', 'PRR15L', 'MYO5B', 'PTGDR', 'SCNN1B', 'CAVIN2', 'CXCL10', 'GP2', 'ITGAM', 'FAM161A', 'KCNG3', 'NETO2', 'SLFN11', 'C1QA', 'RNF213', 'TNK1', 'TCERG1L', 'GRIN1', 'SAMD9L', 'KCNJ10', 'PARP10', 'ZFPM1', 'PSTK', 'PLD5', 'MTURN', 'SGSH', 'SLC9A9', 'EPGN', 'PPP1R27', 'TMEM121B', 'ZNF730', 'SRPK3', 'LPAR5', 'C12ORF56', 'POU3F1', 'AGMO', 'RTL4', 'RELN', 'ZNF33A', 'TECPR2', 'ANXA6', 'C2CD4A', 'SOWAHA', 'KIAA1671-AS1', 'RIPPLY2', 'LST1', 'MICA', 'FAM201A', 'PCDHA9', 'ENSG00000205625', 'ENSG00000206149', 'HCP5', 'HLA-A', 'IGKC', 'IGKV4-1', 'IGLV1-51', 'IGLV2-14', 'IGLC1', 'IGLC2', 'IGHG4', 'IGHG1', 'IGHV3-21', 'IGHV3-23', 'IGHV4-39', 'LINC02347', 'ANKRD36BP1', 'EML6', 'UBE2QL1', 'TNFRSF25', 'MTCH1P1', 'SNORA81', 'ELFN1', 'ENSG00000226281', 'LINC01748', 'ENSG00000228395', 'ALMS1-IT1', 'HLA-DPA1', 'DIRC3', 'SMIM26', 'ENSG00000233179', 'CHRM3-AS2', 'TMEM238', 'ARHGEF7-IT1', 'HRAT17', 'ENSG00000234640', 'SLC26A5-AS1', 'HLA-B', 'ENSG00000236230', 'POU5F1P5', 'CKMT1B', 'KCNMB2-AS1', 'TNFRSF14-AS1', 'IGKV3-20', 'RN7SL403P', 'IGKV3-11', 'SLC5A4-AS1', 'PEG10', 'PCDHGC4', 'LILRA6', 'ODCP', 'NEAT1', 'MTND4P12', 'LINC02242', 'ENSG00000249807', 'PCDHA10', 'LINC01612', 'ENSG00000250992', 'LINC02728', 'FOXD1', 'IGKV1D-39', 'EID3', 'ENSG00000255595', 'ENSG00000256654', 'LINC02955', 'FNBP1P1', 'ENSG00000259453', 'ENSG00000259692', 'ZFHX3-AS1', 'ENSG00000260198', 'ENSG00000260331', 'ENSG00000260947', 'LINC01572', 'LINC01003', 'ENSG00000261628', 'ENSG00000263011', 'RN7SL716P', 'RN7SL850P', 'ENSG00000267054', 'ENSG00000267317', 'ENSG00000267401', 'ENSG00000268218', 'ENSG00000268945', 'ENSG00000269947', 'ENSG00000272077', 'ENSG00000272682', 'ENSG00000272791', 'ENSG00000273312', 'LINC01297', 'ENSG00000277701', 'ENSG00000278041', 'ENSG00000279196', 'FAM223A', 'ENSG00000279286', 'ENSG00000279591', 'ENSG00000280063', 'ENSG00000282865', 'ENSG00000282951', 'ENSG00000285906', 'PARTICL', 'ENSG00000286625', 'ENSG00000286678', 'ENSG00000286806', 'ENSG00000287393', 'ENSG00000288054', 'ENSG00000288792', 'ENSG00000288886', 'ENSG00000289494', 'ENSG00000289528', 'ENSG00000290053', 'ENSG00000290107', 'ENSG00000290117', 'ENSG00000290326', 'ENSG00000290482', 'PMCHL2', 'ENSG00000290858', 'ENSG00000291173']

# Filter training and testing sets to include only significant genes
X_train_significant = X_train[significant_genes]
X_test_significant = X_test[significant_genes]
# Step 3: Find the optimal alpha value **once** using full training set
# Step 1: Define a wide range of alpha values (either log or linear scale)
from sksurv.metrics import concordance_index_censored

def c_index_scorer_func(y_true, y_pred):
    return concordance_index_censored(y_true["event"], y_true["time"], y_pred)[0]  # Extract C-index

# Step 4: Convert the function into a scikit-learn scorer
c_index_scorer = make_scorer(c_index_scorer_func, greater_is_better=True)

alpha_values = np.logspace(-3, 0, 50)  # 50 values from 0.001 to 1 (log scale)
# alpha_values = np.linspace(0.001, 1, 50)  # Alternative: linear scale

# Step 2: Define Stratified K-Fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Step 3: Perform Grid Search with the defined alpha values
gcv = GridSearchCV(
    make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, max_iter=1000)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[alpha] for alpha in alpha_values]},
    cv=cv,
    error_score=0.5,
    n_jobs=-1,
    return_train_score=True,
    scoring=c_index_scorer  # Use a survival-appropriate metric
)

# Step 4: Fit Grid Search to find the best alpha(s)
gcv.fit(X_train_significant, y_train)

# Step 5: Extract the best model and all evaluated models
best_model = gcv.best_estimator_  # Model with best alpha
best_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]  # Best alpha value
all_models = gcv.cv_results_  # Store all models evaluated

# Print best alpha and top 5 models
print(f"Best alpha: {best_alpha:.5f}")
print("\nTop 5 models based on CV performance:")
top_5_idx = np.argsort(gcv.cv_results_["mean_test_score"])[-5:][::-1]
for idx in top_5_idx:
    alpha = gcv.cv_results_["param_coxnetsurvivalanalysis__alphas"][idx][0]
    score = gcv.cv_results_["mean_test_score"][idx]
    print(f"Alpha: {alpha:.5f}, Score: {score:.5f}")

# Store all evaluated alphas and corresponding scores
all_model_results = {
    "alphas": [gcv.cv_results_["param_coxnetsurvivalanalysis__alphas"][i][0] for i in range(len(alpha_values))],
    "scores": gcv.cv_results_["mean_test_score"]
}
# Extract all models from GridSearchCV
all_models = gcv.cv_results_

# Create a dictionary to store model coefficients
model_coefficients = {}

# Iterate through each model evaluated in GridSearchCV
for idx in range(len(all_models["param_coxnetsurvivalanalysis__alphas"])):
    alpha = all_models["param_coxnetsurvivalanalysis__alphas"][idx][0]  # Get alpha value
    score = all_models["mean_test_score"][idx]  # Get C-index score
    
    # Extract the fitted model from GridSearchCV pipeline
    fitted_pipeline = gcv.best_estimator_  # This gives the best pipeline
    fitted_model = fitted_pipeline.named_steps["coxnetsurvivalanalysis"]  # Extract the fitted Cox model
    
    # Extract coefficients
    coefs = fitted_model.coef_.flatten()
    
    # Select only genes with non-zero coefficients
    selected_genes = X_train_significant.columns[np.abs(coefs) > 0]
    selected_coefs = coefs[np.abs(coefs) > 0]
    
    # Store in dictionary
    model_coefficients[alpha] = {
        "C-index": score,
        "genes": list(selected_genes),
        "coefficients": list(selected_coefs)
    }

# Convert results into a DataFrame
coef_df_list = []
for alpha, data in model_coefficients.items():
    for gene, coef in zip(data["genes"], data["coefficients"]):
        coef_df_list.append({"Alpha": alpha, "Gene": gene, "Coefficient": coef, "C-index": data["C-index"]})

coef_df = pd.DataFrame(coef_df_list)

# Print results
print("\nTop models and their selected genes:")
for alpha, data in model_coefficients.items():
    print(f"\nModel with Alpha: {alpha:.5f}, C-index: {data['C-index']:.5f}")
    print("Genes and Coefficients:")
    for gene, coef in zip(data["genes"], data["coefficients"]):
        print(f"  {gene}: {coef:.5f}")

# Optionally, save results to CSV
coef_df.to_csv("coxnet_selected_genes.csv", index=False)

# TODO: See all models and get the coefs from each model to see if this is it...

# # Fit initial Coxnet model to get a range of alpha values
# initial_model = CoxnetSurvivalAnalysis(
#     l1_ratio=0.95,  # Higher L1 ratio for stronger sparsity
#     alpha_min_ratio=0.05,  # Reduce variability
#     max_iter=1000
# )
# initial_model.fit(X_train_significant, y_train)

# # Select the most stable alpha (median value)
# best_alpha = np.median(initial_model.alphas_)
# print(f"Selected fixed alpha: {best_alpha:.5f}")

# # Step 4: Perform gene selection using the fixed alpha over 1000 iterations
# n_iterations = 1000
# gene_set_frequencies = defaultdict(int)  # Track frequency of each gene set
# all_selected_sets = []

# for i in tqdm(range(n_iterations), desc="Elastic Net Iterations"):

#     # Fit Coxnet model with fixed alpha
#     cox_model = CoxnetSurvivalAnalysis(
#         l1_ratio=0.95,
#         alphas=[best_alpha],  # Use the pre-selected alpha
#         max_iter=1000
#     )
#     cox_model.fit(X_train, y_train)

#     # Extract selected genes (non-zero coefficients)
#     selected_genes = frozenset(X_train_significant.columns[np.abs(cox_model.coef_.flatten()) > 0])
#     all_selected_sets.append(selected_genes)

# # Step 5: Count frequencies of unique gene sets
# unique_sets = set(all_selected_sets)
# for gene_set in unique_sets:
#     freq = all_selected_sets.count(gene_set)
#     gene_set_frequencies[gene_set] = freq

# # Report top gene sets
# print(f"\nFound {len(unique_sets)} unique gene sets across {n_iterations} iterations")
# print("\nMost stable gene sets:")
# for gene_set, freq in sorted(gene_set_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]:
#     freq_pct = (freq / n_iterations) * 100
#     print(f"\nModel appearing {freq} times ({freq_pct:.1f}%):")
#     print(f"Number of genes: {len(gene_set)}")
#     print("Genes:", ", ".join(sorted(gene_set)))

# # Step 5: Select the most frequent gene set and fit the final model
# most_frequent_set = max(gene_set_frequencies, key=gene_set_frequencies.get)
# print(f"\nSelected gene set with highest frequency ({gene_set_frequencies[most_frequent_set]} times):")
# print(f"Genes: {', '.join(sorted(most_frequent_set))}")

# # Filter data to the selected gene set
# X_train_selected = X_train_significant[list(most_frequent_set)]
# X_test_selected = X_test_significant[list(most_frequent_set)]
# X_all_selected = X[list(most_frequent_set)]  # Full dataset for total evaluation

# # Fit the final Coxnet model on the training set with the selected genes
# final_model = CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True)
# final_model.fit(X_train_selected, y_train)

# # Step 6: Calculate risk scores for training, testing, and total datasets
# risk_scores_train = final_model.predict(X_train_selected).flatten()
# risk_scores_test = final_model.predict(X_test_selected).flatten()
# risk_scores_all = final_model.predict(X_all_selected).flatten()

# # Step 7: Evaluate with time-dependent ROC and AUC at 1, 3, and 5 years
# times = [1.0, 3.0, 5.0]  # Years for evaluation (adjust based on your time scale)

# # Ensure all y arrays are structured
# assert y_train.dtype.names == ('event', 'time'), "y_train is not a structured array"
# assert y_test.dtype.names == ('event', 'time'), "y_test is not a structured array"
# assert y_structured.dtype.names == ('event', 'time'), "y_structured is not a structured array"

# # Training set AUC
# auc_train, mean_auc_train = cumulative_dynamic_auc(y_train, y_train, risk_scores_train, times)
# # Testing set AUC (using y_train as reference for survival function)
# auc_test, mean_auc_test = cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)
# # Total dataset AUC
# y_all = y_structured  # Full survival data
# auc_all, mean_auc_all = cumulative_dynamic_auc(y_train, y_all, risk_scores_all, times)

# # Print AUC results
# print("\nAUC Results:")
# print("Training Set:")
# for t, auc in zip(times, auc_train):
#     print(f"  {t}-year AUC: {auc:.3f}")
# print("Testing Set:")
# for t, auc in zip(times, auc_test):
#     print(f"  {t}-year AUC: {auc:.3f}")
# print("Total Dataset:")
# for t, auc in zip(times, auc_all):
#     print(f"  {t}-year AUC: {auc:.3f}")

# # Step 8: Calculate C-index for training, testing, and total datasets
# c_index_train = concordance_index_censored(y_train["event"], y_train["time"], risk_scores_train)[0]
# c_index_test = concordance_index_censored(y_test["event"], y_test["time"], risk_scores_test)[0]
# c_index_all = concordance_index_censored(y_all["event"], y_all["time"], risk_scores_all)[0]

# print("\nC-index Results:")
# print(f"Training Set: {c_index_train:.3f}")
# print(f"Testing Set: {c_index_test:.3f}")
# print(f"Total Dataset: {c_index_all:.3f}")
# # Optional: Plot ROC curves (for visualization)
# for i, t in enumerate(times):
#     plt.figure(figsize=(8, 6))
#     # Simplified ROC plotting (requires additional computation for TPR/FPR, not directly provided by sksurv)
#     plt.title(f"ROC Curve at {t} Years (Training Set AUC: {auc_train[i]:.3f})")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.grid(True)
#     plt.show()  # Note: Full ROC curve plotting requires manual computation of TPR/FPR

# Optional: Evaluate the selected gene set on the test set (e.g., ROC, C-index)
# This can be implemented using sksurv.metrics as a next step
###################### Test ElasticNet model #################

# results = []
# # Dictionary to store best models and their info
# best_models = {}

# expr = pd.DataFrame(
#     adata_nmf_cp.layers["normalized_counts"], index=adata_nmf_cp.obs.index, columns=adata_nmf_cp.var.index
# ).reset_index()

# endpoint = "os"
# X = expr
# merged_data = adata_nmf_cp.obs.copy()
# # Store ID_Sample separately and set it as index for X_model

# status_map = {
#     "os": {"Dead": True, "Alive": False},
#     "ttp": {"Progressed": True, "Did not progress": False},
# }
# status_bool = merged_data[f"status-{endpoint}"].map(status_map[endpoint])
# time = merged_data[f"tend_{endpoint}"]

# y = merged_data[f"status-{endpoint}"]

# y_structured = np.zeros(
#     len(time), dtype=[("status", bool), ("time", float)]
# )
# y_structured["status"] = status_bool
# y_structured["time"] = time

# # Split the data
# X_train, X_test, y_train_idx, y_test_idx = train_test_split(
#     X,
#     np.arange(len(y)), #y_structured
#     test_size=0.2,
#     random_state=42,
#     stratify=status_bool,
# )

# # Create structured arrays for train and test sets
# y_train = y[y_train_idx]
# y_test = y[y_test_idx]

# from feature_engine.selection import MRMR
# mrmr_sel = MRMR(method="MIQ", regression=False, random_state=3)
# X_t = mrmr_sel.fit_transform(X_train, y_train)

# %% -------------------------------------------------------
# 1) Map status, prepare X and y
# -------------------------------------------------------

from sksurv.util import Surv

# Remove gene names from adata.obs (keep them in layers)
adata_nmf_cp.obs = adata_nmf_cp.obs.drop(columns=adata_nmf_cp.var_names)

# Map survival status
status_map = {
    "os": {"Dead": True, "Alive": False},
    "ttp": {"Progressed": True, "Did not progress": False},
}
endpoint = "os"
adata_nmf_cp.obs[f"status-{endpoint}"] = adata_nmf_cp.obs[f"status-{endpoint}"].map(status_map[endpoint])

# Extract gene expression data (log-normalized)
X = pd.DataFrame(adata_nmf_cp.layers["lognormalized_counts"], index=adata_nmf_cp.obs_names, columns=adata_nmf_cp.var_names)

# Define thresholds
presence_threshold = 0.75  # Keep genes present in at least 75% of samples

# List of key SCLC biomarkers to retain (add more if needed)
important_biomarkers = {"ASCL1", "NEUROD1", "POU2F3", "YAP1"}

# Calculate presence of each gene
percent_expressed = (X > 0).mean(axis=0)  # Fraction of nonzero samples per gene

# Select genes that pass the threshold OR are known biomarkers
expressed_genes = percent_expressed[(percent_expressed >= presence_threshold) | percent_expressed.index.isin(important_biomarkers)].index.tolist()
removed_genes = percent_expressed[~percent_expressed.index.isin(expressed_genes)].index.tolist()

# Filter the data
X = X[expressed_genes].copy()

# Summary statistics
print(f"Original number of genes: {X.shape[1]}")
print(f"Number of genes expressed in >{presence_threshold*100}% of samples: {len(expressed_genes)}")
print(f"Number of genes removed: {len(removed_genes)}")
print(f"Percentage of genes retained: {len(expressed_genes)/X.shape[1]*100:.2f}%")

# Create a DataFrame of removed genes and their expression percentages
removed_genes_df = pd.DataFrame({
    'Gene': removed_genes,
    'Percent_Expressed': percent_expressed[removed_genes].values
}).sort_values('Percent_Expressed', ascending=False)

# Display top removed genes (closest to threshold)
print("\nTop removed genes (closest to threshold):")
print(removed_genes_df.head(10))

# Histogram of expression percentages
plt.figure(figsize=(8, 5))
plt.hist(percent_expressed * 100, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(presence_threshold * 100, color='red', linestyle='dashed', label=f"Threshold ({presence_threshold*100}%)")
plt.xlabel("Percentage of Samples with Gene Expressed")
plt.ylabel("Number of Genes")
plt.title("Distribution of Gene Expression Across Samples")
plt.legend()
plt.show()

# Update AnnData object
adata_nmf_cp = adata_nmf_cp[:, expressed_genes].copy()

# Prepare structured survival data
time = adata_nmf_cp.obs[f"tend_{endpoint}"]
status = adata_nmf_cp.obs[f"status-{endpoint}"]
y_structured = Surv.from_arrays(event=status, time=time)


# %% -------------------------------------------------------
# 2) Split once into training (70%) and testing (30%)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_structured, test_size=0.20, random_state=42, stratify=status
)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# %% -------------------------------------------------------
# 3) DEG on the training set to get up and down genes
# -------------------------------------------------------
adata_subset_train = adata_nmf_cp[X_train.index].copy()

deg = DEGAnalysis(
    adata = adata_subset_train,
    design_factor=f"status-{endpoint}",
    layer="raw_counts",
    output_dir=f"./deg_analysis_biomarker_{endpoint}",
)
deg.create_dds()
deg.run_comparisons()
deg.save_results()
results_deg = deg.get_results()

top_genes_df = results_deg["Alive_vs_Dead"].results_df[
results_deg["Alive_vs_Dead"].results_df["log2FoldChange"] > 1].sort_values("log2FoldChange", ascending=False
).index.tolist()

bottom_genes_df = results_deg["Alive_vs_Dead"].results_df[
results_deg["Alive_vs_Dead"].results_df["log2FoldChange"] < - 1].sort_values("log2FoldChange", ascending=False).index.tolist()

# %% -------------------------------------------------------
# 4) Univariate Cox on training set to find p<0.05 genes
# -------------------------------------------------------
significant_genes = []
cox_model = CoxPHFitter()

cox_dict = {}
for gene in X_train.columns:
    data = pd.DataFrame({
        "time": y_train["time"],
        "status": y_train["event"],
        gene: X_train[gene]
    })
    cox_model.fit(data, duration_col="time", event_col="status")
    pval = cox_model.summary.loc[gene, "p"]

    if gene in ["NEUROD1","POU2F3","YAP1","ASCL1"]:
        print(cox_model.summary)

    cox_dict[gene] = pval

cox_pvals_df = pd.DataFrame.from_dict(cox_dict, orient="index", columns = ["pval"])
from statsmodels.stats.multitest import \
     multipletests

_, corrected_pvals, _, _ = multipletests(cox_pvals_df["pval"].tolist(), method="holm")

# Store the adjusted p-values in the DataFrame
cox_pvals_df["adjusted_pval"] = corrected_pvals

for g in cox_pvals_df.index:
    if cox_pvals_df.loc[g,"pval"] < 0.05 and "ENSG" not in g:  # more stringent
        significant_genes.append(g)

print(f"Significant genes from univariate Cox (p<0.05): {len(significant_genes)}")
print(significant_genes)

X_train_significant = X_train[significant_genes].copy()
X_test_significant  = X_test[significant_genes].copy()

# %% -------------------------------------------------------
# 5.a) Train ElasticNet Cox model - Find best alpha with 10-fold CV on X_train_significant
# -------------------------------------------------------
def c_index_scorer_func(y_true, y_pred):
    return concordance_index_censored(y_true["event"], y_true["time"], y_pred)[0]

c_index_scorer = make_scorer(c_index_scorer_func, greater_is_better=True)

l1_ratio = 1
alpha_min_ratio = 0.01
max_iter = 10000

# Train a first model to find the alphas range
coxnet_pipe = make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio="auto",n_alphas=10000))
coxnet_pipe.fit(X=X_train_significant, y=y_train)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

# Train CV elastic net model to find the actual best alpha based on the initial set originated

gcv = GridSearchCV(
    CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,       
        alpha_min_ratio=alpha_min_ratio,
    ),
    param_grid={"alphas": [[v] for v in estimated_alphas]},
    scoring=c_index_scorer,
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
    error_score=0.5
)
gcv.fit(X_train_significant, y_train)
cv_results = pd.DataFrame(gcv.cv_results_)

best_model = gcv.best_estimator_  # The pipeline, so best_model.named_steps['coxnetsurvivalanalysis'] if needed
best_model_coef = best_model.coef_.ravel()
best_non_null_coefs = np.where(best_model_coef != 0)
best_selected_features = list(X_train_significant.columns[best_model_coef !=0])
coef_dict = {}
for coef, gene in zip(best_model_coef[best_non_null_coefs], best_selected_features):
    print(f"{coef:.5f} * {gene} +")
    coef_dict[gene] = coef

best_alpha = gcv.best_params_["alphas"][0]

print("Best alpha:", best_alpha)

# Print the best alpha selected
alphas = cv_results.param_alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)
plt.show()

# Evaluate the best model on the test set to check for generalization
risk_scores = best_model.predict(X_test_significant)

# Calculate the C-index
c_index = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)
print("C-index on test set:", c_index[0])

# Calculate the risk scores as coef * lognormalized counts for the whole dataset    

coef_df = pd.DataFrame.from_dict(coef_dict, orient='index', columns=['coefficient'])
coef_df['coefficient'] = coef_df['coefficient'].round(5)
# Find the optimal cutpoint by:

# %%
scores_risk_df = calculate_risk_scores(adata_nmf_cp, coef_df, layer="lognormalized_counts", cutpoint_method="youden", upper_bound=91, lower_bound=15)

# %% -------------------------------------------------------
# 5.b) Train RFS
# -------------------------------------------------------

from sklearn import set_config
from sksurv.ensemble import RandomSurvivalForest
rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42
)
rsf.fit(X_train[significant_genes], y_train)

rsf.score(X_test[significant_genes], y_test)

from sklearn.inspection import permutation_importance

result = permutation_importance(rsf, X_test[significant_genes], y_test, n_repeats=15, random_state=42)

pd.DataFrame(
    {
        k: result[k]
        for k in (
            "importances_mean",
            "importances_std",
        )
    },
    index=X_test[significant_genes].columns,
).sort_values(by="importances_mean", ascending=False)
# %%
