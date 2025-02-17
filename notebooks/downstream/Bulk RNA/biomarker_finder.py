
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
import liana as ln
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
import PyComplexHeatmap as pch
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
from tqdm import tqdm
import umap

# Local imports
from cnmf import cNMF
from catboost import CatBoostClassifier

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

# Functions

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

def evaluate_gene_sets(
    merged_data, adata, top_genes, layer="lognormalized_counts", endpoint="os"
):
    """
    Evaluate different numbers of top genes for each comparison using Cox models.
    Includes risk score calculation, KM curves, and time-dependent ROC analysis.

    Parameters:
    -----------
    merged_data : pandas.DataFrame
        DataFrame containing the survival data and gene expression data
    top_genes : dict
        Dictionary with comparisons as keys and lists of gene names as values

    Returns:
    --------
    dict
        Dictionary containing results_df, best_models, coefficient_dfs, and risk scores
    """

    # Store results
    results = []
    # Dictionary to store best models and their info
    best_models = {}

    expr = pd.DataFrame(
        adata.layers[layer], index=adata.obs.index, columns=adata.var.index
    ).reset_index()
    expr.set_index("ID_Sample", inplace=True)

    # Loop through each comparison in top_genes
    for comparison in top_genes.keys():
        print(f"\nProcessing comparison: {comparison}")
        comparison_best_score = -np.inf
        comparison_best_info = None

        # Loop through different numbers of genes
        for n_genes in range(5, len(top_genes[comparison]), 5):
            print(f"Testing with top {n_genes} genes")

            # Select features
            selected_genes = top_genes[comparison][:n_genes]
            try:
                X = expr[selected_genes]
                print(X)
            except Exception as e:
                print(f"{e}")
                continue
            # Store ID_Sample separately and set it as index for X_model
            # sample_ids = merged_data.index
            # X_model.index = sample_ids

            status_map = {
                "os": {"Dead": True, "Alive": False},
                "ttp": {"Progressed": True, "Did not progress": False},
            }
            status_bool = merged_data[f"status-{endpoint}"].map(status_map[endpoint])
            time = merged_data[f"tend_{endpoint}"]

            y_structured = np.zeros(
                len(time), dtype=[("status", bool), ("time", float)]
            )
            y_structured["status"] = status_bool
            y_structured["time"] = time

            # Split the data
            X_train, X_test, y_train_idx, y_test_idx = train_test_split(
                X,
                np.arange(len(y_structured)),
                test_size=0.2,
                random_state=42,
                stratify=status_bool,
            )

            # Create structured arrays for train and test sets
            y_train = y_structured[y_train_idx]
            y_test = y_structured[y_test_idx]

            # Fit initial model to get alphas
            coxnet_pipe = make_pipeline(
                CoxnetSurvivalAnalysis(
                    l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=1000
                )
            )
            coxnet_pipe.fit(X_train, y_train)
            estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

            # Perform cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            gcv = GridSearchCV(
                make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9)),
                param_grid={
                    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]
                },
                cv=cv,
                error_score=0.5,
                n_jobs=-1,
                verbose=0,
            ).fit(X_train, y_train)
            alphas = [
                alpha[0]
                for alpha in gcv.cv_results_["param_coxnetsurvivalanalysis__alphas"]
            ]

            # Accessing the mean and std test scores from grid search results
            mean = gcv.cv_results_["mean_test_score"]
            std = gcv.cv_results_["std_test_score"]

            # Plotting the performance
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean, label="Mean Concordance Index")
            ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
            ax.set_xscale("log")  # Log scale for the x-axis (alphas)
            ax.set_ylabel("Concordance Index")  # Y-axis label
            ax.set_xlabel("Alpha")  # X-axis label (alpha values)

            # Plotting the best alpha from grid search
            best_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
            ax.axvline(best_alpha, c="C1", label="Best Alpha")  # Line at best alpha

            # Adding a horizontal line at concordance index 0.5
            ax.axhline(0.5, color="grey", linestyle="--", label="Random Concordance")

            # Add grid and legend
            ax.grid(True)
            ax.legend()

            # Display the plot
            plt.show()
            # Get best model
            best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]

            # Make predictions
            train_predictions = best_model.predict(X_train)
            test_predictions = best_model.predict(X_test)

            # Calculate concordance indices
            train_cindex = concordance_index_censored(
                y_train["status"], y_train["time"], train_predictions
            )[0]

            test_cindex = concordance_index_censored(
                y_test["status"], y_test["time"], test_predictions
            )[0]

            # Use predictions directly as risk scores
            train_risk_scores = train_predictions
            test_risk_scores = test_predictions

            # Find optimal cutoff using median in training set
            risk_cutoff = np.median(train_risk_scores)

            # Assign risk groups
            train_risk_groups = (train_risk_scores > risk_cutoff).astype(int)
            test_risk_groups = (test_risk_scores > risk_cutoff).astype(int)

            # Store results
            results.append(
                {
                    "comparison": comparison,
                    "n_genes": n_genes,
                    "train_cindex": train_cindex,
                    "test_cindex": test_cindex,
                    "best_alpha": gcv.best_params_["coxnetsurvivalanalysis__alphas"][0],
                    "cv_score": gcv.best_score_,
                    "risk_cutoff": risk_cutoff,
                    "train_high_risk_n": sum(train_risk_groups),
                    "test_high_risk_n": sum(test_risk_groups),
                }
            )

            # Update best model info if this is the best test score so far for this comparison
            if test_cindex > comparison_best_score:
                comparison_best_score = test_cindex
                comparison_best_info = {
                    "model": best_model,
                    "n_genes": n_genes,
                    "test_cindex": test_cindex,
                    "train_cindex": train_cindex,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "selected_genes": selected_genes,
                }

            print(f"Train C-index: {train_cindex:.3f}")
            print(f"Test C-index: {test_cindex:.3f}")

        # Store best model info for this comparison
        best_models[comparison] = comparison_best_info

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create coefficient DataFrames for each best model
    coef_dfs = {}
    for comparison, model_info in best_models.items():
        best_coefs = pd.DataFrame(
            model_info["model"].coef_,
            index=model_info["X_train"].columns,
            columns=["coefficient"],
        )
        coef_dfs[comparison] = best_coefs

    # Plot results for best models
    for comparison, model_info in best_models.items():
        # Get the best model data
        best_model = model_info["model"]
        X_train = model_info["X_train"]
        X_test = model_info["X_test"]
        y_train = model_info["y_train"]
        y_test = model_info["y_test"]
        n_genes = model_info["n_genes"]

        # Recalculate predictions and use them directly as risk scores
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Use predictions directly as risk scores
        train_risk_scores = train_predictions
        test_risk_scores = test_predictions

        # Find optimal cutoff using median in training set
        risk_cutoff = np.median(train_risk_scores)

        # Assign risk groups
        train_risk_groups = (train_risk_scores > risk_cutoff).astype(int)
        test_risk_groups = (test_risk_scores > risk_cutoff).astype(int)

        # Plot KM curves for best model
        plt.figure(figsize=(10, 6))

        for group in [0, 1]:
            mask = train_risk_groups == group
            if np.any(mask):
                time, survival_prob = kaplan_meier_estimator(
                    y_train["status"][mask], y_train["time"][mask]
                )
                plt.step(
                    time,
                    survival_prob,
                    label=f"{'High' if group else 'Low'} Risk (train)",
                )

        for group in [0, 1]:
            mask = test_risk_groups == group
            if np.any(mask):
                time, survival_prob = kaplan_meier_estimator(
                    y_test["status"][mask], y_test["time"][mask]
                )
                plt.step(
                    time,
                    survival_prob,
                    "--",
                    label=f"{'High' if group else 'Low'} Risk (test)",
                )

        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title(
            f"Kaplan-Meier Curves by Risk Group\nBest Model for {comparison} ({n_genes} genes) using the median as the risk cutoff"
        )
        plt.grid(True)
        plt.legend()
        plt.show()

        # Calculate time-dependent ROC curve
        times = np.array([365, 730, 1095])  # 1, 2, and 3 years

        fig, ax = plt.subplots(figsize=(10, 6))

        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_predictions, times)

        for i, t in enumerate(times):
            if i == 0:
                ax.plot(times, auc, marker="o", color="crimson", label="AUC")
            else:
                ax.plot(times, auc, marker="o", color="crimson")
            ax.text(
                times[i],
                auc[i] + 0.02,
                f"{t/365:.0f} year AUC={auc[i]:.3f}",
                ha="center",
            )

        ax.plot([times[0], times[-1]], [0.5, 0.5], color="grey", linestyle="--")
        ax.set_xlabel("Days")
        ax.set_ylabel("Time-dependent AUC")
        ax.set_title(
            f"Time-dependent ROC\nBest Model for {comparison} ({n_genes} genes)"
        )
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    # Print best results and plot coefficients for each comparison
    fig, axes = plt.subplots(len(best_models), 1, figsize=(10, 6 * len(best_models)))
    if len(best_models) == 1:
        axes = [axes]

    for idx, (comparison, model_info) in enumerate(best_models.items()):
        print(f"\n=== Best Model for {comparison} ===")
        print(f"Number of genes: {model_info['n_genes']}")
        print(f"Test C-index: {model_info['test_cindex']:.3f}")
        print(f"Train C-index: {model_info['train_cindex']:.3f}")

        # Get non-zero coefficients
        best_coefs = coef_dfs[comparison]
        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        print(f"Number of non-zero coefficients: {len(non_zero_coefs)}")

        # Plot coefficients
        non_zero_coefs.loc[coef_order].plot.barh(ax=axes[idx], legend=False)
        axes[idx].set_xlabel("coefficient")
        axes[idx].set_title(f"Coefficients for {comparison}")
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()

    # Create performance plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for comparison in top_genes.keys():
        comparison_results = results_df[results_df["comparison"] == comparison]
        ax.plot(
            comparison_results["n_genes"],
            comparison_results["test_cindex"],
            marker="o",
            label=f"{comparison} (test)",
        )
        ax.plot(
            comparison_results["n_genes"],
            comparison_results["train_cindex"],
            marker="o",
            linestyle="--",
            label=f"{comparison} (train)",
        )

    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Concordance Index")
    ax.set_title("Performance vs Number of Genes")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "results_df": results_df,
        "best_models": best_models,
        "coefficient_dfs": coef_dfs,
    }

def calculate_risk_scores(
    adata, coef_df, layer="raw_counts", endpoint="os", lower_bound=30, upper_bound=71
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
                {"Alive": 0, "Dead": 1}
                if endpoint == "os"
                else {"Did not progress": 0, "Progressed": 1}
            ),
        },
        index=adata.obs.index,
    )

    print(f"Risk scores data:\n {risk_scores_df}")

    # Add expression values for selected genes
    risk_scores_df = pd.concat([risk_scores_df, expr], axis=1)

    # Find optimal cutpoint using log-rank test
    cutpoints = np.percentile(risk_scores, np.arange(lower_bound, upper_bound, 1))
    max_statistic = 0
    optimal_cutpoint = None
    optimal_pvalue = None

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
            optimal_pvalue = pvalue

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
    stats_text = f"HR = {hr:.2f}\n" f"Log-rank P = {optimal_pvalue:.2e}"
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
    print(f"Log-rank test p-value: {optimal_pvalue:.2e}")
    print(f"Hazard Ratio: {hr:.2f}")

    return risk_scores_df


def plot_risk_profile(
    risk_scores_df, adata, gene_coefficients, optimal_cutpoint, layer="raw_counts"
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
        "Risk Score Profile using the maximally selected rank statistic using log rank test to find the optimal cutoff point",
        y=1.02,
    )

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Load Data
km_data_ttp = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_ttp.csv"
)
km_data_os = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/data_os.csv"
)
log_norm_counts = pd.read_csv(
    r"/home/rafaed/work/RO_src/Projects/THORA/StatisticalAnalysis/scripts/log_normalized_counts_df.csv"
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



# Run PyDESeq2
deg_os = DEGAnalysis(
    adata_nmf_cp,
    design_factor="status-os",
    layer="raw_counts",
    output_dir="./deg_analysis_os",
)
deg_os.create_dds()
deg_os.run_comparisons()
deg_os.save_results()
deg_os.create_volcano_grid()
results_deg_os = deg_os.get_results()
os_top_genes = get_top_genes(
    source_paths={
        "Alive vs Dead": "/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/deg_analysis_os/deg_results_Alive_vs_Dead.csv"
    },
    n_genes=100,
    mode="bottom",
    sort_by="stat",
)

deg_ttp = DEGAnalysis(
    adata_nmf_cp,
    design_factor="status-ttp",
    layer="raw_counts",
    output_dir="./deg_analysis_ttp",
)
deg_ttp.create_dds()
deg_ttp.run_comparisons()
deg_ttp.save_results()
deg_ttp.create_volcano_grid()
results_deg_ttp = deg_ttp.get_results()
ttp_top_genes = get_top_genes(
    source_paths={
        "Did not progress vs Progressed": "/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/deg_analysis_ttp/deg_results_Did not progress_vs_Progressed.csv"
    },
    n_genes=100,
    mode="bottom",
    sort_by="stat",
)


# Run ElasticNet model

results = []
# Dictionary to store best models and their info
best_models = {}

expr = pd.DataFrame(
    adata_nmf_cp.layers["normalized_counts"], index=adata_nmf_cp.obs.index, columns=adata_nmf_cp.var.index
).reset_index()
expr.set_index("ID_Sample", inplace=True)

# Select genes from the DEG that are FDR < 0.05
ttp_sig_genes = results_deg_ttp["Did not progress vs Progressed"].results_df[results_deg_ttp["Did not progress vs Progressed"].results_df["padj"] <= 0.05]
os_sig_genes = results_deg_os["Alive vs Dead"].results_df

# Remove genes with unconfirmed expression (TPM=0) in more than 70% of the samples
zero_percentage = (expr == 0).mean(axis=1)
expr_filtered = expr.loc[zero_percentage <= 0.7]

# Calculate the correlation of all gene pairs, remove the genes that have the highest absolute sum of correlation coefficients with other genes, so that all gene pairs have a correlation of less than 0.8

corr_matrix = expr_filtered.corr()
# (1) correlation coefficients for all gene pairs were calculated; (2) the gene pair with the highest correlation coefficient was identified; (3) among these pairs, the gene with the largest absolute sum of correlation coefficients with other genes was removed; and (4) steps 2 and 3 were repeated until the correlation coefficient of all gene pairs was less than 0.8.

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop features 
expr.drop(to_drop, axis=1, inplace=True)
## Testing MRMR
endpoint = "os"
X = expr
merged_data = adata_nmf_cp.obs.copy()
# Store ID_Sample separately and set it as index for X_model

status_map = {
    "os": {"Dead": True, "Alive": False},
    "ttp": {"Progressed": True, "Did not progress": False},
}
status_bool = merged_data[f"status-{endpoint}"].map(status_map[endpoint])
time = merged_data[f"tend_{endpoint}"]

y = merged_data[f"status-{endpoint}"]

y_structured = np.zeros(
    len(time), dtype=[("status", bool), ("time", float)]
)
y_structured["status"] = status_bool
y_structured["time"] = time

# Split the data
X_train, X_test, y_train_idx, y_test_idx = train_test_split(
    X,
    np.arange(len(y)), #y_structured
    test_size=0.2,
    random_state=42,
    stratify=status_bool,
)

# Create structured arrays for train and test sets
y_train = y[y_train_idx]
y_test = y[y_test_idx]

from feature_engine.selection import MRMR
mrmr_sel = MRMR(method="MIQ", regression=False, random_state=3)
X_t = mrmr_sel.fit_transform(X_train, y_train)


#######################################
print("Correlation reduction complete.")
# Train elastic net
results_ttp = evaluate_gene_sets(
    merged_data, adata_nmf_cp, ttp_top_genes, endpoint="ttp"
)
results_df_ttp = results_ttp["results_df"]
best_models_ttp = results_ttp["best_models"]
coefficient_dfs_ttp = results_ttp["coefficient_dfs"]

results_os = evaluate_gene_sets(merged_data, adata_nmf_cp, os_top_genes, endpoint="os")
results_df_os = results_os["results_df"]
best_models_os = results_os["best_models"]
coefficient_dfs_os = results_os["coefficient_dfs"]

# Compute Risk Scores

# Retrieve the risk scores
risk_scores_df_ttp = calculate_risk_scores(
    adata=adata_nmf_cp,
    coef_df=coefficient_dfs_ttp["Did not progress vs Progressed"],
    layer="lognormalized_counts",
    endpoint="ttp",
)

risk_scores_df_os = calculate_risk_scores(
    adata=adata_nmf_cp,
    coef_df=coefficient_dfs_os["Alive vs Dead"],
    layer="lognormalized_counts",
    endpoint="os",
)


#Test
data = pd.read_excel(
    "/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/markers/chemo_immuno_gene_set.xlsx"
)
immune_genes = data["Symbol"].tolist()
adata_nmf = sc.read_h5ad(
    r"/mnt/work/RO_src/Projects/THORA/DataProcessing/data/processed/adata_nmf_cp_81_samp.h5ad"
)
adata_nmf_cp = adata_nmf.copy()
adata_nmf_cp.obs.set_index("ID_Sample", inplace=True)

from collections import defaultdict

# First create gene:category mapping
gene_to_category = {
    g: l for g, l in zip(data["Symbol"], data["Column1"]) if g in adata_nmf_cp.var_names
}

# Convert to category:genes mapping
categories = defaultdict(list)
for gene, category in gene_to_category.items():
    categories[category].append(gene)

# Convert defaultdict to regular dict
gene_categories = dict(categories)

# Get list of genes to plot
genes_to_plot = list(gene_to_category.keys())

# Create the plot
fig = plot_gene_expression_heatmap(
    adata=adata_nmf_cp,
    genes_to_plot=genes_to_plot,
    gene_categories=gene_categories,
    group_column="SCLC-Subtype-de-novo",
)
plt.show()

# Split data into high and low risk groups + create KM curves

df = pd.read_excel(
    r"/home/rafaed/work/RO_src/STAnalysis/notebooks/downstream/Bulk RNA/markers/grn_genes.xlsx"
)
markers = {
    "grn": {
        "up": [
            str(gene).upper().strip()
            for gene, r in zip(df["GeneName"], df["R"])
            if r >= 0
        ],
        "down": [
            str(gene).upper().strip()
            for gene, r in zip(df["GeneName"], df["R"])
            if r < 0
        ],
    }
}

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Filter genes to only those present in the data
available_genes = adata_nmf_cp.var_names
up_genes = [gene for gene in markers["grn"]["up"] if gene in available_genes]
down_genes = [gene for gene in markers["grn"]["down"] if gene in available_genes]

# Create separate matrices for up and down genes
expr_matrix_up = pd.DataFrame(
    adata_nmf_cp.layers["raw_counts"][:, adata_nmf_cp.var_names.isin(up_genes)],
    index=adata_nmf_cp.obs_names,
    columns=adata_nmf_cp.var_names[adata_nmf_cp.var_names.isin(up_genes)],
)

expr_matrix_down = pd.DataFrame(
    adata_nmf_cp.layers["raw_counts"][:, adata_nmf_cp.var_names.isin(down_genes)],
    index=adata_nmf_cp.obs_names,
    columns=adata_nmf_cp.var_names[adata_nmf_cp.var_names.isin(down_genes)],
)

# Concatenate them in order (UP first, then DOWN)
expr_matrix = pd.concat([expr_matrix_up, expr_matrix_down], axis=1)

# Calculate z-scores globally
expr_matrix_zscore = (expr_matrix - expr_matrix.mean()) / expr_matrix.std()

# Create a list of colors for the genes (red for up, blue for down)
gene_colors = ["red"] * len(expr_matrix_up.columns) + ["blue"] * len(
    expr_matrix_down.columns
)

# Create row colors
row_colors = pd.DataFrame({"Regulation": gene_colors}, index=expr_matrix_zscore.columns)

# Print some info about the genes
print(f"Number of UP genes found: {len(expr_matrix_up.columns)}")
print(f"Number of DOWN genes found: {len(expr_matrix_down.columns)}")

# Create clustermap with row colors
g = sns.clustermap(
    expr_matrix_zscore.T,  # Transpose to get genes on y-axis
    cmap="RdBu_r",
    center=0,
    row_cluster=False,  # don't cluster genes to maintain order
    col_cluster=True,  # cluster patients
    dendrogram_ratio=(0.1, 0.2),
    vmin=-3,  # set min value for color scale
    vmax=3,  # set max value for color scale
    xticklabels=True,
    yticklabels=True,
    figsize=(15, 45),  # taller figure
    method="complete",
    row_colors=row_colors,
)  # Add row colors to indicate up/down

# Customize plot
g.ax_heatmap.set_xlabel("Patients")
g.ax_heatmap.set_ylabel("Genes")
plt.suptitle(
    "Gene Expression Z-scores (Patients Clustered)\nUP regulated genes (top, red bar), DOWN regulated genes (bottom, blue bar)",
    y=1.02,
)

# Rotate x-axis labels for better readability
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

plt.show()


def calculate_gene_set_scores(
    adata, up_genes, down_genes, name, layer="raw_counts", combined_score=True
):
    """
    Calculate scores using up and down regulated genes for any gene set

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    up_genes : list
        List of upregulated genes
    down_genes : list
        List of downregulated genes
    name : str
        Name of the gene set (used as prefix for column names)
    layer : str, optional (default="raw_counts")
        Layer to use for calculation

    Returns
    -------
    adata : AnnData
        Updated annotated data matrix with new scores and groups
    """
    # Make a copy of the expression data
    adata.X = adata.layers[layer].copy()

    # Normalize and scale the data
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    if combined_score == False:
        # Filter genes present in the data
        up_genes_filtered = [gene for gene in up_genes if gene in adata.var_names]
        down_genes_filtered = [gene for gene in down_genes if gene in adata.var_names]
        # Calculate scores for up and down regulated genes
        if up_genes_filtered:
            sc.tl.score_genes(adata, up_genes_filtered, score_name=f"{name}_up_score")
            adata.obs[f"{name}_up_normalized"] = adata.obs[f"{name}_up_score"] / len(
                up_genes_filtered
            )
        else:
            adata.obs[f"{name}_up_normalized"] = 0

        if down_genes_filtered:
            sc.tl.score_genes(
                adata, down_genes_filtered, score_name=f"{name}_down_score"
            )
            adata.obs[f"{name}_down_normalized"] = adata.obs[
                f"{name}_down_score"
            ] / len(down_genes_filtered)
        else:
            adata.obs[f"{name}_down_normalized"] = 0
        # Calculate combined score
        adata.obs[f"{name}_combined_score"] = (
            adata.obs[f"{name}_up_normalized"] - adata.obs[f"{name}_down_normalized"]
        )

        # Split into high/low groups using median
        median_score = adata.obs[f"{name}_combined_score"].median()
        adata.obs[f"{name}_group"] = [
            f"{name}-high" if x > median_score else f"{name}-low"
            for x in adata.obs[f"{name}_combined_score"]
        ]
        adata.obs[f"{name}_group"] = pd.Categorical(adata.obs[f"{name}_group"])
    else:
        selected_genes = [gene for gene in up_genes if gene in adata.var_names]
        sc.tl.score_genes(adata, selected_genes, score_name=f"{name}_score")
        median_score = adata.obs[f"{name}_score"].median()
        adata.obs[f"{name}_group"] = [
            f"{name}-high" if x > median_score else f"{name}-low"
            for x in adata.obs[f"{name}_score"]
        ]
        adata.obs[f"{name}_group"] = pd.Categorical(adata.obs[f"{name}_group"])

    return adata


# Calculate GRN scores and assign groups
adata_nmf_cp = calculate_gene_set_scores(
    adata=adata_nmf_cp,
    up_genes=markers["grn"]["up"],
    down_genes=markers["grn"]["down"],
    name="grn",
    layer="raw_counts",
    combined_score=True,
)
# Gene score with DEG and CV

# Create DEG with PyDESeq2 between TTP and OS using just the training


def perform_deg_analysis(adata_train, endpoint="status-os"):

    deg = DEGAnalysis(
        adata_train,
        design_factor=endpoint,
        layer="raw_counts",
        output_dir="./deg_analysis_os_new",
    )
    deg.create_dds()
    deg.run_comparisons()
    deg.create_volcano_grid()
    deg.save_results()
    return deg.get_results()


# Calculate the score for each patient as the average of the top X genes (lognorm) - the bottom X genes (lognorm)
def calculate_gene_score(adata, top_genes, bottom_genes, layer="lognormalized_counts"):
    """
    Calculate risk score where higher score = higher risk
    top_genes: protective genes (upregulated in survivors)
    bottom_genes: risk genes (upregulated in non-survivors)
    """
    protective_expr = (
        -adata[:, top_genes].layers[layer].mean(axis=1)
    )  # Negative because higher expression = better survival. We do *-1 because we want their expression to decrease the risk (we subtract the protective effect to the risk effect)
    risk_expr = (
        adata[:, bottom_genes].layers[layer].mean(axis=1)
    )  # Already positive, higher expression = worse survival

    # We now get a risk score
    return protective_expr + risk_expr


def assign_score_group(adata, risk_scores_df, optimal_cutpoint, endpoint="os"):

    risk_scores_df["group"] = (risk_scores_df["risk_score"] > optimal_cutpoint).map(
        {True: "high", False: "low"}
    )

    # Sort by risk score for visualization
    risk_scores_df = risk_scores_df.sort_values("risk_score")
    risk_scores_df["rank"] = range(len(risk_scores_df))

    print(f"\nMedian time until event by risk scores group (high risk score vs low):")
    for group in risk_scores_df["group"].unique():
        group_df = risk_scores_df[risk_scores_df["group"] == group]
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
        risk_scores_df_with_subtype.groupby(["group", "SCLC_Subtype_de_novo"])
        .agg({"time": "mean", "status": ["count", "sum"]})
        .rename(columns={"sum": "events", "count": "total"})
    )

    # Now status=1 means event occurred, so sum gives us events
    # And status=0 means censored, so we can calculate censored as total - events
    summary["status", "censored"] = (
        summary["status", "total"] - summary["status", "events"]
    )

    print(summary)


# Perform CV (c-index) from 5, 10, 15, 20... top and bottom genes to calculate the score to see which one offers the best training and test scores

endpoint = "os"
status_map = {
    "os": {"Dead": True, "Alive": False},
    "ttp": {"Progressed": True, "Did not progress": False},
}
status_bool = adata_nmf_cp.obs[f"status-{endpoint}"].map(status_map[endpoint])
time = adata_nmf_cp.obs[f"tend_{endpoint}"]

y_structured = np.zeros(len(time), dtype=[("status", bool), ("time", float)])
y_structured["status"] = status_bool
y_structured["time"] = time

X = adata_nmf_cp.obs.index

stratify_columns = pd.DataFrame(
    {"SCLC_Subtype": adata_nmf_cp.obs["SCLC_Subtype_de_novo"], "status": status_bool}
).apply(tuple, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_structured, test_size=0.4, random_state=42, stratify=stratify_columns
)

# Subset adata to the indices chosen on X_train
adata_train = adata_nmf_cp[X_train]
adata_test = adata_nmf_cp[X_test]

# Perform DEG analysis on training data
deg_results = perform_deg_analysis(adata_train, endpoint=f"status-{endpoint}")

deg_results_df = deg_results["Alive_vs_Dead"].results_df.sort_values(
    "log2FoldChange", ascending=False
)

top_no_event_genes = deg_results_df.head(40).index.tolist()
bottom_event_genes = deg_results_df.tail(40).index.tolist()

scores_train = calculate_gene_score(adata_train, top_no_event_genes, bottom_event_genes)
# Find optimal cut off point for 10 genes using max stat log rank
# Create risk scores DataFrame with metadata
risk_scores_df = pd.DataFrame(
    {
        "risk_score": scores_train,
        "time": adata_train.obs["tend_os"],
        "status": adata_train.obs[f"status-{endpoint}"].map(
            {"Alive": 0, "Dead": 1}
            if endpoint == "os"
            else {"Did not progress": 0, "Progressed": 1}
        ),
    },
    index=adata_train.obs.index,
)

# Find optimal cutpoint using log-rank test
cutpoints = np.percentile(scores_train, np.arange(20, 81, 1))
max_statistic = 0
optimal_cutpoint = None
optimal_pvalue = None

for cutpoint in cutpoints:
    groups = scores_train > cutpoint

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
        optimal_pvalue = pvalue

assign_score_group(adata_train, risk_scores_df, optimal_cutpoint, endpoint="os")


# Test on unseen test set using the best cutpoint
scores_test = calculate_gene_score(adata_test, top_no_event_genes, bottom_event_genes)

risk_scores_df_test = pd.DataFrame(
    {
        "risk_score": scores_test,
        "time": adata_test.obs["tend_os"],
        "status": adata_test.obs[f"status-{endpoint}"].map(
            {"Alive": 0, "Dead": 1}
            if endpoint == "os"
            else {"Did not progress": 0, "Progressed": 1}
        ),
    },
    index=adata_test.obs.index,
)
assign_score_group(adata_test, risk_scores_df_test, optimal_cutpoint, endpoint="os")


# Find optimal number of genes to calculate the score
def get_score_statistics(adata, risk_scores_df, optimal_cutpoint, n_genes):
    """Get statistics for a particular gene set size"""
    risk_scores_df["group"] = (risk_scores_df["risk_score"] > optimal_cutpoint).map(
        {True: "high", False: "low"}
    )

    # Calculate statistics for each group
    stats = {}
    for group in ["high", "low"]:
        group_df = risk_scores_df[risk_scores_df["group"] == group]
        n_total = len(group_df)
        if n_total == 0:
            stats[f"{group}_median_survival"] = np.nan
            stats[f"{group}_n_total"] = 0
            stats[f"{group}_n_events"] = 0
            stats[f"{group}_n_censored"] = 0
            stats[f"{group}_event_rate"] = np.nan
            stats[f"{group}_censored_rate"] = np.nan
            continue

        n_events = group_df["status"].sum()
        n_censored = n_total - n_events

        stats[f"{group}_median_survival"] = group_df["time"].median()
        stats[f"{group}_n_total"] = n_total
        stats[f"{group}_n_events"] = n_events
        stats[f"{group}_n_censored"] = n_censored
        stats[f"{group}_event_rate"] = n_events / n_total * 100
        stats[f"{group}_censored_rate"] = n_censored / n_total * 100

    stats["n_genes"] = n_genes
    stats["optimal_cutpoint"] = optimal_cutpoint

    # Calculate log-rank test - Note the <= here
    groups = risk_scores_df["risk_score"] > optimal_cutpoint
    if len(np.unique(groups)) > 1:
        survival_data = np.zeros(
            len(risk_scores_df), dtype=[("status", bool), ("time", float)]
        )
        survival_data["status"] = risk_scores_df["status"].values.astype(bool)
        survival_data["time"] = risk_scores_df["time"].values
        chisq, pvalue = compare_survival(survival_data, groups)
        stats["log_rank_statistic"] = chisq
        stats["log_rank_pvalue"] = pvalue
    else:
        stats["log_rank_statistic"] = np.nan
        stats["log_rank_pvalue"] = np.nan

    return stats


# Initialize lists to store results
train_results = []
test_results = []

# Loop through different numbers of genes
for n_genes in range(5, 101, 5):  # 5, 10, 15, ..., 100
    print(f"Testing with {n_genes} genes...")

    # Get top and bottom genes
    top_no_event_genes = deg_results_df.head(n_genes).index.tolist()
    bottom_event_genes = deg_results_df.tail(n_genes).index.tolist()

    # Calculate scores for training set
    scores_train = calculate_gene_score(
        adata_train, top_no_event_genes, bottom_event_genes
    )

    # Create risk scores DataFrame for training
    risk_scores_df_train = pd.DataFrame(
        {
            "risk_score": scores_train,
            "time": adata_train.obs["tend_os"],
            "status": adata_train.obs[f"status-{endpoint}"].map(
                {"Alive": 0, "Dead": 1}
                if endpoint == "os"
                else {"Did not progress": 0, "Progressed": 1}
            ),
        },
        index=adata_train.obs.index,
    )

    # Find optimal cutpoint using log-rank test
    cutpoints = np.percentile(scores_train, np.arange(20, 81, 1))
    max_statistic = 0
    optimal_cutpoint = None

    for cutpoint in cutpoints:
        groups = scores_train > cutpoint
        survival_data = np.zeros(
            len(risk_scores_df_train), dtype=[("status", bool), ("time", float)]
        )
        survival_data["status"] = risk_scores_df_train["status"].values.astype(bool)
        survival_data["time"] = risk_scores_df_train["time"].values
        chisq, _ = compare_survival(survival_data, groups)
        if chisq > max_statistic:
            max_statistic = chisq
            optimal_cutpoint = cutpoint

    # Get training statistics
    train_stats = get_score_statistics(
        adata_train, risk_scores_df_train, optimal_cutpoint, n_genes
    )
    train_stats["set"] = "train"
    train_results.append(train_stats)

    # Calculate scores for test set
    scores_test = calculate_gene_score(
        adata_test, top_no_event_genes, bottom_event_genes
    )

    # Create risk scores DataFrame for test
    risk_scores_df_test = pd.DataFrame(
        {
            "risk_score": scores_test,
            "time": adata_test.obs["tend_os"],
            "status": adata_test.obs[f"status-{endpoint}"].map(
                {"Alive": 0, "Dead": 1}
                if endpoint == "os"
                else {"Did not progress": 0, "Progressed": 1}
            ),
        },
        index=adata_test.obs.index,
    )

    # Get test statistics using same cutpoint
    test_stats = get_score_statistics(
        adata_test, risk_scores_df_test, optimal_cutpoint, n_genes
    )
    test_stats["set"] = "test"
    test_results.append(test_stats)

# Convert results to DataFrames
results_df = pd.DataFrame(train_results + test_results)

# Sort by log-rank p-value and log-rank statistic
print("\nTop 5 performing gene sets by log-rank p-value in training set:")
print(
    results_df[results_df["set"] == "train"]
    .sort_values("log_rank_pvalue", ascending=False)
    .head()
)

print("\nTop 5 performing gene sets by log-rank statistic in training set:")
print(
    results_df[results_df["set"] == "train"]
    .sort_values("log_rank_statistic", ascending=False)
    .head()
)

# Save results to CSV
results_df.to_excel("gene_set_size_analysis.xlsx")

