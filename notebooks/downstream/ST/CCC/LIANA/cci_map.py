from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patches as patches
import pandas as pd
import scanpy as sc
import decoupler as dc
import liana as li
import scanpy as sc

adata = sc.read_h5ad("./adata_liana_v2.h5ad")


def CCIMap(
    interaction_data,
    lr: str = None,
    pos: dict = None,
    return_pos: bool = False,
    cmap: str = "tab10",
    font_size: int = 12,
    node_size_exp: int = 1,
    node_size_scaler: int = 1,
    min_counts: int = 0,
    sig_interactions: bool = True,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    pad=0.25,
    title: str = None,
    figsize: tuple = (10, 10),
):
    """Circular celltype-celltype interaction network based on LR-CCI analysis using LIANA results adapted from stLearn (https://stlearn.readthedocs.io/en/stable/tutorials/stLearn-CCI.html)."""
    # Filter interactions based on the LR pair if specified
    if sig_interactions:
        interaction_data = interaction_data[
            (interaction_data["cellphone_pvals"] < 0.05)
            & (interaction_data["lr_logfc"] > 0.5)
        ]

    if lr is not None:
        ligand, receptor = lr.split("^")
        interaction_data = interaction_data[
            (interaction_data["ligand_complex"] == ligand)
            & (interaction_data["receptor_complex"] == receptor)
        ]

    # Creating the interaction graph #
    graph = nx.MultiDiGraph()
    cell_types = np.unique(
        np.concatenate([interaction_data["source"], interaction_data["target"]])
    )

    for cell in cell_types:
        graph.add_node(cell)

    for _, row in interaction_data.iterrows():
        ligand = row["source"]
        receptor = row["target"]
        count = row["lr_means"]

        if count > min_counts:
            graph.add_edge(ligand, receptor, weight=count)

    if pos is None:
        pos = nx.circular_layout(graph)

    total = sum([d["weight"] for (u, v, d) in graph.edges(data=True)])

    if total == 0:
        print("No interactions found with the specified criteria.")
        return

    node_sizes = np.array(
        [
            (
                (sum([d["weight"] for _, _, d in graph.edges(cell, data=True)]) / total)
                * 10000
                * node_size_scaler
            )
            ** node_size_exp
            for cell in graph.nodes()
        ]
    )
    node_sizes[node_sizes == 0] = 0.1  # pseudocount

    edge_weights = [d["weight"] / total for (_, _, d) in graph.edges(data=True)]

    #### Drawing the graph #####
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=[0.7, 0.7, 0.7, 0.4])

    color_map = plt.get_cmap(cmap)
    node_colors = [color_map(i % color_map.N) for i, _ in enumerate(graph.nodes())]

    # Adding in the self-loops #
    z = 55
    for u, v, d in graph.edges(data=True):
        if u == v:
            x, y = pos[u]
            angle = np.degrees(np.arctan(y / x))
            if x > 0:
                angle += 180
            arc = patches.Arc(
                xy=(x, y),
                width=0.3,
                height=0.025,
                lw=5,
                ec=plt.cm.get_cmap("Blues")(d["weight"] / total),
                angle=angle,
                theta1=z,
                theta2=360 - z,
            )
            ax.add_patch(arc)

    nx.draw_networkx(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        arrowstyle="->",
        arrowsize=50,
        width=5,
        font_size=font_size,
        font_weight="bold",
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        ax=ax,
    )

    if title is None:
        title = f"CCI Interaction Network for ({lr})"
    fig.suptitle(title, fontsize=30)
    plt.tight_layout()

    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0] - pad, xlims[1] + pad)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0] - pad, ylims[1] + pad)

    if return_pos:
        return pos


ligrec_pairs = li.mt.cellphonedb(
    adata,
    groupby="cell_type",
    resource_name="consensus",
    expr_prop=0.1,
    verbose=True,
    use_raw=False,
    key_added="cpdb_res",
)

liana_results = li.mt.rank_aggregate(
    adata,
    groupby="cell_type",
    use_raw=False,
    groupby_pairs=adata.uns["cpdb_res"],
)

lr_pair = "ADCYAP1^RAMP2"
CCIMap(interaction_data=adata.uns["liana_res"], lr=lr_pair, sig_interactions=False)

plt.savefig("./test3.png")
