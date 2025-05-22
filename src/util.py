import numpy as np
import pandas as pd

import plotnine as p9
import plotly.graph_objects as go

def compare_dmaps(dmap1, dmap2):
    """Compare two diffusion maps, correcting the signs of the components"""
    maximal_idx = (np.argmax(np.abs(dmap1), axis=0), np.arange(dmap1.shape[1]))
    dmap1p = dmap1 * np.sign(dmap1[maximal_idx]) * np.sign(dmap2[maximal_idx])
    return np.max(np.abs(dmap1p - dmap2))

theme_chill = (
        p9.theme_bw()
        + p9.theme(
    plot_background=p9.element_rect(fill=(0,0,0,0), size=0),
    panel_background = p9.element_rect(fill=(0,0,0,0)),
    legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
))

theme_arguelles = (
    p9.theme_bw(base_size=14) +
    p9.theme(
        text=p9.element_text(family="Alegreya", color="#343A40"),
        axis_text=p9.element_text(family="sans"),
        plot_background=p9.element_rect(fill=(0, 0, 0, 0), size=0),
        panel_background=p9.element_rect(fill=(0, 0, 0, 0)),
        legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
        strip_background=p9.element_rect(fill="#343A40"),
        strip_text=p9.element_text(color="#F8F9FA"),
    )
)

theme_arguelles_sans = (
        p9.theme_bw(base_size=14) +
        p9.theme(
            text=p9.element_text(family="Alegreya Sans", color="#343A40"),
            axis_text=p9.element_text(family="sans", size=9, color="#4D4D4D"),
            plot_background=p9.element_rect(fill=(0, 0, 0, 0), size=0),
            panel_background=p9.element_rect(fill=(0, 0, 0, 0)),
            legend_background=p9.element_rect(fill=(0, 0, 0, 0)),
            strip_background=p9.element_rect(fill="#343A40"),
            strip_text=p9.element_text(color="#F8F9FA"),
        )
)

def visualize_graph_3d(coords, adjacency_matrix, threshold):
    """

    Args:
        coords: coordinates of the nodes. A (n, 3) array
        adjacency_matrix: Adjacency matrix describing the weights of the edges. A (n, n) array
        threshold: Only entries which are larger than the threshold are displayed

    Returns:

    """
    # TODO: enable encoding weight by line size in bins
    max_weight = np.max(adjacency_matrix)

    edge_x, edge_y, edge_z, edge_colors = [], [], [], []
    for i, j in np.ndindex(adjacency_matrix.shape):
        weight = adjacency_matrix[i, j]
        if adjacency_matrix[i, j] < threshold:
            continue
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        logweight = np.log(weight)
        edge_colors.extend([logweight, logweight, logweight])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(width=5, color=edge_colors, colorscale="viridis_r"),
        opacity=0.7,
        hoverinfo="skip",
    )
    return edge_trace


def array_to_df(data, prefix: str, indexing = 1):
    """
    Convert a 2D array into a pandas DataFrame with prefixed column names.

    Parameters:
    - data (array-like): A two-dimensional array to convert into a DataFrame.
    - prefix (str): Prefix for the column names.
    - indexing (int, optional): Starting index for column naming. Defaults to 1.

    Returns:
    - pd.DataFrame: DataFrame with columns named using the prefix and index.
    """
    if len(data.shape) != 2:
        raise ValueError("Need two-dimensional array to transform into DataFrame")
    n_columns = data.shape[1]
    columns = [prefix+str(i) for i in range(indexing, indexing + n_columns)]
    return pd.DataFrame(data, columns=columns)