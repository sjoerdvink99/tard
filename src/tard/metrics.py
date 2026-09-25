from collections import defaultdict
from collections.abc import Hashable
from numbers import Integral

import networkx as nx
import numpy as np
import pandas as pd

from tard.result import DistortionResult

Features = np.ndarray | pd.DataFrame


def compute(
    graph: nx.Graph,
    attributes: Features,
    embeddings: Features,
    max_hops: int = 4,
) -> DistortionResult:
    """Compare attribute and embedding cosine similarity over exact k-hop shells.

    DataFrames are aligned to graph nodes by index and must contain exactly the
    graph's nodes. Arrays must list rows in ``graph.nodes`` iteration order.
    """
    _validate_graph(graph, max_hops)
    nodes = list(graph.nodes)
    attribute_matrix = _align_to_nodes(attributes, nodes, "attributes")
    embedding_matrix = _align_to_nodes(embeddings, nodes, "embeddings")
    row_of = {node: row for row, node in enumerate(nodes)}

    distortion = np.full((len(nodes), max_hops), np.nan)
    signed_change = np.full((len(nodes), max_hops), np.nan)

    for row, node in enumerate(nodes):
        for hop, shell in hop_shells(graph, node, max_hops).items():
            shell_rows = [row_of[member] for member in shell]
            attribute_similarity = cosine_similarity(
                attribute_matrix[row], attribute_matrix[shell_rows]
            )
            embedding_similarity = cosine_similarity(
                embedding_matrix[row], embedding_matrix[shell_rows]
            )
            change = embedding_similarity - attribute_similarity
            distortion[row, hop - 1] = np.abs(change).mean()
            signed_change[row, hop - 1] = change.mean()

    scales = list(range(1, max_hops + 1))
    columns = [f"{hop}-hop" for hop in scales]
    return DistortionResult(
        distortion=pd.DataFrame(distortion, index=nodes, columns=columns),
        signed_change=pd.DataFrame(signed_change, index=nodes, columns=columns),
        nodes=nodes,
        scales=scales,
    )


def hop_shells(graph: nx.Graph, node: Hashable, max_hops: int) -> dict[int, list[Hashable]]:
    """Nodes at exactly k shortest-path hops from ``node``, for k = 1..max_hops."""
    distances = nx.single_source_shortest_path_length(graph, node, cutoff=max_hops)
    shells: dict[int, list[Hashable]] = defaultdict(list)
    for other, distance in distances.items():
        if distance > 0:
            shells[distance].append(other)
    return dict(shells)


def cosine_similarity(vector: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``vector`` to each row of ``others``; zero vectors give 0."""
    norms = np.linalg.norm(others, axis=1) * np.linalg.norm(vector)
    dot_products = others @ vector
    return np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms > 0)


def _validate_graph(graph: nx.Graph, max_hops: int) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise ValueError("graph must be an undirected networkx.Graph")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if isinstance(max_hops, bool) or not isinstance(max_hops, Integral) or max_hops < 1:
        raise ValueError("max_hops must be a positive integer")


def _align_to_nodes(features: Features, nodes: list[Hashable], name: str) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        if features.index.has_duplicates:
            raise ValueError(f"{name} index contains duplicate nodes")
        missing = [node for node in nodes if node not in features.index]
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} graph nodes, e.g. {missing[:3]}")
        if len(features.index) != len(nodes):
            raise ValueError(f"{name} contains rows for nodes that are not in the graph")
        matrix = features.loc[nodes].to_numpy(dtype=float)
    else:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(nodes):
            raise ValueError(
                f"{name} must have shape (n_nodes, n_features) = ({len(nodes)}, ...), "
                f"got {matrix.shape}"
            )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix
