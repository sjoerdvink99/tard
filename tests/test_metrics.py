import networkx as nx
import numpy as np
import pandas as pd
import pytest

import tard
from tard.metrics import cosine_similarity, hop_shells


def path_graph() -> nx.Graph:
    return nx.path_graph(["a", "b", "c", "d"])


def features(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def test_identical_representations_have_zero_distortion():
    attributes = features({"a": [1, 0], "b": [0, 1], "c": [1, 1], "d": [2, -1]})

    result = tard.compute(path_graph(), attributes, attributes.copy(), max_hops=3)

    assert np.nanmax(np.abs(result.distortion.to_numpy())) == pytest.approx(0.0)
    assert np.nanmax(np.abs(result.signed_change.to_numpy())) == pytest.approx(0.0)
    assert result.global_distortion == pytest.approx(0.0)


def test_changed_representation_produces_expected_distortion_and_sign():
    graph = nx.Graph([("a", "b")])
    attributes = features({"a": [1, 0], "b": [0, 1]})
    pulled_together = features({"a": [1, 0], "b": [1, 0]})
    pushed_apart = features({"a": [1, 0], "b": [-1, 0]})

    together = tard.compute(graph, attributes, pulled_together, max_hops=1)
    apart = tard.compute(graph, attributes, pushed_apart, max_hops=1)

    assert together.distortion.loc["a", "1-hop"] == pytest.approx(1.0)
    assert together.signed_change.loc["a", "1-hop"] == pytest.approx(1.0)
    assert apart.distortion.loc["a", "1-hop"] == pytest.approx(1.0)
    assert apart.signed_change.loc["a", "1-hop"] == pytest.approx(-1.0)


def test_cells_use_exact_hop_shells_only():
    attributes = features({"a": [1, 0], "b": [1, 0], "c": [1, 0], "d": [1, 0]})
    embeddings = features({"a": [1, 0], "b": [1, 0], "c": [0, 1], "d": [1, 0]})

    result = tard.compute(path_graph(), attributes, embeddings, max_hops=3)

    assert hop_shells(path_graph(), "a", 3) == {1: ["b"], 2: ["c"], 3: ["d"]}
    assert result.distortion.loc["a"].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_missing_shells_are_nan_and_ignored_globally():
    graph = nx.Graph([("a", "b")])
    graph.add_node("isolated")
    attributes = features({"a": [1, 0], "b": [0, 1], "isolated": [1, 1]})
    embeddings = features({"a": [1, 0], "b": [1, 0], "isolated": [1, 1]})

    result = tard.compute(graph, attributes, embeddings, max_hops=2)

    assert result.distortion["2-hop"].isna().all()
    assert result.distortion.loc["isolated"].isna().all()
    assert result.signed_change.loc["isolated"].isna().all()
    assert result.global_distortion == pytest.approx(1.0)


def test_dataframes_are_aligned_to_graph_nodes_by_index():
    attributes = features({"a": [1, 0], "b": [0, 1], "c": [1, 1], "d": [2, -1]})
    embeddings = features({"a": [0, 1], "b": [1, 1], "c": [1, 0], "d": [1, 2]})
    graph = path_graph()

    aligned = tard.compute(graph, attributes, embeddings)
    shuffled = tard.compute(graph, attributes.iloc[::-1], embeddings.sample(frac=1, random_state=0))
    from_arrays = tard.compute(graph, attributes.to_numpy(), embeddings.to_numpy())

    assert list(aligned.distortion.index) == list(graph.nodes)
    pd.testing.assert_frame_equal(aligned.distortion, shuffled.distortion)
    pd.testing.assert_frame_equal(aligned.distortion, from_arrays.distortion)


def test_zero_vectors_have_zero_similarity():
    similarity = cosine_similarity(np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]]))

    assert similarity.tolist() == [0.0, 0.0]


def test_invalid_inputs_raise_value_error():
    attributes = features({"a": [1, 0], "b": [0, 1], "c": [1, 1], "d": [0, 0]})
    graph = path_graph()

    with pytest.raises(ValueError, match="missing"):
        tard.compute(graph, attributes.drop("d"), attributes)
    with pytest.raises(ValueError, match="not in the graph"):
        tard.compute(graph, attributes, pd.concat([attributes, features({"e": [1, 1]})]))
    with pytest.raises(ValueError, match="duplicate"):
        tard.compute(graph, attributes, pd.concat([attributes, attributes.iloc[:1]]))
    with pytest.raises(ValueError, match="shape"):
        tard.compute(graph, np.ones((4, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="NaN"):
        tard.compute(graph, attributes, attributes.replace(0, np.nan))
    with pytest.raises(ValueError, match="max_hops"):
        tard.compute(graph, attributes, attributes, max_hops=0)
    with pytest.raises(ValueError, match="max_hops"):
        tard.compute(graph, attributes, attributes, max_hops=2.5)
    with pytest.raises(ValueError, match="undirected"):
        tard.compute(nx.DiGraph([("a", "b")]), np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="no nodes"):
        tard.compute(nx.Graph(), np.ones((0, 2)), np.ones((0, 2)))
