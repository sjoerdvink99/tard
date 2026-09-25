import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import tard


@pytest.fixture
def result() -> tard.DistortionResult:
    graph = nx.path_graph(["a", "b", "c"])
    attributes = pd.DataFrame({"x": [1.0, 0.0, 1.0], "y": [0.0, 1.0, 1.0]}, index=["a", "b", "c"])
    embeddings = pd.DataFrame({"x": [1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}, index=["a", "b", "c"])
    return tard.compute(graph, attributes, embeddings, max_hops=3)


def test_result_shape_and_labels(result):
    assert result.nodes == ["a", "b", "c"]
    assert result.scales == [1, 2, 3]
    assert list(result.distortion.columns) == ["1-hop", "2-hop", "3-hop"]
    assert result.signed_change.shape == result.distortion.shape
    assert result.distortion["3-hop"].isna().all()


def test_global_distortion_is_mean_of_valid_cells(result):
    expected = np.nanmean(result.distortion.to_numpy())

    assert result.global_distortion == pytest.approx(expected)


def test_plots_return_figure_and_axes(result):
    for figure, ax in (
        tard.plot_distortion_map(result, title="Distortion"),
        tard.plot_signed_map(result),
        tard.plot_node_profile(result, "a"),
    ):
        assert ax.figure is figure
        plt.close(figure)


def test_maps_accept_a_shared_color_scale(result):
    distortion_figure, distortion_ax = tard.plot_distortion_map(result, vmax=0.25)
    signed_figure, signed_ax = tard.plot_signed_map(result, limit=0.25)

    distortion_norm = distortion_ax.collections[0].norm
    signed_norm = signed_ax.collections[0].norm
    assert (distortion_norm.vmin, distortion_norm.vmax) == (0.0, 0.25)
    assert (signed_norm.vmin, signed_norm.vmax) == (-0.25, 0.25)
    plt.close(distortion_figure)
    plt.close(signed_figure)


def test_signed_map_is_centered_on_zero_by_default(result):
    figure, ax = tard.plot_signed_map(result)

    norm = ax.collections[0].norm
    assert norm.vmin == -norm.vmax
    assert norm.vmax == pytest.approx(np.nanmax(np.abs(result.signed_change.to_numpy())))
    plt.close(figure)
