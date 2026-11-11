"""Test importing and verifying all dashboard views."""

from unittest.mock import MagicMock, patch
import pytest

from dashboard.views.home import render_home_view
from dashboard.views.model_comparison import render_model_comparison_view
from dashboard.views.telemetry_explorer import render_telemetry_explorer_view
from dashboard.views.prediction_playground import render_prediction_playground_view
from dashboard.views.evaluation_testing import render_evaluation_testing_view
from dashboard.views.research_info import render_research_info_view


def test_views_callable():
    assert callable(render_home_view)
    assert callable(render_model_comparison_view)
    assert callable(render_telemetry_explorer_view)
    assert callable(render_prediction_playground_view)
    assert callable(render_evaluation_testing_view)
    assert callable(render_research_info_view)


def _make_mock_st(dataset_mode="Local Borg 2011 — Available"):
    mock_st = MagicMock()
    mock_st.session_state = {"dataset_mode": dataset_mode}

    def mock_cols(spec):
        n = len(spec) if isinstance(spec, list) else int(spec)
        return [MagicMock() for _ in range(n)]

    mock_st.columns.side_effect = mock_cols
    mock_st.radio.return_value = "Validation Split"
    mock_st.selectbox.return_value = "Aligned Stacked LSTM (128h, 2l)"
    mock_st.multiselect.return_value = ["cpu_avg_usage", "memory_canonical_usage"]
    mock_st.button.return_value = False
    mock_st.text_input.side_effect = lambda label, value="", **kwargs: value
    return mock_st


def test_render_views_local_mode():
    mode = "Local Borg 2011 — Available"
    views = [
        (render_home_view, "dashboard.views.home.st"),
        (render_model_comparison_view, "dashboard.views.model_comparison.st"),
        (render_telemetry_explorer_view, "dashboard.views.telemetry_explorer.st"),
        (render_prediction_playground_view, "dashboard.views.prediction_playground.st"),
        (render_evaluation_testing_view, "dashboard.views.evaluation_testing.st"),
        (render_research_info_view, "dashboard.views.research_info.st"),
    ]
    for fn, target in views:
        with patch(target, _make_mock_st(mode)):
            fn()  # Must execute without exception (e.g. NameError, TypeError)


def test_render_views_google2019_mode():
    mode = "Google Cluster Trace 2019 — Access pending"
    views = [
        (render_home_view, "dashboard.views.home.st"),
        (render_model_comparison_view, "dashboard.views.model_comparison.st"),
        (render_telemetry_explorer_view, "dashboard.views.telemetry_explorer.st"),
        (render_prediction_playground_view, "dashboard.views.prediction_playground.st"),
        (render_evaluation_testing_view, "dashboard.views.evaluation_testing.st"),
        (render_research_info_view, "dashboard.views.research_info.st"),
    ]
    for fn, target in views:
        with patch(target, _make_mock_st(mode)):
            fn()  # Must execute without exception


