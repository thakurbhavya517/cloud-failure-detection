"""Test importing and verifying all dashboard views."""

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
