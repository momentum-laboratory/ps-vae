"""Public API surface for the staged ``analyze_uncertainty`` refactor.

Importing ``_analyze_uncertainty`` exposes the refactored helpers so existing
call-sites can switch from ``analyze_uncertainty`` to the package entry point
incrementally::

	import _analyze_uncertainty as au
	grid = au.get_nrmse_grid(...)

The re-exports below are grouped by the module that owns the implementation so
you can immediately see where each symbol lives.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from . import apps, core, metrics, viz, helpers
from .core import (
	AnalyzeUQconfig,
	au_config,
	extract_nn_estimate_of_posterior,
	get_nrmse_grid,
	get_progressive_pdf,
	nrmse_to_CIs,
)
from .metrics import (
	analyze_scatter,
	compare_posteriors,	
	get_cross_mahas,
	plot_CI_intersect,
	plot_fk_CI_intersect,
	analyze_CI_intersections,
	summarize_mahas,
)
from .viz import (
	cbarhist,
	demonstrate_ellipses,
	plot_ellipse,
	plot_map,
	plot_CI_maps,	
	plot_two_points,
	juxtapose_two_voxels_posteriors,
	simple_error_map,
	single_voxel_detailed_viz,
	viz_posteriors,
	viz_cov,
)
from .apps import (
	demo_classification_app,
	demo_posterior_sampling,
	plot_CRs_decrease_along_sequence,
	plot_progressive_maps_calc_MAPEs,
)

__all__ = [
	# submodules
	"apps",
	"core",
	"metrics",
	"viz",
	# core
	"AnalyzeUQconfig",
	"au_config",
	"extract_nn_estimate_of_posterior",
	"get_nrmse_grid",
	"get_progressive_pdf",
	"nrmse_to_CIs",
	# metrics
	"analyze_scatter",
	"compare_posteriors",
	"farthest_point_sampling",
	"get_cross_mahas",
	"plot_CI_intersect",
	"plot_fk_CI_intersect",
	"analyze_CI_intersections",
	"summarize_mahas",
	# viz
	"cbarhist",
	"demonstrate_ellipses",
	"plot_ellipse",
	"plot_map",
	"plot_CI_maps",
	"plot_progressive_pdfs",
	"plot_two_points",
	"juxtapose_two_voxels_posteriors",
	"simple_error_map",
	"single_voxel_detailed_viz",
	"viz_posteriors",
	"viz_cov",
	# apps
	"demo_classification_app",
	"demo_posterior_sampling",
	"plot_CRs_decrease_along_sequence",
]
