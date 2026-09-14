"""RA N-P-V estimation toolkit.

Implements exactly the estimators written in "Stage 2 Task for RAs"
(Discovering the Physics of LLM Creativity), section "The N-P-V framework":

* N  - bias-corrected Chao1 lower bound, sample coverage, discovery curve
* P  - Good-Turing coverage-adjusted idea probabilities, unseen mass, pi_0
* V  - inverse-detection-weighted valuable share r_V, N_V, Q_V
"""

from .estimators import (  # noqa: F401
    NEstimate,
    PEstimate,
    VEstimate,
    chao1_bias_corrected,
    chao1_bootstrap,
    detection_probability,
    discovery_curves,
    empirical_accumulation,
    estimate_N,
    estimate_P,
    estimate_V,
    expected_distinct,
    frequency_counts,
    rarefaction_curve,
    reconstructed_landscape,
    sample_coverage,
)

__version__ = "0.1.0"
