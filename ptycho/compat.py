"""Compatibility shims across ``quantem`` versions.

The tutorial notebook targets an unreleased ``quantem`` that exposes
``quantem.diffractive_imaging.PtychoObjConstraintParams``.  Released wheels
(<= 0.1.9) instead accept a plain nested dict ``{"object": {...}}`` and use a
slightly different set of constraint keys.  This module resolves whichever
spelling is available so the pipeline runs on both.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keys accepted by quantem <= 0.1.9 for the object model.
_LEGACY_OBJECT_CONSTRAINTS = {
    "positivity",
    "fix_potential_baseline",
    "fix_potential_baseline_factor",
    "identical_slices",
    "apply_fov_mask",
    "tv_weight_z",
    "tv_weight_xy",
    "surface_zero_weight",
    "gaussian_sigma",
    "butterworth_order",
    "q_lowpass",
    "q_highpass",
}

_LEGACY_ALIASES = {
    # Newer API                    legacy equivalent (None -> drop)
    "positivity_mode": None,
}


def _legacy_defaults() -> set[str]:
    try:
        from quantem.diffractive_imaging.object_models import ObjectConstraints

        return set(ObjectConstraints.DEFAULT_CONSTRAINTS)
    except Exception:  # pragma: no cover - defensive
        return set(_LEGACY_OBJECT_CONSTRAINTS)


def _filter_for_legacy(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = _legacy_defaults()
    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in allowed:
            filtered[key] = value
            continue
        alias = _LEGACY_ALIASES.get(key, KeyError)
        if alias is KeyError:
            logger.warning("Dropping unsupported object constraint %r for this quantem version", key)
        elif alias is not None:
            filtered[alias] = value
    return filtered


def object_constraints(**kwargs: Any) -> Any:
    """Build the object constraint payload used by ``Ptychography.reconstruct``.

    Returns whatever the installed ``quantem`` expects as the value of
    ``constraints["object"]``.
    """
    try:
        from quantem.diffractive_imaging import PtychoObjConstraintParams

        return PtychoObjConstraintParams.Raster(**kwargs)
    except ImportError:
        return _filter_for_legacy(kwargs)
