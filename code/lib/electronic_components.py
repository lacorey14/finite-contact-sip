"""Electronic-component bookkeeping for topology-controlled SIP models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def canonical_component_ids(component_ids: Sequence[int], n_particles: int) -> np.ndarray:
    """Validate labels and relabel components by first appearance as 0..K-1."""
    labels = np.asarray(component_ids, dtype=int)
    if labels.shape != (n_particles,):
        raise ValueError(
            f"component_ids must have length {n_particles}; got shape {labels.shape}"
        )
    if np.any(labels < 0):
        raise ValueError("component_ids must be non-negative")
    mapping: dict[int, int] = {}
    canonical = np.empty(n_particles, dtype=int)
    for index, label in enumerate(labels.tolist()):
        mapping.setdefault(label, len(mapping))
        canonical[index] = mapping[label]
    return canonical


def combine_facet_groups(
    particle_facets: Sequence[np.ndarray], component_ids: Sequence[int] | None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Combine particle surface facets belonging to one equipotential component.

    With ``component_ids=None``, every particle remains an independent floating
    conductor, reproducing the original solver behavior.
    """
    n_particles = len(particle_facets)
    labels = (
        np.arange(n_particles, dtype=int)
        if component_ids is None
        else canonical_component_ids(component_ids, n_particles)
    )
    components = []
    for component in range(int(labels.max()) + 1 if n_particles else 0):
        members = [
            np.asarray(particle_facets[i], dtype=int)
            for i in range(n_particles)
            if labels[i] == component
        ]
        components.append(np.concatenate(members) if members else np.array([], dtype=int))
    return components, labels


def contact_laplacian(
    n_components: int, contact_edges: Sequence[tuple[int, int, float]] | None
) -> np.ndarray:
    """Return the symmetric conductance-graph Laplacian in siemens."""
    laplacian = np.zeros((n_components, n_components), dtype=float)
    for edge in contact_edges or ():
        if len(edge) != 3:
            raise ValueError("each contact edge must be (component_i, component_j, G_S)")
        i, j, conductance = int(edge[0]), int(edge[1]), float(edge[2])
        if i == j:
            raise ValueError("a contact edge must join different components")
        if not (0 <= i < n_components and 0 <= j < n_components):
            raise ValueError(f"contact edge {(i, j)} outside 0..{n_components - 1}")
        if not np.isfinite(conductance) or conductance < 0.0:
            raise ValueError("contact conductance must be finite and non-negative")
        laplacian[i, i] += conductance
        laplacian[j, j] += conductance
        laplacian[i, j] -= conductance
        laplacian[j, i] -= conductance
    return laplacian
