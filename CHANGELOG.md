# Changelog

This project follows [Semantic Versioning](https://semver.org/). Before 1.0,
minor releases may change optimizer internals or checkpoint schemas; such
changes are called out here.

## [Unreleased]

## [0.4.0] - 2026-07-31

### Added

- Native float32 K-FAC factors and decompositions for MLX Linear layers.
- Homogeneous-coordinate bias handling and AdamW fallback ownership.
- Conv1d, Conv2d, Embedding, and MultiHeadAttention curvature blocks.
- Factor partitioning, head-aware attention blocks, masking, momentum,
  KL clipping, adaptive damping, and distributed sufficient-stat aggregation.
- Cholesky and eigendecomposition solve paths with `mx.compile`-compatible
  optimizer state and validated checkpoint restoration.
- Public dual-graph compilation helpers that avoid decomposition work between
  scheduled inverse refreshes.
- Pre-allocation state-memory estimation, a 1024-wide default factor cap, and
  a configurable optimizer-state memory guard.
- Finite hyperparameter validation, bounded damping, decomposition floors, and
  transactional model/optimizer rollback after failed eager or compiled
  updates.
- Distributed collective validation and coordinated failure recovery.
- Distributed gradient-schema coordination before rank-dependent collectives,
  including externally pre-aggregated compiled training.
- Packaging, compatibility, CI, and trusted-publishing release infrastructure.

[Unreleased]: https://github.com/primateria/mlx-kfac/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/primateria/mlx-kfac/releases/tag/v0.4.0
