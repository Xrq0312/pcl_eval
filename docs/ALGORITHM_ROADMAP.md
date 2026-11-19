# Staged fusion roadmap

The ordering deliberately isolates causal effects instead of enabling every
idea at once.

## Stage 0: reproducibility and frozen baseline

- Preserve upstream artifacts and document missing source.
- Freeze seeds, splits, build configuration and evaluator schema.
- Report instance detection, typed detection, mIoU, cylinder parameter errors,
  unassigned ratio and wall time.

## Stage 1: segmentation-independent candidate generation

- Generate global and spatially local hypotheses from oriented minimal
  samples, so a region-growing failure does not eliminate a primitive.
- Use a deterministic pilot subset to rank hypotheses before full-cloud
  scoring, bounding CPU cost.
- Add a three-oriented-point cylinder sampler and robust cylinder refit.
- Ablate global-only, local-only and global+local proposals.

This stage draws on Efficient RANSAC/minimal quadric fitting and the CPFN
global/local decomposition, but it remains a non-learning CPU implementation.

## Stage 2: progressive multi-model management

- Maintain a proposal pool rather than deleting points after the first fit.
- Alternate model selection, point labeling and robust parameter refitting.
- Penalize model count and overlapping explanations; prune duplicate models.
- Stop when the probability of finding a new supported model falls below a
  configured bound or the time budget expires.

This is a bounded approximation of the Progressive-X/PEARL principle.  Full
graph-cut alpha expansion is not required for the first CPU ablation; the
energy and every approximation must be explicit.

## Stage 3: boundary-aware labeling

- Build a sparse neighborhood graph once.
- Derive a geometric boundary likelihood from normal jumps, curvature and
  competing-model residual margins.
- Reduce smoothing across high-boundary edges and reject merges without local
  adjacency support.
- Keep disconnected coplanar surfaces separate unless there is positive
  occlusion/continuity evidence.

## Stage 4: global relation refinement

- Discover high-confidence parallel, perpendicular, coplanar and coaxial
  relations after segmentation is stable.
- Select a non-conflicting relation set and optimize relations jointly with
  data residuals.
- Accept the transaction only if support and robust residual gates pass.

This follows GlobFit's relation-graph idea.  Relation snapping must never be
used to merge instances by itself.

## Stage 5: dataset expansion and final audit

- Tune only on designated development splits.
- Keep TraceParts/ABCParts test splits and real Primitive Object samples as
  frozen audits where licensing/download availability permits.
- Report dataset-native limitations.  In particular, BA and GCCao do not
  provide official point-level multi-instance labels, while Primitive Object
  provides object pose/dimensions but no point mask.

