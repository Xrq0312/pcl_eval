# Staged fusion roadmap

The ordering deliberately isolates causal effects instead of enabling every
idea at once.

## Stage 0: reproducibility and frozen baseline

- Preserve upstream artifacts and document missing source.
- Freeze seeds, splits, build configuration and evaluator schema.
- Report instance detection, typed detection, mIoU, cylinder parameter errors,
  unassigned ratio and wall time.

## Stage 1: segmentation-independent candidate generation

Status: implemented as `global_ransac` and `global_local`.

- Generate global and spatially local hypotheses from oriented minimal
  samples, so a region-growing failure does not eliminate a primitive.
- Use a deterministic pilot subset to rank hypotheses before full-cloud
  scoring, bounding CPU cost.
- Add a three-oriented-point cylinder sampler and robust cylinder refit.
- Use optional organized-depth normals/adjacency when a pixel map is supplied;
  retain the KNN path for ordinary point clouds.
- Reject geometrically unobservable strip and large-radius hypotheses.
- Ablate global-only, local-only and global+local proposals.

This stage draws on Efficient RANSAC/minimal quadric fitting and the CPFN
global/local decomposition, but it remains a non-learning CPU implementation.

## Stage 2: progressive multi-model management

Status: first CPU approximation implemented as `joint_opt`.

- Maintain a proposal pool rather than deleting points after the first fit.
- Alternate model selection, point labeling and robust parameter refitting.
- Penalize model count and overlapping explanations; prune duplicate models.
- Stop when the probability of finding a new supported model falls below a
  configured bound or the time budget expires.
- The current approximation uses support-aware assignment, adaptive bounded
  noise scales, duplicate pruning and alternating refit. Spatial Potts terms
  and explicit label-cost optimization remain future work.

This is a bounded approximation of the Progressive-X/PEARL principle.  Full
graph-cut alpha expansion is not required for the first CPU ablation; the
energy and every approximation must be explicit.

## Stage 3: boundary-aware labeling

Status: an experimental `boundary_aware` stage was implemented and rejected
by the first 10-sample ablation. It changed target mIoU from 0.3191 to 0.3224
but reduced detection from 0.2661 to 0.2452 and typed detection from 0.2229 to
0.1922. A more conservative rerun also failed. The accepted default therefore
remains `joint_opt`; this negative result is not presented as a completed
SED-style boundary solution.

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

