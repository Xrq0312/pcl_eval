#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace pcl_eval::features {

struct Spectrum3 {
    // Ascending eigenvalues and matching column eigenvectors.
    Eigen::Vector3d values = Eigen::Vector3d::Zero();
    Eigen::Matrix3d vectors = Eigen::Matrix3d::Identity();
};

struct LowRankEstimate {
    bool valid = false;
    double signed_radius = 0.0;
    Eigen::Vector3d anchor = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    Spectrum3 compensated_spectrum;
    double rmse = 0.0;
    double normalized_rmse = 0.0;
};

struct ShapeOperatorEstimate {
    bool valid = false;
    // Signed principal curvatures, ordered by decreasing absolute value.
    Eigen::Vector2d curvatures = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 3, 2> directions =
        Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double residual = 0.0;
    double tangent_condition = 0.0;
};

struct PointPairEstimate {
    bool sphere_valid = false;
    bool cylinder_valid = false;
    double sphere_signed_radius = 0.0;
    double sphere_residual = 0.0;
    double cylinder_signed_radius = 0.0;
    double cylinder_residual = 0.0;
    double plane_normal_change = 0.0;
    std::size_t pairs = 0;
};

struct MultiScaleShapeOperatorEstimate {
    std::size_t neighbors = 0;
    double radius = 0.0;
    ShapeOperatorEstimate aso;
};

struct CurvatureBoundaryEstimate {
    bool valid = false;
    double observed_normal_change = 0.0;
    double predicted_normal_change = 0.0;
    double residual = 0.0;
    double relative_residual = 0.0;
};

struct PatchFeatures {
    Spectrum3 position_spectrum;
    Spectrum3 normal_spectrum;
    LowRankEstimate sphere_low_rank;
    LowRankEstimate cylinder_low_rank;
    ShapeOperatorEstimate normal_ls_shape_operator;
    PointPairEstimate point_pairs;
};

// All routines expect one geometrically coherent patch with consistently
// oriented normals. A common global sign ambiguity is harmless (the estimated
// radius changes sign), while per-point sign flips are intentionally not hidden:
// they are an important failure mode measured by the experiments. This module
// is deliberately independent from the production primitive pipeline.
PatchFeatures compute_patch_features(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs = 4096);

LowRankEstimate estimate_sphere_low_rank(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals);

LowRankEstimate estimate_cylinder_low_rank(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    int alternating_iterations = 4);

ShapeOperatorEstimate estimate_normal_ls_shape_operator(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals);

// Author implementation of Lejemble et al.'s Algebraic Shape Operator (ASO),
// evaluated at one query point with a compact-support Euclidean radius.
ShapeOperatorEstimate estimate_algebraic_shape_operator(
    const Eigen::Vector3d& query,
    double radius,
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals);

std::vector<MultiScaleShapeOperatorEstimate> estimate_multiscale_aso(
    std::size_t query_index,
    const std::vector<std::size_t>& neighbor_counts,
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals);

// Compare an observed normal difference to the first-order variation predicted
// by a local shape operator. Large unexplained residual is a boundary cue, not
// by itself a proof of an instance boundary.
CurvatureBoundaryEstimate estimate_curvature_boundary(
    const Eigen::Vector3d& point_i,
    const Eigen::Vector3d& normal_i,
    const Eigen::Vector3d& point_j,
    const Eigen::Vector3d& normal_j,
    const ShapeOperatorEstimate& shape_operator,
    double noise_floor = 1e-6);

PointPairEstimate estimate_point_pair_invariants(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    const Eigen::Vector3d& cylinder_axis,
    std::size_t max_pairs = 4096);

}  // namespace pcl_eval::features
