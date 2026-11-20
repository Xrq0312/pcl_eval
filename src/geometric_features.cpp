#include "pcl_eval/geometric_features.hpp"

#include "algebraic_shape_operator/AlgebraicShapeOperator.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace pcl_eval::features {
namespace {

constexpr double kEpsilon = 1e-12;

struct PreparedPatch {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
};

PreparedPatch prepare(const std::vector<Eigen::Vector3d>& points,
                      const std::vector<Eigen::Vector3d>& normals) {
    PreparedPatch result;
    if (points.size() != normals.size()) return result;
    result.points.reserve(points.size());
    result.normals.reserve(normals.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!points[i].allFinite() || !normals[i].allFinite()) continue;
        const double length = normals[i].norm();
        if (length <= kEpsilon) continue;
        result.points.push_back(points[i]);
        result.normals.push_back(normals[i] / length);
    }
    if (result.points.size() < 4) return {};
    return result;
}

Eigen::Vector3d mean(const std::vector<Eigen::Vector3d>& values) {
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (const auto& value : values) result += value;
    return result / static_cast<double>(values.size());
}

Eigen::Matrix3d covariance(const std::vector<Eigen::Vector3d>& values,
                           const Eigen::Vector3d& center) {
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    for (const auto& value : values) {
        const Eigen::Vector3d delta = value - center;
        result.noalias() += delta * delta.transpose();
    }
    return result / static_cast<double>(values.size());
}

Spectrum3 spectrum(const Eigen::Matrix3d& matrix) {
    Spectrum3 result;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(
        0.5 * (matrix + matrix.transpose()));
    if (solver.info() == Eigen::Success) {
        result.values = solver.eigenvalues().cwiseMax(0.0);
        result.vectors = solver.eigenvectors();
    }
    return result;
}

double patch_scale(const std::vector<Eigen::Vector3d>& points) {
    const Eigen::Vector3d center = mean(points);
    return std::sqrt(std::max(kEpsilon, covariance(points, center).trace()));
}

double solve_radius(const std::vector<Eigen::Vector3d>& points,
                    const std::vector<Eigen::Vector3d>& normals,
                    const Eigen::Matrix3d& projector) {
    const Eigen::Vector3d point_mean = mean(points);
    const Eigen::Vector3d normal_mean = mean(normals);
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d dp = projector * (points[i] - point_mean);
        const Eigen::Vector3d dn = projector * (normals[i] - normal_mean);
        numerator += dp.dot(dn);
        denominator += dn.squaredNorm();
    }
    if (denominator <= kEpsilon) return std::numeric_limits<double>::quiet_NaN();
    return numerator / denominator;
}

std::vector<Eigen::Vector3d> compensate(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    double radius) {
    std::vector<Eigen::Vector3d> result(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        result[i] = points[i] - radius * normals[i];
    return result;
}

Eigen::Matrix<double, 3, 2> tangent_basis(const Eigen::Vector3d& normal) {
    Eigen::Vector3d helper = std::abs(normal.z()) < 0.8
        ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = normal.cross(helper).normalized();
    basis.col(1) = normal.cross(basis.col(0)).normalized();
    return basis;
}

}  // namespace

LowRankEstimate estimate_sphere_low_rank(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals) {
    const PreparedPatch patch = prepare(points, normals);
    LowRankEstimate result;
    if (patch.points.empty()) return result;
    result.signed_radius = solve_radius(
        patch.points, patch.normals, Eigen::Matrix3d::Identity());
    if (!std::isfinite(result.signed_radius)) return result;
    const auto q = compensate(patch.points, patch.normals, result.signed_radius);
    result.anchor = mean(q);
    result.compensated_spectrum = spectrum(covariance(q, result.anchor));
    result.rmse = std::sqrt(result.compensated_spectrum.values.sum());
    result.normalized_rmse = result.rmse / patch_scale(patch.points);
    result.valid = std::isfinite(result.normalized_rmse);
    return result;
}

LowRankEstimate estimate_cylinder_low_rank(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    int alternating_iterations) {
    const PreparedPatch patch = prepare(points, normals);
    LowRankEstimate result;
    if (patch.points.empty()) return result;

    const Eigen::Vector3d normal_mean = mean(patch.normals);
    Eigen::Vector3d axis = spectrum(covariance(patch.normals, normal_mean))
                               .vectors.col(0).normalized();
    double radius = std::numeric_limits<double>::quiet_NaN();
    for (int iteration = 0; iteration < std::max(1, alternating_iterations); ++iteration) {
        const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() -
            axis * axis.transpose();
        radius = solve_radius(patch.points, patch.normals, projector);
        if (!std::isfinite(radius)) return result;
        const auto q = compensate(patch.points, patch.normals, radius);
        const auto q_spectrum = spectrum(covariance(q, mean(q)));
        Eigen::Vector3d proposed = q_spectrum.vectors.col(2).normalized();
        if (proposed.dot(axis) < 0.0) proposed = -proposed;
        // Normal covariance remains informative when a short axial extent makes
        // the largest compensated eigenvector ill-conditioned.
        if (q_spectrum.values(2) > 4.0 * q_spectrum.values(1)) axis = proposed;
    }

    const auto q = compensate(patch.points, patch.normals, radius);
    result.anchor = mean(q);
    result.compensated_spectrum = spectrum(covariance(q, result.anchor));
    result.axis = axis;
    result.signed_radius = radius;
    result.rmse = std::sqrt(result.compensated_spectrum.values(0) +
                            result.compensated_spectrum.values(1));
    result.normalized_rmse = result.rmse / patch_scale(patch.points);
    result.valid = std::isfinite(result.normalized_rmse);
    return result;
}

ShapeOperatorEstimate estimate_normal_ls_shape_operator(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals) {
    const PreparedPatch patch = prepare(points, normals);
    ShapeOperatorEstimate result;
    if (patch.points.empty()) return result;

    Eigen::Vector3d normal = mean(patch.normals);
    if (normal.norm() <= kEpsilon) return result;
    normal.normalize();
    const auto basis = tangent_basis(normal);
    const Eigen::Vector3d point_mean = mean(patch.points);
    const Eigen::Vector3d normal_mean = mean(patch.normals);
    Eigen::Matrix2d xx = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d nx = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < patch.points.size(); ++i) {
        const Eigen::Vector2d x = basis.transpose() * (patch.points[i] - point_mean);
        const Eigen::Vector2d dn = basis.transpose() * (patch.normals[i] - normal_mean);
        xx.noalias() += x * x.transpose();
        nx.noalias() += dn * x.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> xx_solver(xx);
    if (xx_solver.info() != Eigen::Success || xx_solver.eigenvalues()(0) <= kEpsilon)
        return result;
    result.tangent_condition = xx_solver.eigenvalues()(1) /
                               xx_solver.eigenvalues()(0);
    Eigen::Matrix2d shape = -nx * xx.inverse();
    shape = 0.5 * (shape + shape.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(shape);
    if (solver.info() != Eigen::Success) return result;
    int first = std::abs(solver.eigenvalues()(1)) >
                std::abs(solver.eigenvalues()(0)) ? 1 : 0;
    const int second = 1 - first;
    result.curvatures << solver.eigenvalues()(first), solver.eigenvalues()(second);
    result.directions.col(0) = basis * solver.eigenvectors().col(first);
    result.directions.col(1) = basis * solver.eigenvectors().col(second);
    result.normal = normal;

    double squared_error = 0.0;
    double squared_signal = 0.0;
    for (std::size_t i = 0; i < patch.points.size(); ++i) {
        const Eigen::Vector2d x = basis.transpose() * (patch.points[i] - point_mean);
        const Eigen::Vector2d dn = basis.transpose() * (patch.normals[i] - normal_mean);
        squared_error += (dn + shape * x).squaredNorm();
        squared_signal += dn.squaredNorm();
    }
    result.residual = std::sqrt(squared_error /
        std::max(kEpsilon, squared_signal));
    result.valid = std::isfinite(result.residual);
    return result;
}

ShapeOperatorEstimate estimate_algebraic_shape_operator(
    const Eigen::Vector3d& query,
    double radius,
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals) {
    ShapeOperatorEstimate result;
    if (!query.allFinite() || !std::isfinite(radius) || radius <= 0.0) return result;
    const PreparedPatch patch = prepare(points, normals);
    if (patch.points.empty()) return result;
    const auto properties = aso::compute(
        query, radius,
        patch.points.cbegin(), patch.points.cend(),
        patch.normals.cbegin(), patch.normals.cend());
    if (!std::isfinite(properties.k1()) || !std::isfinite(properties.k2()) ||
        !properties.d1().allFinite() || !properties.d2().allFinite() ||
        !properties.n().allFinite() || properties.n().norm() <= kEpsilon)
        return result;
    result.curvatures << properties.k1(), properties.k2();
    result.directions.col(0) = properties.d1();
    result.directions.col(1) = properties.d2();
    result.normal = properties.n();
    result.valid = true;
    return result;
}

PointPairEstimate estimate_point_pair_invariants(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    const Eigen::Vector3d& cylinder_axis,
    std::size_t max_pairs) {
    const PreparedPatch patch = prepare(points, normals);
    PointPairEstimate result;
    if (patch.points.empty()) return result;
    const Eigen::Vector3d axis = cylinder_axis.normalized();
    const Eigen::Matrix3d transverse = Eigen::Matrix3d::Identity() -
        axis * axis.transpose();
    double sphere_num = 0.0, sphere_den = 0.0;
    double cylinder_num = 0.0, cylinder_den = 0.0;
    double normal_change = 0.0;

    const std::size_t n = patch.points.size();
    const std::size_t total = n * (n - 1) / 2;
    const std::size_t stride = std::max<std::size_t>(1, total / std::max<std::size_t>(1, max_pairs));
    std::size_t serial = 0;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> samples;
    samples.reserve(std::min(total, max_pairs));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j, ++serial) {
            if (serial % stride != 0 || samples.size() >= max_pairs) continue;
            const Eigen::Vector3d dp = patch.points[j] - patch.points[i];
            const Eigen::Vector3d dn = patch.normals[j] - patch.normals[i];
            if (dp.squaredNorm() <= kEpsilon) continue;
            samples.emplace_back(dp, dn);
            sphere_num += dp.dot(dn);
            sphere_den += dn.squaredNorm();
            const Eigen::Vector3d tdp = transverse * dp;
            const Eigen::Vector3d tdn = transverse * dn;
            cylinder_num += tdp.dot(tdn);
            cylinder_den += tdn.squaredNorm();
            normal_change += dn.squaredNorm();
        }
    }
    result.pairs = samples.size();
    if (samples.empty()) return result;
    result.plane_normal_change = std::sqrt(normal_change / samples.size());
    if (sphere_den > kEpsilon) {
        result.sphere_signed_radius = sphere_num / sphere_den;
        double error = 0.0, signal = 0.0;
        for (const auto& [dp, dn] : samples) {
            error += (dp - result.sphere_signed_radius * dn).squaredNorm();
            signal += dp.squaredNorm();
        }
        result.sphere_residual = std::sqrt(error / std::max(kEpsilon, signal));
        result.sphere_valid = true;
    }
    if (cylinder_den > kEpsilon) {
        result.cylinder_signed_radius = cylinder_num / cylinder_den;
        double error = 0.0, signal = 0.0;
        for (const auto& [dp, dn] : samples) {
            const Eigen::Vector3d tdp = transverse * dp;
            const Eigen::Vector3d tdn = transverse * dn;
            error += (tdp - result.cylinder_signed_radius * tdn).squaredNorm();
            signal += tdp.squaredNorm();
        }
        result.cylinder_residual = std::sqrt(error / std::max(kEpsilon, signal));
        result.cylinder_valid = true;
    }
    return result;
}

PatchFeatures compute_patch_features(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs) {
    PatchFeatures result;
    const PreparedPatch patch = prepare(points, normals);
    if (patch.points.empty()) return result;
    result.position_spectrum = spectrum(covariance(patch.points, mean(patch.points)));
    result.normal_spectrum = spectrum(covariance(patch.normals, mean(patch.normals)));
    result.sphere_low_rank = estimate_sphere_low_rank(patch.points, patch.normals);
    result.cylinder_low_rank = estimate_cylinder_low_rank(patch.points, patch.normals);
    result.normal_ls_shape_operator =
        estimate_normal_ls_shape_operator(patch.points, patch.normals);
    result.point_pairs = estimate_point_pair_invariants(
        patch.points, patch.normals, result.cylinder_low_rank.axis, max_pairs);
    return result;
}

}  // namespace pcl_eval::features
