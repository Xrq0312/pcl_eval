#include "pcl_eval/robust_candidates.hpp"

#include "pcl_eval/geometric_features.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace pcl_eval::candidates {
namespace {

constexpr double kEpsilon = 1e-12;

struct Patch {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
};

Patch prepare(const std::vector<Eigen::Vector3d>& points,
              const std::vector<Eigen::Vector3d>& normals) {
    Patch result;
    if (points.size() != normals.size()) return result;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!points[i].allFinite() || !normals[i].allFinite() ||
            normals[i].norm() <= kEpsilon) continue;
        result.points.push_back(points[i]);
        result.normals.push_back(normals[i].normalized());
    }
    if (result.points.size() < 6) return {};
    return result;
}

Eigen::Vector3d mean(const std::vector<Eigen::Vector3d>& values,
                     const std::vector<double>* weights = nullptr) {
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    double total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double weight = weights == nullptr ? 1.0 : (*weights)[i];
        result += weight * values[i];
        total += weight;
    }
    if (total <= kEpsilon) return Eigen::Vector3d::Zero();
    return result / total;
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = 0.5 * (result + *lower);
    }
    return result;
}

double patch_scale(const std::vector<Eigen::Vector3d>& points) {
    const Eigen::Vector3d center = mean(points);
    double sum = 0.0;
    for (const auto& point : points) sum += (point - center).squaredNorm();
    return std::sqrt(std::max(kEpsilon, sum / points.size()));
}

std::pair<double, double> residual_summary(std::vector<double> residuals) {
    if (residuals.empty()) return {0.0, 0.0};
    const double med = median(residuals);
    std::sort(residuals.begin(), residuals.end());
    const std::size_t keep = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(0.9 * residuals.size())));
    double squared = 0.0;
    for (std::size_t i = 0; i < keep; ++i) squared += residuals[i] * residuals[i];
    return {med, std::sqrt(squared / keep)};
}

std::vector<double> sphere_residuals(const std::vector<Eigen::Vector3d>& points,
                                     const Eigen::Vector3d& center,
                                     double radius) {
    std::vector<double> result;
    result.reserve(points.size());
    for (const auto& point : points)
        result.push_back(std::abs((point - center).norm() - radius));
    return result;
}

std::vector<double> cylinder_residuals(const std::vector<Eigen::Vector3d>& points,
                                       const Eigen::Vector3d& axis_point,
                                       const Eigen::Vector3d& axis,
                                       double radius) {
    std::vector<double> result;
    result.reserve(points.size());
    for (const auto& point : points) {
        const Eigen::Vector3d delta = point - axis_point;
        result.push_back(std::abs((delta - delta.dot(axis) * axis).norm() - radius));
    }
    return result;
}

double normalized_tls_slope(const std::vector<Eigen::Vector3d>& x,
                            const std::vector<Eigen::Vector3d>& y,
                            const Eigen::Matrix3d& projector,
                            double y_scale,
                            double* condition) {
    const Eigen::Vector3d x_mean = mean(x);
    const Eigen::Vector3d y_mean = mean(y);
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < x.size(); ++i) {
        const Eigen::Vector3d dx = projector * (x[i] - x_mean);
        const Eigen::Vector3d dy = projector * ((y[i] - y_mean) / y_scale);
        scatter(0, 0) += dx.squaredNorm();
        scatter(0, 1) += dx.dot(dy);
        scatter(1, 1) += dy.squaredNorm();
    }
    scatter(1, 0) = scatter(0, 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    if (solver.info() != Eigen::Success) return std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector2d direction = solver.eigenvectors().col(1);
    if (condition != nullptr)
        *condition = solver.eigenvalues()(1) /
            std::max(kEpsilon, solver.eigenvalues()(0));
    if (std::abs(direction(0)) <= kEpsilon)
        return std::numeric_limits<double>::quiet_NaN();
    return y_scale * direction(1) / direction(0);
}

std::vector<double> pair_radius_seeds(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    const Eigen::Matrix3d& projector,
    std::size_t max_pairs) {
    const std::size_t n = points.size();
    const std::size_t total = n * (n - 1) / 2;
    const std::size_t stride = std::max<std::size_t>(1,
        total / std::max<std::size_t>(1, max_pairs));
    std::vector<double> strict, fallback;
    strict.reserve(std::min(total, max_pairs));
    fallback.reserve(std::min(total, max_pairs));
    std::size_t serial = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j, ++serial) {
            if (serial % stride != 0 || fallback.size() >= max_pairs) continue;
            const Eigen::Vector3d dp = projector * (points[j] - points[i]);
            const Eigen::Vector3d dn = projector * (normals[j] - normals[i]);
            const double denominator = dn.squaredNorm();
            if (denominator <= 1e-8 || dp.squaredNorm() <= 1e-12) continue;
            const double radius = dp.dot(dn) / denominator;
            if (!std::isfinite(radius)) continue;
            fallback.push_back(radius);
            const double relative = (dp - radius * dn).norm() / dp.norm();
            if (relative < 0.35) strict.push_back(radius);
        }
    }
    return strict.size() >= 8 ? strict : fallback;
}

double weighted_radius(const Patch& patch, const Eigen::Matrix3d& projector,
                       const std::vector<double>& weights) {
    const Eigen::Vector3d point_mean = mean(patch.points, &weights);
    const Eigen::Vector3d normal_mean = mean(patch.normals, &weights);
    double numerator = 0.0, denominator = 0.0;
    for (std::size_t i = 0; i < patch.points.size(); ++i) {
        const Eigen::Vector3d dp = projector * (patch.points[i] - point_mean);
        const Eigen::Vector3d dn = projector * (patch.normals[i] - normal_mean);
        numerator += weights[i] * dp.dot(dn);
        denominator += weights[i] * dn.squaredNorm();
    }
    return denominator > kEpsilon ? numerator / denominator
                                  : std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> huber_weights(const std::vector<double>& residuals) {
    const double scale = std::max(1e-9, 1.4826 * median(residuals));
    const double cutoff = 1.345 * scale;
    std::vector<double> weights(residuals.size(), 1.0);
    for (std::size_t i = 0; i < residuals.size(); ++i)
        if (residuals[i] > cutoff) weights[i] = cutoff / residuals[i];
    return weights;
}

SphereCandidate make_sphere(const Patch& patch, Estimator estimator,
                            double radius, const std::vector<double>* weights,
                            double condition, double runtime_ms) {
    SphereCandidate result;
    result.estimator = estimator;
    if (!std::isfinite(radius) || std::abs(radius) <= kEpsilon) return result;
    const Eigen::Vector3d point_mean = mean(patch.points, weights);
    const Eigen::Vector3d normal_mean = mean(patch.normals, weights);
    result.center = point_mean - radius * normal_mean;
    result.radius = std::abs(radius);
    const auto summary = residual_summary(
        sphere_residuals(patch.points, result.center, result.radius));
    result.median_residual = summary.first;
    result.trimmed_rmse = summary.second;
    result.condition = condition;
    result.runtime_ms = runtime_ms;
    result.valid = result.center.allFinite() && std::isfinite(result.trimmed_rmse);
    return result;
}

CylinderCandidate make_cylinder(const Patch& patch, Estimator estimator,
                                const Eigen::Vector3d& axis, double radius,
                                const std::vector<double>* weights,
                                double condition, double runtime_ms) {
    CylinderCandidate result;
    result.estimator = estimator;
    if (!axis.allFinite() || !std::isfinite(radius) || std::abs(radius) <= kEpsilon)
        return result;
    result.axis_direction = axis.normalized();
    std::vector<Eigen::Vector3d> compensated(patch.points.size());
    for (std::size_t i = 0; i < patch.points.size(); ++i)
        compensated[i] = patch.points[i] - radius * patch.normals[i];
    result.axis_point = mean(compensated, weights);
    result.radius = std::abs(radius);
    const auto summary = residual_summary(cylinder_residuals(
        patch.points, result.axis_point, result.axis_direction, result.radius));
    result.median_residual = summary.first;
    result.trimmed_rmse = summary.second;
    result.condition = condition;
    result.runtime_ms = runtime_ms;
    result.valid = result.axis_point.allFinite() && std::isfinite(result.trimmed_rmse);
    return result;
}

Eigen::Vector3d sign_invariant_axis(const std::vector<Eigen::Vector3d>& normals,
                                    double* condition) {
    Eigen::Matrix3d moment = Eigen::Matrix3d::Zero();
    for (const auto& normal : normals) moment += normal * normal.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(moment);
    if (solver.info() != Eigen::Success) return Eigen::Vector3d::UnitZ();
    if (condition != nullptr)
        *condition = solver.eigenvalues()(1) /
            std::max(kEpsilon, solver.eigenvalues()(0));
    return solver.eigenvectors().col(0).normalized();
}

SphereCandidate point_only_sphere(const std::vector<Eigen::Vector3d>& points) {
    const auto started = std::chrono::steady_clock::now();
    SphereCandidate result;
    result.estimator = Estimator::PointOnly;
    if (points.size() < 5) return result;
    Eigen::MatrixXd design(points.size(), 4);
    Eigen::VectorXd target(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        design.row(i) << 2.0 * points[i].transpose(), 1.0;
        target(static_cast<Eigen::Index>(i)) = points[i].squaredNorm();
    }
    const Eigen::Vector4d parameters = design.colPivHouseholderQr().solve(target);
    const double squared_radius = parameters.head<3>().squaredNorm() + parameters(3);
    if (squared_radius <= kEpsilon || !std::isfinite(squared_radius)) return result;
    result.center = parameters.head<3>();
    result.radius = std::sqrt(squared_radius);
    const auto summary = residual_summary(
        sphere_residuals(points, result.center, result.radius));
    result.median_residual = summary.first;
    result.trimmed_rmse = summary.second;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(design);
    const auto singular = svd.singularValues();
    result.condition = singular(0) /
        std::max(kEpsilon, singular(singular.size() - 1));
    result.runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.valid = result.center.allFinite() && std::isfinite(result.trimmed_rmse) &&
                   std::isfinite(result.condition) && result.condition < 1e8 &&
                   result.radius < 1e4 * patch_scale(points);
    return result;
}

double model_bic(std::vector<double> residuals, int parameters, double scale) {
    std::sort(residuals.begin(), residuals.end());
    const std::size_t keep = std::max<std::size_t>(parameters + 1,
        static_cast<std::size_t>(std::ceil(0.9 * residuals.size())));
    double squared = 0.0;
    for (std::size_t i = 0; i < std::min(keep, residuals.size()); ++i)
        squared += residuals[i] * residuals[i];
    const double variance = squared / std::min(keep, residuals.size());
    const double normalized = std::max(1e-16, variance / (scale * scale));
    return static_cast<double>(keep) * std::log(normalized) +
           parameters * std::log(static_cast<double>(keep));
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> split_halves(
    const std::vector<Eigen::Vector3d>& values) {
    std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> result;
    result.first.reserve((values.size() + 1) / 2);
    result.second.reserve(values.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i)
        (i % 2 == 0 ? result.first : result.second).push_back(values[i]);
    return result;
}

const SphereCandidate* find_sphere(
    const std::vector<SphereCandidate>& candidates, Estimator estimator) {
    const auto found = std::find_if(candidates.begin(), candidates.end(),
        [&](const SphereCandidate& candidate) {
            return candidate.estimator == estimator && candidate.valid;
        });
    return found == candidates.end() ? nullptr : &*found;
}

const CylinderCandidate* find_cylinder(
    const std::vector<CylinderCandidate>& candidates, Estimator estimator) {
    const auto found = std::find_if(candidates.begin(), candidates.end(),
        [&](const CylinderCandidate& candidate) {
            return candidate.estimator == estimator && candidate.valid;
        });
    return found == candidates.end() ? nullptr : &*found;
}

double sphere_stability(const SphereCandidate* left, const SphereCandidate* right,
                        double scale) {
    if (left == nullptr || right == nullptr) return 2.0;
    return ((left->center - right->center).norm() +
            std::abs(left->radius - right->radius)) / scale;
}

double cylinder_stability(const CylinderCandidate* left,
                          const CylinderCandidate* right, double scale) {
    if (left == nullptr || right == nullptr) return 2.0;
    const double cosine = std::clamp(std::abs(left->axis_direction.dot(
        right->axis_direction)), 0.0, 1.0);
    const double angle = std::acos(cosine);
    const double offset = ((left->axis_point - right->axis_point).cross(
        right->axis_direction)).norm() / scale;
    return angle + offset + std::abs(left->radius - right->radius) / scale;
}

double plane_stability(const PlaneCandidate& left, const PlaneCandidate& right,
                       double scale) {
    if (!left.valid || !right.valid) return 2.0;
    double sign = left.normal.dot(right.normal) < 0.0 ? -1.0 : 1.0;
    const double cosine = std::clamp(left.normal.dot(sign * right.normal), 0.0, 1.0);
    return std::acos(cosine) + std::abs(left.offset - sign * right.offset) / scale;
}

double stability_penalty(double stability, std::size_t point_count) {
    // A continuous complexity term: stable models are almost unchanged, while
    // hypotheses that depend on one half of the samples pay a substantial cost.
    return static_cast<double>(point_count) * std::log1p(stability * stability);
}

Eigen::Vector3d soft_geometry_costs(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals, double scale) {
    Eigen::Vector3d costs = Eigen::Vector3d::Zero();
    if (points.size() < 8 || points.size() != normals.size()) return costs;
    Eigen::Vector3d center = mean(points);
    std::size_t query = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
        if ((points[i] - center).squaredNorm() < (points[query] - center).squaredNorm())
            query = i;
    const auto estimates = features::estimate_multiscale_aso(
        query, {16, 32, 64}, points, normals);
    std::vector<Eigen::Vector2d> valid;
    for (const auto& estimate : estimates)
        if (estimate.aso.valid)
            valid.push_back(estimate.aso.curvatures.cwiseAbs() * scale);
    if (valid.empty()) return costs;
    Eigen::Vector2d curvature = valid[valid.size() / 2];
    double major = std::max(curvature(0), curvature(1));
    double minor = std::min(curvature(0), curvature(1));
    const double curvedness = major + minor;
    const double flat_cost_for_curved_models = 1.0 / (1.0 + 20.0 * curvedness);
    const double balance = (major - minor) / std::max(1e-9, curvedness);
    const double minor_fraction = minor / std::max(1e-9, major);
    // Plane, sphere, cylinder respectively. These continuous costs never make
    // a hard type decision; they only break close geometric-residual ties.
    costs(0) = curvedness * curvedness / (1.0 + curvedness * curvedness);
    costs(1) = flat_cost_for_curved_models + balance * balance;
    costs(2) = flat_cost_for_curved_models + minor_fraction * minor_fraction;
    return costs.cwiseMin(Eigen::Vector3d::Constant(2.0));
}

}  // namespace

std::vector<SphereCandidate> generate_sphere_candidates(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs) {
    const Patch patch = prepare(points, normals);
    if (patch.points.empty()) return {};
    std::vector<SphereCandidate> result;
    const double scale = patch_scale(patch.points);

    auto started = std::chrono::steady_clock::now();
    const auto ls = features::estimate_sphere_low_rank(patch.points, patch.normals);
    result.push_back(make_sphere(patch, Estimator::NormalLs, ls.signed_radius,
        nullptr, 0.0, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count()));

    started = std::chrono::steady_clock::now();
    double condition = 0.0;
    const double tls_radius = normalized_tls_slope(patch.normals, patch.points,
        Eigen::Matrix3d::Identity(), scale, &condition);
    result.push_back(make_sphere(patch, Estimator::NormalizedTls, tls_radius,
        nullptr, condition, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count()));

    started = std::chrono::steady_clock::now();
    const auto seeds = pair_radius_seeds(patch.points, patch.normals,
        Eigen::Matrix3d::Identity(), max_pairs);
    if (!seeds.empty()) {
        double radius = median(seeds);
        std::vector<double> weights(patch.points.size(), 1.0);
        for (int iteration = 0; iteration < 8; ++iteration) {
            const Eigen::Vector3d center = mean(patch.points, &weights) -
                radius * mean(patch.normals, &weights);
            std::vector<double> residuals(patch.points.size());
            for (std::size_t i = 0; i < patch.points.size(); ++i)
                residuals[i] = (patch.points[i] - radius * patch.normals[i] - center).norm();
            weights = huber_weights(residuals);
            radius = weighted_radius(patch, Eigen::Matrix3d::Identity(), weights);
        }
        result.push_back(make_sphere(patch, Estimator::PairIrls, radius, &weights,
            0.0, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count()));
    } else {
        SphereCandidate invalid;
        invalid.estimator = Estimator::PairIrls;
        result.push_back(invalid);
    }
    result.push_back(point_only_sphere(patch.points));
    return result;
}

std::vector<CylinderCandidate> generate_cylinder_candidates(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs) {
    const Patch patch = prepare(points, normals);
    if (patch.points.empty()) return {};
    std::vector<CylinderCandidate> result;
    const double scale = patch_scale(patch.points);

    auto started = std::chrono::steady_clock::now();
    const auto ls = features::estimate_cylinder_low_rank(patch.points, patch.normals);
    result.push_back(make_cylinder(patch, Estimator::NormalLs, ls.axis,
        ls.signed_radius, nullptr, 0.0,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count()));

    double axis_condition = 0.0;
    const Eigen::Vector3d axis = sign_invariant_axis(patch.normals, &axis_condition);
    const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - axis * axis.transpose();
    started = std::chrono::steady_clock::now();
    double tls_condition = 0.0;
    const double tls_radius = normalized_tls_slope(patch.normals, patch.points,
        projector, scale, &tls_condition);
    result.push_back(make_cylinder(patch, Estimator::NormalizedTls, axis,
        tls_radius, nullptr, std::min(axis_condition, tls_condition),
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count()));

    started = std::chrono::steady_clock::now();
    const auto seeds = pair_radius_seeds(patch.points, patch.normals,
        projector, max_pairs);
    if (!seeds.empty()) {
        double radius = median(seeds);
        std::vector<double> weights(patch.points.size(), 1.0);
        for (int iteration = 0; iteration < 8; ++iteration) {
            std::vector<Eigen::Vector3d> q(patch.points.size());
            for (std::size_t i = 0; i < patch.points.size(); ++i)
                q[i] = patch.points[i] - radius * patch.normals[i];
            const Eigen::Vector3d anchor = mean(q, &weights);
            std::vector<double> residuals(q.size());
            for (std::size_t i = 0; i < q.size(); ++i)
                residuals[i] = (projector * (q[i] - anchor)).norm();
            weights = huber_weights(residuals);
            radius = weighted_radius(patch, projector, weights);
        }
        result.push_back(make_cylinder(patch, Estimator::PairIrls, axis,
            radius, &weights, axis_condition,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count()));
    } else {
        CylinderCandidate invalid;
        invalid.estimator = Estimator::PairIrls;
        result.push_back(invalid);
    }
    return result;
}

PlaneCandidate fit_plane(const std::vector<Eigen::Vector3d>& points) {
    PlaneCandidate result;
    if (points.size() < 3) return result;
    const Eigen::Vector3d center = mean(points);
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& point : points) {
        const Eigen::Vector3d delta = point - center;
        covariance += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) return result;
    result.normal = solver.eigenvectors().col(0).normalized();
    result.offset = -result.normal.dot(center);
    std::vector<double> residuals;
    residuals.reserve(points.size());
    for (const auto& point : points)
        residuals.push_back(std::abs(result.normal.dot(point) + result.offset));
    const auto summary = residual_summary(std::move(residuals));
    result.median_residual = summary.first;
    result.trimmed_rmse = summary.second;
    result.valid = true;
    return result;
}

CompetitionResult compete_models(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    double maximum_normalized_rmse,
    double minimum_bic_margin_per_point,
    double stability_weight,
    double soft_geometry_prior_weight) {
    CompetitionResult result;
    if (points.size() < 6 || points.size() != normals.size()) return result;
    const double scale = patch_scale(points);
    const Eigen::Vector3d geometry_cost = soft_geometry_prior_weight != 0.0
        ? soft_geometry_costs(points, normals, scale) : Eigen::Vector3d::Zero();
    const PlaneCandidate plane = fit_plane(points);
    const auto spheres = generate_sphere_candidates(points, normals);
    const auto cylinders = generate_cylinder_candidates(points, normals);
    const auto point_halves = split_halves(points);
    const auto normal_halves = split_halves(normals);
    const PlaneCandidate plane_left = fit_plane(point_halves.first);
    const PlaneCandidate plane_right = fit_plane(point_halves.second);
    const auto spheres_left = generate_sphere_candidates(
        point_halves.first, normal_halves.first);
    const auto spheres_right = generate_sphere_candidates(
        point_halves.second, normal_halves.second);
    const auto cylinders_left = generate_cylinder_candidates(
        point_halves.first, normal_halves.first);
    const auto cylinders_right = generate_cylinder_candidates(
        point_halves.second, normal_halves.second);

    if (plane.valid) {
        std::vector<double> residuals;
        residuals.reserve(points.size());
        for (const auto& point : points)
            residuals.push_back(std::abs(plane.normal.dot(point) + plane.offset));
        const double stability = plane_stability(plane_left, plane_right, scale);
        result.scores.push_back({ModelType::Plane, true,
            model_bic(residuals, 3, scale) + stability_weight *
                stability_penalty(stability, points.size()) +
                soft_geometry_prior_weight * points.size() * geometry_cost(0),
            plane.trimmed_rmse / scale, stability, 3, "point_pca"});
    }
    for (const auto& sphere : spheres) if (sphere.valid) {
        const auto residuals = sphere_residuals(points, sphere.center, sphere.radius);
        const double stability = sphere_stability(
            find_sphere(spheres_left, sphere.estimator),
            find_sphere(spheres_right, sphere.estimator), scale);
        result.scores.push_back({ModelType::Sphere, true,
            model_bic(residuals, 4, scale) + stability_weight *
                stability_penalty(stability, points.size()) +
                soft_geometry_prior_weight * points.size() * geometry_cost(1),
            sphere.trimmed_rmse / scale, stability, 4, to_string(sphere.estimator)});
    }
    for (const auto& cylinder : cylinders) if (cylinder.valid) {
        const auto residuals = cylinder_residuals(points, cylinder.axis_point,
            cylinder.axis_direction, cylinder.radius);
        const double stability = cylinder_stability(
            find_cylinder(cylinders_left, cylinder.estimator),
            find_cylinder(cylinders_right, cylinder.estimator), scale);
        result.scores.push_back({ModelType::Cylinder, true,
            model_bic(residuals, 5, scale) + stability_weight *
                stability_penalty(stability, points.size()) +
                soft_geometry_prior_weight * points.size() * geometry_cost(2),
            cylinder.trimmed_rmse / scale, stability, 5, to_string(cylinder.estimator)});
    }
    if (result.scores.empty()) return result;
    std::sort(result.scores.begin(), result.scores.end(),
        [](const ModelScore& left, const ModelScore& right) { return left.bic < right.bic; });
    const ModelScore& best = result.scores.front();
    const auto competing_type = std::find_if(
        result.scores.begin() + 1, result.scores.end(),
        [&](const ModelScore& score) { return score.type != best.type; });
    const double margin = competing_type != result.scores.end()
        ? (competing_type->bic - best.bic) / points.size()
        : std::numeric_limits<double>::infinity();
    result.confidence = margin;
    result.rejected = best.normalized_rmse > maximum_normalized_rmse ||
                      margin < minimum_bic_margin_per_point;
    result.selected = result.rejected ? ModelType::Rejected : best.type;
    return result;
}

std::string to_string(Estimator estimator) {
    switch (estimator) {
        case Estimator::NormalLs: return "normal_ls";
        case Estimator::NormalizedTls: return "normalized_tls";
        case Estimator::PairIrls: return "pair_irls";
        case Estimator::PointOnly: return "point_only";
    }
    return "unknown";
}

std::string to_string(ModelType type) {
    switch (type) {
        case ModelType::Plane: return "plane";
        case ModelType::Sphere: return "sphere";
        case ModelType::Cylinder: return "cylinder";
        case ModelType::Rejected: return "rejected";
    }
    return "rejected";
}

}  // namespace pcl_eval::candidates
