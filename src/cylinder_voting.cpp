#include "pcl_eval/cylinder_voting.hpp"

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace pcl_eval::voting {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;

struct Projection {
    Eigen::Vector3d axis;
    Eigen::Vector3d e1;
    Eigen::Vector3d e2;
    std::vector<Eigen::Vector2d> uv;
    std::vector<double> axial;
};

struct Circle {
    bool valid = false;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    double radius = 0.0;
};

double scene_scale(const std::vector<Eigen::Vector3d>& points) {
    Eigen::Vector3d minimum = points.front();
    Eigen::Vector3d maximum = points.front();
    for (const auto& point : points) {
        minimum = minimum.cwiseMin(point);
        maximum = maximum.cwiseMax(point);
    }
    return std::max(kEpsilon, (maximum - minimum).norm());
}

std::vector<Eigen::Vector3d> hemisphere_directions(int count) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(std::max(1, count));
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < std::max(1, count); ++i) {
        const double z = (i + 0.5) / std::max(1, count);
        const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double azimuth = golden * i;
        result.emplace_back(radial * std::cos(azimuth),
                            radial * std::sin(azimuth), z);
    }
    return result;
}

Projection project(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& axis) {
    Projection result;
    result.axis = axis.normalized();
    result.e1 = result.axis.unitOrthogonal();
    result.e2 = result.axis.cross(result.e1).normalized();
    result.uv.reserve(points.size());
    result.axial.reserve(points.size());
    for (const auto& point : points) {
        result.uv.emplace_back(point.dot(result.e1), point.dot(result.e2));
        result.axial.push_back(point.dot(result.axis));
    }
    return result;
}

Circle circle_from_three(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                         const Eigen::Vector2d& c) {
    Circle result;
    Eigen::Matrix2d matrix;
    matrix.row(0) = 2.0 * (b - a).transpose();
    matrix.row(1) = 2.0 * (c - a).transpose();
    if (std::abs(matrix.determinant()) < 1e-10) return result;
    Eigen::Vector2d target(b.squaredNorm() - a.squaredNorm(),
                           c.squaredNorm() - a.squaredNorm());
    result.center = matrix.fullPivLu().solve(target);
    result.radius = (a - result.center).norm();
    result.valid = result.center.allFinite() && std::isfinite(result.radius);
    return result;
}

Circle refine_circle(const std::vector<Eigen::Vector2d>& points,
                     const std::vector<std::size_t>& support) {
    Circle result;
    if (support.size() < 3) return result;
    Eigen::MatrixXd design(support.size(), 3);
    Eigen::VectorXd target(support.size());
    for (std::size_t row = 0; row < support.size(); ++row) {
        const Eigen::Vector2d point = points[support[row]];
        design.row(static_cast<Eigen::Index>(row)) << 2.0 * point.x(),
            2.0 * point.y(), 1.0;
        target(static_cast<Eigen::Index>(row)) = point.squaredNorm();
    }
    const Eigen::Vector3d parameters = design.colPivHouseholderQr().solve(target);
    const double squared_radius = parameters.head<2>().squaredNorm() + parameters(2);
    if (!parameters.allFinite() || squared_radius <= kEpsilon) return result;
    result.center = parameters.head<2>();
    result.radius = std::sqrt(squared_radius);
    result.valid = std::isfinite(result.radius);
    return result;
}

std::vector<std::size_t> support_for(const Projection& projection,
                                     const Circle& circle, double threshold) {
    std::vector<std::size_t> support;
    support.reserve(projection.uv.size());
    for (std::size_t i = 0; i < projection.uv.size(); ++i)
        if (std::abs((projection.uv[i] - circle.center).norm() - circle.radius) <= threshold)
            support.push_back(i);
    return support;
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

Cylinder make_candidate(const Projection& projection, Circle circle,
                        double threshold, int angular_bins,
                        std::size_t minimum_support) {
    Cylinder result;
    auto support = support_for(projection, circle, threshold);
    if (support.size() < minimum_support) return result;
    circle = refine_circle(projection.uv, support);
    if (!circle.valid) return result;
    support = support_for(projection, circle, threshold);
    if (support.size() < minimum_support) return result;

    std::vector<unsigned char> occupied(std::max(4, angular_bins), 0);
    std::vector<double> residuals;
    residuals.reserve(support.size());
    double axial_mean = 0.0;
    double axial_minimum = std::numeric_limits<double>::infinity();
    double axial_maximum = -std::numeric_limits<double>::infinity();
    for (const std::size_t index : support) {
        const Eigen::Vector2d delta = projection.uv[index] - circle.center;
        double angle = std::atan2(delta.y(), delta.x());
        if (angle < 0.0) angle += 2.0 * kPi;
        const int bin = std::min(static_cast<int>(occupied.size()) - 1,
            static_cast<int>(angle / (2.0 * kPi) * occupied.size()));
        occupied[bin] = 1;
        residuals.push_back(std::abs(delta.norm() - circle.radius));
        axial_mean += projection.axial[index];
        axial_minimum = std::min(axial_minimum, projection.axial[index]);
        axial_maximum = std::max(axial_maximum, projection.axial[index]);
    }
    axial_mean /= support.size();
    result.axis_direction = projection.axis;
    result.axis_point = circle.center.x() * projection.e1 +
        circle.center.y() * projection.e2 + axial_mean * projection.axis;
    result.radius = circle.radius;
    result.median_residual = median(std::move(residuals));
    result.angular_coverage_degrees = 360.0 * std::accumulate(
        occupied.begin(), occupied.end(), 0) / occupied.size();
    result.axial_span = axial_maximum - axial_minimum;
    result.support = std::move(support);
    result.score = result.support.size() /
        (1.0 + result.median_residual / std::max(kEpsilon, threshold));
    return result;
}

double axis_angle(const Cylinder& left, const Cylinder& right) {
    return std::acos(std::clamp(std::abs(left.axis_direction.dot(
        right.axis_direction)), 0.0, 1.0));
}

double line_distance(const Cylinder& left, const Cylinder& right) {
    return ((left.axis_point - right.axis_point).cross(right.axis_direction)).norm();
}

double support_overlap(const Cylinder& left, const Cylinder& right) {
    std::size_t i = 0, j = 0, intersection = 0;
    while (i < left.support.size() && j < right.support.size()) {
        if (left.support[i] == right.support[j]) { ++intersection; ++i; ++j; }
        else if (left.support[i] < right.support[j]) ++i;
        else ++j;
    }
    return static_cast<double>(intersection) /
        std::max<std::size_t>(1, std::min(left.support.size(), right.support.size()));
}

double local_cylinder_fraction(const Projection& projection,
                               const Cylinder& candidate) {
    if (candidate.support.empty()) return 0.0;
    const std::size_t samples = std::min<std::size_t>(24, candidate.support.size());
    std::size_t passed = 0;
    for (std::size_t serial = 0; serial < samples; ++serial) {
        const std::size_t support_offset = std::min(candidate.support.size() - 1,
            static_cast<std::size_t>((serial + 0.5) * candidate.support.size() / samples));
        const std::size_t query = candidate.support[support_offset];
        std::vector<std::pair<double, std::size_t>> distances;
        distances.reserve(projection.uv.size());
        for (std::size_t i = 0; i < projection.uv.size(); ++i) {
            const double squared = (projection.uv[i] - projection.uv[query]).squaredNorm() +
                std::pow(projection.axial[i] - projection.axial[query], 2.0);
            distances.emplace_back(squared, i);
        }
        const std::size_t neighbours = std::min<std::size_t>(12, distances.size());
        std::nth_element(distances.begin(), distances.begin() + neighbours - 1,
                         distances.end());
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        for (std::size_t k = 0; k < neighbours; ++k) {
            const std::size_t i = distances[k].second;
            center += Eigen::Vector3d(projection.uv[i].x(), projection.uv[i].y(),
                                      projection.axial[i]);
        }
        center /= static_cast<double>(neighbours);
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (std::size_t k = 0; k < neighbours; ++k) {
            const std::size_t i = distances[k].second;
            const Eigen::Vector3d point(projection.uv[i].x(), projection.uv[i].y(),
                                        projection.axial[i]);
            const Eigen::Vector3d delta = point - center;
            covariance += delta * delta.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success) continue;
        const Eigen::Vector3d normal = solver.eigenvectors().col(0);
        Eigen::Vector3d radial(projection.uv[query].x(), projection.uv[query].y(), 0.0);
        const Eigen::Vector3d axis_point = candidate.axis_point;
        radial.x() -= axis_point.dot(projection.e1);
        radial.y() -= axis_point.dot(projection.e2);
        if (radial.head<2>().norm() <= kEpsilon) continue;
        radial.normalize();
        if (std::abs(normal.z()) < 0.35 && std::abs(normal.dot(radial)) > 0.75)
            ++passed;
    }
    return static_cast<double>(passed) / samples;
}

bool cylinder_wins_local_model_competition(const Projection& projection,
                                           const Cylinder& candidate) {
    if (candidate.support.size() < 8) return false;
    std::vector<Eigen::Vector3d> points;
    points.reserve(candidate.support.size());
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double cylinder_squared = 0.0;
    for (const std::size_t index : candidate.support) {
        Eigen::Vector3d point(projection.uv[index].x(), projection.uv[index].y(),
                              projection.axial[index]);
        points.push_back(point);
        center += point;
        const double radial = (projection.uv[index] - Eigen::Vector2d(
            candidate.axis_point.dot(projection.e1),
            candidate.axis_point.dot(projection.e2))).norm();
        cylinder_squared += std::pow(radial - candidate.radius, 2.0);
    }
    center /= static_cast<double>(points.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& point : points) {
        const Eigen::Vector3d delta = point - center;
        covariance += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    const Eigen::Vector3d plane_normal = solver.eigenvectors().col(0);
    double plane_squared = 0.0;
    for (const auto& point : points)
        plane_squared += std::pow(plane_normal.dot(point - center), 2.0);

    Eigen::MatrixXd design(points.size(), 4);
    Eigen::VectorXd target(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        design.row(static_cast<Eigen::Index>(i)) << 2.0 * points[i].transpose(), 1.0;
        target(static_cast<Eigen::Index>(i)) = points[i].squaredNorm();
    }
    const Eigen::Vector4d parameters = design.colPivHouseholderQr().solve(target);
    const double squared_radius = parameters.head<3>().squaredNorm() + parameters(3);
    double sphere_squared = std::numeric_limits<double>::infinity();
    if (parameters.allFinite() && squared_radius > kEpsilon) {
        const double radius = std::sqrt(squared_radius);
        sphere_squared = 0.0;
        for (const auto& point : points)
            sphere_squared += std::pow((point - parameters.head<3>()).norm() - radius, 2.0);
    }
    const double cylinder_rmse = std::sqrt(cylinder_squared / points.size());
    const double plane_rmse = std::sqrt(plane_squared / points.size());
    const double sphere_rmse = std::sqrt(sphere_squared / points.size());
    // This is a conservative proposal gate, not a hard surface classifier:
    // reject only when an equally supported simpler model is clearly better.
    return cylinder_rmse <= 0.85 * plane_rmse &&
           cylinder_rmse <= 0.85 * sphere_rmse;
}

}  // namespace

std::vector<Cylinder> detect(const std::vector<Eigen::Vector3d>& points,
                             const Options& options) {
    if (points.size() < options.minimum_support || points.size() < 3) return {};
    const double scale = scene_scale(points);
    const double threshold = options.distance_threshold_fraction * scale;
    const double minimum_radius = options.minimum_radius_fraction * scale;
    const double maximum_radius = options.maximum_radius_fraction * scale;
    std::vector<Cylinder> proposals;
    std::mt19937_64 random(options.random_seed);
    std::uniform_int_distribution<std::size_t> index(0, points.size() - 1);

    for (const auto& direction : hemisphere_directions(options.direction_count)) {
        const Projection projection = project(points, direction);
        Cylinder best;
        for (int hypothesis = 0; hypothesis < options.hypotheses_per_direction;
             ++hypothesis) {
            const std::size_t i = index(random), j = index(random), k = index(random);
            if (i == j || i == k || j == k) continue;
            const Circle circle = circle_from_three(
                projection.uv[i], projection.uv[j], projection.uv[k]);
            if (!circle.valid || circle.radius < minimum_radius ||
                circle.radius > maximum_radius) continue;
            Cylinder candidate = make_candidate(projection, circle, threshold,
                options.angular_bins, options.minimum_support);
            if (candidate.radius < minimum_radius || candidate.radius > maximum_radius ||
                candidate.angular_coverage_degrees <
                    options.minimum_angular_coverage_degrees ||
                candidate.axial_span < 4.0 * threshold) continue;
            if (candidate.score > best.score) best = std::move(candidate);
        }
        if (best.score > 0.0 && best.median_residual < 0.55 * threshold &&
            best.axial_span > 0.35 * best.radius &&
            local_cylinder_fraction(projection, best) >= 0.60 &&
            cylinder_wins_local_model_competition(projection, best))
            proposals.push_back(std::move(best));
    }

    std::sort(proposals.begin(), proposals.end(),
        [](const Cylinder& left, const Cylinder& right) { return left.score > right.score; });
    std::vector<Cylinder> result;
    for (auto& proposal : proposals) {
        bool duplicate = false;
        for (const auto& accepted : result) {
            const double radius_relative = std::abs(proposal.radius - accepted.radius) /
                std::max(proposal.radius, accepted.radius);
            if (axis_angle(proposal, accepted) < 6.0 * kPi / 180.0 &&
                radius_relative < 0.15 && line_distance(proposal, accepted) < 0.03 * scale &&
                support_overlap(proposal, accepted) > 0.35) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) result.push_back(std::move(proposal));
        if (result.size() >= options.maximum_candidates) break;
    }
    return result;
}

}  // namespace pcl_eval::voting
