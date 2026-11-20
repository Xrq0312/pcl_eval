#include "pcl_eval/geometric_features.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_exact_sphere() {
    const Eigen::Vector3d center(1.2, -0.7, 2.5);
    constexpr double radius = 0.8;
    std::vector<Eigen::Vector3d> points, normals;
    for (int latitude = 2; latitude < 18; ++latitude) {
        for (int longitude = 0; longitude < 28; ++longitude) {
            const double phi = kPi * latitude / 20.0;
            const double theta = 2.0 * kPi * longitude / 28.0;
            Eigen::Vector3d n(std::sin(phi) * std::cos(theta),
                              std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(center + radius * n);
            normals.push_back(n);
        }
    }
    const auto features = pcl_eval::features::compute_patch_features(points, normals);
    require(features.sphere_low_rank.valid, "sphere low-rank estimate invalid");
    require(std::abs(std::abs(features.sphere_low_rank.signed_radius) - radius) < 1e-9,
            "sphere radius must be exact");
    require((features.sphere_low_rank.anchor - center).norm() < 1e-9,
            "sphere center must be exact");
    require(features.sphere_low_rank.rmse < 1e-9, "sphere q must be rank zero");
    require(features.point_pairs.sphere_residual < 1e-9,
            "sphere pair invariant must be exact");
    const auto aso = pcl_eval::features::estimate_algebraic_shape_operator(
        points[points.size() / 2], 0.55, points, normals);
    require(aso.valid, "ASO sphere estimate invalid");
    require(std::abs(std::abs(aso.curvatures(0)) - 1.0 / radius) < 0.12,
            "ASO sphere k1 inaccurate");
    require(std::abs(std::abs(aso.curvatures(1)) - 1.0 / radius) < 0.12,
            "ASO sphere k2 inaccurate");
}

void test_exact_cylinder() {
    const Eigen::Vector3d axis_point(-0.4, 0.9, 1.1);
    const Eigen::Vector3d axis = Eigen::Vector3d(0.3, -0.2, 0.93).normalized();
    const Eigen::Vector3d e1 = axis.unitOrthogonal();
    const Eigen::Vector3d e2 = axis.cross(e1).normalized();
    constexpr double radius = 0.35;
    std::vector<Eigen::Vector3d> points, normals;
    for (int iz = 0; iz < 17; ++iz) {
        const double z = -0.7 + 1.4 * iz / 16.0;
        for (int ia = 0; ia < 32; ++ia) {
            const double angle = 2.0 * kPi * ia / 32.0;
            const Eigen::Vector3d n = std::cos(angle) * e1 + std::sin(angle) * e2;
            points.push_back(axis_point + z * axis + radius * n);
            normals.push_back(n);
        }
    }
    const auto features = pcl_eval::features::compute_patch_features(points, normals);
    require(features.cylinder_low_rank.valid, "cylinder low-rank estimate invalid");
    require(std::abs(std::abs(features.cylinder_low_rank.signed_radius) - radius) < 1e-9,
            "cylinder radius must be exact");
    require(std::abs(features.cylinder_low_rank.axis.dot(axis)) > 1.0 - 1e-9,
            "cylinder axis must be exact");
    require(features.cylinder_low_rank.rmse < 1e-7, "cylinder q must be rank one");
    require(features.point_pairs.cylinder_residual < 1e-9,
            "cylinder pair invariant must be exact");
    const auto aso = pcl_eval::features::estimate_algebraic_shape_operator(
        points[points.size() / 2], 0.45, points, normals);
    require(aso.valid, "ASO cylinder estimate invalid");
    require(std::abs(std::abs(aso.curvatures(0)) - 1.0 / radius) < 0.30,
            "ASO cylinder k1 inaccurate");
    require(std::abs(aso.curvatures(1)) < 0.15,
            "ASO cylinder k2 should be zero");
}

void test_plane() {
    std::vector<Eigen::Vector3d> points, normals;
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            points.emplace_back(0.03 * x, 0.03 * y, 1.2);
            normals.emplace_back(0, 0, 1);
        }
    }
    const auto features = pcl_eval::features::compute_patch_features(points, normals);
    require(features.position_spectrum.values(0) < 1e-12,
            "plane position covariance must be rank two");
    require(features.point_pairs.plane_normal_change < 1e-12,
            "plane normals should be constant");
    require(features.normal_ls_shape_operator.valid,
            "plane shape operator estimate invalid");
    require(features.normal_ls_shape_operator.curvatures.norm() < 1e-12,
            "plane curvatures must be zero");
}

void test_multiscale_aso_and_curvature_boundary() {
    const Eigen::Vector3d center = Eigen::Vector3d::Zero();
    constexpr double radius = 2.0;
    std::vector<Eigen::Vector3d> points, normals;
    for (int latitude = 1; latitude < 25; ++latitude) {
        const double phi = kPi * latitude / 25.0;
        for (int longitude = 0; longitude < 48; ++longitude) {
            const double theta = 2.0 * kPi * longitude / 48.0;
            const Eigen::Vector3d normal(std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(center + radius * normal);
            normals.push_back(normal);
        }
    }
    const std::size_t query = points.size() / 2;
    const auto scales = pcl_eval::features::estimate_multiscale_aso(
        query, {24, 48, 96}, points, normals);
    require(scales.size() == 3, "expected three ASO scales");
    for (const auto& scale : scales) {
        require(scale.aso.valid, "multiscale ASO invalid");
        require(std::abs(std::abs(scale.aso.curvatures(0)) - 1.0 / radius) < 0.08,
                "multiscale sphere k1 inaccurate");
        require(std::abs(std::abs(scale.aso.curvatures(1)) - 1.0 / radius) < 0.08,
                "multiscale sphere k2 inaccurate");
    }

    std::size_t neighbor = query + 1;
    const auto smooth = pcl_eval::features::estimate_curvature_boundary(
        points[query], normals[query], points[neighbor], normals[neighbor],
        scales.front().aso);
    require(smooth.valid && smooth.residual < 0.02,
            "smooth sphere normal change should be predicted by ASO");

    const Eigen::AngleAxisd crease(30.0 * kPi / 180.0,
        scales.front().aso.directions.col(0).normalized());
    const auto boundary = pcl_eval::features::estimate_curvature_boundary(
        points[query], normals[query], points[neighbor],
        crease * normals[neighbor], scales.front().aso);
    require(boundary.valid && boundary.residual > 0.35,
            "unexplained crease normal jump should yield a boundary residual");
}

}  // namespace

int main() {
    try {
        test_exact_sphere();
        test_exact_cylinder();
        test_plane();
        test_multiscale_aso_and_curvature_boundary();
        std::cout << "ALL GEOMETRIC FEATURE TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
