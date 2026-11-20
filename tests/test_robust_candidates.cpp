#include "pcl_eval/robust_candidates.hpp"

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

void add_sphere(std::vector<Eigen::Vector3d>& points,
                std::vector<Eigen::Vector3d>& normals, double flip_fraction) {
    const Eigen::Vector3d center(0.4, -0.2, 0.7);
    constexpr double radius = 0.8;
    int serial = 0;
    for (int latitude = 2; latitude < 20; ++latitude)
        for (int longitude = 0; longitude < 36; ++longitude, ++serial) {
            const double phi = kPi * latitude / 22.0;
            const double theta = 2.0 * kPi * longitude / 36.0;
            Eigen::Vector3d normal(std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(center + radius * normal);
            if (flip_fraction > 0.0 && serial % static_cast<int>(1.0 / flip_fraction) == 0)
                normal = -normal;
            normals.push_back(normal);
        }
}

void add_cylinder(std::vector<Eigen::Vector3d>& points,
                  std::vector<Eigen::Vector3d>& normals, double flip_fraction) {
    const Eigen::Vector3d axis = Eigen::Vector3d(0.2, -0.4, 0.89).normalized();
    const Eigen::Vector3d e1 = axis.unitOrthogonal();
    const Eigen::Vector3d e2 = axis.cross(e1);
    constexpr double radius = 0.35;
    int serial = 0;
    for (int z = 0; z < 24; ++z)
        for (int angle_index = 0; angle_index < 40; ++angle_index, ++serial) {
            const double angle = 2.0 * kPi * angle_index / 40.0;
            Eigen::Vector3d normal = std::cos(angle) * e1 + std::sin(angle) * e2;
            points.push_back((z / 23.0 - 0.5) * axis + radius * normal);
            if (flip_fraction > 0.0 && serial % static_cast<int>(1.0 / flip_fraction) == 0)
                normal = -normal;
            normals.push_back(normal);
        }
}

void test_robust_sphere_flips() {
    std::vector<Eigen::Vector3d> points, normals;
    add_sphere(points, normals, 0.10);
    const auto candidates = pcl_eval::candidates::generate_sphere_candidates(points, normals);
    require(candidates.size() == 4, "expected four sphere candidate methods");
    require(candidates[0].valid && candidates[2].valid && candidates[3].valid,
            "sphere candidates invalid");
    const double ls_error = std::abs(candidates[0].radius - 0.8);
    const double robust_error = std::abs(candidates[2].radius - 0.8);
    const double point_only_error = std::abs(candidates[3].radius - 0.8);
    require(robust_error < ls_error, "pair IRLS should improve flipped-normal sphere LS");
    require(robust_error < 0.03, "robust sphere radius inaccurate");
    require(point_only_error < 1e-8, "point-only sphere should ignore normal flips");
}

void test_robust_cylinder_flips() {
    std::vector<Eigen::Vector3d> points, normals;
    add_cylinder(points, normals, 0.10);
    const auto candidates = pcl_eval::candidates::generate_cylinder_candidates(points, normals);
    require(candidates.size() == 3, "expected three cylinder candidate methods");
    require(candidates[2].valid, "robust cylinder candidate invalid");
    require(std::abs(candidates[2].radius - 0.35) < 0.03,
            "robust cylinder radius inaccurate under flips");
}

void test_model_competition() {
    std::vector<Eigen::Vector3d> points, normals;
    add_sphere(points, normals, 0.05);
    const auto sphere = pcl_eval::candidates::compete_models(points, normals);
    if (sphere.selected != pcl_eval::candidates::ModelType::Sphere) {
        std::cerr << "sphere competition selected "
                  << pcl_eval::candidates::to_string(sphere.selected) << '\n';
        for (const auto& score : sphere.scores)
            std::cerr << "  " << pcl_eval::candidates::to_string(score.type)
                      << '/' << score.source << " bic=" << score.bic
                      << " nrmse=" << score.normalized_rmse << '\n';
    }
    require(sphere.selected == pcl_eval::candidates::ModelType::Sphere,
            "competition should select sphere");

    points.clear(); normals.clear();
    add_cylinder(points, normals, 0.05);
    const auto cylinder = pcl_eval::candidates::compete_models(points, normals);
    require(cylinder.selected == pcl_eval::candidates::ModelType::Cylinder,
            "competition should select cylinder");

    points.clear(); normals.clear();
    for (int y = 0; y < 25; ++y)
        for (int x = 0; x < 25; ++x) {
            points.emplace_back(0.03 * x, 0.03 * y, 1.2);
            normals.emplace_back(0.0, 0.0, 1.0);
        }
    const auto plane = pcl_eval::candidates::compete_models(points, normals);
    if (plane.selected != pcl_eval::candidates::ModelType::Plane) {
        std::cerr << "plane competition selected "
                  << pcl_eval::candidates::to_string(plane.selected) << '\n';
        for (const auto& score : plane.scores)
            std::cerr << "  " << pcl_eval::candidates::to_string(score.type)
                      << '/' << score.source << " bic=" << score.bic
                      << " nrmse=" << score.normalized_rmse << '\n';
    }
    require(plane.selected == pcl_eval::candidates::ModelType::Plane,
            "competition should select plane");
}

}  // namespace

int main() {
    try {
        test_robust_sphere_flips();
        test_robust_cylinder_flips();
        test_model_competition();
        std::cout << "ALL ROBUST CANDIDATE TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
