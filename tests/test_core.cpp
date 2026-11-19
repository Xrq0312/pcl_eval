#include "pcl_eval/primitive_pipeline.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void add_plane(std::vector<Eigen::Vector3d>& points,
               std::vector<Eigen::Vector3d>& normals,
               double x_offset) {
    for (int iy = 0; iy < 18; ++iy) {
        for (int ix = 0; ix < 18; ++ix) {
            points.emplace_back(x_offset + 0.035 * ix, -0.3 + 0.035 * iy, 0.0);
            normals.emplace_back(0.0, 0.0, 1.0);
        }
    }
}

void add_sphere(std::vector<Eigen::Vector3d>& points,
                std::vector<Eigen::Vector3d>& normals) {
    const Eigen::Vector3d center(0.3, 2.0, 0.4);
    const double radius = 0.32;
    for (int latitude = 1; latitude < 18; ++latitude) {
        const double phi = kPi * latitude / 18.0;
        for (int longitude = 0; longitude < 32; ++longitude) {
            const double theta = 2.0 * kPi * longitude / 32.0;
            Eigen::Vector3d normal(std::sin(phi) * std::cos(theta),
                                   std::sin(phi) * std::sin(theta),
                                   std::cos(phi));
            points.push_back(center + radius * normal);
            normals.push_back(normal);
        }
    }
}

void add_cylinder(std::vector<Eigen::Vector3d>& points,
                  std::vector<Eigen::Vector3d>& normals) {
    const Eigen::Vector3d center(2.5, 2.0, 0.0);
    const double radius = 0.24;
    for (int iz = 0; iz < 20; ++iz) {
        const double z = -0.45 + 0.9 * iz / 19.0;
        for (int ia = 0; ia < 32; ++ia) {
            const double angle = 2.0 * kPi * ia / 32.0;
            Eigen::Vector3d normal(std::cos(angle), std::sin(angle), 0.0);
            points.push_back(center + Eigen::Vector3d(radius * normal.x(),
                                                       radius * normal.y(), z));
            normals.push_back(normal);
        }
    }
}

void test_global_local_pipeline() {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    add_plane(points, normals, -2.0);
    add_plane(points, normals, 0.8);  // disjoint but exactly coplanar
    add_sphere(points, normals);
    add_cylinder(points, normals);

    pcl_eval::Config config;
    config.stage = pcl_eval::PipelineStage::GlobalLocal;
    config.distance_threshold = 0.008;
    config.min_points = 70;
    config.global_iterations = 1500;
    config.local_iterations = 1500;
    config.proposal_pool_size = 64;
    config.max_models = 12;
    config.seed = 41;

    const auto result = pcl_eval::fit_primitives(points, config, &normals);
    int planes = 0, spheres = 0, cylinders = 0;
    for (const auto& model : result.models) {
        if (model.type == pcl_eval::PrimitiveType::Plane) ++planes;
        if (model.type == pcl_eval::PrimitiveType::Sphere) ++spheres;
        if (model.type == pcl_eval::PrimitiveType::Cylinder) ++cylinders;
        require(std::isfinite(model.rmse), "model RMSE must be finite");
    }
    require(planes >= 2, "disconnected coplanar instances must stay separate");
    require(spheres >= 1, "sphere should be detected");
    require(cylinders >= 1, "cylinder should be detected");
    const int assigned = static_cast<int>(std::count_if(
        result.labels.begin(), result.labels.end(), [](int label) { return label >= 0; }));
    require(assigned > static_cast<int>(points.size() * 0.9), "expected >90% point coverage");

    const auto repeated = pcl_eval::fit_primitives(points, config, &normals);
    require(result.labels == repeated.labels, "fixed-seed labels must be deterministic");
}

}  // namespace

int main() {
    try {
        test_global_local_pipeline();
        std::cout << "ALL CORE TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
