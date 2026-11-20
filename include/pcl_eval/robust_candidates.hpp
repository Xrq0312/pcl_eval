#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

namespace pcl_eval::candidates {

enum class Estimator { NormalLs, NormalizedTls, PairIrls, PointOnly };
enum class ModelType { Plane, Sphere, Cylinder, Rejected };

struct SphereCandidate {
    bool valid = false;
    Estimator estimator = Estimator::NormalLs;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double radius = 0.0;
    double median_residual = 0.0;
    double trimmed_rmse = 0.0;
    double condition = 0.0;
    double runtime_ms = 0.0;
};

struct CylinderCandidate {
    bool valid = false;
    Estimator estimator = Estimator::NormalLs;
    Eigen::Vector3d axis_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis_direction = Eigen::Vector3d::UnitZ();
    double radius = 0.0;
    double median_residual = 0.0;
    double trimmed_rmse = 0.0;
    double condition = 0.0;
    double runtime_ms = 0.0;
};

struct PlaneCandidate {
    bool valid = false;
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double offset = 0.0;
    double median_residual = 0.0;
    double trimmed_rmse = 0.0;
};

struct ModelScore {
    ModelType type = ModelType::Rejected;
    bool valid = false;
    double bic = 0.0;
    double normalized_rmse = 0.0;
    double stability = 0.0;
    int parameters = 0;
    std::string source;
};

struct CompetitionResult {
    ModelType selected = ModelType::Rejected;
    bool rejected = true;
    double confidence = 0.0;
    std::vector<ModelScore> scores;
};

std::vector<SphereCandidate> generate_sphere_candidates(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs = 8192);

std::vector<CylinderCandidate> generate_cylinder_candidates(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    std::size_t max_pairs = 8192);

PlaneCandidate fit_plane(const std::vector<Eigen::Vector3d>& points);

CompetitionResult compete_models(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals,
    double maximum_normalized_rmse = 0.08,
    double minimum_bic_margin_per_point = 0.02,
    double stability_weight = 0.0,
    double soft_geometry_prior_weight = 0.0);

std::string to_string(Estimator estimator);
std::string to_string(ModelType type);

}  // namespace pcl_eval::candidates
