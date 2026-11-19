#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace pcl_eval {

enum class PrimitiveType { Plane, Sphere, Cylinder };
enum class PipelineStage { GlobalRansac, GlobalLocal, JointOptimization };

struct Config {
    PipelineStage stage = PipelineStage::GlobalLocal;
    double distance_threshold = 0.0;  // 0: infer from median point spacing
    double distance_spacing_factor = 0.35;
    double normal_angle_deg = 25.0;
    int normal_k = 24;
    int min_points = 40;
    int global_iterations = 1800;
    int local_iterations = 1800;
    int pilot_size = 4096;
    int proposal_pool_size = 96;
    int region_k = 12;
    double region_normal_angle_deg = 18.0;
    int max_models = 48;
    int refinement_iterations = 2;
    double joint_support_prior = 0.4;
    double connectivity_spacing_factor = 12.0;
    std::uint64_t seed = 1337;
    int threads = 0;
};

struct OrganizedGrid {
    int width = 0;
    int height = 0;
    // Row-major image cells; -1 marks an invalid depth pixel.
    std::vector<int> pixel_to_point;
};

struct PrimitiveModel {
    PrimitiveType type = PrimitiveType::Plane;
    // plane: nx ny nz d, n.p + d = 0
    // sphere: cx cy cz r
    // cylinder: axis_point xyz, axis_direction xyz, radius
    std::vector<double> coefficients;
    std::vector<int> points;
    double rmse = 0.0;
    double score = 0.0;
    double fit_threshold = 0.0;
    std::string proposal_source;
    int parent_model = -1;
};

struct Timing {
    double normals = 0.0;
    double proposals = 0.0;
    double selection = 0.0;
    double refinement = 0.0;
    double connectivity = 0.0;
    double total = 0.0;
};

struct FitResult {
    std::vector<PrimitiveModel> premerge_models;
    std::vector<PrimitiveModel> models;
    std::vector<int> premerge_labels;
    std::vector<int> labels;
    std::vector<Eigen::Vector3d> normals;
    double spacing = 0.0;
    double distance_threshold = 0.0;
    Timing timing;
};

FitResult fit_primitives(
    const std::vector<Eigen::Vector3d>& points,
    const Config& config,
    const std::vector<Eigen::Vector3d>* supplied_normals = nullptr,
    const OrganizedGrid* organized_grid = nullptr);

std::string to_string(PrimitiveType type);
std::string to_string(PipelineStage stage);
PipelineStage parse_stage(const std::string& value);

}  // namespace pcl_eval

