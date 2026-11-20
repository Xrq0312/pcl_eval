#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pcl_eval::voting {

struct Options {
    int direction_count = 192;
    int hypotheses_per_direction = 256;
    int angular_bins = 24;
    double distance_threshold_fraction = 0.006;
    double minimum_radius_fraction = 0.01;
    double maximum_radius_fraction = 0.50;
    double minimum_angular_coverage_degrees = 120.0;
    std::size_t minimum_support = 80;
    std::size_t maximum_candidates = 24;
    std::uint64_t random_seed = 5489;
};

struct Cylinder {
    Eigen::Vector3d axis_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis_direction = Eigen::Vector3d::UnitZ();
    double radius = 0.0;
    double median_residual = 0.0;
    double angular_coverage_degrees = 0.0;
    double axial_span = 0.0;
    double score = 0.0;
    std::vector<std::size_t> support;
};

// Point-only scene proposal. It does not consume normals, labels, regions, or
// primitive types. Directions are voted globally, circles are generated in
// axis-orthogonal projections, then compatible hypotheses are suppressed.
std::vector<Cylinder> detect(const std::vector<Eigen::Vector3d>& points,
                             const Options& options = {});

}  // namespace pcl_eval::voting
