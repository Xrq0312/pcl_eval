#include "pcl_eval/geometric_features.hpp"
#include "pcl_eval/robust_candidates.hpp"

#include <nlohmann/json.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Arguments {
    fs::path sample_dir;
    int normal_k = 24;
    int orient_k = 12;
    int aso_k = 64;
    int max_points = 4096;
    int min_points = 64;
};

struct Label {
    int instance = -1;
    std::string type;
};

Arguments parse_args(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() {
            if (++i >= argc) throw std::runtime_error("missing value after " + key);
            return std::string(argv[i]);
        };
        if (key == "--sample-dir") result.sample_dir = value();
        else if (key == "--normal-k") result.normal_k = std::stoi(value());
        else if (key == "--orient-k") result.orient_k = std::stoi(value());
        else if (key == "--aso-k") result.aso_k = std::stoi(value());
        else if (key == "--max-points") result.max_points = std::stoi(value());
        else if (key == "--min-points") result.min_points = std::stoi(value());
        else if (key == "--help" || key == "-h") {
            std::cout << "geometry_feature_probe --sample-dir DIR [--normal-k 24] "
                         "[--orient-k 12] [--aso-k 64] [--max-points 4096] "
                         "[--min-points 64]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + key);
    }
    if (result.sample_dir.empty()) throw std::runtime_error("--sample-dir is required");
    return result;
}

std::vector<Eigen::Vector3d> load_points(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::vector<Eigen::Vector3d> points;
    std::string line;
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        Eigen::Vector3d point;
        if (row >> point.x() >> point.y() >> point.z()) points.push_back(point);
    }
    return points;
}

std::vector<Label> load_labels(const fs::path& path, std::size_t point_count) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::vector<Label> labels(point_count);
    std::string line;
    std::getline(stream, line);
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        std::size_t index = 0;
        Label label;
        if (row >> index >> label.instance >> label.type) {
            if (index >= labels.size()) throw std::runtime_error("label index out of range");
            labels[index] = label;
        }
    }
    return labels;
}

std::vector<Eigen::Vector3d> subsample(const std::vector<Eigen::Vector3d>& input,
                                       int max_points) {
    if (static_cast<int>(input.size()) <= max_points) return input;
    std::vector<Eigen::Vector3d> output;
    output.reserve(max_points);
    for (int i = 0; i < max_points; ++i) {
        const std::size_t index = static_cast<std::size_t>(
            (static_cast<double>(i) + 0.5) * input.size() / max_points);
        output.push_back(input[std::min(index, input.size() - 1)]);
    }
    return output;
}

std::vector<Eigen::Vector3d> estimate_oriented_normals(
    const std::vector<Eigen::Vector3d>& points, int normal_k, int orient_k) {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(points.size());
    for (const auto& point : points)
        cloud->emplace_back(static_cast<float>(point.x()),
                            static_cast<float>(point.y()),
                            static_cast<float>(point.z()));
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;
    estimator.setNumberOfThreads(0);
    estimator.setInputCloud(cloud);
    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    estimator.setSearchMethod(tree);
    estimator.setKSearch(std::min<int>(normal_k, static_cast<int>(points.size()) - 1));
    pcl::PointCloud<pcl::Normal> pcl_normals;
    estimator.compute(pcl_normals);

    std::vector<Eigen::Vector3d> normals(points.size(), Eigen::Vector3d::Zero());
    for (std::size_t i = 0; i < points.size(); ++i) {
        normals[i] = Eigen::Vector3d(pcl_normals[i].normal_x,
                                     pcl_normals[i].normal_y,
                                     pcl_normals[i].normal_z);
        if (normals[i].allFinite() && normals[i].norm() > 1e-12) normals[i].normalize();
    }

    // Local sign propagation preserves the normal field of closed curved
    // surfaces, unlike alignment to one global average normal.
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<unsigned char> visited(points.size(), 0);
    struct Edge {
        double confidence;
        int parent;
        int child;
        bool operator<(const Edge& other) const { return confidence < other.confidence; }
    };
    std::priority_queue<Edge> pending;
    std::vector<int> indices(std::max(2, orient_k));
    std::vector<float> distances(indices.size());
    for (std::size_t seed = 0; seed < points.size(); ++seed) {
        if (visited[seed] || normals[seed].norm() < 0.5) continue;
        pending.push({2.0, -1, static_cast<int>(seed)});
        while (!pending.empty()) {
            const Edge edge = pending.top();
            pending.pop();
            const int current = edge.child;
            if (visited[current]) continue;
            if (edge.parent >= 0 && normals[edge.parent].dot(normals[current]) < 0.0)
                normals[current] = -normals[current];
            visited[current] = 1;
            const int found = kdtree.nearestKSearch(
                (*cloud)[current], static_cast<int>(indices.size()), indices, distances);
            for (int k = 1; k < found; ++k) {
                const int next = indices[k];
                if (visited[next] || normals[next].norm() < 0.5) continue;
                pending.push({std::abs(normals[current].dot(normals[next])),
                              current, next});
            }
        }
    }
    return normals;
}

std::unordered_map<int, json> load_primitives(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) return {};
    json root;
    stream >> root;
    std::unordered_map<int, json> result;
    for (const auto& primitive : root.value("primitives", json::array()))
        result[primitive.value("instance_id", -1)] = primitive;
    return result;
}

double vector_angle_deg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double cosine = std::clamp(std::abs(a.normalized().dot(b.normalized())), 0.0, 1.0);
    return std::acos(cosine) * 180.0 / 3.14159265358979323846;
}

double line_distance(const Eigen::Vector3d& point,
                     const Eigen::Vector3d& line_point,
                     const Eigen::Vector3d& line_axis) {
    return ((point - line_point).cross(line_axis.normalized())).norm();
}

json sphere_candidates_json(
    const std::vector<pcl_eval::candidates::SphereCandidate>& candidates) {
    json output = json::array();
    for (const auto& candidate : candidates) {
        output.push_back({
            {"estimator", pcl_eval::candidates::to_string(candidate.estimator)},
            {"valid", candidate.valid}, {"radius", candidate.radius},
            {"center", {candidate.center.x(), candidate.center.y(), candidate.center.z()}},
            {"median_residual", candidate.median_residual},
            {"trimmed_rmse", candidate.trimmed_rmse},
            {"condition", candidate.condition}, {"runtime_ms", candidate.runtime_ms}});
    }
    return output;
}

json cylinder_candidates_json(
    const std::vector<pcl_eval::candidates::CylinderCandidate>& candidates) {
    json output = json::array();
    for (const auto& candidate : candidates) {
        output.push_back({
            {"estimator", pcl_eval::candidates::to_string(candidate.estimator)},
            {"valid", candidate.valid}, {"radius", candidate.radius},
            {"axis_point", {candidate.axis_point.x(), candidate.axis_point.y(),
                            candidate.axis_point.z()}},
            {"axis_direction", {candidate.axis_direction.x(), candidate.axis_direction.y(),
                                candidate.axis_direction.z()}},
            {"median_residual", candidate.median_residual},
            {"trimmed_rmse", candidate.trimmed_rmse},
            {"condition", candidate.condition}, {"runtime_ms", candidate.runtime_ms}});
    }
    return output;
}

json competition_json(const pcl_eval::candidates::CompetitionResult& competition) {
    json scores = json::array();
    for (const auto& score : competition.scores) {
        scores.push_back({{"type", pcl_eval::candidates::to_string(score.type)},
            {"source", score.source}, {"valid", score.valid}, {"bic", score.bic},
            {"normalized_rmse", score.normalized_rmse},
            {"stability", score.stability}, {"parameters", score.parameters}});
    }
    return {{"selected", pcl_eval::candidates::to_string(competition.selected)},
        {"rejected", competition.rejected}, {"confidence", competition.confidence},
        {"scores", scores}};
}

json probe_instance(const std::string& sample_name, int instance,
                    const std::string& type,
                    const std::vector<Eigen::Vector3d>& original_points,
                    const json* ground_truth, const Arguments& args) {
    const auto points = subsample(original_points, args.max_points);
    const auto started = std::chrono::steady_clock::now();
    const auto normals = estimate_oriented_normals(points, args.normal_k, args.orient_k);
    const auto normal_done = std::chrono::steady_clock::now();
    const auto features = pcl_eval::features::compute_patch_features(points, normals);
    const auto feature_done = std::chrono::steady_clock::now();
    Eigen::Vector3d point_mean = Eigen::Vector3d::Zero();
    for (const auto& point : points) point_mean += point;
    point_mean /= static_cast<double>(points.size());
    std::size_t query_index = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
        if ((points[i] - point_mean).squaredNorm() <
            (points[query_index] - point_mean).squaredNorm()) query_index = i;
    std::vector<double> squared_distances;
    squared_distances.reserve(points.size());
    for (const auto& point : points)
        squared_distances.push_back((point - points[query_index]).squaredNorm());
    const std::size_t radius_index = std::min<std::size_t>(
        std::max(4, args.aso_k), squared_distances.size() - 1);
    std::nth_element(squared_distances.begin(),
        squared_distances.begin() + radius_index, squared_distances.end());
    const double aso_radius = 1.001 * std::sqrt(squared_distances[radius_index]);
    const auto aso = pcl_eval::features::estimate_algebraic_shape_operator(
        points[query_index], aso_radius, points, normals);
    const auto aso_done = std::chrono::steady_clock::now();
    const auto sphere_candidates = pcl_eval::candidates::generate_sphere_candidates(
        points, normals);
    const auto cylinder_candidates = pcl_eval::candidates::generate_cylinder_candidates(
        points, normals);
    const auto competition = pcl_eval::candidates::compete_models(
        points, normals, 0.08, 0.02, 1.0, 0.0);
    const auto competition_without_stability = pcl_eval::candidates::compete_models(
        points, normals, 0.08, 0.02, 0.0);
    const auto competition_soft_geometry = pcl_eval::candidates::compete_models(
        points, normals, 0.08, 0.02, 0.0, 0.25);
    const auto competition_combined = pcl_eval::candidates::compete_models(
        points, normals, 0.08, 0.02, 1.0, 0.25);
    const auto candidates_done = std::chrono::steady_clock::now();
    auto milliseconds = [](auto duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };
    json output = {
        {"sample", sample_name}, {"instance_id", instance}, {"type", type},
        {"source_points", original_points.size()}, {"used_points", points.size()},
        {"normal_ms", milliseconds(normal_done - started)},
        {"feature_ms", milliseconds(feature_done - normal_done)},
        {"candidate_and_competition_ms", milliseconds(candidates_done - aso_done)},
        {"aso_ms", milliseconds(aso_done - feature_done)},
        {"aso", {
            {"valid", aso.valid}, {"radius", aso_radius},
            {"k1", aso.curvatures(0)}, {"k2", aso.curvatures(1)},
            {"scaled_abs_k1", std::abs(aso.curvatures(0)) * aso_radius},
            {"abs_curvature_ratio", std::abs(aso.curvatures(1)) /
                std::max(1e-12, std::abs(aso.curvatures(0)))}}},
        {"position_eigenvalues", {
            features.position_spectrum.values(0), features.position_spectrum.values(1),
            features.position_spectrum.values(2)}},
        {"normal_eigenvalues", {
            features.normal_spectrum.values(0), features.normal_spectrum.values(1),
            features.normal_spectrum.values(2)}},
        {"sphere_low_rank", {
            {"valid", features.sphere_low_rank.valid},
            {"radius", std::abs(features.sphere_low_rank.signed_radius)},
            {"normalized_rmse", features.sphere_low_rank.normalized_rmse},
            {"center", {features.sphere_low_rank.anchor.x(),
                         features.sphere_low_rank.anchor.y(),
                         features.sphere_low_rank.anchor.z()}}}},
        {"cylinder_low_rank", {
            {"valid", features.cylinder_low_rank.valid},
            {"radius", std::abs(features.cylinder_low_rank.signed_radius)},
            {"normalized_rmse", features.cylinder_low_rank.normalized_rmse},
            {"axis", {features.cylinder_low_rank.axis.x(),
                       features.cylinder_low_rank.axis.y(),
                       features.cylinder_low_rank.axis.z()}}}},
        {"normal_ls", {
            {"valid", features.normal_ls_shape_operator.valid},
            {"k1", features.normal_ls_shape_operator.curvatures(0)},
            {"k2", features.normal_ls_shape_operator.curvatures(1)},
            {"residual", features.normal_ls_shape_operator.residual},
            {"condition", features.normal_ls_shape_operator.tangent_condition}}},
        {"point_pairs", {
            {"pairs", features.point_pairs.pairs},
            {"normal_change", features.point_pairs.plane_normal_change},
            {"sphere_radius", std::abs(features.point_pairs.sphere_signed_radius)},
            {"sphere_residual", features.point_pairs.sphere_residual},
            {"cylinder_radius", std::abs(features.point_pairs.cylinder_signed_radius)},
            {"cylinder_residual", features.point_pairs.cylinder_residual}}},
        {"robust_sphere_candidates", sphere_candidates_json(sphere_candidates)},
        {"robust_cylinder_candidates", cylinder_candidates_json(cylinder_candidates)},
        {"model_competition", competition_json(competition)}
        ,{"model_competition_without_stability",
            competition_json(competition_without_stability)}
        ,{"model_competition_soft_geometry",
            competition_json(competition_soft_geometry)}
        ,{"model_competition_combined",
            competition_json(competition_combined)}
    };

    if (ground_truth != nullptr && ground_truth->value("has_params", false)) {
        json errors = json::object();
        if (type == "sphere" && ground_truth->contains("radius") &&
            ground_truth->contains("center")) {
            const double radius = ground_truth->at("radius").get<double>();
            const auto center_values = ground_truth->at("center").get<std::vector<double>>();
            const Eigen::Vector3d center(center_values[0], center_values[1], center_values[2]);
            errors["sphere_radius_relative"] =
                std::abs(std::abs(features.sphere_low_rank.signed_radius) - radius) / radius;
            errors["sphere_center_relative"] =
                (features.sphere_low_rank.anchor - center).norm() / radius;
            std::vector<Eigen::Vector3d> oracle_normals;
            oracle_normals.reserve(points.size());
            double angular_error = 0.0;
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Eigen::Vector3d oracle = (points[i] - center).normalized();
                oracle_normals.push_back(oracle);
                angular_error += vector_angle_deg(normals[i], oracle);
            }
            const auto oracle = pcl_eval::features::compute_patch_features(
                points, oracle_normals);
            errors["estimated_normal_mean_abs_degrees"] = angular_error / points.size();
            output["oracle_normal_low_rank"] = {
                {"radius", std::abs(oracle.sphere_low_rank.signed_radius)},
                {"radius_relative_error",
                    std::abs(std::abs(oracle.sphere_low_rank.signed_radius) - radius) / radius},
                {"center_relative_error",
                    (oracle.sphere_low_rank.anchor - center).norm() / radius},
                {"normalized_rmse", oracle.sphere_low_rank.normalized_rmse}};
            for (std::size_t i = 0; i < sphere_candidates.size(); ++i) {
                output["robust_sphere_candidates"][i]["radius_relative_error"] =
                    std::abs(sphere_candidates[i].radius - radius) / radius;
                output["robust_sphere_candidates"][i]["center_relative_error"] =
                    (sphere_candidates[i].center - center).norm() / radius;
            }
        } else if (type == "cylinder" && ground_truth->contains("radius") &&
                   ground_truth->contains("axis_direction") &&
                   ground_truth->contains("axis_point")) {
            const double radius = ground_truth->at("radius").get<double>();
            const auto av = ground_truth->at("axis_direction").get<std::vector<double>>();
            const auto pv = ground_truth->at("axis_point").get<std::vector<double>>();
            const Eigen::Vector3d axis(av[0], av[1], av[2]);
            const Eigen::Vector3d axis_point(pv[0], pv[1], pv[2]);
            errors["cylinder_radius_relative"] =
                std::abs(std::abs(features.cylinder_low_rank.signed_radius) - radius) / radius;
            errors["cylinder_axis_degrees"] =
                vector_angle_deg(features.cylinder_low_rank.axis, axis);
            errors["cylinder_axis_offset_relative"] =
                line_distance(features.cylinder_low_rank.anchor, axis_point, axis) / radius;
            std::vector<Eigen::Vector3d> oracle_normals;
            oracle_normals.reserve(points.size());
            double angular_error = 0.0;
            const Eigen::Vector3d unit_axis = axis.normalized();
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Eigen::Vector3d delta = points[i] - axis_point;
                const Eigen::Vector3d oracle =
                    (delta - delta.dot(unit_axis) * unit_axis).normalized();
                oracle_normals.push_back(oracle);
                angular_error += vector_angle_deg(normals[i], oracle);
            }
            const auto oracle = pcl_eval::features::compute_patch_features(
                points, oracle_normals);
            errors["estimated_normal_mean_abs_degrees"] = angular_error / points.size();
            output["oracle_normal_low_rank"] = {
                {"radius", std::abs(oracle.cylinder_low_rank.signed_radius)},
                {"radius_relative_error",
                    std::abs(std::abs(oracle.cylinder_low_rank.signed_radius) - radius) / radius},
                {"axis_degrees", vector_angle_deg(oracle.cylinder_low_rank.axis, axis)},
                {"axis_offset_relative",
                    line_distance(oracle.cylinder_low_rank.anchor, axis_point, axis) / radius},
                {"normalized_rmse", oracle.cylinder_low_rank.normalized_rmse}};
            for (std::size_t i = 0; i < cylinder_candidates.size(); ++i) {
                output["robust_cylinder_candidates"][i]["radius_relative_error"] =
                    std::abs(cylinder_candidates[i].radius - radius) / radius;
                output["robust_cylinder_candidates"][i]["axis_degrees"] =
                    vector_angle_deg(cylinder_candidates[i].axis_direction, axis);
                output["robust_cylinder_candidates"][i]["axis_offset_relative"] =
                    line_distance(cylinder_candidates[i].axis_point, axis_point, axis) / radius;
            }
        }
        output["parameter_errors"] = errors;
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        const auto points = load_points(args.sample_dir / "points.txt");
        const auto labels = load_labels(args.sample_dir / "gt_labels.csv", points.size());
        const auto ground_truth = load_primitives(args.sample_dir / "gt_primitives.json");
        std::map<std::pair<int, std::string>, std::vector<Eigen::Vector3d>> groups;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto& label = labels[i];
            if (label.instance < 0) continue;
            if (label.type != "plane" && label.type != "sphere" && label.type != "cylinder")
                continue;
            groups[{label.instance, label.type}].push_back(points[i]);
        }
        json records = json::array();
        for (const auto& [key, group_points] : groups) {
            if (static_cast<int>(group_points.size()) < args.min_points) continue;
            const auto gt = ground_truth.find(key.first);
            const json* gt_pointer = gt == ground_truth.end() ? nullptr : &gt->second;
            records.push_back(probe_instance(args.sample_dir.filename().string(),
                key.first, key.second, group_points, gt_pointer, args));
        }
        std::cout << std::setw(2) << json{{"schema_version", 1},
            {"evaluation_mode", "oracle_instance_patch"},
            {"normal_policy", "PCL PCA kNN followed by local sign propagation"},
            {"records", records}} << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "feature probe failed: " << error.what() << '\n';
        return 1;
    }
}
