#include "pcl_eval/cylinder_voting.hpp"

#include <nlohmann/json.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Label { int instance = -1; std::string type; };

struct Arguments {
    fs::path sample_dir;
    int max_points = 5000;
    int directions = 192;
    int hypotheses = 256;
    int minimum_support = 80;
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
        else if (key == "--max-points") result.max_points = std::stoi(value());
        else if (key == "--directions") result.directions = std::stoi(value());
        else if (key == "--hypotheses") result.hypotheses = std::stoi(value());
        else if (key == "--minimum-support") result.minimum_support = std::stoi(value());
        else throw std::runtime_error("unknown option: " + key);
    }
    if (result.sample_dir.empty()) throw std::runtime_error("--sample-dir is required");
    return result;
}

std::vector<Eigen::Vector3d> load_points(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::vector<Eigen::Vector3d> result;
    std::string line;
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        Eigen::Vector3d point;
        if (row >> point.x() >> point.y() >> point.z()) result.push_back(point);
    }
    return result;
}

std::vector<Label> load_labels(const fs::path& path, std::size_t count) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::vector<Label> result(count);
    std::string line;
    std::getline(stream, line);
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        std::size_t index = 0;
        Label label;
        if (row >> index >> label.instance >> label.type && index < count)
            result[index] = label;
    }
    return result;
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

double angle_degrees(const Eigen::Vector3d& left, const Eigen::Vector3d& right) {
    return std::acos(std::clamp(std::abs(left.normalized().dot(right.normalized())),
        0.0, 1.0)) * 180.0 / kPi;
}

double line_distance(const Eigen::Vector3d& point, const Eigen::Vector3d& line_point,
                     const Eigen::Vector3d& axis) {
    return ((point - line_point).cross(axis.normalized())).norm();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        const auto all_points = load_points(args.sample_dir / "points.txt");
        const auto all_labels = load_labels(args.sample_dir / "gt_labels.csv", all_points.size());
        const auto primitives = load_primitives(args.sample_dir / "gt_primitives.json");
        std::vector<Eigen::Vector3d> points;
        std::vector<Label> labels;
        const std::size_t used = std::min<std::size_t>(all_points.size(), args.max_points);
        points.reserve(used);
        labels.reserve(used);
        for (std::size_t i = 0; i < used; ++i) {
            const std::size_t index = std::min(all_points.size() - 1,
                static_cast<std::size_t>((i + 0.5) * all_points.size() / used));
            points.push_back(all_points[index]);
            labels.push_back(all_labels[index]);
        }
        std::map<int, std::size_t> gt_counts;
        for (const auto& label : labels)
            if (label.instance >= 0 && label.type == "cylinder") ++gt_counts[label.instance];

        pcl_eval::voting::Options options;
        options.direction_count = args.directions;
        options.hypotheses_per_direction = args.hypotheses;
        options.minimum_support = std::min<std::size_t>(args.minimum_support,
            std::max<std::size_t>(12, points.size() / 4));
        const auto started = std::chrono::steady_clock::now();
        const auto detected = pcl_eval::voting::detect(points, options);
        const double runtime_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        json candidates = json::array();
        for (const auto& candidate : detected) {
            int best_instance = -1;
            double best_iou = 0.0;
            for (const auto& [instance, gt_count] : gt_counts) {
                const std::size_t intersection = std::count_if(candidate.support.begin(),
                    candidate.support.end(), [&](std::size_t index) {
                        return labels[index].instance == instance && labels[index].type == "cylinder";
                    });
                const double iou = static_cast<double>(intersection) /
                    std::max<std::size_t>(1, candidate.support.size() + gt_count - intersection);
                if (iou > best_iou) { best_iou = iou; best_instance = instance; }
            }
            json row = {{"axis_point", {candidate.axis_point.x(), candidate.axis_point.y(),
                         candidate.axis_point.z()}},
                {"axis_direction", {candidate.axis_direction.x(), candidate.axis_direction.y(),
                                    candidate.axis_direction.z()}},
                {"radius", candidate.radius}, {"support", candidate.support.size()},
                {"median_residual", candidate.median_residual},
                {"angular_coverage_degrees", candidate.angular_coverage_degrees},
                {"axial_span", candidate.axial_span}, {"score", candidate.score},
                {"best_gt_instance", best_instance}, {"best_gt_iou", best_iou}};
            const auto gt = primitives.find(best_instance);
            if (gt != primitives.end() && gt->second.value("has_params", false) &&
                gt->second.contains("radius") && gt->second.contains("axis_direction") &&
                gt->second.contains("axis_point")) {
                const double radius = gt->second.at("radius").get<double>();
                const auto av = gt->second.at("axis_direction").get<std::vector<double>>();
                const auto pv = gt->second.at("axis_point").get<std::vector<double>>();
                const Eigen::Vector3d axis(av[0], av[1], av[2]);
                const Eigen::Vector3d axis_point(pv[0], pv[1], pv[2]);
                row["radius_relative_error"] = std::abs(candidate.radius - radius) / radius;
                row["axis_degrees"] = angle_degrees(candidate.axis_direction, axis);
                row["axis_offset_relative"] =
                    line_distance(candidate.axis_point, axis_point, axis) / radius;
            }
            candidates.push_back(std::move(row));
        }
        json gt_instances = json::array();
        for (const auto& [instance, count] : gt_counts)
            gt_instances.push_back({{"instance_id", instance}, {"sampled_points", count}});
        std::cout << std::setw(2) << json{{"schema_version", 1},
            {"evaluation_mode", "full_scene_point_only_candidate_detection"},
            {"candidate_inputs", "xyz_only_no_normals_no_gt_no_regions"},
            {"sample", args.sample_dir.filename().string()},
            {"source_points", all_points.size()}, {"used_points", points.size()},
            {"runtime_ms", runtime_ms}, {"gt_cylinders", gt_instances},
            {"candidates", candidates}} << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cylinder voting probe failed: " << error.what() << '\n';
        return 1;
    }
}
