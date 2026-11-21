#include "pcl_eval/primitive_pipeline.hpp"

#include <nlohmann/json.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Arguments {
    fs::path input;
    fs::path pixel_map;
    fs::path output_prefix;
    pcl_eval::Config config;
};

struct LoadedCloud {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    bool has_normals = false;
};

void usage() {
    std::cout
        << "native_primitive_fit --input points.txt|cloud.pcd --output-prefix PATH\n"
        << "  [--stage global_ransac|global_local|joint_opt|boundary_aware|adaptive_local|balanced_select] [--distance VALUE]\n"
        << "  [--pixel-map pixel_map.txt]\n"
        << "  [--min-points N] [--small-min-points N] [--global-iterations N]\n"
        << "  [--local-iterations N] [--adaptive-local-anchors N]\n"
        << "  [--max-refit-rmse-factor VALUE] (0 disables the post-refit gate)\n"
        << "  [--normal-k N] [--seed N] [--threads N]\n";
}

template <typename T>
T parse_number(const std::string& text, const char* option) {
    std::istringstream stream(text);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + text);
    }
    return value;
}

Arguments parse_args(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + key);
            return argv[i];
        };
        if (key == "--input") args.input = value();
        else if (key == "--pixel-map") args.pixel_map = value();
        else if (key == "--output-prefix") args.output_prefix = value();
        else if (key == "--stage") args.config.stage = pcl_eval::parse_stage(value());
        else if (key == "--distance") args.config.distance_threshold = parse_number<double>(value(), "--distance");
        else if (key == "--min-points") args.config.min_points = parse_number<int>(value(), "--min-points");
        else if (key == "--small-min-points")
            args.config.small_min_points = parse_number<int>(value(), "--small-min-points");
        else if (key == "--global-iterations") args.config.global_iterations = parse_number<int>(value(), "--global-iterations");
        else if (key == "--local-iterations") args.config.local_iterations = parse_number<int>(value(), "--local-iterations");
        else if (key == "--adaptive-local-anchors")
            args.config.adaptive_local_anchors = parse_number<int>(value(), "--adaptive-local-anchors");
        else if (key == "--normal-k") args.config.normal_k = parse_number<int>(value(), "--normal-k");
        else if (key == "--max-refit-rmse-factor")
            args.config.max_refit_rmse_factor = parse_number<double>(
                value(), "--max-refit-rmse-factor");
        else if (key == "--seed") args.config.seed = parse_number<std::uint64_t>(value(), "--seed");
        else if (key == "--threads") args.config.threads = parse_number<int>(value(), "--threads");
        else if (key == "--help" || key == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown option: " + key);
    }
    if (args.input.empty() || args.output_prefix.empty()) {
        usage();
        throw std::runtime_error("--input and --output-prefix are required");
    }
    return args;
}

pcl_eval::OrganizedGrid load_pixel_map(const fs::path& path, std::size_t point_count) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open pixel map: " + path.string());
    pcl_eval::OrganizedGrid grid;
    if (!(stream >> grid.width >> grid.height) || grid.width <= 0 || grid.height <= 0)
        throw std::runtime_error("invalid pixel map header: " + path.string());
    grid.pixel_to_point.resize(static_cast<std::size_t>(grid.width) * grid.height);
    std::vector<unsigned char> seen(point_count, 0);
    for (int& index : grid.pixel_to_point) {
        if (!(stream >> index)) throw std::runtime_error("truncated pixel map: " + path.string());
        if (index < -1 || index >= static_cast<int>(point_count))
            throw std::runtime_error("pixel map index out of range");
        if (index >= 0) {
            if (seen[index]) throw std::runtime_error("duplicate point in pixel map");
            seen[index] = 1;
        }
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end())
        throw std::runtime_error("pixel map does not reference every input point");
    return grid;
}

LoadedCloud load_cloud(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    LoadedCloud result;
    if (extension == ".ply") {
        pcl::PointCloud<pcl::PointNormal> cloud;
        if (pcl::io::loadPLYFile(path.string(), cloud) < 0) {
            throw std::runtime_error("cannot load PLY: " + path.string());
        }
        result.points.reserve(cloud.size());
        result.normals.reserve(cloud.size());
        bool all_normals_valid = true;
        for (const auto& point : cloud) {
            const Eigen::Vector3d p(point.x, point.y, point.z);
            if (!p.allFinite()) continue;
            const Eigen::Vector3d n(point.normal_x, point.normal_y, point.normal_z);
            result.points.push_back(p);
            result.normals.push_back(n);
            all_normals_valid = all_normals_valid && n.allFinite() && n.norm() > 1e-12;
        }
        result.has_normals = all_normals_valid &&
            result.normals.size() == result.points.size();
    } else if (extension == ".pcd") {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        if (pcl::io::loadPCDFile(path.string(), cloud) < 0) {
            throw std::runtime_error("cannot load PCD: " + path.string());
        }
        result.points.reserve(cloud.size());
        for (const auto& point : cloud) {
            const Eigen::Vector3d p(point.x, point.y, point.z);
            if (p.allFinite()) result.points.push_back(p);
        }
    } else {
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("cannot open input: " + path.string());
        std::string line;
        while (std::getline(stream, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            Eigen::Vector3d p;
            if (row >> p.x() >> p.y() >> p.z()) {
                if (p.allFinite()) result.points.push_back(p);
            }
        }
    }
    if (result.points.size() < 4)
        throw std::runtime_error("input has fewer than four finite points");
    return result;
}

json model_json(const pcl_eval::PrimitiveModel& model, int id) {
    json output = {
        {"id", id}, {"type", pcl_eval::to_string(model.type)},
        {"points", model.points.size()}, {"regions", 1}, {"rmse", model.rmse},
        {"score", model.score}, {"fit_threshold", model.fit_threshold},
        {"coefficients", model.coefficients},
        {"proposal_source", model.proposal_source}, {"parent_model", model.parent_model}
    };
    if (model.local_scope) {
        output["local_scope_center"] = {
            model.scope_center.x(), model.scope_center.y(), model.scope_center.z()};
        output["local_scope_radius"] = model.scope_radius;
    }
    return output;
}

json model_array_json(const std::vector<pcl_eval::PrimitiveModel>& models) {
    json output = json::array();
    for (std::size_t i = 0; i < models.size(); ++i)
        output.push_back(model_json(models[i], static_cast<int>(i)));
    return output;
}

void write_outputs(const fs::path& prefix,
                   const std::vector<Eigen::Vector3d>& points,
                   const pcl_eval::Config& config,
                   const pcl_eval::FitResult& result) {
    if (!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
    std::vector<int> region(points.size(), -1);

    std::ofstream labels(prefix.string() + "_labels.csv");
    labels << "point_index,x,y,z,normal_x,normal_y,normal_z,region_id,premerge_primitive_id,primitive_id\n";
    labels << std::setprecision(12);
    for (std::size_t i = 0; i < points.size(); ++i) {
        labels << i << ',' << points[i].x() << ',' << points[i].y() << ','
               << points[i].z() << ',' << result.normals[i].x() << ','
               << result.normals[i].y() << ',' << result.normals[i].z() << ','
               << region[i] << ','
               << result.premerge_labels[i] << ',' << result.labels[i] << '\n';
    }

    std::ofstream models(prefix.string() + "_models.csv");
    models << "primitive_id,type,points,merged_regions,rmse,coefficients\n";
    models << std::setprecision(12);
    for (std::size_t i = 0; i < result.models.size(); ++i) {
        const auto& model = result.models[i];
        models << i << ',' << pcl_eval::to_string(model.type) << ','
               << model.points.size() << ",1," << model.rmse << ",\"";
        for (std::size_t j = 0; j < model.coefficients.size(); ++j) {
            if (j) models << ' ';
            models << model.coefficients[j];
        }
        models << "\"\n";
    }

    const json premerge = model_array_json(result.premerge_models);
    const json final = model_array_json(result.models);
    const json metadata = {
        {"implementation", "independent-stage1-global-local"},
        {"config", {
            {"stage", pcl_eval::to_string(config.stage)},
            {"distance", result.distance_threshold}, {"spacing", result.spacing},
            {"min_points", config.min_points}, {"normal_k", config.normal_k},
            {"small_min_points", config.small_min_points},
            {"adaptive_local_anchors", config.adaptive_local_anchors},
            {"normal_angle_deg", config.normal_angle_deg},
            {"global_iterations", config.global_iterations},
            {"local_iterations", config.local_iterations}, {"seed", config.seed},
            {"max_refit_rmse_factor", config.max_refit_rmse_factor}
        }},
        {"num_points", points.size()},
        {"diagnostics", {
            {"screened_proposals", model_array_json(result.screened_proposals)},
            {"refit_failed_proposals", model_array_json(result.refit_failed_proposals)},
            {"filtered_proposals", model_array_json(result.filtered_proposals)},
            {"proposals", model_array_json(result.proposals)},
            {"selected_models", model_array_json(result.selected_models)}
        }},
        {"premerge_models", premerge}, {"models", final},
        {"failed_regions", json::array()},
        {"timing_seconds", {
            {"normals", result.timing.normals}, {"proposals", result.timing.proposals},
            {"selection", result.timing.selection}, {"refinement", result.timing.refinement},
            {"connectivity", result.timing.connectivity}, {"total", result.timing.total}
        }}
    };
    std::ofstream primitives(prefix.string() + "_primitives.json");
    primitives << std::setw(2) << metadata << '\n';

    pcl::PointCloud<pcl::PointXYZRGB> segmented;
    segmented.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = static_cast<float>(points[i].x());
        point.y = static_cast<float>(points[i].y());
        point.z = static_cast<float>(points[i].z());
        const int label = result.labels[i];
        if (label < 0) {
            point.r = 70; point.g = 70; point.b = 70;
        } else {
            const std::uint32_t hash = 0x9e3779b9u * static_cast<std::uint32_t>(label + 1);
            point.r = static_cast<std::uint8_t>(55 + (hash & 0x9f));
            point.g = static_cast<std::uint8_t>(55 + ((hash >> 8) & 0x9f));
            point.b = static_cast<std::uint8_t>(55 + ((hash >> 16) & 0x9f));
        }
        segmented.push_back(point);
    }
    segmented.width = static_cast<std::uint32_t>(segmented.size());
    segmented.height = 1;
    if (pcl::io::savePLYFileBinary(prefix.string() + "_segmented.ply", segmented) < 0)
        throw std::runtime_error("cannot write segmented PLY");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_args(argc, argv);
        const auto loaded = load_cloud(arguments.input);
        pcl_eval::OrganizedGrid grid;
        const pcl_eval::OrganizedGrid* grid_ptr = nullptr;
        if (!arguments.pixel_map.empty()) {
            grid = load_pixel_map(arguments.pixel_map, loaded.points.size());
            grid_ptr = &grid;
        }
        const auto* normals = loaded.has_normals ? &loaded.normals : nullptr;
        const auto result = pcl_eval::fit_primitives(
            loaded.points, arguments.config, normals, grid_ptr);
        write_outputs(arguments.output_prefix, loaded.points, arguments.config, result);
        std::cout << "stage=" << pcl_eval::to_string(arguments.config.stage)
                  << " points=" << loaded.points.size() << " models=" << result.models.size()
                  << " distance=" << result.distance_threshold
                  << " seconds=" << result.timing.total << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
