#include "pcl_eval/geometric_features.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Sample {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
};

Eigen::Vector3d noisy_normal(const Eigen::Vector3d& normal, double degrees,
                             std::mt19937_64& random) {
    if (degrees == 0.0) return normal;
    std::normal_distribution<double> standard(0.0, 1.0);
    Eigen::Vector3d tangent(standard(random), standard(random), standard(random));
    tangent -= tangent.dot(normal) * normal;
    if (tangent.norm() < 1e-12) tangent = normal.unitOrthogonal();
    tangent.normalize();
    const double angle = standard(random) * degrees * kPi / 180.0;
    return (std::cos(angle) * normal + std::sin(angle) * tangent).normalized();
}

void perturb(Sample& sample, double position_sigma, double normal_degrees,
             double flip_fraction, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::normal_distribution<double> gaussian(0.0, position_sigma);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (std::size_t i = 0; i < sample.points.size(); ++i) {
        sample.points[i] += Eigen::Vector3d(
            gaussian(random), gaussian(random), gaussian(random));
        sample.normals[i] = noisy_normal(sample.normals[i], normal_degrees, random);
        if (uniform(random) < flip_fraction) sample.normals[i] = -sample.normals[i];
    }
}

Sample sphere_sample(int count, double cap_degrees) {
    Sample result;
    result.points.reserve(count);
    result.normals.reserve(count);
    const double cap = cap_degrees * kPi / 180.0;
    const double min_z = std::cos(cap);
    constexpr double golden = 2.39996322972865332;
    const Eigen::Vector3d center(0.3, -0.2, 0.5);
    for (int i = 0; i < count; ++i) {
        const double z = 1.0 - (1.0 - min_z) * (i + 0.5) / count;
        const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle = golden * i;
        const Eigen::Vector3d normal(radial * std::cos(angle),
                                     radial * std::sin(angle), z);
        result.points.push_back(center + normal);
        result.normals.push_back(normal);
    }
    return result;
}

Sample cylinder_sample(int count, double arc_degrees, double length) {
    Sample result;
    result.points.reserve(count);
    result.normals.reserve(count);
    const int around = std::max(8, static_cast<int>(std::sqrt(count * 2.0)));
    const int along = std::max(2, count / around);
    const double arc = arc_degrees * kPi / 180.0;
    for (int iz = 0; iz < along && static_cast<int>(result.points.size()) < count; ++iz) {
        const double z = length * ((iz + 0.5) / along - 0.5);
        for (int ia = 0; ia < around && static_cast<int>(result.points.size()) < count; ++ia) {
            const double angle = arc * ((ia + 0.5) / around - 0.5);
            const Eigen::Vector3d normal(std::cos(angle), std::sin(angle), 0.0);
            result.points.emplace_back(normal.x(), normal.y(), z);
            result.normals.push_back(normal);
        }
    }
    return result;
}

Sample plane_sample(int count) {
    Sample result;
    const int side = static_cast<int>(std::ceil(std::sqrt(count)));
    for (int y = 0; y < side && static_cast<int>(result.points.size()) < count; ++y) {
        for (int x = 0; x < side && static_cast<int>(result.points.size()) < count; ++x) {
            result.points.emplace_back((x + 0.5) / side - 0.5,
                                       (y + 0.5) / side - 0.5, 0.0);
            result.normals.emplace_back(0.0, 0.0, 1.0);
        }
    }
    return result;
}

double axis_angle(const Eigen::Vector3d& axis) {
    const double cosine = std::clamp(std::abs(axis.normalized().z()), 0.0, 1.0);
    return std::acos(cosine) * 180.0 / kPi;
}

void emit(std::ostream& output, const std::string& type, const std::string& suite,
          int count, double coverage, double length, double position_sigma,
          double normal_degrees, double flip_fraction, std::uint64_t seed) {
    Sample sample;
    if (type == "sphere") sample = sphere_sample(count, coverage);
    else if (type == "cylinder") sample = cylinder_sample(count, coverage, length);
    else sample = plane_sample(count);
    perturb(sample, position_sigma, normal_degrees, flip_fraction, seed);
    const auto started = std::chrono::steady_clock::now();
    const auto f = pcl_eval::features::compute_patch_features(sample.points, sample.normals);
    const double feature_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    Eigen::Vector3d point_mean = Eigen::Vector3d::Zero();
    for (const auto& point : sample.points) point_mean += point;
    point_mean /= static_cast<double>(sample.points.size());
    std::size_t query_index = 0;
    for (std::size_t i = 1; i < sample.points.size(); ++i)
        if ((sample.points[i] - point_mean).squaredNorm() <
            (sample.points[query_index] - point_mean).squaredNorm()) query_index = i;
    const auto aso_started = std::chrono::steady_clock::now();
    const auto aso = pcl_eval::features::estimate_algebraic_shape_operator(
        sample.points[query_index], 0.35, sample.points, sample.normals);
    const double aso_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - aso_started).count();
    const double sphere_radius_error =
        std::abs(std::abs(f.sphere_low_rank.signed_radius) - 1.0);
    const double sphere_center_error =
        (f.sphere_low_rank.anchor - Eigen::Vector3d(0.3, -0.2, 0.5)).norm();
    const double cylinder_radius_error =
        std::abs(std::abs(f.cylinder_low_rank.signed_radius) - 1.0);
    const double cylinder_axis_offset =
        f.cylinder_low_rank.anchor.head<2>().norm();
    output << type << ',' << suite << ',' << count << ',' << coverage << ','
           << length << ',' << position_sigma << ',' << normal_degrees << ','
           << flip_fraction << ',' << seed << ',' << feature_elapsed << ','
           << f.sphere_low_rank.normalized_rmse << ',' << sphere_radius_error << ','
           << sphere_center_error << ',' << f.cylinder_low_rank.normalized_rmse << ','
           << cylinder_radius_error << ',' << axis_angle(f.cylinder_low_rank.axis) << ','
           << cylinder_axis_offset << ','
           << f.normal_ls_shape_operator.curvatures(0) << ','
           << f.normal_ls_shape_operator.curvatures(1) << ','
           << f.normal_ls_shape_operator.residual << ','
           << aso_elapsed << ',' << (aso.valid ? 1 : 0) << ','
           << aso.curvatures(0) << ',' << aso.curvatures(1) << ','
           << f.point_pairs.plane_normal_change << ','
           << f.point_pairs.sphere_residual << ','
           << f.point_pairs.cylinder_residual << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--output" && ++i < argc) output_path = argv[i];
            else if (key == "--help" || key == "-h") {
                std::cout << "geometry_feature_synthetic [--output FILE]\n";
                return 0;
            } else throw std::runtime_error("unknown or incomplete option: " + key);
        }
        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
            file.open(output_path);
            if (!file) throw std::runtime_error("cannot create " + output_path.string());
            output = &file;
        }
        *output << std::setprecision(10)
            << "type,suite,points,coverage_deg,length,position_sigma,normal_sigma_deg,"
               "flip_fraction,seed,runtime_ms,sphere_q_nrmse,sphere_radius_abs_error,"
               "sphere_center_error,cylinder_q_nrmse,cylinder_radius_abs_error,"
               "cylinder_axis_deg,cylinder_axis_offset,normal_ls_k1,normal_ls_k2,"
               "normal_ls_residual,aso_ms,aso_valid,aso_k1,aso_k2,"
               "pair_normal_change,pair_sphere_residual,"
               "pair_cylinder_residual\n";
        std::uint64_t seed = 1000;
        for (const std::string type : {"plane", "sphere", "cylinder"}) {
            const std::vector<double> coverages = type == "sphere"
                ? std::vector<double>{20, 45, 90, 180}
                : type == "cylinder" ? std::vector<double>{15, 30, 60, 120, 360}
                                      : std::vector<double>{0};
            const std::vector<double> lengths = type == "cylinder"
                ? std::vector<double>{0.2, 1.0, 4.0} : std::vector<double>{1.0};
            for (const int count : {256, 1024})
                for (const double coverage : coverages)
                    for (const double length : lengths)
                        for (const double position_sigma : {0.0, 0.001, 0.005, 0.01})
                            for (const double normal_degrees : {0.0, 1.0, 3.0, 5.0})
                                emit(*output, type, "noise", count, coverage, length,
                                     position_sigma, normal_degrees, 0.0, seed++);
        }
        for (const std::string type : {"plane", "sphere", "cylinder"})
            for (const double flips : {0.01, 0.05, 0.10, 0.25})
                emit(*output, type, "normal_sign", 1024,
                     type == "sphere" ? 180.0 : type == "cylinder" ? 360.0 : 0.0,
                     type == "cylinder" ? 2.0 : 1.0, 0.0, 0.0, flips, seed++);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "synthetic feature benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
