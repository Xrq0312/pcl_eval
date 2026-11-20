#include "pcl_eval/geometric_features.hpp"

#include <Eigen/Geometry>

#include <algorithm>
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
    std::size_t query = 0;
    std::size_t neighbor = 0;
};

Sample make_plane() {
    Sample sample;
    constexpr int side = 33;
    constexpr double spacing = 0.025;
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            sample.points.emplace_back((x - side / 2) * spacing,
                                       (y - side / 2) * spacing, 0.0);
            sample.normals.emplace_back(0.0, 0.0, 1.0);
        }
    sample.query = (side / 2) * side + side / 2;
    sample.neighbor = sample.query + 1;
    return sample;
}

Sample make_sphere() {
    Sample sample;
    constexpr int latitudes = 32;
    constexpr int longitudes = 64;
    for (int latitude = 1; latitude < latitudes; ++latitude) {
        const double phi = kPi * latitude / latitudes;
        for (int longitude = 0; longitude < longitudes; ++longitude) {
            const double theta = 2.0 * kPi * longitude / longitudes;
            const Eigen::Vector3d normal(std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta), std::cos(phi));
            sample.points.push_back(normal);
            sample.normals.push_back(normal);
        }
    }
    const Eigen::Vector3d target(1.0, 0.0, 0.0);
    sample.query = 0;
    for (std::size_t i = 1; i < sample.points.size(); ++i)
        if ((sample.points[i] - target).squaredNorm() <
            (sample.points[sample.query] - target).squaredNorm()) sample.query = i;
    sample.neighbor = sample.query + 1;
    return sample;
}

Eigen::Vector3d rotate_normal_noise(const Eigen::Vector3d& normal,
                                    double sigma_degrees,
                                    std::mt19937_64& random) {
    if (sigma_degrees == 0.0) return normal;
    std::normal_distribution<double> standard(0.0, 1.0);
    Eigen::Vector3d axis(standard(random), standard(random), standard(random));
    axis -= axis.dot(normal) * normal;
    if (axis.norm() < 1e-12) axis = normal.unitOrthogonal();
    axis.normalize();
    const double angle = standard(random) * sigma_degrees * kPi / 180.0;
    return Eigen::AngleAxisd(angle, axis) * normal;
}

void perturb(Sample& sample, double position_sigma, double normal_sigma,
             std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::normal_distribution<double> position_noise(0.0, position_sigma);
    for (std::size_t i = 0; i < sample.points.size(); ++i) {
        sample.points[i] += Eigen::Vector3d(position_noise(random),
            position_noise(random), position_noise(random));
        sample.normals[i] = rotate_normal_noise(sample.normals[i], normal_sigma, random);
    }
}

void emit(std::ostream& output, const std::string& geometry,
          double position_sigma, double normal_sigma, double crease_degrees,
          std::size_t neighbors, std::uint64_t seed) {
    Sample sample = geometry == "sphere" ? make_sphere() : make_plane();
    perturb(sample, position_sigma, normal_sigma, seed);
    const auto scales = pcl_eval::features::estimate_multiscale_aso(
        sample.query, {neighbors}, sample.points, sample.normals);
    if (scales.empty()) return;
    const auto& aso = scales.front().aso;
    const auto smooth = pcl_eval::features::estimate_curvature_boundary(
        sample.points[sample.query], sample.normals[sample.query],
        sample.points[sample.neighbor], sample.normals[sample.neighbor], aso, 1e-4);
    Eigen::Vector3d crease_axis = aso.valid ? aso.directions.col(0)
                                           : sample.normals[sample.query].unitOrthogonal();
    if (crease_axis.norm() < 1e-12) crease_axis = sample.normals[sample.query].unitOrthogonal();
    crease_axis.normalize();
    const Eigen::Vector3d boundary_normal = Eigen::AngleAxisd(
        crease_degrees * kPi / 180.0, crease_axis) * sample.normals[sample.neighbor];
    const auto boundary = pcl_eval::features::estimate_curvature_boundary(
        sample.points[sample.query], sample.normals[sample.query],
        sample.points[sample.neighbor], boundary_normal, aso, 1e-4);
    output << geometry << ',' << position_sigma << ',' << normal_sigma << ','
           << crease_degrees << ',' << neighbors << ',' << seed << ','
           << (aso.valid ? 1 : 0) << ',' << scales.front().radius << ','
           << aso.curvatures(0) << ',' << aso.curvatures(1) << ','
           << smooth.residual << ',' << smooth.relative_residual << ','
           << boundary.residual << ',' << boundary.relative_residual << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--output" && ++i < argc) output_path = argv[i];
            else if (key == "--help" || key == "-h") {
                std::cout << "geometry_boundary_synthetic [--output FILE]\n";
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
            << "geometry,position_sigma,normal_sigma_deg,crease_deg,neighbors,seed,"
               "aso_valid,radius,k1,k2,smooth_residual,smooth_relative_residual,"
               "boundary_residual,boundary_relative_residual\n";
        std::uint64_t seed = 7000;
        for (const std::string geometry : {"plane", "sphere"})
            for (const double position_sigma : {0.0, 0.001, 0.005, 0.01})
                for (const double normal_sigma : {0.0, 1.0, 3.0, 5.0})
                    for (const double crease : {5.0, 15.0, 30.0, 60.0})
                        for (const std::size_t neighbors : {32, 64, 128, 256})
                            for (int repeat = 0; repeat < 8; ++repeat)
                                emit(*output, geometry, position_sigma, normal_sigma,
                                     crease, neighbors, seed++);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "boundary benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
