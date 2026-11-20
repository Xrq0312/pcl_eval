#include "pcl_eval/robust_candidates.hpp"

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
namespace candidates = pcl_eval::candidates;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadius = 1.0;
const Eigen::Vector3d kSphereCenter(0.3, -0.2, 0.5);
const Eigen::Vector3d kCylinderAxis = Eigen::Vector3d(0.2, -0.4, 0.89).normalized();

struct Sample {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
};

Eigen::Vector3d random_unit(std::mt19937_64& random) {
    std::normal_distribution<double> gaussian(0.0, 1.0);
    Eigen::Vector3d value(gaussian(random), gaussian(random), gaussian(random));
    return value.normalized();
}

Sample make_sphere(int count, double cap_degrees) {
    Sample result;
    const double min_z = std::cos(cap_degrees * kPi / 180.0);
    constexpr double golden = 2.39996322972865332;
    for (int i = 0; i < count; ++i) {
        const double z = 1.0 - (1.0 - min_z) * (i + 0.5) / count;
        const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle = golden * i;
        const Eigen::Vector3d normal(radial * std::cos(angle),
                                     radial * std::sin(angle), z);
        result.points.push_back(kSphereCenter + kRadius * normal);
        result.normals.push_back(normal);
    }
    return result;
}

Sample make_cylinder(int count, double arc_degrees, double length) {
    Sample result;
    const Eigen::Vector3d e1 = kCylinderAxis.unitOrthogonal();
    const Eigen::Vector3d e2 = kCylinderAxis.cross(e1);
    const int around = std::max(8, static_cast<int>(std::sqrt(2.0 * count)));
    const int along = std::max(2, count / around);
    const double arc = arc_degrees * kPi / 180.0;
    for (int z = 0; z < along && static_cast<int>(result.points.size()) < count; ++z) {
        const double axial = length * ((z + 0.5) / along - 0.5);
        for (int a = 0; a < around && static_cast<int>(result.points.size()) < count; ++a) {
            const double angle = arc * ((a + 0.5) / around - 0.5);
            const Eigen::Vector3d normal = std::cos(angle) * e1 + std::sin(angle) * e2;
            result.points.push_back(axial * kCylinderAxis + kRadius * normal);
            result.normals.push_back(normal);
        }
    }
    return result;
}

Sample make_plane(int count) {
    Sample result;
    const int side = static_cast<int>(std::ceil(std::sqrt(count)));
    for (int y = 0; y < side && static_cast<int>(result.points.size()) < count; ++y)
        for (int x = 0; x < side && static_cast<int>(result.points.size()) < count; ++x) {
            result.points.emplace_back((x + 0.5) / side - 0.5,
                                       (y + 0.5) / side - 0.5, 0.0);
            result.normals.emplace_back(0.0, 0.0, 1.0);
        }
    return result;
}

void perturb(Sample& sample, double position_sigma, double normal_sigma_degrees,
             double flip_fraction, double outlier_fraction, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::normal_distribution<double> position_noise(0.0, position_sigma);
    std::normal_distribution<double> angle_noise(
        0.0, normal_sigma_degrees * kPi / 180.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> cube(-1.5, 1.5);
    for (std::size_t i = 0; i < sample.points.size(); ++i) {
        if (unit(random) < outlier_fraction) {
            sample.points[i] = Eigen::Vector3d(cube(random), cube(random), cube(random));
            sample.normals[i] = random_unit(random);
            continue;
        }
        sample.points[i] += Eigen::Vector3d(position_noise(random),
            position_noise(random), position_noise(random));
        Eigen::Vector3d tangent = random_unit(random);
        tangent -= tangent.dot(sample.normals[i]) * sample.normals[i];
        if (tangent.norm() < 1e-12) tangent = sample.normals[i].unitOrthogonal();
        tangent.normalize();
        const double angle = angle_noise(random);
        sample.normals[i] = (std::cos(angle) * sample.normals[i] +
            std::sin(angle) * tangent).normalized();
        if (unit(random) < flip_fraction) sample.normals[i] = -sample.normals[i];
    }
}

double axis_degrees(const Eigen::Vector3d& axis) {
    return std::acos(std::clamp(std::abs(axis.dot(kCylinderAxis)), 0.0, 1.0)) *
        180.0 / kPi;
}

double axis_offset(const Eigen::Vector3d& point) {
    return (point.cross(kCylinderAxis)).norm();
}

void emit_row(std::ostream& output, const std::string& truth,
              const std::string& suite, int count, double coverage, double length,
              double position_sigma, double normal_sigma, double flips,
              double outliers, std::uint64_t seed, const std::string& family,
              const std::string& estimator, bool valid, double radius_error,
              double location_error, double axis_error, double nrmse,
              double condition, double runtime_ms,
              const candidates::CompetitionResult& competition) {
    double best_stability = 0.0;
    if (!competition.scores.empty()) best_stability = competition.scores.front().stability;
    output << truth << ',' << suite << ',' << count << ',' << coverage << ',' << length
           << ',' << position_sigma << ',' << normal_sigma << ',' << flips << ','
           << outliers << ',' << seed << ',' << family << ',' << estimator << ','
           << (valid ? 1 : 0) << ',' << radius_error << ',' << location_error << ','
           << axis_error << ',' << nrmse << ',' << condition << ',' << runtime_ms << ','
           << candidates::to_string(competition.selected) << ','
           << (competition.rejected ? 1 : 0) << ',' << competition.confidence << ','
           << best_stability << '\n';
}

void run_case(std::ostream& output, const std::string& truth, const std::string& suite,
              int count, double coverage, double length, double position_sigma,
              double normal_sigma, double flips, double outliers, std::uint64_t seed) {
    Sample sample = truth == "sphere" ? make_sphere(count, coverage)
        : truth == "cylinder" ? make_cylinder(count, coverage, length)
                               : make_plane(count);
    perturb(sample, position_sigma, normal_sigma, flips, outliers, seed);
    const auto competition = candidates::compete_models(sample.points, sample.normals);
    if (truth == "sphere") {
        for (const auto& candidate : candidates::generate_sphere_candidates(
                 sample.points, sample.normals)) {
            emit_row(output, truth, suite, count, coverage, length, position_sigma,
                normal_sigma, flips, outliers, seed, "sphere",
                candidates::to_string(candidate.estimator), candidate.valid,
                std::abs(candidate.radius - kRadius),
                (candidate.center - kSphereCenter).norm(), 0.0,
                candidate.trimmed_rmse, candidate.condition, candidate.runtime_ms,
                competition);
        }
    } else if (truth == "cylinder") {
        for (const auto& candidate : candidates::generate_cylinder_candidates(
                 sample.points, sample.normals)) {
            emit_row(output, truth, suite, count, coverage, length, position_sigma,
                normal_sigma, flips, outliers, seed, "cylinder",
                candidates::to_string(candidate.estimator), candidate.valid,
                std::abs(candidate.radius - kRadius), axis_offset(candidate.axis_point),
                axis_degrees(candidate.axis_direction), candidate.trimmed_rmse,
                candidate.condition, candidate.runtime_ms, competition);
        }
    } else {
        emit_row(output, truth, suite, count, coverage, length, position_sigma,
            normal_sigma, flips, outliers, seed, "competition", "none", true,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, competition);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--output" && ++i < argc) output_path = argv[i];
            else throw std::runtime_error("unknown or incomplete option: " + key);
        }
        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
            file.open(output_path);
            if (!file) throw std::runtime_error("cannot create output");
            output = &file;
        }
        *output << std::setprecision(10)
            << "truth,suite,points,coverage_deg,length,position_sigma,normal_sigma_deg,"
               "flip_fraction,outlier_fraction,seed,family,estimator,valid,"
               "radius_abs_error,location_error,axis_degrees,trimmed_rmse,condition,"
               "runtime_ms,competition_selected,rejected,confidence,best_stability\n";
        std::uint64_t seed = 9000;
        for (const std::string truth : {"plane", "sphere", "cylinder"}) {
            const std::vector<double> coverages = truth == "sphere"
                ? std::vector<double>{20, 45, 90, 180}
                : truth == "cylinder" ? std::vector<double>{15, 30, 60, 120, 360}
                                       : std::vector<double>{0};
            for (const double coverage : coverages)
                for (const int count : {128, 512, 2048})
                    run_case(*output, truth, "coverage_density", count, coverage,
                        truth == "cylinder" ? 2.0 : 1.0, 0.002, 2.0, 0.0, 0.0, seed++);
            for (const double normal_sigma : {0.0, 3.0, 8.0, 15.0})
                for (const double flips : {0.0, 0.10, 0.25})
                    run_case(*output, truth, "normal_robustness", 1024,
                        truth == "sphere" ? 90.0 : truth == "cylinder" ? 120.0 : 0.0,
                        truth == "cylinder" ? 2.0 : 1.0, 0.005, normal_sigma,
                        flips, 0.0, seed++);
            for (const double outliers : {0.05, 0.10, 0.20})
                run_case(*output, truth, "point_outliers", 1024,
                    truth == "sphere" ? 90.0 : truth == "cylinder" ? 120.0 : 0.0,
                    truth == "cylinder" ? 2.0 : 1.0, 0.005, 3.0, 0.05,
                    outliers, seed++);
        }
        for (const double length : {0.1, 0.25, 0.5, 1.0, 4.0})
            run_case(*output, "cylinder", "short_extent", 1024, 60.0, length,
                0.005, 3.0, 0.05, 0.0, seed++);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "robust candidate synthetic benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
