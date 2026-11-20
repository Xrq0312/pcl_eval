#include "pcl_eval/cylinder_voting.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

double angle_degrees(const Eigen::Vector3d& left, const Eigen::Vector3d& right) {
    return std::acos(std::clamp(std::abs(left.dot(right)), 0.0, 1.0)) * 180.0 / kPi;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const Eigen::Vector3d axis = Eigen::Vector3d(0.25, -0.35, 0.902).normalized();
        const Eigen::Vector3d e1 = axis.unitOrthogonal();
        const Eigen::Vector3d e2 = axis.cross(e1);
        const Eigen::Vector3d anchor(0.2, -0.1, 0.3);
        constexpr double radius = 0.42;
        std::vector<Eigen::Vector3d> points;
        std::mt19937_64 random(44);
        std::normal_distribution<double> noise(0.0, 0.001);
        for (int z = 0; z < 30; ++z)
            for (int a = 0; a < 48; ++a) {
                const double angle = 2.0 * kPi * a / 48.0;
                Eigen::Vector3d point = anchor + (z / 29.0 - 0.5) * 1.5 * axis +
                    radius * (std::cos(angle) * e1 + std::sin(angle) * e2);
                point += Eigen::Vector3d(noise(random), noise(random), noise(random));
                points.push_back(point);
            }
        std::uniform_real_distribution<double> clutter(-1.0, 1.0);
        for (int i = 0; i < 300; ++i)
            points.emplace_back(clutter(random), clutter(random), clutter(random));

        pcl_eval::voting::Options options;
        options.direction_count = 256;
        options.hypotheses_per_direction = 320;
        options.minimum_support = 250;
        options.distance_threshold_fraction = 0.008;
        const auto detected = pcl_eval::voting::detect(points, options);
        require(!detected.empty(), "no point-only cylinder candidate");
        const auto found = std::find_if(detected.begin(), detected.end(), [&](const auto& c) {
            return angle_degrees(c.axis_direction, axis) < 4.0 &&
                   std::abs(c.radius - radius) / radius < 0.10;
        });
        require(found != detected.end(), "true cylinder not recalled");
        std::cout << "ALL CYLINDER VOTING TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
