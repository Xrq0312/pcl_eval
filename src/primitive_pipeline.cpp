#include "pcl_eval/primitive_pipeline.hpp"

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace pcl_eval {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kEps = 1e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Observations {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> curvature;
    std::vector<unsigned char> normal_valid;
    int grid_width = 0;
    int grid_height = 0;
    std::vector<int> pixel_to_point;
    std::vector<int> point_to_pixel;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
};

struct Candidate {
    PrimitiveModel model;
};

double seconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

Eigen::Vector3d canonical_direction(Eigen::Vector3d value) {
    const double norm = value.norm();
    if (!(norm > kEps) || !value.allFinite()) return Eigen::Vector3d::Zero();
    value /= norm;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(value[i]) > 1e-10) {
            if (value[i] < 0.0) value = -value;
            break;
        }
    }
    return value;
}

Eigen::Vector3d vec3(const std::vector<double>& c, std::size_t offset) {
    return Eigen::Vector3d(c[offset], c[offset + 1], c[offset + 2]);
}

void set_vec3(std::vector<double>& c, std::size_t offset, const Eigen::Vector3d& value) {
    c[offset] = value.x(); c[offset + 1] = value.y(); c[offset + 2] = value.z();
}

bool finite_coefficients(const PrimitiveModel& model) {
    return !model.coefficients.empty() &&
           std::all_of(model.coefficients.begin(), model.coefficients.end(),
                       [](double value) { return std::isfinite(value); });
}

double residual(const PrimitiveModel& model, const Eigen::Vector3d& point) {
    if (model.type == PrimitiveType::Plane) {
        return std::abs(vec3(model.coefficients, 0).dot(point) + model.coefficients[3]);
    }
    if (model.type == PrimitiveType::Sphere) {
        return std::abs((point - vec3(model.coefficients, 0)).norm() - model.coefficients[3]);
    }
    const Eigen::Vector3d origin = vec3(model.coefficients, 0);
    const Eigen::Vector3d axis = vec3(model.coefficients, 3);
    const Eigen::Vector3d delta = point - origin;
    return std::abs((delta - axis * axis.dot(delta)).norm() - model.coefficients[6]);
}

double normal_alignment(const PrimitiveModel& model, const Eigen::Vector3d& point,
                        const Eigen::Vector3d& normal) {
    Eigen::Vector3d expected;
    if (model.type == PrimitiveType::Plane) {
        expected = vec3(model.coefficients, 0);
    } else if (model.type == PrimitiveType::Sphere) {
        expected = point - vec3(model.coefficients, 0);
    } else {
        const Eigen::Vector3d origin = vec3(model.coefficients, 0);
        const Eigen::Vector3d axis = vec3(model.coefficients, 3);
        const Eigen::Vector3d delta = point - origin;
        expected = delta - axis * axis.dot(delta);
    }
    const double expected_norm = expected.norm();
    if (!(expected_norm > kEps)) return 0.0;
    return std::abs(expected.dot(normal) / expected_norm);
}

bool circle_from_three(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                       const Eigen::Vector2d& c, Eigen::Vector2d& center, double& radius) {
    Eigen::Matrix2d matrix;
    matrix.row(0) = 2.0 * (b - a).transpose();
    matrix.row(1) = 2.0 * (c - a).transpose();
    if (std::abs(matrix.determinant()) < 1e-10) return false;
    Eigen::Vector2d rhs(b.squaredNorm() - a.squaredNorm(),
                        c.squaredNorm() - a.squaredNorm());
    center = matrix.fullPivLu().solve(rhs);
    radius = (a - center).norm();
    return center.allFinite() && std::isfinite(radius) && radius > kEps;
}

bool sphere_from_four(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
                      Eigen::Vector3d& center, double& radius) {
    Eigen::Matrix3d matrix;
    matrix.row(0) = 2.0 * (p1 - p0).transpose();
    matrix.row(1) = 2.0 * (p2 - p0).transpose();
    matrix.row(2) = 2.0 * (p3 - p0).transpose();
    if (std::abs(matrix.determinant()) < 1e-12) return false;
    Eigen::Vector3d rhs(p1.squaredNorm() - p0.squaredNorm(),
                        p2.squaredNorm() - p0.squaredNorm(),
                        p3.squaredNorm() - p0.squaredNorm());
    center = matrix.fullPivLu().solve(rhs);
    radius = (p0 - center).norm();
    return center.allFinite() && std::isfinite(radius) && radius > kEps;
}

bool sphere_from_oriented_pair(const Eigen::Vector3d& p0, const Eigen::Vector3d& n0,
                               const Eigen::Vector3d& p1, const Eigen::Vector3d& n1,
                               Eigen::Vector3d& center, double& radius) {
    const double dot = n0.dot(n1);
    const double denominator = 1.0 - dot * dot;
    if (denominator < 1e-5) return false;
    const Eigen::Vector3d w = p0 - p1;
    const double d = n0.dot(w);
    const double e = n1.dot(w);
    const double s = (dot * e - d) / denominator;
    const double t = (e - dot * d) / denominator;
    const Eigen::Vector3d c0 = p0 + s * n0;
    const Eigen::Vector3d c1 = p1 + t * n1;
    center = 0.5 * (c0 + c1);
    radius = 0.5 * ((p0 - center).norm() + (p1 - center).norm());
    const double mismatch = (c0 - c1).norm();
    return center.allFinite() && std::isfinite(radius) && radius > kEps &&
           mismatch < std::max(0.2 * radius, 1e-5);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> perpendicular_frame(const Eigen::Vector3d& axis) {
    Eigen::Vector3d helper = std::abs(axis.z()) < 0.8 ? Eigen::Vector3d::UnitZ()
                                                       : Eigen::Vector3d::UnitY();
    Eigen::Vector3d x = canonical_direction(axis.cross(helper));
    Eigen::Vector3d y = axis.cross(x).normalized();
    return {x, y};
}

bool model_from_sample(PrimitiveType type, const Observations& obs,
                       const std::vector<int>& sample, double max_radius,
                       PrimitiveModel& model) {
    model = PrimitiveModel{};
    model.type = type;
    if (type == PrimitiveType::Plane) {
        if (sample.empty()) return false;
        Eigen::Vector3d normal = obs.normal_valid[sample[0]]
            ? canonical_direction(obs.normals[sample[0]]) : Eigen::Vector3d::Zero();
        if (normal.isZero(kEps) && sample.size() >= 3) {
            normal = canonical_direction(
                (obs.points[sample[1]] - obs.points[sample[0]])
                    .cross(obs.points[sample[2]] - obs.points[sample[0]]));
        }
        if (normal.isZero(kEps)) return false;
        model.coefficients = {normal.x(), normal.y(), normal.z(),
                              -normal.dot(obs.points[sample[0]])};
    } else if (type == PrimitiveType::Sphere) {
        Eigen::Vector3d center;
        double radius = 0.0;
        bool ok = false;
        if (sample.size() >= 2 && obs.normal_valid[sample[0]] && obs.normal_valid[sample[1]]) {
            ok = sphere_from_oriented_pair(obs.points[sample[0]], obs.normals[sample[0]],
                                           obs.points[sample[1]], obs.normals[sample[1]],
                                           center, radius);
        }
        if (!ok && sample.size() >= 4) {
            ok = sphere_from_four(obs.points[sample[0]], obs.points[sample[1]],
                                  obs.points[sample[2]], obs.points[sample[3]], center, radius);
        }
        if (!ok || radius > max_radius) return false;
        model.coefficients = {center.x(), center.y(), center.z(), radius};
    } else {
        if (sample.size() < 3 || !obs.normal_valid[sample[0]] ||
            !obs.normal_valid[sample[1]] || !obs.normal_valid[sample[2]]) return false;
        Eigen::Matrix3d normals;
        for (int row = 0; row < 3; ++row) normals.row(row) = obs.normals[sample[row]].transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(normals, Eigen::ComputeFullV);
        Eigen::Vector3d axis = canonical_direction(svd.matrixV().col(2));
        if (axis.isZero(kEps) || svd.singularValues()[1] < 1e-3) return false;
        const auto frame = perpendicular_frame(axis);
        Eigen::Vector2d projected[3];
        for (int i = 0; i < 3; ++i) {
            projected[i] = {obs.points[sample[i]].dot(frame.first),
                            obs.points[sample[i]].dot(frame.second)};
        }
        Eigen::Vector2d center2;
        double radius = 0.0;
        if (!circle_from_three(projected[0], projected[1], projected[2], center2, radius) ||
            radius > max_radius) return false;
        Eigen::Vector3d origin = center2.x() * frame.first + center2.y() * frame.second;
        origin -= axis * axis.dot(origin);  // canonical closest point to world origin
        model.coefficients = {origin.x(), origin.y(), origin.z(),
                              axis.x(), axis.y(), axis.z(), radius};
    }
    return finite_coefficients(model);
}

void evaluate_model(PrimitiveModel& model, const Observations& obs,
                    const std::vector<int>& indices, double threshold,
                    double min_normal_alignment) {
    model.points.clear();
    model.points.reserve(indices.size());
    double score = 0.0;
    double squared_error = 0.0;
    for (int index : indices) {
        const double distance = residual(model, obs.points[index]);
        if (!(distance <= threshold)) continue;
        double alignment = 1.0;
        if (obs.normal_valid[index]) {
            alignment = normal_alignment(model, obs.points[index], obs.normals[index]);
            if (alignment < min_normal_alignment) continue;
        }
        const double normalized = distance / std::max(threshold, kEps);
        score += (1.0 - normalized * normalized) * (0.5 + 0.5 * alignment);
        squared_error += distance * distance;
        model.points.push_back(index);
    }
    model.score = score;
    model.rmse = model.points.empty()
        ? std::numeric_limits<double>::infinity()
        : std::sqrt(squared_error / static_cast<double>(model.points.size()));
}

bool refit_plane(const Observations& obs, const std::vector<int>& indices, PrimitiveModel& model) {
    if (indices.size() < 3) return false;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int i : indices) centroid += obs.points[i];
    centroid /= static_cast<double>(indices.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int i : indices) {
        const Eigen::Vector3d delta = obs.points[i] - centroid;
        covariance.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) return false;
    Eigen::Vector3d normal = canonical_direction(solver.eigenvectors().col(0));
    if (normal.isZero(kEps)) return false;
    model.type = PrimitiveType::Plane;
    model.coefficients = {normal.x(), normal.y(), normal.z(), -normal.dot(centroid)};
    return true;
}

bool refit_sphere(const Observations& obs, const std::vector<int>& indices, PrimitiveModel& model) {
    if (indices.size() < 4) return false;
    Eigen::MatrixXd matrix(indices.size(), 4);
    Eigen::VectorXd rhs(indices.size());
    for (std::size_t row = 0; row < indices.size(); ++row) {
        const Eigen::Vector3d& p = obs.points[indices[row]];
        matrix.row(row) << 2.0 * p.transpose(), 1.0;
        rhs[row] = p.squaredNorm();
    }
    const Eigen::Vector4d solution = matrix.colPivHouseholderQr().solve(rhs);
    const Eigen::Vector3d center = solution.head<3>();
    const double radius_squared = solution[3] + center.squaredNorm();
    if (!center.allFinite() || !(radius_squared > kEps) || !std::isfinite(radius_squared)) return false;
    model.type = PrimitiveType::Sphere;
    model.coefficients = {center.x(), center.y(), center.z(), std::sqrt(radius_squared)};
    return true;
}

bool refit_cylinder(const Observations& obs, const std::vector<int>& indices,
                    PrimitiveModel& model) {
    if (indices.size() < 6) return false;
    Eigen::Matrix3d normal_scatter = Eigen::Matrix3d::Zero();
    int valid_normals = 0;
    for (int i : indices) {
        if (!obs.normal_valid[i]) continue;
        normal_scatter.noalias() += obs.normals[i] * obs.normals[i].transpose();
        ++valid_normals;
    }
    if (valid_normals < 3) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(normal_scatter);
    if (solver.info() != Eigen::Success) return false;
    Eigen::Vector3d axis = canonical_direction(solver.eigenvectors().col(0));
    if (axis.isZero(kEps)) return false;
    const auto frame = perpendicular_frame(axis);
    Eigen::MatrixXd matrix(indices.size(), 3);
    Eigen::VectorXd rhs(indices.size());
    for (std::size_t row = 0; row < indices.size(); ++row) {
        const Eigen::Vector3d& p = obs.points[indices[row]];
        const double x = p.dot(frame.first);
        const double y = p.dot(frame.second);
        matrix.row(row) << 2.0 * x, 2.0 * y, 1.0;
        rhs[row] = x * x + y * y;
    }
    const Eigen::Vector3d solution = matrix.colPivHouseholderQr().solve(rhs);
    const double radius_squared = solution[2] + solution.head<2>().squaredNorm();
    if (!solution.allFinite() || !(radius_squared > kEps)) return false;
    Eigen::Vector3d origin = solution.x() * frame.first + solution.y() * frame.second;
    origin -= axis * axis.dot(origin);
    model.type = PrimitiveType::Cylinder;
    model.coefficients = {origin.x(), origin.y(), origin.z(),
                          axis.x(), axis.y(), axis.z(), std::sqrt(radius_squared)};
    return finite_coefficients(model);
}

bool refit_model(const Observations& obs, const std::vector<int>& indices,
                 PrimitiveModel& model) {
    if (model.type == PrimitiveType::Plane) return refit_plane(obs, indices, model);
    if (model.type == PrimitiveType::Sphere) return refit_sphere(obs, indices, model);
    return refit_cylinder(obs, indices, model);
}

bool geometry_is_observable(const PrimitiveModel& model, const Observations& obs,
                            const std::vector<int>& support, double threshold) {
    if (support.empty()) return false;
    Eigen::AlignedBox3d bounds;
    Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    int valid_normals = 0;
    for (int index : support) {
        bounds.extend(obs.points[index]);
        centroid += obs.points[index];
        if (obs.normal_valid[index]) {
            scatter.noalias() += obs.normals[index] * obs.normals[index].transpose();
            ++valid_normals;
        }
    }
    centroid /= static_cast<double>(support.size());
    Eigen::Matrix3d point_scatter = Eigen::Matrix3d::Zero();
    for (int index : support) {
        const Eigen::Vector3d delta = obs.points[index] - centroid;
        point_scatter.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> point_solver(point_scatter);
    if (point_solver.info() != Eigen::Success) return false;
    if (model.type == PrimitiveType::Plane) {
        const auto values = point_solver.eigenvalues();
        // Range-image rows and other nearly one-dimensional strips can have
        // zero residual yet are not observable plane instances.
        return values[1] / std::max(values[2], kEps) >= 0.002;
    }
    const double diameter = bounds.diagonal().norm();
    const double radius = model.coefficients.back();
    if (!(radius > threshold) || radius > std::max(6.0 * diameter, 20.0 * threshold)) return false;
    if (valid_normals < 6) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
    if (solver.info() != Eigen::Success) return false;
    const auto eigenvalues = solver.eigenvalues();
    const double largest = std::max(eigenvalues[2], kEps);
    // A plane fitted as a huge sphere/cylinder has effectively rank-one normals.
    // Requiring measurable normal variation rejects that ill-conditioned fit.
    return eigenvalues[1] / largest >= 0.012;
}

Observations prepare_observations(const std::vector<Eigen::Vector3d>& points,
                                  const Config& config,
                                  const std::vector<Eigen::Vector3d>* supplied,
                                  const OrganizedGrid* grid) {
    Observations obs;
    obs.points = points;
    obs.cloud->reserve(points.size());
    for (const auto& p : points) obs.cloud->push_back(pcl::PointXYZ(
        static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())));
    obs.cloud->width = static_cast<std::uint32_t>(obs.cloud->size());
    obs.cloud->height = 1;
    obs.tree.setInputCloud(obs.cloud);
    obs.normals.resize(points.size(), Eigen::Vector3d::Zero());
    obs.curvature.resize(points.size(), 0.0);
    obs.normal_valid.resize(points.size(), 0);
    if (grid) {
        if (grid->width <= 0 || grid->height <= 0 ||
            grid->pixel_to_point.size() !=
                static_cast<std::size_t>(grid->width * grid->height))
            throw std::runtime_error("invalid organized grid dimensions");
        obs.grid_width = grid->width;
        obs.grid_height = grid->height;
        obs.pixel_to_point = grid->pixel_to_point;
        obs.point_to_pixel.assign(points.size(), -1);
        for (std::size_t pixel = 0; pixel < obs.pixel_to_point.size(); ++pixel) {
            const int index = obs.pixel_to_point[pixel];
            if (index < 0) continue;
            if (index >= static_cast<int>(points.size()) || obs.point_to_pixel[index] >= 0)
                throw std::runtime_error("organized grid contains an invalid or duplicate point index");
            obs.point_to_pixel[index] = static_cast<int>(pixel);
        }
        if (std::find(obs.point_to_pixel.begin(), obs.point_to_pixel.end(), -1) !=
            obs.point_to_pixel.end())
            throw std::runtime_error("organized grid does not reference every input point");
    }
    if (supplied) {
        if (supplied->size() != points.size()) throw std::runtime_error("supplied normal count mismatch");
        for (std::size_t i = 0; i < supplied->size(); ++i) {
            Eigen::Vector3d normal = (*supplied)[i];
            if (normal.allFinite() && normal.norm() > kEps) {
                obs.normals[i] = normal.normalized();
                obs.normal_valid[i] = 1;
            }
        }
    } else if (grid) {
        auto organized = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        organized->width = static_cast<std::uint32_t>(grid->width);
        organized->height = static_cast<std::uint32_t>(grid->height);
        organized->is_dense = false;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        organized->points.assign(grid->pixel_to_point.size(), pcl::PointXYZ(nan, nan, nan));
        for (std::size_t pixel = 0; pixel < grid->pixel_to_point.size(); ++pixel) {
            const int index = grid->pixel_to_point[pixel];
            if (index < 0) continue;
            const auto& p = points[index];
            organized->points[pixel] = pcl::PointXYZ(static_cast<float>(p.x()),
                                                      static_cast<float>(p.y()),
                                                      static_cast<float>(p.z()));
        }
        pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> estimator;
        estimator.setNormalEstimationMethod(estimator.AVERAGE_3D_GRADIENT);
        estimator.setMaxDepthChangeFactor(0.02f);
        estimator.setNormalSmoothingSize(8.0f);
        estimator.setDepthDependentSmoothing(true);
        estimator.setInputCloud(organized);
        pcl::PointCloud<pcl::Normal> normals;
        estimator.compute(normals);
        for (std::size_t pixel = 0; pixel < grid->pixel_to_point.size(); ++pixel) {
            const int index = grid->pixel_to_point[pixel];
            if (index < 0) continue;
            const auto& value = normals[pixel];
            Eigen::Vector3d normal(value.normal_x, value.normal_y, value.normal_z);
            if (normal.allFinite() && normal.norm() > kEps) {
                obs.normals[index] = normal.normalized();
                obs.normal_valid[index] = 1;
                obs.curvature[index] = std::isfinite(value.curvature)
                    ? static_cast<double>(value.curvature) : 0.0;
            }
        }
    } else {
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;
        if (config.threads > 0) estimator.setNumberOfThreads(static_cast<unsigned int>(config.threads));
        estimator.setInputCloud(obs.cloud);
        auto search = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        estimator.setSearchMethod(search);
        estimator.setKSearch(std::max(6, config.normal_k));
        pcl::PointCloud<pcl::Normal> normals;
        estimator.compute(normals);
        for (std::size_t i = 0; i < normals.size(); ++i) {
            Eigen::Vector3d normal(normals[i].normal_x, normals[i].normal_y, normals[i].normal_z);
            if (normal.allFinite() && normal.norm() > kEps) {
                obs.normals[i] = normal.normalized();
                obs.normal_valid[i] = 1;
                obs.curvature[i] = std::isfinite(normals[i].curvature)
                    ? static_cast<double>(normals[i].curvature) : 0.0;
            }
        }
    }
    return obs;
}

std::vector<std::vector<int>> smooth_regions(Observations& obs, const Config& config,
                                             double threshold, double spacing) {
    const int n = static_cast<int>(obs.points.size());
    std::vector<int> seeds(n);
    std::iota(seeds.begin(), seeds.end(), 0);
    std::stable_sort(seeds.begin(), seeds.end(), [&](int a, int b) {
        if (obs.curvature[a] != obs.curvature[b]) return obs.curvature[a] < obs.curvature[b];
        return a < b;
    });
    std::vector<unsigned char> assigned(n, 0);
    std::vector<std::vector<int>> regions;
    const int k = std::max(6, config.region_k);
    const double minimum_alignment = std::cos(
        config.region_normal_angle_deg * kPi / 180.0);
    std::vector<int> neighbors(k);
    std::vector<float> squared(k);
    for (int seed : seeds) {
        if (assigned[seed] || !obs.normal_valid[seed]) continue;
        regions.emplace_back();
        std::deque<int> queue{seed};
        assigned[seed] = 1;
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop_front();
            regions.back().push_back(current);
            int found = 0;
            if (obs.grid_width > 0) {
                const int pixel = obs.point_to_pixel[current];
                const int x = pixel % obs.grid_width;
                const int y = pixel / obs.grid_width;
                neighbors.clear();
                squared.clear();
                neighbors.push_back(current);
                squared.push_back(0.0f);
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if ((dx == 0 && dy == 0) || std::abs(dx) + std::abs(dy) > 3) continue;
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= obs.grid_width || yy >= obs.grid_height)
                            continue;
                        const int neighbor = obs.pixel_to_point[yy * obs.grid_width + xx];
                        if (neighbor < 0) continue;
                        neighbors.push_back(neighbor);
                        squared.push_back(static_cast<float>(
                            (obs.points[neighbor] - obs.points[current]).squaredNorm()));
                    }
                }
                found = static_cast<int>(neighbors.size());
            } else {
                found = obs.tree.nearestKSearch(current, k, neighbors, squared);
            }
            if (found < 2) continue;
            const double local_radius = obs.grid_width > 0
                ? std::max(25.0 * spacing, 15.0 * threshold)
                : std::max(2.0 * spacing,
                           1.6 * std::sqrt(static_cast<double>(squared[found - 1])));
            for (int j = 1; j < found; ++j) {
                const int neighbor = neighbors[j];
                if (assigned[neighbor] || !obs.normal_valid[neighbor]) continue;
                const Eigen::Vector3d delta = obs.points[neighbor] - obs.points[current];
                const double edge = delta.norm();
                if (edge > local_radius) continue;
                if (std::abs(obs.normals[current].dot(obs.normals[neighbor])) < minimum_alignment)
                    continue;
                const double tangent_error = std::min(
                    std::abs(delta.dot(obs.normals[current])),
                    std::abs(delta.dot(obs.normals[neighbor])));
                if (tangent_error > std::max(2.0 * threshold, 0.3 * edge)) continue;
                assigned[neighbor] = 1;
                queue.push_back(neighbor);
            }
        }
        if (static_cast<int>(regions.back().size()) < std::max(12, config.min_points / 3))
            regions.pop_back();
    }
    std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a.front() < b.front();
    });
    return regions;
}

double estimate_spacing(Observations& obs, std::uint64_t seed) {
    std::vector<int> order(obs.points.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed ^ 0x8d58ac26afe12e47ULL);
    std::shuffle(order.begin(), order.end(), rng);
    if (order.size() > 4096) order.resize(4096);
    std::vector<double> distances;
    distances.reserve(order.size());
    std::vector<int> neighbors(2);
    std::vector<float> squared(2);
    for (int index : order) {
        if (obs.tree.nearestKSearch(index, 2, neighbors, squared) == 2 && squared[1] > 0.0f) {
            distances.push_back(std::sqrt(static_cast<double>(squared[1])));
        }
    }
    if (distances.empty()) throw std::runtime_error("cannot estimate nonzero point spacing");
    const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), middle, distances.end());
    return *middle;
}

std::vector<int> sample_distinct(const std::vector<int>& source, int count,
                                 std::mt19937_64& rng) {
    if (static_cast<int>(source.size()) < count) return {};
    std::vector<int> result;
    result.reserve(count);
    std::uniform_int_distribution<std::size_t> pick(0, source.size() - 1);
    for (int attempts = 0; static_cast<int>(result.size()) < count && attempts < count * 20; ++attempts) {
        const int value = source[pick(rng)];
        if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
    }
    return static_cast<int>(result.size()) == count ? result : std::vector<int>{};
}

void trim_pool(std::vector<Candidate>& pool, int limit) {
    std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
        if (a.model.score != b.model.score) return a.model.score > b.model.score;
        if (a.model.rmse != b.model.rmse) return a.model.rmse < b.model.rmse;
        return a.model.coefficients < b.model.coefficients;
    });
    if (static_cast<int>(pool.size()) > limit) pool.resize(limit);
}

std::vector<Candidate> generate_proposals(Observations& obs, const Config& config,
                                          double threshold, double spacing) {
    const int n = static_cast<int>(obs.points.size());
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::mt19937_64 rng(config.seed);
    std::vector<int> pilot = all;
    std::shuffle(pilot.begin(), pilot.end(), rng);
    if (static_cast<int>(pilot.size()) > config.pilot_size) pilot.resize(config.pilot_size);
    std::sort(pilot.begin(), pilot.end());
    Eigen::AlignedBox3d bounds;
    for (const auto& point : obs.points) bounds.extend(point);
    const double max_radius = std::max(bounds.diagonal().norm() * 2.0, spacing * 10.0);
    const double minimum_alignment = std::cos(config.normal_angle_deg * kPi / 180.0);
    const int pilot_min = std::max(6, static_cast<int>(std::floor(
        config.min_points * static_cast<double>(pilot.size()) / n * 0.55)));
    std::vector<Candidate> pools[3];
    std::vector<Candidate> region_candidates;

    auto propose = [&](PrimitiveType type, const std::vector<int>& source,
                       const std::string& origin) {
        int sample_count = type == PrimitiveType::Plane ? 1
                         : type == PrimitiveType::Sphere ? 4 : 3;
        auto sample = sample_distinct(source, sample_count, rng);
        if (sample.empty()) return;
        PrimitiveModel model;
        if (!model_from_sample(type, obs, sample, max_radius, model)) return;
        model.proposal_source = origin;
        evaluate_model(model, obs, pilot, threshold, minimum_alignment);
        if (static_cast<int>(model.points.size()) < pilot_min) return;
        if (model.points.size() >= 12 &&
            !geometry_is_observable(model, obs, model.points, threshold)) return;
        const int slot = type == PrimitiveType::Plane ? 0 : type == PrimitiveType::Sphere ? 1 : 2;
        pools[slot].push_back({std::move(model)});
        if (static_cast<int>(pools[slot].size()) > 2 * config.proposal_pool_size)
            trim_pool(pools[slot], config.proposal_pool_size);
    };

    for (int iteration = 0; iteration < config.global_iterations; ++iteration) {
        const PrimitiveType type = iteration % 3 == 0 ? PrimitiveType::Plane
                                  : iteration % 3 == 1 ? PrimitiveType::Sphere
                                                       : PrimitiveType::Cylinder;
        propose(type, all, "global");
    }

    if (config.stage != PipelineStage::GlobalRansac) {
        const int scales[] = {64, 256, 1024, 4096, 16384};
        std::uniform_int_distribution<int> anchor_pick(0, n - 1);
        std::vector<int> neighbors;
        std::vector<float> squared;
        for (int iteration = 0; iteration < config.local_iterations; ++iteration) {
            const int anchor = anchor_pick(rng);
            const int k = std::min(n, scales[iteration % 5]);
            neighbors.resize(k); squared.resize(k);
            const int found = obs.tree.nearestKSearch(anchor, k, neighbors, squared);
            neighbors.resize(std::max(0, found));
            const PrimitiveType type = iteration % 3 == 0 ? PrimitiveType::Plane
                                      : iteration % 3 == 1 ? PrimitiveType::Sphere
                                                           : PrimitiveType::Cylinder;
            propose(type, neighbors, "local_k" + std::to_string(k));
        }

        // Region growing is deliberately only a proposal source.  The final
        // support is still re-evaluated globally, so the detector does not
        // inherit a hard segmentation boundary from this inexpensive cue.
        const auto regions = smooth_regions(obs, config, threshold, spacing);
        for (const auto& region : regions) {
            if (static_cast<int>(region.size()) < config.min_points) continue;
            for (PrimitiveType type : {PrimitiveType::Plane, PrimitiveType::Sphere,
                                       PrimitiveType::Cylinder}) {
                PrimitiveModel model;
                model.type = type;
                if (!refit_model(obs, region, model)) continue;
                model.proposal_source = "smooth_region";
                evaluate_model(model, obs, region, threshold, minimum_alignment);
                if (static_cast<int>(model.points.size()) < config.min_points) continue;
                if (!geometry_is_observable(model, obs, model.points, threshold)) continue;
                region_candidates.push_back({std::move(model)});
            }
        }
    }

    std::vector<Candidate> result;
    for (auto& pool : pools) {
        trim_pool(pool, config.proposal_pool_size);
        region_candidates.insert(region_candidates.end(),
                                 std::make_move_iterator(pool.begin()),
                                 std::make_move_iterator(pool.end()));
    }
    for (auto& candidate : region_candidates) {
        evaluate_model(candidate.model, obs, all, threshold, minimum_alignment);
        for (int pass = 0; pass < 2 &&
             static_cast<int>(candidate.model.points.size()) >= config.min_points; ++pass) {
            const auto support = candidate.model.points;
            if (!refit_model(obs, support, candidate.model)) break;
            candidate.model.proposal_source += "+refit";
            const bool adaptive = config.stage == PipelineStage::JointOptimization ||
                                  config.stage == PipelineStage::BoundaryAware;
            const double adaptive_threshold = adaptive
                ? std::clamp(4.0 * candidate.model.rmse, threshold, 3.0 * threshold)
                : threshold;
            candidate.model.fit_threshold = adaptive_threshold;
            evaluate_model(candidate.model, obs, all, adaptive_threshold, minimum_alignment);
        }
        if (static_cast<int>(candidate.model.points.size()) >= config.min_points &&
            finite_coefficients(candidate.model) &&
            geometry_is_observable(candidate.model, obs, candidate.model.points, threshold))
            result.push_back(std::move(candidate));
    }
    return result;
}

double support_iou(const std::vector<int>& a, const std::vector<int>& b) {
    std::size_t i = 0, j = 0, intersection = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { ++intersection; ++i; ++j; }
        else if (a[i] < b[j]) ++i;
        else ++j;
    }
    const std::size_t union_size = a.size() + b.size() - intersection;
    return union_size ? static_cast<double>(intersection) / union_size : 0.0;
}

std::vector<PrimitiveModel> select_models(std::vector<Candidate> candidates,
                                          const Config& config, int point_count) {
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        const double qa = a.model.score / std::sqrt(std::max<std::size_t>(1, a.model.points.size()));
        const double qb = b.model.score / std::sqrt(std::max<std::size_t>(1, b.model.points.size()));
        if (qa != qb) return qa > qb;
        return a.model.rmse < b.model.rmse;
    });
    std::vector<unsigned char> covered(point_count, 0);
    std::vector<PrimitiveModel> selected;
    for (auto& candidate : candidates) {
        if (static_cast<int>(selected.size()) >= config.max_models) break;
        bool duplicate = false;
        for (const auto& existing : selected) {
            if (existing.type == candidate.model.type &&
                support_iou(existing.points, candidate.model.points) > 0.72) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        const int novel = static_cast<int>(std::count_if(
            candidate.model.points.begin(), candidate.model.points.end(),
            [&](int index) { return !covered[index]; }));
        if (novel < std::max(8, config.min_points / 2)) continue;
        for (int index : candidate.model.points) covered[index] = 1;
        selected.push_back(std::move(candidate.model));
    }
    return selected;
}

std::vector<int> assign_points(const Observations& obs,
                               const std::vector<PrimitiveModel>& models,
                               double threshold, double minimum_alignment,
                               double support_prior_weight) {
    std::vector<int> labels(obs.points.size(), -1);
    for (std::size_t i = 0; i < obs.points.size(); ++i) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best = -1;
        for (std::size_t m = 0; m < models.size(); ++m) {
            const double model_threshold = models[m].fit_threshold > 0.0
                ? models[m].fit_threshold : threshold;
            const double distance = residual(models[m], obs.points[i]);
            if (distance > model_threshold) continue;
            double alignment = 1.0;
            if (obs.normal_valid[i]) {
                alignment = normal_alignment(models[m], obs.points[i], obs.normals[i]);
                if (alignment < minimum_alignment) continue;
            }
            const double support_prior = std::log1p(
                static_cast<double>(models[m].points.size()));
            const double cost = distance / std::max(model_threshold, kEps) +
                0.2 * (1.0 - alignment) - support_prior_weight * support_prior;
            if (cost < best_cost) { best_cost = cost; best = static_cast<int>(m); }
        }
        labels[i] = best;
    }
    return labels;
}

void compact_and_refit(Observations& obs, const Config& config, double threshold,
                       double support_prior_weight, std::vector<PrimitiveModel>& models,
                       std::vector<int>& labels) {
    std::vector<PrimitiveModel> kept;
    std::vector<int> remap(models.size(), -1);
    for (std::size_t model_id = 0; model_id < models.size(); ++model_id) {
        std::vector<int> support;
        for (std::size_t i = 0; i < labels.size(); ++i)
            if (labels[i] == static_cast<int>(model_id)) support.push_back(static_cast<int>(i));
        if (static_cast<int>(support.size()) < config.min_points) continue;
        PrimitiveModel model = models[model_id];
        const double previous_threshold = model.fit_threshold > 0.0
            ? model.fit_threshold : threshold;
        if (!refit_model(obs, support, model)) continue;
        model.points = std::move(support);
        double squared = 0.0;
        for (int index : model.points) { const double r = residual(model, obs.points[index]); squared += r * r; }
        model.rmse = std::sqrt(squared / model.points.size());
        // A least-squares update can jump to a different basin, especially
        // for short cylinder arcs. Never keep a refit that no longer explains
        // the support under the tolerance that admitted it.
        if (config.max_refit_rmse_factor > 0.0 &&
            model.rmse > config.max_refit_rmse_factor * previous_threshold) continue;
        if (config.stage == PipelineStage::JointOptimization ||
            config.stage == PipelineStage::BoundaryAware) {
            model.fit_threshold = std::clamp(4.0 * model.rmse, threshold, 3.0 * threshold);
        } else {
            model.fit_threshold = threshold;
        }
        remap[model_id] = static_cast<int>(kept.size());
        kept.push_back(std::move(model));
    }
    for (int& label : labels) label = label >= 0 ? remap[label] : -1;
    models = std::move(kept);
    const double minimum_alignment = std::cos(config.normal_angle_deg * kPi / 180.0);
    labels = assign_points(obs, models, threshold, minimum_alignment,
                           support_prior_weight);
}

std::vector<std::vector<int>> connected_components(const Observations& obs,
                                                   const std::vector<int>& support,
                                                   double radius) {
    std::unordered_set<int> allowed(support.begin(), support.end());
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> components;
    std::vector<int> neighbors;
    std::vector<float> squared;
    for (int seed : support) {
        if (visited.count(seed)) continue;
        components.emplace_back();
        std::deque<int> queue{seed};
        visited.insert(seed);
        while (!queue.empty()) {
            const int current = queue.front(); queue.pop_front();
            components.back().push_back(current);
            neighbors.clear(); squared.clear();
            if (obs.grid_width > 0) {
                const int pixel = obs.point_to_pixel[current];
                const int x = pixel % obs.grid_width;
                const int y = pixel / obs.grid_width;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= obs.grid_width || yy >= obs.grid_height)
                            continue;
                        const int neighbor = obs.pixel_to_point[yy * obs.grid_width + xx];
                        if (neighbor < 0) continue;
                        const double edge = (obs.points[neighbor] - obs.points[current]).norm();
                        if (edge > radius) continue;
                        if (obs.normal_valid[current] && obs.normal_valid[neighbor] &&
                            std::abs(obs.normals[current].dot(obs.normals[neighbor])) < 0.75)
                            continue;
                        neighbors.push_back(neighbor);
                    }
                }
            } else {
                obs.tree.radiusSearch(current, radius, neighbors, squared);
            }
            for (int neighbor : neighbors) {
                if (allowed.count(neighbor) && !visited.count(neighbor)) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        std::sort(components.back().begin(), components.back().end());
    }
    std::sort(components.begin(), components.end(), [](const auto& a, const auto& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a.front() < b.front();
    });
    return components;
}

void split_disconnected(Observations& obs, const Config& config, double threshold,
                        double spacing, const std::vector<PrimitiveModel>& source_models,
                        const std::vector<int>& source_labels,
                        std::vector<PrimitiveModel>& output_models,
                        std::vector<int>& output_labels) {
    output_labels.assign(source_labels.size(), -1);
    const double radius = std::max(config.connectivity_spacing_factor * spacing,
                                   6.0 * threshold);
    for (std::size_t model_id = 0; model_id < source_models.size(); ++model_id) {
        std::vector<int> support;
        for (std::size_t i = 0; i < source_labels.size(); ++i)
            if (source_labels[i] == static_cast<int>(model_id)) support.push_back(static_cast<int>(i));
        auto components = connected_components(obs, support, radius);
        const std::size_t major_min = std::max<std::size_t>(
            static_cast<std::size_t>(config.min_points),
            static_cast<std::size_t>(std::ceil(0.10 * support.size())));
        std::size_t major_count = 0;
        std::vector<Eigen::AlignedBox3d> major_bounds;
        for (const auto& component : components) {
            if (component.size() >= major_min) {
                ++major_count;
                Eigen::AlignedBox3d component_bounds;
                for (int index : component) component_bounds.extend(obs.points[index]);
                major_bounds.push_back(component_bounds);
            }
        }
        bool well_separated = major_bounds.size() >= 2;
        if (well_separated) {
            Eigen::Vector3d gap = Eigen::Vector3d::Zero();
            for (int axis = 0; axis < 3; ++axis) {
                if (major_bounds[0].max()[axis] < major_bounds[1].min()[axis])
                    gap[axis] = major_bounds[1].min()[axis] - major_bounds[0].max()[axis];
                else if (major_bounds[1].max()[axis] < major_bounds[0].min()[axis])
                    gap[axis] = major_bounds[0].min()[axis] - major_bounds[1].max()[axis];
            }
            const double local_extent = std::min(
                major_bounds[0].diagonal().norm(), major_bounds[1].diagonal().norm());
            const double occlusion_gap_factor = obs.grid_width > 0 ? 16.0 : 3.0;
            well_separated = gap.norm() > std::max(
                occlusion_gap_factor * radius, 0.25 * local_extent);
        }
        // Connectivity is a boundary cue, not permission to explode a noisy
        // support into dozens of scan fragments. Split only when at least two
        // substantial components explain most of the parent model.
        const bool should_split = major_count >= 2 && well_separated;
        if (!should_split && static_cast<int>(support.size()) >= config.min_points) {
            PrimitiveModel model = source_models[model_id];
            model.points = support;
            const int new_id = static_cast<int>(output_models.size());
            for (int index : support) output_labels[index] = new_id;
            output_models.push_back(std::move(model));
            continue;
        }
        bool emitted = false;
        for (const auto& component : components) {
            if (component.size() < major_min) continue;
            PrimitiveModel model = source_models[model_id];
            if (!refit_model(obs, component, model)) continue;
            model.points = component;
            model.parent_model = static_cast<int>(model_id);
            model.proposal_source += "+component";
            double squared = 0.0;
            for (int index : component) { const double r = residual(model, obs.points[index]); squared += r * r; }
            model.rmse = std::sqrt(squared / component.size());
            const int new_id = static_cast<int>(output_models.size());
            for (int index : component) output_labels[index] = new_id;
            output_models.push_back(std::move(model));
            emitted = true;
        }
        if (!emitted && static_cast<int>(support.size()) >= config.min_points) {
            PrimitiveModel model = source_models[model_id];
            model.points = support;
            const int new_id = static_cast<int>(output_models.size());
            for (int index : support) output_labels[index] = new_id;
            output_models.push_back(std::move(model));
        }
    }
}

}  // namespace

std::string to_string(PrimitiveType type) {
    if (type == PrimitiveType::Plane) return "plane";
    if (type == PrimitiveType::Sphere) return "sphere";
    return "cylinder";
}

std::string to_string(PipelineStage stage) {
    if (stage == PipelineStage::GlobalRansac) return "global_ransac";
    if (stage == PipelineStage::GlobalLocal) return "global_local";
    if (stage == PipelineStage::JointOptimization) return "joint_opt";
    return "boundary_aware";
}

PipelineStage parse_stage(const std::string& value) {
    if (value == "global_ransac") return PipelineStage::GlobalRansac;
    if (value == "global_local") return PipelineStage::GlobalLocal;
    if (value == "joint_opt") return PipelineStage::JointOptimization;
    if (value == "boundary_aware") return PipelineStage::BoundaryAware;
    throw std::runtime_error("unknown stage: " + value);
}

FitResult fit_primitives(const std::vector<Eigen::Vector3d>& points,
                         const Config& config,
                         const std::vector<Eigen::Vector3d>* supplied_normals,
                         const OrganizedGrid* organized_grid) {
    if (points.size() < 4) throw std::runtime_error("at least four points are required");
    if (config.min_points < 3 || config.global_iterations < 0 || config.local_iterations < 0)
        throw std::runtime_error("invalid primitive fitting configuration");
    const auto total_begin = Clock::now();
    FitResult result;

    const auto normals_begin = Clock::now();
    Observations obs = prepare_observations(points, config, supplied_normals, organized_grid);
    result.spacing = estimate_spacing(obs, config.seed);
    result.distance_threshold = config.distance_threshold > 0.0
        ? config.distance_threshold
        : config.distance_spacing_factor * result.spacing;
    if (!(result.distance_threshold > 0.0) || !std::isfinite(result.distance_threshold))
        throw std::runtime_error("invalid inferred distance threshold");
    result.normals = obs.normals;
    result.timing.normals = seconds(normals_begin, Clock::now());

    const auto proposal_begin = Clock::now();
    auto candidates = generate_proposals(obs, config, result.distance_threshold, result.spacing);
    result.timing.proposals = seconds(proposal_begin, Clock::now());

    const auto selection_begin = Clock::now();
    auto models = select_models(std::move(candidates), config, static_cast<int>(points.size()));
    const double minimum_alignment = std::cos(config.normal_angle_deg * kPi / 180.0);
    const bool joint_stage = config.stage == PipelineStage::JointOptimization ||
                             config.stage == PipelineStage::BoundaryAware;
    const double support_prior_weight = joint_stage
        ? config.joint_support_prior : 0.0;
    auto labels = assign_points(obs, models, result.distance_threshold, minimum_alignment,
                                support_prior_weight);
    result.timing.selection = seconds(selection_begin, Clock::now());

    const auto refinement_begin = Clock::now();
    for (int pass = 0; pass < config.refinement_iterations; ++pass) {
        compact_and_refit(obs, config, result.distance_threshold, support_prior_weight,
                          models, labels);
    }
    compact_and_refit(obs, config, result.distance_threshold, support_prior_weight,
                      models, labels);
    result.premerge_models = models;
    result.premerge_labels = labels;
    result.timing.refinement = seconds(refinement_begin, Clock::now());

    const auto connectivity_begin = Clock::now();
    if (config.stage == PipelineStage::GlobalLocal ||
        config.stage == PipelineStage::BoundaryAware) {
        split_disconnected(obs, config, result.distance_threshold, result.spacing,
                           models, labels, result.models, result.labels);
    } else {
        result.models = std::move(models);
        result.labels = std::move(labels);
    }
    result.timing.connectivity = seconds(connectivity_begin, Clock::now());
    result.timing.total = seconds(total_begin, Clock::now());
    return result;
}

}  // namespace pcl_eval
