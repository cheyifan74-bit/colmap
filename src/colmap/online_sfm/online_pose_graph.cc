#include "colmap/online_sfm/online_pose_graph.h"

#include <memory>

#include <ceres/ceres.h>

namespace colmap {
namespace {

struct RelativePoseCost {
  RelativePoseCost(const Rigid3d& j_from_i,
                   const double rotation_weight,
                   const double translation_weight)
      : q_meas(j_from_i.rotation()),
        t_meas(j_from_i.translation()),
        w_rot(rotation_weight),
        w_trans(translation_weight) {}

  template <typename T>
  bool operator()(const T* const qi_ptr,
                  const T* const ti_ptr,
                  const T* const qj_ptr,
                  const T* const tj_ptr,
                  T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> qi(qi_ptr);
    const Eigen::Map<const Eigen::Quaternion<T>> qj(qj_ptr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> ti(ti_ptr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tj(tj_ptr);

    const Eigen::Quaternion<T> q_ji = qj * qi.conjugate();
    const Eigen::Matrix<T, 3, 1> t_ji = tj - q_ji * ti;
    const Eigen::Quaternion<T> dq = q_meas.cast<T>().conjugate() * q_ji;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residuals);
    r.template head<3>() = T(2.0 * w_rot) * dq.vec();
    r.template tail<3>() = T(w_trans) * (t_ji - t_meas.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Rigid3d& j_from_i,
                                     const double rotation_weight,
                                     const double translation_weight) {
    return new ceres::AutoDiffCostFunction<RelativePoseCost, 6, 4, 3, 4, 3>(
        new RelativePoseCost(j_from_i, rotation_weight, translation_weight));
  }

  const Eigen::Quaterniond q_meas;
  const Eigen::Vector3d t_meas;
  const double w_rot;
  const double w_trans;
};

}  // namespace

PoseGraphSummary OptimizeSE3PoseGraph(
    const PoseGraphOptions& options,
    const std::vector<PoseGraphEdge>& edges,
    const std::unordered_set<image_t>& fixed_ids,
    std::unordered_map<image_t, Rigid3d>* cam_from_world) {
  PoseGraphSummary summary;
  if (cam_from_world == nullptr || edges.empty()) {
    return summary;
  }

  ceres::Problem problem;
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
  bool manifold_owned = false;
  std::unordered_set<image_t> nodes;

  const auto AddNode = [&](const image_t id) {
    Rigid3d& pose = cam_from_world->at(id);
    if (!nodes.insert(id).second) {
      return;
    }
    problem.AddParameterBlock(pose.params.data(), 4, quaternion_manifold);
    problem.AddParameterBlock(pose.params.data() + 4, 3);
    manifold_owned = true;
    if (fixed_ids.count(id) > 0) {
      problem.SetParameterBlockConstant(pose.params.data());
      problem.SetParameterBlockConstant(pose.params.data() + 4);
      ++summary.num_fixed;
    }
  };

  for (const PoseGraphEdge& edge : edges) {
    if (cam_from_world->count(edge.image_id_i) == 0 ||
        cam_from_world->count(edge.image_id_j) == 0 ||
        edge.image_id_i == edge.image_id_j) {
      continue;
    }
    AddNode(edge.image_id_i);
    AddNode(edge.image_id_j);
    Rigid3d& pi = cam_from_world->at(edge.image_id_i);
    Rigid3d& pj = cam_from_world->at(edge.image_id_j);
    problem.AddResidualBlock(
        RelativePoseCost::Create(edge.j_from_i,
                                 edge.weight * options.rotation_weight,
                                 edge.weight * options.translation_weight),
        nullptr,
        pi.params.data(),
        pi.params.data() + 4,
        pj.params.data(),
        pj.params.data() + 4);
    ++summary.num_edges;
  }
  if (!manifold_owned) {
    delete quaternion_manifold;
  }
  summary.num_nodes = static_cast<int>(nodes.size());
  if (summary.num_edges == 0 || summary.num_fixed == 0 ||
      summary.num_fixed == summary.num_nodes) {
    return summary;
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.logging_type = ceres::LoggingType::SILENT;
  solver_options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary ceres_summary;
  ceres::Solve(solver_options, &problem, &ceres_summary);

  for (const image_t id : nodes) {
    cam_from_world->at(id).rotation().normalize();
  }
  summary.initial_cost = ceres_summary.initial_cost;
  summary.final_cost = ceres_summary.final_cost;
  summary.success = ceres_summary.IsSolutionUsable();
  return summary;
}

}  // namespace colmap
