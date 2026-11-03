#include "colmap/online_sfm/online_pose_graph.h"

#include "colmap/util/logging.h"

#include <cmath>
#include <memory>

#include <ceres/ceres.h>

namespace colmap {
namespace {

// Parameters: cam_from_world quaternion / translation of i and j.
struct RelativePoseCost {
  RelativePoseCost(const Rigid3d& i_from_j,
                   const double rot_sigma,
                   const double trans_sigma)
      : q_meas(i_from_j.rotation()),
        t_meas(i_from_j.translation()),
        inv_rot_sigma(1.0 / rot_sigma),
        inv_trans_sigma(1.0 / trans_sigma) {}

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

    const Eigen::Quaternion<T> q_ij = qi * qj.conjugate();
    const Eigen::Matrix<T, 3, 1> t_ij = ti - q_ij * tj;
    const Eigen::Quaternion<T> dq = q_meas.cast<T>().conjugate() * q_ij;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> r(residuals);
    r.template head<3>() = T(2.0 * inv_rot_sigma) * dq.vec();
    r.template tail<3>() = T(inv_trans_sigma) * (t_ij - t_meas.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Rigid3d& i_from_j,
                                     const double rot_sigma,
                                     const double trans_sigma) {
    return new ceres::AutoDiffCostFunction<RelativePoseCost, 6, 4, 3, 4, 3>(
        new RelativePoseCost(i_from_j, rot_sigma, trans_sigma));
  }

  const Eigen::Quaterniond q_meas;
  const Eigen::Vector3d t_meas;
  const double inv_rot_sigma;
  const double inv_trans_sigma;
};

Eigen::Matrix3d RotZ(const double yaw) {
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

double YawOf(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

// Parameters: yaw increment about world +z and camera center of i and j.
// R_world_from_cam = Rz(yaw) * R0, with R0 the input orientation.
struct FourDofRelativePoseCost {
  FourDofRelativePoseCost(const Rigid3d& i_from_j,
                          const Eigen::Matrix3d& R0_world_from_i,
                          const Eigen::Matrix3d& R0_world_from_j,
                          const double rot_sigma,
                          const double trans_sigma)
      : R0_i_from_world(R0_world_from_i.transpose()),
        t_meas(i_from_j.translation()),
        yaw_offset(YawOf(R0_world_from_j *
                         i_from_j.rotation().toRotationMatrix().transpose() *
                         R0_world_from_i.transpose())),
        inv_rot_sigma(1.0 / rot_sigma),
        inv_trans_sigma(1.0 / trans_sigma) {}

  template <typename T>
  bool operator()(const T* const yaw_i,
                  const T* const ci_ptr,
                  const T* const yaw_j,
                  const T* const cj_ptr,
                  T* residuals) const {
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> ci(ci_ptr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> cj(cj_ptr);
    const T c = ceres::cos(yaw_i[0]);
    const T s = ceres::sin(yaw_i[0]);
    const Eigen::Matrix<T, 3, 1> d = cj - ci;
    // Rz(yaw_i)^T * d
    const Eigen::Matrix<T, 3, 1> d_rot(c * d.x() + s * d.y(),
                                       -s * d.x() + c * d.y(),
                                       d.z());
    const Eigen::Matrix<T, 3, 1> p_j_in_i = R0_i_from_world.cast<T>() * d_rot;

    const T e = yaw_j[0] - yaw_i[0] + T(yaw_offset);
    residuals[0] = T(inv_rot_sigma) * ceres::atan2(ceres::sin(e), ceres::cos(e));
    Eigen::Map<Eigen::Matrix<T, 3, 1>> r_t(residuals + 1);
    r_t = T(inv_trans_sigma) * (p_j_in_i - t_meas.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Rigid3d& i_from_j,
                                     const Eigen::Matrix3d& R0_world_from_i,
                                     const Eigen::Matrix3d& R0_world_from_j,
                                     const double rot_sigma,
                                     const double trans_sigma) {
    return new ceres::
        AutoDiffCostFunction<FourDofRelativePoseCost, 4, 1, 3, 1, 3>(
            new FourDofRelativePoseCost(i_from_j,
                                        R0_world_from_i,
                                        R0_world_from_j,
                                        rot_sigma,
                                        trans_sigma));
  }

  const Eigen::Matrix3d R0_i_from_world;
  const Eigen::Vector3d t_meas;
  const double yaw_offset;
  const double inv_rot_sigma;
  const double inv_trans_sigma;
};

struct FourDofNode {
  double yaw = 0.0;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R0_world_from_cam = Eigen::Matrix3d::Identity();
};

}  // namespace

double PoseGraphEdgeChi2Quantile(const PoseGraphOptions& options,
                                 const double probability) {
  // Chi-square inverse CDF for 4 and 6 dof.
  if (probability >= 0.99) {
    return options.four_dof ? 13.2767 : 16.8119;
  }
  THROW_CHECK_GE(probability, 0.95);
  return options.four_dof ? 9.4877 : 12.5916;
}

PoseGraphSummary OptimizePoseGraph(
    const PoseGraphOptions& options,
    const std::vector<PoseGraphEdge>& edges,
    const std::unordered_set<image_t>& fixed_ids,
    std::unordered_map<image_t, Rigid3d>* cam_from_world) {
  PoseGraphSummary summary;
  summary.edge_chi2.assign(edges.size(), -1.0);
  if (cam_from_world == nullptr || edges.empty()) {
    return summary;
  }

  ceres::Problem problem;
  std::unordered_set<image_t> nodes;
  std::unordered_map<image_t, FourDofNode> four_dof_nodes;
  std::vector<ceres::ResidualBlockId> block_ids(edges.size(), nullptr);
  const double huber_scale =
      std::sqrt(PoseGraphEdgeChi2Quantile(options, 0.95));

  const auto AddNode = [&](const image_t id) {
    if (!nodes.insert(id).second) {
      return;
    }
    const bool fixed = fixed_ids.count(id) > 0;
    if (fixed) {
      ++summary.num_fixed;
    }
    Rigid3d& pose = cam_from_world->at(id);
    if (options.four_dof) {
      FourDofNode& node = four_dof_nodes[id];
      node.R0_world_from_cam =
          pose.rotation().normalized().toRotationMatrix().transpose();
      node.center = pose.TgtOriginInSrc();
      problem.AddParameterBlock(&node.yaw, 1);
      problem.AddParameterBlock(node.center.data(), 3);
      if (fixed) {
        problem.SetParameterBlockConstant(&node.yaw);
        problem.SetParameterBlockConstant(node.center.data());
      }
      return;
    }
    problem.AddParameterBlock(
        pose.params.data(), 4, new ceres::EigenQuaternionManifold);
    problem.AddParameterBlock(pose.params.data() + 4, 3);
    if (fixed) {
      problem.SetParameterBlockConstant(pose.params.data());
      problem.SetParameterBlockConstant(pose.params.data() + 4);
    }
  };

  for (size_t k = 0; k < edges.size(); ++k) {
    const PoseGraphEdge& edge = edges[k];
    if (cam_from_world->count(edge.image_id_i) == 0 ||
        cam_from_world->count(edge.image_id_j) == 0 ||
        edge.image_id_i == edge.image_id_j || !(edge.rot_sigma > 0.0) ||
        !(edge.trans_sigma > 0.0)) {
      continue;
    }
    AddNode(edge.image_id_i);
    AddNode(edge.image_id_j);
    ceres::LossFunction* loss =
        edge.robust ? new ceres::HuberLoss(huber_scale) : nullptr;
    const Rigid3d i_from_j = Inverse(edge.j_from_i);
    if (options.four_dof) {
      FourDofNode& ni = four_dof_nodes.at(edge.image_id_i);
      FourDofNode& nj = four_dof_nodes.at(edge.image_id_j);
      block_ids[k] = problem.AddResidualBlock(
          FourDofRelativePoseCost::Create(i_from_j,
                                          ni.R0_world_from_cam,
                                          nj.R0_world_from_cam,
                                          edge.rot_sigma,
                                          edge.trans_sigma),
          loss,
          &ni.yaw,
          ni.center.data(),
          &nj.yaw,
          nj.center.data());
    } else {
      Rigid3d& pi = cam_from_world->at(edge.image_id_i);
      Rigid3d& pj = cam_from_world->at(edge.image_id_j);
      block_ids[k] = problem.AddResidualBlock(
          RelativePoseCost::Create(i_from_j, edge.rot_sigma, edge.trans_sigma),
          loss,
          pi.params.data(),
          pi.params.data() + 4,
          pj.params.data(),
          pj.params.data() + 4);
    }
    ++summary.num_edges;
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

  for (size_t k = 0; k < edges.size(); ++k) {
    if (block_ids[k] == nullptr) {
      continue;
    }
    double cost = 0.0;
    if (problem.EvaluateResidualBlock(
            block_ids[k], /*apply_loss_function=*/false, &cost, nullptr,
            nullptr)) {
      summary.edge_chi2[k] = 2.0 * cost;
    }
  }

  for (const image_t id : nodes) {
    if (fixed_ids.count(id) > 0) {
      continue;
    }
    Rigid3d& pose = cam_from_world->at(id);
    if (options.four_dof) {
      const FourDofNode& node = four_dof_nodes.at(id);
      const Eigen::Matrix3d R_cam_from_world =
          (RotZ(node.yaw) * node.R0_world_from_cam).transpose();
      pose = Rigid3d(Eigen::Quaterniond(R_cam_from_world),
                     -R_cam_from_world * node.center);
    }
    pose.rotation().normalize();
  }
  summary.initial_cost = ceres_summary.initial_cost;
  summary.final_cost = ceres_summary.final_cost;
  summary.success = ceres_summary.IsSolutionUsable();
  return summary;
}

}  // namespace colmap
