#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {

// Relative-pose edge: cam_j_from_cam_i measured before loop correction
// (odometry / covisibility) or from the loop PnP pose (loop edge).
// Residual = [rotation error / rot_sigma, (p_j_in_i - p_meas) / trans_sigma],
// p_j_in_i = camera j center in camera i. In 4-DoF mode the rotation error is
// the yaw about world +z only.
struct PoseGraphEdge {
  image_t image_id_i = kInvalidImageId;
  image_t image_id_j = kInvalidImageId;
  Rigid3d j_from_i;
  // Isotropic 1-sigma, radians / meters.
  double rot_sigma = 1.0;
  double trans_sigma = 1.0;
  // Huber at the 95% chi-square quantile of the whitened residual.
  bool robust = false;
};

struct PoseGraphOptions {
  int max_num_iterations = 50;
  // Optimize camera center + yaw about world +z per node; roll / pitch stay at
  // their input values. Requires a gravity-aligned (VIO) world frame.
  bool four_dof = false;
};

struct PoseGraphSummary {
  bool success = false;
  int num_nodes = 0;
  int num_fixed = 0;
  int num_edges = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  // Whitened squared residual per input edge after the solve (-1 if the edge
  // was not added). Robust edges are reported without the loss.
  std::vector<double> edge_chi2;
};

// Chi-square quantile of one edge residual (dof 6, or 4 in 4-DoF mode).
double PoseGraphEdgeChi2Quantile(const PoseGraphOptions& options,
                                 double probability);

// Pose graph over cam_from_world poses. Nodes in `fixed_ids` stay constant;
// every other node referenced by an edge is optimized in place.
PoseGraphSummary OptimizePoseGraph(
    const PoseGraphOptions& options,
    const std::vector<PoseGraphEdge>& edges,
    const std::unordered_set<image_t>& fixed_ids,
    std::unordered_map<image_t, Rigid3d>* cam_from_world);

}  // namespace colmap
