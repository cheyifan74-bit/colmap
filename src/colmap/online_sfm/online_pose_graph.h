#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {

// Relative-pose edge: cam_j_from_cam_i measured before loop correction
// (odometry / covisibility) or from the loop PnP pose (loop edge).
struct PoseGraphEdge {
  image_t image_id_i = kInvalidImageId;
  image_t image_id_j = kInvalidImageId;
  Rigid3d j_from_i;
  double weight = 1.0;
  // Huber knee on the residual L2 norm. <=0 keeps a squared residual.
  // Loop edges set this in weighted-residual units so a bad historical
  // measurement cannot dominate the odometry chain.
  double robust_loss_scale = 0.0;
};

struct PoseGraphOptions {
  int max_num_iterations = 50;
  // Residual = [w_rot * 2 * vec(dq), w_trans * dt]; 1 rad ~ w_rot / w_trans m.
  double rotation_weight = 1.0;
  double translation_weight = 1.0;
};

struct PoseGraphSummary {
  bool success = false;
  int num_nodes = 0;
  int num_fixed = 0;
  int num_edges = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
};

// SE3 pose graph over cam_from_world poses. Nodes in `fixed_ids` stay
// constant; every other node referenced by an edge is optimized in place.
PoseGraphSummary OptimizeSE3PoseGraph(
    const PoseGraphOptions& options,
    const std::vector<PoseGraphEdge>& edges,
    const std::unordered_set<image_t>& fixed_ids,
    std::unordered_map<image_t, Rigid3d>* cam_from_world);

}  // namespace colmap
