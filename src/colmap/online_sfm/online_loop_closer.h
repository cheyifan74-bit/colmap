// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/online_sfm/online_feature_matcher.h"
#include "colmap/online_sfm/online_pose_graph.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class TimingStats;

struct OnlineLoopCloserOptions {
  bool enabled = true;
  // Required TensorRT MixVPR engine (.engine) path. No bundled fallback.
  std::string mixvpr_engine_path;
  bool mixvpr_use_gpu = true;
  std::string mixvpr_gpu_index = "-1";
  int cooldown_num_images = 10;
  // Candidates must be at least this many keyframes older than the query, so
  // temporal neighbours with broken covisibility are never loop candidates.
  int min_frame_gap = 20;
  // Camera-center radius (meters). <=0 disables the distance gate.
  double max_distance = 10.0;
  // Optical-axis angle (degrees). <=0 disables the heading gate.
  double max_view_angle_deg = 60.0;
  // Drop MixVPR 512-D cosines below this. <=0 disables the score gate.
  double min_mixvpr_score = 0.4;
  int topk = 5;
  int min_covisibility_points = 15;
  int min_num_matches = 20;
  // PnP inliers after guided projection matching.
  int min_num_verified_matches = 40;
  // 2D (current) – 3D (loop map) correspondences required to run PnP.
  int min_num_3d_correspondences = 20;
  // PnP RANSAC inliers before guided matching.
  int min_num_3d_inliers = 20;
  bool correct_local = true;
  // Fusion / loop-BA window: corrected frames that see the loop-side map.
  int max_correct_connected = 20;
  // Loop PnP reprojection threshold (pixels).
  double pnp_max_error = 3.0;
  // Guided projection search radius when growing PnP matches (pixels).
  double guided_radius_px = 7.5;
  // Loop rotation (PnP) vs essential-matrix rotation; <=0 disables.
  double max_rot_vs_essential_deg = 3.0;
  // Reject loops whose correction of the current pose is larger than this.
  double max_correction_m = 3.0;
  double max_correction_deg = 20.0;
  // Covisibility edges in the pose graph need at least this many shared points.
  int pose_graph_min_covis = 50;
  // Projection fusion after correction.
  double fuse_radius_px = 6.0;
  double fuse_max_point_distance = 0.2;
};

struct OnlineLoopPair {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  float retrieval_score = 0.0f;
  size_t num_matches = 0;
  size_t num_inliers = 0;
  size_t num_3d_correspondences = 0;
  size_t num_3d_inliers = 0;
  bool has_loop_pose = false;
  // Current-frame pose from PnP against the loop-side map.
  Rigid3d loop_cam_from_world;
  double correction_m = 0.0;
  double correction_deg = 0.0;
  double rot_vs_essential_deg = -1.0;
  // PnP inliers: (current point2D idx, loop-side point3D id).
  std::vector<std::pair<point2D_t, point3D_t>> loop_observations;
};

// Last successful loop correction, consumed by the mapper's anchored BA.
struct OnlineLoopCorrection {
  image_t current_id = kInvalidImageId;
  image_t loop_id = kInvalidImageId;
  std::vector<image_t> variable_ids;
  std::vector<image_t> anchor_ids;
  // Frames the loop-side map was projected into. Post-BA fusion reuses them.
  std::vector<image_t> fusion_ids;
  // Images whose 3D points are the loop-side map for that projection.
  std::vector<image_t> loop_point_image_ids;
};

class MixVprEncoder;

class OnlineLoopCloser {
 public:
  OnlineLoopCloser(std::string database_path, OnlineLoopCloserOptions options);
  ~OnlineLoopCloser();

  OnlineLoopCloser(const OnlineLoopCloser&) = delete;
  OnlineLoopCloser& operator=(const OnlineLoopCloser&) = delete;

  // True when yaml enables loop closing and MixVPR has not been fault-disabled.
  bool IsLoopClosureActive() const { return options_.enabled && !disabled_; }

  bool Process(image_t image_id,
               const std::string& image_abs_path,
               const DatabaseCache& cache,
               Reconstruction& reconstruction,
               ObservationManager* obs_manager,
               OnlineFeatureMatcher* matcher,
               const Bitmap* bitmap = nullptr);

  const std::vector<OnlineLoopPair>& ConfirmedLoops() const {
    return confirmed_loops_;
  }
  const OnlineLoopCorrection& LastCorrection() const {
    return last_correction_;
  }

  // Project the loop-side map into the fusion window again, after loop BA has
  // moved poses and points, and merge pairs now within the fuse distance.
  void FuseAfterLoopBA(Reconstruction& reconstruction,
                       ObservationManager* obs_manager);

 private:
  bool InitMixVprModel();
  bool StoreMixVprDescriptor(image_t image_id, const Eigen::VectorXf& desc);
  std::vector<std::pair<image_t, float>> DetectLoop(
      image_t image_id,
      const Eigen::VectorXf& query,
      const Reconstruction& reconstruction);
  // Matches for every candidate are already written. Reads that pair and runs
  // PnP only when the match count is high enough.
  bool VerifyCandidate(image_t image_id,
                       image_t candidate_id,
                       float retrieval_score,
                       const Reconstruction& reconstruction,
                       OnlineLoopPair* loop_pair);
  bool CorrectLoop(Reconstruction& reconstruction,
                   ObservationManager* obs_manager,
                   const OnlineLoopPair& loop_pair);
  // Candidate + its strongest covisibles (the old map PnP is solved against).
  std::vector<image_t> LoopMapImages(image_t candidate_id,
                                     image_t current_id,
                                     const Reconstruction& reconstruction) const;
  std::unordered_map<image_t, int> CountCovisibilityPoints(
      image_t image_id, const Reconstruction& reconstruction) const;
  std::unordered_set<image_t> CovisibilityGroup(
      image_t image_id, const Reconstruction& reconstruction) const;
  std::vector<image_t> RankedCovisible(
      image_t image_id, const Reconstruction& reconstruction) const;

  // Frozen PnP measurement. `query_from_map` is cam_query_from_cam_map at the
  // moment the loop was accepted, and is never recomputed from later poses.
  struct LoopConstraint {
    image_t map_id = kInvalidImageId;
    image_t query_id = kInvalidImageId;
    Rigid3d query_from_map;
    size_t num_inliers = 0;
  };

  double LoopConstraintWeight(size_t num_inliers, size_t ref_inliers) const;
  // Queries within the cooldown and map frames in one covisibility group are
  // the same place. The pose graph keeps a single edge per cluster.
  bool SameLoopCluster(image_t query_a,
                       image_t map_a,
                       image_t query_b,
                       image_t map_b,
                       const Reconstruction& reconstruction) const;
  bool HistoricalResidualPlausible(const Rigid3d& measured,
                                   const Rigid3d& current_rel) const;
  // Adds `current` plus historical constraints whose query frame is still in
  // `variable_set`. Returns how many historical edges were added.
  size_t AppendLoopEdges(const Reconstruction& reconstruction,
                         const std::unordered_set<image_t>& variable_set,
                         const LoopConstraint& current,
                         const PoseGraphOptions& graph_options,
                         std::unordered_map<image_t, Rigid3d>* poses_old,
                         std::unordered_set<image_t>* fixed_ids,
                         std::vector<PoseGraphEdge>* edges);
  void RememberLoopConstraint(const LoopConstraint& constraint);

  std::string database_path_;
  OnlineLoopCloserOptions options_;
  bool disabled_ = false;
  bool mixvpr_ready_ = false;
  std::unique_ptr<MixVprEncoder> mixvpr_;
  std::unique_ptr<TimingStats> timing_;
  std::vector<OnlineLoopPair> confirmed_loops_;
  // Successful loop corrections only. Not written to the database.
  std::vector<LoopConstraint> loop_constraints_;
  OnlineLoopCorrection last_correction_;
  image_t last_loop_image_id_ = kInvalidImageId;
};

}  // namespace colmap
