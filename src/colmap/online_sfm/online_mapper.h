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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ceres/loss_function.h>
#include <ceres/problem.h>

namespace colmap {

struct OnlineMapperOptions {
  int min_num_inliers = 15;
  int abs_pose_min_num_inliers = 50;
  double abs_pose_max_error = 1.0;
  double abs_pose_min_inlier_ratio = 0.25;
  int ba_local_num_images = 0;
  int ba_min_covisibility_points = 15;
  double filter_max_reproj_error = 1.0;
  double filter_min_tri_angle = 1.5;
  // Local-map projection matching: points seen by the last N registered
  // frames are projected into the current frame and matched by descriptor.
  // Before registration the VIO-predicted pose is used with
  // `local_map_search_radius_px`; after registration the refined pose with
  // `local_map_refine_radius_px`. <= 0 images disables.
  int local_map_num_images = 15;
  double local_map_search_radius_px = 15.0;
  double local_map_refine_radius_px = 4.0;
  // SIFT L2 distance on unit-normalized descriptors.
  double local_map_max_desc_distance = 0.7;
  double local_map_max_ratio = 0.9;
  
  double ba_vio_prior_rot_sigma_deg = 1.0;
  double ba_vio_prior_trans_sigma_m = 0.02;
  double ba_vio_prior_trans_sigma_ratio = 0.03;
  int ba_vio_prior_max_gap = 5;
  // Local BA windows smaller than this (current + covisible frames) are padded
  // with up to `ba_temporal_window_images` most recent registered frames.
  int ba_min_window_images = 4;
  int ba_temporal_window_images = 5;
  // Write the full sparse model every N successful map updates.
  // <=1 writes every frame. Init, loop correction, and shutdown always flush.
  int sparse_write_every_n = 10;
  IncrementalTriangulator::Options triangulation;

  OnlineMapperOptions() { triangulation.ignore_two_view_tracks = false; }
};

class OnlineMapper {
 public:
  OnlineMapper(std::shared_ptr<DatabaseCache> database_cache,
               std::string sparse_path,
               OnlineMapperOptions options = {});
  ~OnlineMapper();

  OnlineMapper(const OnlineMapper&) = delete;
  OnlineMapper& operator=(const OnlineMapper&) = delete;

 
  bool SyncReconstruction();


  void NotifyNewImagePair(image_t image_id1, image_t image_id2);

  // Descriptors in keypoint order, used by local-map projection matching.
  void CacheDescriptors(image_t image_id,
                        std::shared_ptr<const FeatureDescriptors> descriptors);


  bool MappingCurrentImage(image_t image_id,
                           const std::optional<Rigid3d>& vio_prior);

  bool Initialized() const { return initialized_; }
  const Reconstruction& GetReconstruction() const { return *reconstruction_; }
  Reconstruction& GetReconstruction() { return *reconstruction_; }
  ObservationManager* GetObservationManager() { return obs_manager_.get(); }
  void WriteSparse() const;
  // Throttled write used on the per-keyframe path.
  void MaybeWriteSparse();
  // Local BA after loop correction: `variable_ids` refined, `anchor_ids`
  // (loop-side old map) held constant.
  void AdjustLoopBundle(const std::vector<image_t>& variable_ids,
                        const std::vector<image_t>& anchor_ids);

 private:
  struct PriorRegisterResult {
    bool success = false;
    size_t num_inliers = 0;
    Rigid3d cam_from_world;
  };

  bool InitReconstruction();
  bool RegisterImageWithKnownPose(image_t image_id,
                                  const Rigid3d& cam_from_world);
  PriorRegisterResult RegisterNextImageWithPrior(image_t image_id);
  struct ProjectionMatch {
    point2D_t point2D_idx = kInvalidPoint2DIdx;
    point3D_t point3D_id = kInvalidPoint3DId;
  };
  std::vector<ProjectionMatch> SearchLocalMapByProjection(
      image_t image_id,
      const Rigid3d& cam_from_world,
      double radius_px,
      const FlatHashSet<point2D_t>& skip_points2D,
      const FlatHashSet<point3D_t>& skip_points3D) const;
  // After registration: attach local-map points to unassociated keypoints.
  size_t ExtendTracksByProjection(image_t image_id, const char* stage);
  size_t AddVioPriorsToLocalBundle(const std::vector<image_t>& image_ids,
                                   Reconstruction& reconstruction,
                                   ceres::Problem* problem,
                                   ceres::LossFunction* loss) const;
  Rigid3d MapFrameCamFromWorld(image_t image_id, const Rigid3d& vio_prior) const;
  bool RegisterWithRelativeVio(image_t image_id, const Rigid3d& vio_prior);
  image_t FindLastRegisteredWithPrior(image_t image_id) const;
  bool TryInitialize(image_t image_id);
  void UnregisterFrame(image_t image_id);
  size_t TriangulateImage(image_t image_id);
  void MergeAndFilterPoints(image_t image_id);
  void AdjustLocalBundle(image_t image_id);
  bool CollectRegisteredImage(image_t image_id,
                              std::vector<image_t>* images) const;
  void BuildCovisibilityBundle(image_t image_id,
                               std::vector<image_t>* local_ids,
                               std::vector<image_t>* constant_ids) const;
  void AppendLbaPoseRecord(image_t image_id,
                           const std::vector<image_t>& first_level_ids,
                           const std::vector<image_t>& second_level_ids,
                           const Rigid3d& pose_before,
                           const Rigid3d& pose_after,
                           bool usable,
                           bool jumped,
                           bool reverted) const;
  image_t FindInitPartner(image_t image_id) const;

  std::shared_ptr<DatabaseCache> database_cache_;
  std::string sparse_path_;
  OnlineMapperOptions options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::shared_ptr<ObservationManager> obs_manager_;
  std::shared_ptr<IncrementalTriangulator> triangulator_;
  std::unordered_map<image_t, Rigid3d> pose_priors_;
  std::unordered_map<image_t, std::shared_ptr<const FeatureDescriptors>>
      descriptors_;
  std::unique_ptr<TimingStats> lba_timing_;
  bool initialized_ = false;
  // Map updates since the last sparse flush. Reset by WriteSparse().
  mutable int sparse_writes_since_flush_ = 0;
};

}  // namespace colmap
