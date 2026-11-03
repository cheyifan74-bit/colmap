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

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  // Feature scale consistency (BAFS observation model s = f * S / range):
  // the point's size S from its reference observations predicts its scale in
  // the current frame, and keypoints off by more than this factor are not
  // match candidates. <= 1 disables.
  double scale_check_max_ratio = 0.0;
  // Resumed maps: the window also takes up to N previous-session keyframes
  // near the predicted pose, so revisits attach to the old map instead of
  // growing a duplicate layer that only loop fusion would stitch. <= 0
  // disables.
  int local_map_spatial_num_images = 0;
  double local_map_spatial_max_distance_m = 2.0;
  double local_map_spatial_max_angle_deg = 45.0;
  
  double ba_vio_prior_rot_sigma_deg = 1.0;
  double ba_vio_prior_trans_sigma_m = 0.02;
  double ba_vio_prior_trans_sigma_ratio = 0.03;
  int ba_vio_prior_max_gap = 5;
  // Local BA holds the oldest window frame with at least this many 3D points
  // constant, falling back to the oldest frame. 0 = always the oldest.
  int ba_fixed_min_points = 0;
  // Loop BA refines the whole pose-graph-corrected chain, so it needs more
  // iterations than the per-keyframe local BA (fixed at 10).
  int ba_loop_max_num_iterations = 50;
  // Local BA windows smaller than this (current + covisible frames) are padded
  // with up to `ba_temporal_window_images` most recent registered frames.
  int ba_min_window_images = 4;
  int ba_temporal_window_images = 5;
  // Write the full sparse model every N successful map updates.
  // <=1 writes every frame. Init, loop correction, and shutdown always flush.
  int sparse_write_every_n = 10;
  // Redundant keyframe culling over the covisible frames of the newest
  // keyframe. A frame is removed when at least `cull_redundant_ratio` of its
  // 3D points are observed by `cull_min_other_observers` other keyframes from
  // no farther than `cull_max_scale_ratio` x its own viewing distance.
  bool cull_enabled = false;
  double cull_redundant_ratio = 0.9;
  int cull_min_other_observers = 3;
  double cull_max_scale_ratio = 1.2;
  // Frames with fewer 3D points carry the VIO chain through weak texture.
  int cull_min_num_points = 30;
  // The newest N registered keyframes are never culled.
  int cull_protect_recent = 5;
  // The VIO-prior neighbours left around a culled frame must stay this close.
  double cull_max_neighbor_distance_m = 3.0;
  double cull_max_neighbor_angle_deg = 45.0;
  int cull_max_per_frame = 1;
  // Frames restored from a previous session have no VIO prior in this one.
  bool cull_prior_session = false;

  // Keyframe insertion check. A frame registered by PnP next to an existing
  // reference keyframe is dropped when it tracks about as many map points as
  // that keyframe holds (ORB-SLAM's reference ratio) and most of its points
  // are already seen by other keyframes.
  bool insert_check_enabled = false;
  double insert_redundant_ratio = 0.75;
  double insert_ref_tracked_ratio = 0.9;
  // Fewer tracked 3D points always inserts.
  int insert_min_tracked = 40;
  // An existing keyframe this close must share at least half of the tracked
  // points, i.e. the frame is consistent with that part of the map.
  double insert_max_kf_distance_m = 1.0;
  double insert_max_kf_angle_deg = 30.0;
  // Force a keyframe (and a loop-closure query) after this many drops.
  int insert_max_consecutive_skips = 10;

  // Per-point visibility statistics: a point predicted in view but not
  // matched counts as visible only. Points with at least `stale_min_visible`
  // checks and found/visible below `stale_max_found_ratio` are removed.
  // A check counts only when a keypoint was detected near the projection.
  bool point_stats_enabled = false;
  int stale_min_visible = 15;
  double stale_max_found_ratio = 0.25;
  // Only points not observed by any keyframe of this session are judged.
  bool stale_prior_session_only = true;
  // Statistics loaded from the previous session are scaled by this, so recent
  // sessions dominate the found ratio.
  double stale_session_decay = 0.5;

  // Spatial capacity: at most `capacity_max_per_cell` keyframes per voxel and
  // viewing-yaw bin, scored by the sum of their points' found ratios. When a
  // new keyframe overfills a cell it is the one dropped, unless the worst
  // existing keyframe scores below `capacity_replace_ratio` x the new one.
  bool capacity_enabled = false;
  double capacity_voxel_m = 1.0;
  int capacity_yaw_bins = 4;
  int capacity_max_per_cell = 2;
  double capacity_replace_ratio = 0.5;
  // Keyframes of previous sessions are replaced only after this session has
  // mapped this many keyframes since initialization / relocalization.
  int capacity_min_session_frames = 20;
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
  // Keypoint scales in keypoint order, used by the scale consistency check.
  void CacheKeypointScales(image_t image_id, const FeatureKeypoints& keypoints);
  // How many recent registered frames the local-map projection matching uses.
  // Callers resuming a map preload this window's descriptors.
  size_t LocalMapWindowSize() const;


  bool MappingCurrentImage(image_t image_id,
                           const std::optional<Rigid3d>& vio_prior);

  // Load `sparse_path_` from disk and rebuild the runtime state (observation
  // manager / triangulator) so mapping continues on the existing map.
  // `registered_image_names` receives the names of the loaded registered
  // images; the caller uses them to build the filtered database cache.
  bool LoadSparseMap(std::vector<std::string>* registered_image_names);
  // Second half of the restore, after the database cache holds those images.
  bool RebuildAfterRestore();
  // Register a relocalized frame: `cam_from_world` is the map pose from PnP,
  // `vio_cam_from_world` the VIO-frame pose. Only the VIO pose goes into the
  // prior chain, so later frames inherit the map pose and not the VIO one.
  bool RegisterRelocalizedFrame(
      image_t image_id,
      const Rigid3d& cam_from_world,
      const Rigid3d& vio_cam_from_world,
      const std::vector<std::pair<point2D_t, point3D_t>>& loop_observations);
  bool HasMappedFrames() const { return initialized_; }
  void MarkInitialized() { initialized_ = true; }

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

  // Remove keyframes made redundant by `image_id` from its covisible window.
  // `protected_ids` (loop edges, anchors) are kept. Returns the culled ids.
  std::vector<image_t> CullRedundantKeyframes(
      image_t image_id, const std::unordered_set<image_t>& protected_ids);
  // Evict the lowest scored keyframes of the cell `image_id` falls into while
  // the cell holds more than `capacity_max_per_cell`. Returns the evicted ids.
  std::vector<image_t> EnforceSpatialCapacity(
      image_t image_id, const std::unordered_set<image_t>& protected_ids);
  // Deregister a keyframe and drop its mapper state. The database rows stay;
  // the next resume skips it because it is no longer in the sparse model.
  // `reason` / `score` go to culled_keyframes.txt for the database compaction.
  bool RemoveKeyframe(image_t image_id,
                      const char* reason = "manual",
                      double score = 0.0);
  const std::set<image_t>& CulledImageIds() const { return culled_ids_; }
  // True when the last MappingCurrentImage dropped its frame as a
  // non-keyframe (localized against the map, no new information).
  bool LastFrameSkipped() const { return last_frame_skipped_; }
  // Loads descriptors of keyframes that are not cached (previous session),
  // so their points can be matched by projection.
  using DescriptorLoader =
      std::function<std::shared_ptr<const FeatureDescriptors>(image_t)>;
  void SetDescriptorLoader(DescriptorLoader loader) {
    descriptor_loader_ = std::move(loader);
  }
  using KeypointLoader =
      std::function<std::shared_ptr<const FeatureKeypoints>(image_t)>;
  void SetKeypointLoader(KeypointLoader loader) {
    keypoint_loader_ = std::move(loader);
  }
  // tgt_from_src camera motion measured by VIO. Both frames must carry a VIO
  // prior from the same session (this one or a stored one): each session has
  // its own VIO world.
  std::optional<Rigid3d> VioRelativePose(image_t tgt_id, image_t src_id) const;

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
      const FlatHashSet<point3D_t>& skip_points3D);
  // Previous-session keyframes closest to `cam_from_world`, nearest first.
  std::vector<image_t> SpatialLocalMapImages(
      image_t image_id,
      const Rigid3d& cam_from_world,
      const std::vector<image_t>& exclude_ids) const;
  // Projection matching of `candidate_points`. Points that project into the
  // image within the viewing-angle gate, next to at least one unassociated
  // keypoint, are appended to `visible_points`.
  std::vector<ProjectionMatch> SearchPointsByProjection(
      image_t image_id,
      const Rigid3d& cam_from_world,
      double radius_px,
      const FlatHashSet<point2D_t>& skip_points2D,
      const FlatHashSet<point3D_t>& candidate_points,
      std::vector<point3D_t>* visible_points) const;
  // After registration: attach local-map points to unassociated keypoints.
  size_t ExtendTracksByProjection(image_t image_id, const char* stage);
  // Frames whose pose block is missing from `problem` (no observations) are
  // held constant, unless listed in `free_ids`: those are added as variable
  // poses constrained by the VIO priors alone.
  size_t AddVioPriorsToLocalBundle(
      const std::vector<image_t>& image_ids,
      Reconstruction& reconstruction,
      ceres::Problem* problem,
      ceres::LossFunction* loss,
      const FlatHashSet<image_t>* free_ids = nullptr) const;
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
  // Id distance with culled keyframes skipped, so gap thresholds count the
  // keyframes that still exist.
  image_t KeyframeGap(image_t id_i, image_t id_j) const;
  // Fraction of `image_id`'s 3D points seen by enough closer-or-equal other
  // keyframes; `num_points` receives its 3D point count.
  double RedundantPointRatio(image_t image_id, size_t* num_points) const;
  // Nearest registered frames with a VIO prior before / after `image_id`.
  std::pair<image_t, image_t> PriorChainNeighbors(image_t image_id) const;
  // Removing `image_id` keeps its VIO-prior neighbours close and within the
  // prior gap. Frames without a prior are not part of the chain.
  bool CanRemoveFromChain(image_t image_id) const;
  // Shared protection rules of culling and capacity eviction.
  bool IsRemovable(image_t image_id,
                   image_t current_id,
                   const FlatHashSet<image_t>& recent_ids,
                   const std::unordered_set<image_t>& protected_ids,
                   bool allow_prior_session) const;
  FlatHashSet<image_t> RecentRegisteredIds() const;
  void AppendCullRecord(image_t image_id,
                        const char* reason,
                        double score) const;

  // Non-keyframe decision for a frame just registered by PnP.
  bool ShouldSkipKeyframe(image_t image_id) const;
  void SkipFrame(image_t image_id);

  // Visibility statistics of the points around `image_id` (registered) and
  // removal of the stale ones.
  void UpdatePointStats(image_t image_id);
  void EnsureDescriptors(const FlatHashSet<image_t>& image_ids);
  double PointQuality(point3D_t point3D_id) const;
  std::string PointStatsPath() const;
  void WritePointStats() const;
  void ReadPointStats();
  // VIO priors of every registered frame, so resumed sessions keep the
  // relative constraints between previous-session keyframes.
  const Rigid3d* FindVioPrior(image_t image_id, uint32_t* session) const;
  std::string VioPriorsPath() const;
  void WriteVioPriors() const;
  void ReadVioPriors();

  std::shared_ptr<DatabaseCache> database_cache_;
  std::string sparse_path_;
  OnlineMapperOptions options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  std::shared_ptr<ObservationManager> obs_manager_;
  std::shared_ptr<IncrementalTriangulator> triangulator_;
  std::unordered_map<image_t, Rigid3d> pose_priors_;
  // Previous-session priors. `rank` is the frame's position in its session's
  // keyframe chain at load time and replaces KeyframeGap for those frames.
  struct StoredPrior {
    uint32_t session = 0;
    size_t rank = 0;
    Rigid3d vio_cam_from_world;
  };
  std::unordered_map<image_t, StoredPrior> stored_priors_;
  std::unordered_map<image_t, std::shared_ptr<const FeatureDescriptors>>
      descriptors_;
  std::unordered_map<image_t, std::vector<float>> keypoint_scales_;
  // Keypoints skipped by the scale check over the mapper's lifetime.
  mutable size_t num_scale_rejected_ = 0;
  std::unique_ptr<TimingStats> lba_timing_;
  std::set<image_t> culled_ids_;
  // Relocalization anchors tie this session to the stored map.
  std::unordered_set<image_t> reloc_anchor_ids_;

  struct PointStat {
    float num_visible = 0.0f;
    float num_found = 0.0f;
    uint32_t last_seen_session = 0;
  };
  std::unordered_map<point3D_t, PointStat> point_stats_;
  // Incremented every time a map is resumed from disk.
  uint32_t session_index_ = 0;
  DescriptorLoader descriptor_loader_;
  KeypointLoader keypoint_loader_;

  // Pose of the last dropped (non-keyframe) frame. Later frames chain their
  // VIO prior from it when it is newer than the last registered keyframe.
  struct TrackingAnchor {
    image_t image_id = kInvalidImageId;
    Rigid3d vio_cam_from_world;
    Rigid3d cam_from_world;
  };
  std::optional<TrackingAnchor> tracking_anchor_;
  int consecutive_skips_ = 0;
  int consecutive_capacity_drops_ = 0;
  bool last_frame_skipped_ = false;
  // Keyframes mapped since initialization / relocalization in this session.
  size_t session_mapped_frames_ = 0;
  bool initialized_ = false;
  // Map updates since the last sparse flush. Reset by WriteSparse().
  mutable int sparse_writes_since_flush_ = 0;
};

}  // namespace colmap
