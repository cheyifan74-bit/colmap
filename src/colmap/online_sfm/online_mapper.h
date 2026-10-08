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
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/types.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {

struct OnlineMapperOptions {
  int min_num_inliers = 15;
  int abs_pose_min_num_inliers = 50;
  double abs_pose_max_error = 1.0;
  double abs_pose_min_inlier_ratio = 0.25;
  int ba_local_num_images = 0;
  int ba_min_covisibility_points = 15;
  int ba_min_first_level_for_pose = 3;
  double ba_max_pose_center_jump = 0.5;
  double ba_max_pose_angle_deg = 15.0;
  double filter_max_reproj_error = 1.0;
  double filter_min_tri_angle = 1.5;
  IncrementalTriangulator::Options triangulation;

  OnlineMapperOptions() { triangulation.ignore_two_view_tracks = false; }
};

// Online monocular mapping: VIO-metric init, P3P with VIO fallback,
// IncrementalTriangulator, and a simplified local BA (fixed
// intrinsics).
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


  bool MappingCurrentImage(image_t image_id,
                           const std::optional<Rigid3d>& vio_prior);

  bool Initialized() const { return initialized_; }
  const Reconstruction& GetReconstruction() const { return *reconstruction_; }
  Reconstruction& GetReconstruction() { return *reconstruction_; }
  ObservationManager* GetObservationManager() { return obs_manager_.get(); }
  void WriteSparse() const;

 private:
  struct P3PRegisterResult {
    bool success = false;
    size_t num_inliers = 0;
    Rigid3d cam_from_world;
  };

  bool InitReconstruction();
  bool RegisterImageWithKnownPose(image_t image_id,
                                  const Rigid3d& cam_from_world);
  P3PRegisterResult RegisterNextImageP3P(image_t image_id);
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
  bool initialized_ = false;
};

}  // namespace colmap
