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

#include "colmap/geometry/sim3.h"
#include "colmap/online_sfm/online_feature_matcher.h"
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
  // Camera-center radius (meters). <=0 disables the distance gate.
  double max_distance = 10.0;
  // Optical-axis angle (degrees). <=0 disables the heading gate.
  double max_view_angle_deg = 60.0;
  // Drop MixVPR 512-D cosines below this. <=0 disables the score gate.
  double min_mixvpr_score = 0.4;
  int topk = 5;
  int min_covisibility_points = 15;
  int min_num_matches = 20;
  int min_num_verified_matches = 40;
  int min_num_3d_correspondences = 20;
  int min_num_3d_inliers = 20;
  bool correct_local = true;
  int max_correct_connected = 20;
  // false: estimate Sim3 scale (7DoF); true: fix s=1 (SE3 / stereo ORB).
  bool fix_scale = false;
};

struct OnlineLoopPair {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  float retrieval_score = 0.0f;
  size_t num_matches = 0;
  size_t num_inliers = 0;
  size_t num_3d_correspondences = 0;
  size_t num_3d_inliers = 0;
  bool used_sim3 = false;
  double scale = 1.0;
  Sim3d cand_from_cur;
  std::vector<std::pair<point3D_t, point3D_t>> matched_point3Ds;
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

 private:
  bool InitMixVprModel();
  bool StoreMixVprDescriptor(image_t image_id, const Eigen::VectorXf& desc);
  std::vector<std::pair<image_t, float>> DetectLoop(
      image_t image_id,
      const Eigen::VectorXf& query,
      const Reconstruction& reconstruction);
  bool VerifyCandidate(image_t image_id,
                       image_t candidate_id,
                       float retrieval_score,
                       const std::string& query_abs_path,
                       const DatabaseCache& cache,
                       const Reconstruction& reconstruction,
                       OnlineFeatureMatcher* matcher,
                       OnlineLoopPair* loop_pair);
  bool CorrectLocalLoop(Reconstruction& reconstruction,
                        ObservationManager* obs_manager,
                        const OnlineLoopPair& loop_pair);
  // Smallest strong-covisible image id (incl. self). Older map side → smaller.
  image_t MinCovisibilityImageId(image_t image_id,
                                 const Reconstruction& reconstruction) const;
  std::unordered_map<image_t, int> CountCovisibilityPoints(
      image_t image_id, const Reconstruction& reconstruction) const;
  std::unordered_set<image_t> CovisibilityGroup(
      image_t image_id, const Reconstruction& reconstruction) const;
  std::vector<image_t> RankedCovisible(
      image_t image_id, const Reconstruction& reconstruction) const;
  void DumpMixVprTopkImages(
      image_t query_id,
      const std::string& query_abs_path,
      const DatabaseCache& cache,
      const std::vector<std::pair<image_t, float>>& ranked) const;

  std::string database_path_;
  std::string mixvpr_topk_dir_;
  std::string loop_verify_dir_;
  OnlineLoopCloserOptions options_;
  bool disabled_ = false;
  bool mixvpr_ready_ = false;
  std::unique_ptr<MixVprEncoder> mixvpr_;
  std::unique_ptr<TimingStats> timing_;
  std::vector<OnlineLoopPair> confirmed_loops_;
  image_t last_loop_image_id_ = kInvalidImageId;
};

}  // namespace colmap
