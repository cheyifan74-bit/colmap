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

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/types.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/util/types.h"

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {

class TimingStats;

// Linear sequential matching against the previous `overlap` keyframes.
// Spatial pairing can be added later without changing the write path.
// Geometry defaults skip COLMAP's offline F/H/watermark path and use a
// shorter RANSAC budget; `e_only` estimates the essential matrix only.
struct OnlineMatchingOptions {
  int overlap = 10;
  bool e_only = true;
  // Caps GPU max_num_matches together with Database::MaxNumKeypoints().
  int max_num_features_hint = 0;
  FeatureMatchingOptions matching =
      FeatureMatchingOptions(FeatureMatcherType::SIFT_BRUTEFORCE);
  TwoViewGeometryOptions geometry;

  OnlineMatchingOptions() {
    geometry.detect_watermark = false;
    geometry.use_degensac = false;
    geometry.ransac_options.min_num_trials = 30;
    geometry.ransac_options.max_num_trials = 500;
    geometry.ransac_options.confidence = 0.99;
  }
};

// Online 1-to-N matching: current keyframe vs temporal overlap, then
// WriteMatches / WriteTwoViewGeometry. Owns the sliding feature window,
// Faiss index cache, and GPU or CPU matcher workers.
class OnlineFeatureMatcher {
 public:
  OnlineFeatureMatcher(std::string database_path,
                       OnlineMatchingOptions options);
  ~OnlineFeatureMatcher();

  OnlineFeatureMatcher(const OnlineFeatureMatcher&) = delete;
  OnlineFeatureMatcher& operator=(const OnlineFeatureMatcher&) = delete;

  void PutFeatures(image_t image_id,
                   std::shared_ptr<const FeatureKeypoints> keypoints,
                   std::shared_ptr<const FeatureDescriptors> descriptors);

  bool MatchAndWrite(image_t image_id, const DatabaseCache& cache);

 private:
  struct FeatureCacheEntry {
    std::shared_ptr<const FeatureKeypoints> keypoints;
    std::shared_ptr<const FeatureDescriptors> descriptors;
  };
  struct CpuMatchContext;

  bool InitMatchers();
  FeatureCacheEntry LoadFeatures(Database& database, image_t image_id);
  std::vector<image_t> SelectTemporalOverlapImages(
      image_t image_id, const DatabaseCache& cache) const;

  std::string database_path_;
  OnlineMatchingOptions options_;
  std::unique_ptr<FeatureMatcher> gpu_matcher_;
  std::unique_ptr<CpuMatchContext> cpu_match_;
  std::unique_ptr<TimingStats> timing_;
  std::unordered_map<image_t, FeatureCacheEntry> feature_cache_;
  std::deque<image_t> feature_cache_order_;
};

}  // namespace colmap
