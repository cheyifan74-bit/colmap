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

#include "colmap/online_sfm/online_incremental_mapper.h"

#include "colmap/feature/sift.h"
#include "colmap/scene/database_session.h"
#include "colmap/util/logging.h"

#include <exception>

namespace colmap {

OnlineIncrementalMapper::OnlineIncrementalMapper(
    std::string database_path,
    FeatureExtractionOptions extraction_options,
    OnlineMatchingOptions matching_options)
    : database_path_(std::move(database_path)) {
  if (extraction_options.sift) {
    matching_options.max_num_features_hint =
        extraction_options.sift->max_num_features;
  }
  extractor_ = std::make_unique<OnlineFeatureExtractor>(
      database_path_, std::move(extraction_options));
  matcher_ = std::make_unique<OnlineFeatureMatcher>(
      database_path_, std::move(matching_options));
}

OnlineIncrementalMapper::~OnlineIncrementalMapper() = default;

bool OnlineIncrementalMapper::Process(
    const image_t image_id,
    const std::string& image_abs_path,
    const std::optional<Rigid3d>& cam_from_world_prior) {
  if (database_path_.empty()) {
    LOG(ERROR) << "OnlineIncrementalMapper: empty database path";
    return false;
  }
  if (image_id == kInvalidImageId) {
    LOG(ERROR) << "OnlineIncrementalMapper: invalid image_id";
    return false;
  }

  DatabaseCache::Options options;
  options.min_num_matches = 0;
  options.load_all_images = true;

  try {
    DatabaseSession session(database_path_);
    cache_ = DatabaseCache::Create(*session, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to load cache: " << e.what();
    cache_.reset();
    return false;
  }

  if (!cache_ || !cache_->ExistsImage(image_id)) {
    LOG(ERROR) << "OnlineIncrementalMapper: image " << image_id
               << " not in cache after Load (num_images="
               << (cache_ ? cache_->NumImages() : 0) << ")";
    return false;
  }

  last_image_id_ = image_id;
  last_image_abs_path_ = image_abs_path;
  last_cam_from_world_prior_ = cam_from_world_prior;

  LOG(INFO) << "OnlineIncrementalMapper: loaded cache with "
            << cache_->NumImages() << " image(s), current=" << image_id;

  OnlineFeatureExtractor::ExtractedFeatures extracted;
  if (!extractor_->ExtractAndWrite(
          image_id, image_abs_path, *cache_, &extracted)) {
    return false;
  }
  if (extracted.keypoints && extracted.descriptors) {
    matcher_->PutFeatures(
        image_id, extracted.keypoints, extracted.descriptors);
  }

  if (!matcher_->MatchAndWrite(image_id, *cache_)) {
    return false;
  }

  return true;
}

}  // namespace colmap
