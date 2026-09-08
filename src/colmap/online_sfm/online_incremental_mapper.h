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

#include "colmap/feature/extractor.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_cache.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace colmap {

// Online wrapper around incremental SfM. Each Process() call currently:
//   path-lock Open -> full DatabaseCache::Load (load_all_images) -> Close
//   SIFT -> path-lock Open -> WriteKeypoints / WriteDescriptors -> Close
// Match / P3P / triangulate / local BA are added in later steps.
// VIO pose is stored as a prior only; it is not written as a registered pose.
class OnlineIncrementalMapper {
 public:
  explicit OnlineIncrementalMapper(
      std::string database_path,
      FeatureExtractionOptions extraction_options =
          FeatureExtractionOptions(FeatureExtractorType::SIFT))
      : database_path_(std::move(database_path)),
        extraction_options_(std::move(extraction_options)) {
    extraction_options_.type = FeatureExtractorType::SIFT;
  }
  ~OnlineIncrementalMapper() = default;

  OnlineIncrementalMapper(const OnlineIncrementalMapper&) = delete;
  OnlineIncrementalMapper& operator=(const OnlineIncrementalMapper&) = delete;

  // `image_id` must already exist in the database (WriteImage happened first).
  // `cam_from_world_prior` is optional VIO left-camera pose in COLMAP convention.
  bool Process(image_t image_id,
               const std::string& image_abs_path,
               const std::optional<Rigid3d>& cam_from_world_prior =
                   std::nullopt);

  const std::shared_ptr<DatabaseCache>& Cache() const { return cache_; }
  image_t LastImageId() const { return last_image_id_; }
  const std::string& LastImageAbsPath() const { return last_image_abs_path_; }
  const std::optional<Rigid3d>& LastCamFromWorldPrior() const {
    return last_cam_from_world_prior_;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  bool ExtractAndWriteSift(image_t image_id, const std::string& image_abs_path);

  std::string database_path_;
  FeatureExtractionOptions extraction_options_;
  std::shared_ptr<DatabaseCache> cache_;
  std::unique_ptr<FeatureExtractor> sift_extractor_;
  image_t last_image_id_ = kInvalidImageId;
  std::string last_image_abs_path_;
  std::optional<Rigid3d> last_cam_from_world_prior_;
};

}  // namespace colmap
