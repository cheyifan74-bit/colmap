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

#include "colmap/online_sfm/online_incremental_mapper.h"

#include "colmap/feature/extractor.h"
#include "colmap/feature/sift.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/scene/database_session.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

namespace colmap {
namespace {

void ScaleKeypointsToCamera(const Bitmap& bitmap,
                            const Camera& camera,
                            FeatureKeypoints* keypoints) {
  if (static_cast<size_t>(bitmap.Width()) == camera.width &&
      static_cast<size_t>(bitmap.Height()) == camera.height) {
    return;
  }
  const float scale_x =
      static_cast<float>(camera.width) / static_cast<float>(bitmap.Width());
  const float scale_y =
      static_cast<float>(camera.height) / static_cast<float>(bitmap.Height());
  for (auto& keypoint : *keypoints) {
    keypoint.Rescale(scale_x, scale_y);
  }
}

}  // namespace

OnlineIncrementalMapper::OnlineIncrementalMapper(
    std::string database_path, FeatureExtractionOptions extraction_options)
    : database_path_(std::move(database_path)),
      extraction_options_(std::move(extraction_options)),
      extract_timing_(std::make_unique<TimingStats>(
          TimingStats::FileBesideDatabase(database_path_, "extract_timing.txt"),
          "extract",
          std::vector<std::string>{"num_features", "image"})) {
  extract_timing_->AddMeta(
      "extractor",
      std::string(FeatureExtractorTypeToString(extraction_options_.type)));
  extract_timing_->AddMeta("use_gpu",
                           extraction_options_.use_gpu ? "1" : "0");
  extract_timing_->AddMeta("gpu_index", extraction_options_.gpu_index);
  extract_timing_->AddMeta("scope", "FeatureExtractor::Extract");
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

  if (!ExtractAndWriteFeatures(image_id, image_abs_path)) {
    return false;
  }

  return true;
}

bool OnlineIncrementalMapper::ExtractAndWriteFeatures(
    const image_t image_id, const std::string& image_abs_path) {
  {
    DatabaseSession session(database_path_);
    if (session->ExistsKeypoints(image_id) &&
        session->ExistsDescriptors(image_id)) {
      LOG(INFO) << "OnlineIncrementalMapper: skip extract, image " << image_id
                << " already has features";
      return true;
    }
  }

  if (image_abs_path.empty()) {
    LOG(ERROR) << "OnlineIncrementalMapper: empty image path for image "
               << image_id;
    return false;
  }

  Bitmap bitmap;
  if (!bitmap.Read(image_abs_path, extraction_options_.RequiresRGB())) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to read image "
               << image_abs_path;
    return false;
  }

  if (!feature_extractor_) {
    if (extraction_options_.use_gpu) {
#if !defined(COLMAP_GPU_ENABLED)
      LOG(ERROR)
          << "OnlineIncrementalMapper: use_gpu=true but COLMAP was built "
             "without GPU. Rebuild with CUDA_ENABLED=ON.";
      return false;
#endif
      if (extraction_options_.type == FeatureExtractorType::SIFT &&
          (extraction_options_.sift->estimate_affine_shape ||
           extraction_options_.sift->domain_size_pooling ||
           extraction_options_.sift->force_covariant_extractor)) {
        LOG(WARNING)
            << "OnlineIncrementalMapper: GPU SIFT is unavailable with "
               "affine/DSP/covariant SIFT; falling back to CPU covariant SIFT";
      }
    }
    if (!extraction_options_.Check()) {
      LOG(ERROR) << "OnlineIncrementalMapper: invalid extraction options "
                    "type="
                 << FeatureExtractorTypeToString(extraction_options_.type)
                 << " use_gpu=" << extraction_options_.use_gpu
                 << " gpu_index=" << extraction_options_.gpu_index;
      return false;
    }
    feature_extractor_ = FeatureExtractor::Create(extraction_options_);
    if (!feature_extractor_) {
      LOG(ERROR) << "OnlineIncrementalMapper: failed to create extractor "
                    "type="
                 << FeatureExtractorTypeToString(extraction_options_.type)
                 << " use_gpu=" << extraction_options_.use_gpu
                 << " gpu_index=" << extraction_options_.gpu_index;
      return false;
    }
    LOG(INFO) << "OnlineIncrementalMapper: extractor="
              << FeatureExtractorTypeToString(extraction_options_.type)
              << " use_gpu=" << extraction_options_.use_gpu
              << " gpu_index=" << extraction_options_.gpu_index
              << " max_image_size=" << extraction_options_.max_image_size;
  }

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  extract_timing_->Start();
  const bool extract_ok =
      feature_extractor_->Extract(bitmap, &keypoints, &descriptors);
  if (!extract_ok) {
    LOG(ERROR) << "OnlineIncrementalMapper: extraction failed for image "
               << image_id
               << " extract_ms=" << extract_timing_->ElapsedMilliseconds();
    return false;
  }
  extract_timing_->Record(
      std::to_string(image_id),
      {std::to_string(keypoints.size()), image_abs_path});

  if (cache_->ExistsImage(image_id)) {
    const Image& image = cache_->Image(image_id);
    if (image.HasCameraId() && cache_->ExistsCamera(image.CameraId())) {
      ScaleKeypointsToCamera(
          bitmap, cache_->Camera(image.CameraId()), &keypoints);
    }
  }

  try {
    DatabaseSession session(database_path_);
    if (!session->ExistsKeypoints(image_id)) {
      session->WriteKeypoints(image_id, keypoints);
    }
    if (!session->ExistsDescriptors(image_id)) {
      session->WriteDescriptors(image_id, descriptors);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to write features: "
               << e.what();
    return false;
  }

  LOG(INFO) << "OnlineIncrementalMapper: extract image " << image_id << " -> "
            << keypoints.size() << " features type="
            << FeatureExtractorTypeToString(descriptors.type);
  return true;
}

}  // namespace colmap
