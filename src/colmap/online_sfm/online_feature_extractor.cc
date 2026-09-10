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

#include "colmap/online_sfm/online_feature_extractor.h"

#include "colmap/feature/sift.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/scene/database_session.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <exception>
#include <string>
#include <vector>

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

std::unique_ptr<TimingStats> CreateExtractTiming(
    const std::string& database_path,
    const FeatureExtractionOptions& options) {
  auto stats = std::make_unique<TimingStats>(
      TimingStats::FileBesideDatabase(database_path, "extract_timing.txt"),
      "extract",
      std::vector<std::string>{"num_features", "image"});
  stats->AddMeta("extractor",
                 std::string(FeatureExtractorTypeToString(options.type)));
  stats->AddMeta("use_gpu", options.use_gpu ? "1" : "0");
  stats->AddMeta("gpu_index", options.gpu_index);
  stats->AddMeta("scope", "FeatureExtractor::Extract");
  return stats;
}

}  // namespace

OnlineFeatureExtractor::OnlineFeatureExtractor(
    std::string database_path, FeatureExtractionOptions options)
    : database_path_(std::move(database_path)),
      options_(std::move(options)),
      timing_(CreateExtractTiming(database_path_, options_)) {}

OnlineFeatureExtractor::~OnlineFeatureExtractor() = default;

bool OnlineFeatureExtractor::InitExtractor() {
  if (extractor_) {
    return true;
  }

  if (options_.use_gpu) {
#if !defined(COLMAP_GPU_ENABLED)
    LOG(ERROR) << "OnlineFeatureExtractor: use_gpu=true but COLMAP was built "
                  "without GPU. Rebuild with CUDA_ENABLED=ON.";
    return false;
#endif
    if (options_.type == FeatureExtractorType::SIFT &&
        (options_.sift->estimate_affine_shape ||
         options_.sift->domain_size_pooling ||
         options_.sift->force_covariant_extractor)) {
      LOG(WARNING)
          << "OnlineFeatureExtractor: GPU SIFT is unavailable with "
             "affine/DSP/covariant SIFT; falling back to CPU covariant SIFT";
    }
  }
  if (!options_.Check()) {
    LOG(ERROR) << "OnlineFeatureExtractor: invalid extraction options type="
               << FeatureExtractorTypeToString(options_.type)
               << " use_gpu=" << options_.use_gpu
               << " gpu_index=" << options_.gpu_index;
    return false;
  }
  extractor_ = FeatureExtractor::Create(options_);
  if (!extractor_) {
    LOG(ERROR) << "OnlineFeatureExtractor: failed to create extractor type="
               << FeatureExtractorTypeToString(options_.type)
               << " use_gpu=" << options_.use_gpu
               << " gpu_index=" << options_.gpu_index;
    return false;
  }
  LOG(INFO) << "OnlineFeatureExtractor: extractor="
            << FeatureExtractorTypeToString(options_.type)
            << " use_gpu=" << options_.use_gpu
            << " gpu_index=" << options_.gpu_index
            << " max_image_size=" << options_.max_image_size;
  return true;
}

bool OnlineFeatureExtractor::ExtractAndWrite(
    const image_t image_id,
    const std::string& image_abs_path,
    const DatabaseCache& cache,
    ExtractedFeatures* features) {
  if (features != nullptr) {
    *features = {};
  }

  {
    DatabaseSession session(database_path_);
    if (session->ExistsKeypoints(image_id) &&
        session->ExistsDescriptors(image_id)) {
      LOG(INFO) << "OnlineFeatureExtractor: skip extract, image " << image_id
                << " already has features";
      return true;
    }
  }

  if (image_abs_path.empty()) {
    LOG(ERROR) << "OnlineFeatureExtractor: empty image path for image "
               << image_id;
    return false;
  }

  Bitmap bitmap;
  if (!bitmap.Read(image_abs_path, options_.RequiresRGB())) {
    LOG(ERROR) << "OnlineFeatureExtractor: failed to read image "
               << image_abs_path;
    return false;
  }

  if (!InitExtractor()) {
    return false;
  }

  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  timing_->Start();
  const bool extract_ok =
      extractor_->Extract(bitmap, keypoints.get(), descriptors.get());
  if (!extract_ok) {
    LOG(ERROR) << "OnlineFeatureExtractor: extraction failed for image "
               << image_id
               << " extract_ms=" << timing_->ElapsedMilliseconds();
    return false;
  }
  timing_->Record(std::to_string(image_id),
                  {std::to_string(keypoints->size()), image_abs_path});

  if (cache.ExistsImage(image_id)) {
    const Image& image = cache.Image(image_id);
    if (image.HasCameraId() && cache.ExistsCamera(image.CameraId())) {
      ScaleKeypointsToCamera(
          bitmap, cache.Camera(image.CameraId()), keypoints.get());
    }
  }

  try {
    DatabaseSession session(database_path_);
    if (!session->ExistsKeypoints(image_id)) {
      session->WriteKeypoints(image_id, *keypoints);
    }
    if (!session->ExistsDescriptors(image_id)) {
      session->WriteDescriptors(image_id, *descriptors);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineFeatureExtractor: failed to write features: "
               << e.what();
    return false;
  }

  if (features != nullptr) {
    features->keypoints = keypoints;
    features->descriptors = descriptors;
  }

  LOG(INFO) << "OnlineFeatureExtractor: extract image " << image_id << " -> "
            << keypoints->size() << " features type="
            << FeatureExtractorTypeToString(descriptors->type);
  return true;
}

}  // namespace colmap
