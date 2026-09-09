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

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <sstream>

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

std::string FormatMs(const double ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ms;
  return oss.str();
}

FeatureMatches ExtractInlierMatches(const FeatureMatches& matches,
                                    const size_t num_inliers,
                                    const std::vector<char>& inlier_mask) {
  FeatureMatches inlier_matches(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_matches[j] = matches[i];
      j += 1;
    }
  }
  return inlier_matches;
}

// Calibrated essential-matrix only. Skips COLMAP's F, H, and watermark RANSAC.
TwoViewGeometry EstimateEssentialTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  TwoViewGeometry geometry;
  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (matches.size() < min_num_inliers) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  std::vector<Eigen::Vector3d> matched_cam_rays1(matches.size());
  std::vector<Eigen::Vector3d> matched_cam_rays2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_cam_rays1[i] =
        camera1.CamRayFromImg(points1[matches[i].point2D_idx1])
            .value_or(Eigen::Vector3d::Zero());
    matched_cam_rays2[i] =
        camera2.CamRayFromImg(points2[matches[i].point2D_idx2])
            .value_or(Eigen::Vector3d::Zero());
  }

  auto ransac_options = options.ransac_options;
  if (options.min_inlier_ratio > 0) {
    ransac_options.min_inlier_ratio = options.min_inlier_ratio;
  }
  ransac_options.max_error =
      (camera1.CamFromImgThreshold(options.ransac_options.max_error) +
       camera2.CamFromImgThreshold(options.ransac_options.max_error)) /
      2;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(ransac_options);
  const auto E_report = E_ransac.Estimate(matched_cam_rays1, matched_cam_rays2);
  geometry.E = E_report.model;

  if (!E_report.success || E_report.support.num_inliers < min_num_inliers) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
  geometry.inlier_matches = ExtractInlierMatches(
      matches, E_report.support.num_inliers, E_report.inlier_mask);
  if (options.min_inlier_ratio > 0) {
    const double inlier_ratio =
        static_cast<double>(E_report.support.num_inliers) / matches.size();
    if (inlier_ratio < options.min_inlier_ratio) {
      geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
      geometry.inlier_matches.clear();
    }
  }
  return geometry;
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

std::unique_ptr<TimingStats> CreateMatchTiming(
    const std::string& database_path, const OnlineMatchingOptions& options) {
  const auto& ransac = options.geometry.ransac_options;
  auto stats = std::make_unique<TimingStats>(
      TimingStats::FileBesideDatabase(database_path, "match_timing.txt"),
      "match",
      std::vector<std::string>{"pair_image", "match_ms", "tvg_ms",
                               "num_matches", "num_inliers", "config"});
  stats->AddMeta("matcher",
                 std::string(FeatureMatcherTypeToString(options.matching.type)));
  stats->AddMeta("use_gpu", options.matching.use_gpu ? "1" : "0");
  stats->AddMeta("gpu_index", options.matching.gpu_index);
  stats->AddMeta("overlap", std::to_string(options.overlap));
  stats->AddMeta("min_num_inliers",
                 std::to_string(options.geometry.min_num_inliers));
  stats->AddMeta("e_only", options.e_only ? "1" : "0");
  stats->AddMeta("use_degensac", options.geometry.use_degensac ? "1" : "0");
  stats->AddMeta("detect_watermark",
                 options.geometry.detect_watermark ? "1" : "0");
  stats->AddMeta("min_num_trials", std::to_string(ransac.min_num_trials));
  stats->AddMeta("max_num_trials", std::to_string(ransac.max_num_trials));
  stats->AddMeta("confidence", std::to_string(ransac.confidence));
  stats->AddMeta("scope",
                 options.e_only ? "FeatureMatcher::Match + essential-only TVG"
                                : "FeatureMatcher::Match + EstimateTwoViewGeometry");
  return stats;
}

}  // namespace

OnlineIncrementalMapper::OnlineIncrementalMapper(
    std::string database_path,
    FeatureExtractionOptions extraction_options,
    OnlineMatchingOptions matching_options)
    : database_path_(std::move(database_path)),
      extraction_options_(std::move(extraction_options)),
      matching_options_(std::move(matching_options)),
      extract_timing_(
          CreateExtractTiming(database_path_, extraction_options_)),
      match_timing_(CreateMatchTiming(database_path_, matching_options_)) {}

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

  if (!MatchAndWrite(image_id)) {
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
      LoadFeatures(*session, image_id);
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

  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  extract_timing_->Start();
  const bool extract_ok =
      feature_extractor_->Extract(bitmap, keypoints.get(), descriptors.get());
  if (!extract_ok) {
    LOG(ERROR) << "OnlineIncrementalMapper: extraction failed for image "
               << image_id
               << " extract_ms=" << extract_timing_->ElapsedMilliseconds();
    return false;
  }
  extract_timing_->Record(
      std::to_string(image_id),
      {std::to_string(keypoints->size()), image_abs_path});

  if (cache_->ExistsImage(image_id)) {
    const Image& image = cache_->Image(image_id);
    if (image.HasCameraId() && cache_->ExistsCamera(image.CameraId())) {
      ScaleKeypointsToCamera(
          bitmap, cache_->Camera(image.CameraId()), keypoints.get());
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
    LOG(ERROR) << "OnlineIncrementalMapper: failed to write features: "
               << e.what();
    return false;
  }

  CacheFeatures(image_id, keypoints, descriptors);

  LOG(INFO) << "OnlineIncrementalMapper: extract image " << image_id << " -> "
            << keypoints->size() << " features type="
            << FeatureExtractorTypeToString(descriptors->type);
  return true;
}

bool OnlineIncrementalMapper::InitFeatureMatcher() {
  if (feature_matcher_) {
    return true;
  }

  if (matching_options_.matching.use_gpu) {
#if !defined(COLMAP_GPU_ENABLED)
    LOG(ERROR) << "OnlineIncrementalMapper: match use_gpu=true but COLMAP was "
                  "built without GPU. Rebuild with CUDA_ENABLED=ON.";
    return false;
#endif
  }

  matching_options_.matching.guided_matching = false;
  matching_options_.matching.num_threads = 1;

  // GPU SiftMatch allocates O(max_num_matches^2) device memory. COLMAP's
  // default 32768 would request a ~4GB score matrix and Create() returns null.
  int max_keypoints = 0;
  try {
    DatabaseSession session(database_path_);
    max_keypoints = static_cast<int>(session->MaxNumKeypoints());
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineIncrementalMapper: MaxNumKeypoints failed: "
                 << e.what();
  }
  if (extraction_options_.sift) {
    max_keypoints =
        std::max(max_keypoints, extraction_options_.sift->max_num_features);
  }
  const int needed = std::max(max_keypoints * 2, 1024);
  if (matching_options_.matching.max_num_matches > needed) {
    LOG(INFO) << "OnlineIncrementalMapper: cap max_num_matches "
              << matching_options_.matching.max_num_matches << " -> " << needed;
    matching_options_.matching.max_num_matches = needed;
  }

  if (!matching_options_.matching.Check() ||
      !matching_options_.geometry.Check()) {
    LOG(ERROR) << "OnlineIncrementalMapper: invalid matching options type="
               << FeatureMatcherTypeToString(matching_options_.matching.type)
               << " use_gpu=" << matching_options_.matching.use_gpu
               << " gpu_index=" << matching_options_.matching.gpu_index
               << " overlap=" << matching_options_.overlap;
    return false;
  }

  try {
    feature_matcher_ = FeatureMatcher::Create(matching_options_.matching);
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: matcher Create threw: " << e.what();
    feature_matcher_.reset();
    return false;
  }
  if (!feature_matcher_) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to create matcher type="
               << FeatureMatcherTypeToString(matching_options_.matching.type)
               << " use_gpu=" << matching_options_.matching.use_gpu
               << " gpu_index=" << matching_options_.matching.gpu_index
               << " max_num_matches="
               << matching_options_.matching.max_num_matches;
    return false;
  }

  LOG(INFO) << "OnlineIncrementalMapper: matcher="
            << FeatureMatcherTypeToString(matching_options_.matching.type)
            << " use_gpu=" << matching_options_.matching.use_gpu
            << " gpu_index=" << matching_options_.matching.gpu_index
            << " max_num_matches="
            << matching_options_.matching.max_num_matches
            << " overlap=" << matching_options_.overlap;
  return true;
}

void OnlineIncrementalMapper::CacheFeatures(
    const image_t image_id,
    std::shared_ptr<const FeatureKeypoints> keypoints,
    std::shared_ptr<const FeatureDescriptors> descriptors) {
  if (image_id == kInvalidImageId || !keypoints || !descriptors) {
    return;
  }
  if (feature_cache_.find(image_id) == feature_cache_.end()) {
    feature_cache_order_.push_back(image_id);
  }
  feature_cache_[image_id] = FeatureCacheEntry{std::move(keypoints),
                                               std::move(descriptors)};

  const size_t max_n =
      static_cast<size_t>(std::max(matching_options_.overlap, 0)) + 1;
  while (feature_cache_order_.size() > max_n) {
    const image_t old_id = feature_cache_order_.front();
    feature_cache_order_.pop_front();
    feature_cache_.erase(old_id);
  }
}

OnlineIncrementalMapper::FeatureCacheEntry
OnlineIncrementalMapper::LoadFeatures(Database& database,
                                      const image_t image_id) {
  const auto cached = feature_cache_.find(image_id);
  if (cached != feature_cache_.end()) {
    return cached->second;
  }
  if (!database.ExistsKeypoints(image_id) ||
      !database.ExistsDescriptors(image_id)) {
    return {};
  }

  auto keypoints =
      std::make_shared<FeatureKeypoints>(database.ReadKeypoints(image_id));
  auto descriptors =
      std::make_shared<FeatureDescriptors>(database.ReadDescriptors(image_id));
  CacheFeatures(image_id, keypoints, descriptors);
  return FeatureCacheEntry{std::move(keypoints), std::move(descriptors)};
}

std::vector<image_t> OnlineIncrementalMapper::SelectTemporalOverlapImages(
    const image_t image_id) const {
  std::vector<image_t> previous_ids;
  if (!cache_ || matching_options_.overlap <= 0) {
    return previous_ids;
  }

  previous_ids.reserve(cache_->NumImages());
  for (const auto& [id, image] : cache_->Images()) {
    (void)image;
    if (id < image_id) {
      previous_ids.push_back(id);
    }
  }
  std::sort(previous_ids.begin(), previous_ids.end());
  if (previous_ids.size() >
      static_cast<size_t>(matching_options_.overlap)) {
    previous_ids.erase(previous_ids.begin(),
                       previous_ids.end() - matching_options_.overlap);
  }
  return previous_ids;
}

bool OnlineIncrementalMapper::MatchAndWrite(const image_t image_id) {
  if (matching_options_.overlap <= 0) {
    LOG(INFO) << "OnlineIncrementalMapper: skip match, overlap="
              << matching_options_.overlap;
    return true;
  }

  const std::vector<image_t> previous_ids = SelectTemporalOverlapImages(image_id);
  if (previous_ids.empty()) {
    LOG(INFO) << "OnlineIncrementalMapper: skip match, image " << image_id
              << " has no previous keyframes";
    return true;
  }

  if (!InitFeatureMatcher()) {
    return false;
  }

  try {
    DatabaseSession session(database_path_);
    const FeatureCacheEntry current = LoadFeatures(*session, image_id);
    if (!current.keypoints || !current.descriptors ||
        current.keypoints->empty() || current.descriptors->data.rows() == 0) {
      LOG(WARNING) << "OnlineIncrementalMapper: skip match, image " << image_id
                   << " has no features";
      return true;
    }

    if (!cache_->ExistsImage(image_id) ||
        !cache_->Image(image_id).HasCameraId() ||
        !cache_->ExistsCamera(cache_->Image(image_id).CameraId())) {
      LOG(ERROR) << "OnlineIncrementalMapper: image " << image_id
                 << " missing camera for matching";
      return false;
    }
    const Camera& camera1 = cache_->Camera(cache_->Image(image_id).CameraId());

    for (const image_t prev_id : previous_ids) {
      try {
        if (!session->ExistsKeypoints(prev_id) ||
            !session->ExistsDescriptors(prev_id)) {
          LOG(WARNING) << "OnlineIncrementalMapper: skip pair " << prev_id
                       << "-" << image_id << ", previous image has no features";
          continue;
        }

        if (!cache_->ExistsImage(prev_id) ||
            !cache_->Image(prev_id).HasCameraId() ||
            !cache_->ExistsCamera(cache_->Image(prev_id).CameraId())) {
          LOG(WARNING) << "OnlineIncrementalMapper: skip pair " << prev_id
                       << "-" << image_id << ", previous image missing camera";
          continue;
        }

        const bool exists_matches = session->ExistsMatches(prev_id, image_id);
        const bool exists_tvg =
            session->ExistsTwoViewGeometry(prev_id, image_id);
        if (exists_matches && exists_tvg) {
          LOG(INFO) << "OnlineIncrementalMapper: skip pair " << prev_id << "-"
                    << image_id << ", already matched";
          continue;
        }
        if (exists_matches) {
          session->DeleteMatches(prev_id, image_id);
        }
        if (exists_tvg) {
          session->DeleteTwoViewGeometry(prev_id, image_id);
        }

        const FeatureCacheEntry previous = LoadFeatures(*session, prev_id);
        if (!previous.keypoints || !previous.descriptors ||
            previous.keypoints->empty() ||
            previous.descriptors->data.rows() == 0) {
          LOG(WARNING) << "OnlineIncrementalMapper: skip pair " << prev_id
                       << "-" << image_id << ", failed to load previous features";
          continue;
        }

        const Camera& camera2 =
            cache_->Camera(cache_->Image(prev_id).CameraId());

        FeatureMatches matches;
        match_timing_->Start();
        feature_matcher_->Match(
            {prev_id, &camera2, previous.keypoints, previous.descriptors},
            {image_id, &camera1, current.keypoints, current.descriptors},
            &matches);
        const double match_ms = match_timing_->ElapsedMilliseconds();

        TwoViewGeometry two_view_geometry;
        double tvg_ms = 0.0;
        if (matches.size() >=
            static_cast<size_t>(matching_options_.geometry.min_num_inliers)) {
          match_timing_->Start();
          const auto points2 =
              FeatureKeypointsToPointsVector(*previous.keypoints);
          const auto points1 =
              FeatureKeypointsToPointsVector(*current.keypoints);
          if (matching_options_.e_only) {
            two_view_geometry = EstimateEssentialTwoViewGeometry(
                camera2, points2, camera1, points1, matches,
                matching_options_.geometry);
          } else {
            two_view_geometry = EstimateTwoViewGeometry(
                camera2, points2, camera1, points1, matches,
                matching_options_.geometry);
          }
          tvg_ms = match_timing_->ElapsedMilliseconds();
        }

        match_timing_->Record(
            std::to_string(image_id), match_ms + tvg_ms,
            {std::to_string(prev_id), FormatMs(match_ms), FormatMs(tvg_ms),
             std::to_string(matches.size()),
             std::to_string(two_view_geometry.inlier_matches.size()),
             std::to_string(two_view_geometry.config)});

        if (matches.size() <
            static_cast<size_t>(matching_options_.geometry.min_num_inliers)) {
          matches.clear();
        }
        if (two_view_geometry.inlier_matches.size() <
            static_cast<size_t>(matching_options_.geometry.min_num_inliers)) {
          two_view_geometry = TwoViewGeometry();
        }

        session->WriteMatches(prev_id, image_id, matches);
        session->WriteTwoViewGeometry(prev_id, image_id, two_view_geometry);

        LOG(INFO) << "OnlineIncrementalMapper: match " << prev_id << "-"
                  << image_id << " matches=" << matches.size()
                  << " inliers=" << two_view_geometry.inlier_matches.size()
                  << " config=" << two_view_geometry.config;
      } catch (const std::exception& e) {
        LOG(WARNING) << "OnlineIncrementalMapper: pair " << prev_id << "-"
                     << image_id << " failed: " << e.what();
        try {
          if (!session->ExistsMatches(prev_id, image_id)) {
            session->WriteMatches(prev_id, image_id, FeatureMatches());
          }
          if (!session->ExistsTwoViewGeometry(prev_id, image_id)) {
            session->WriteTwoViewGeometry(
                prev_id, image_id, TwoViewGeometry());
          }
        } catch (const std::exception& write_error) {
          LOG(WARNING) << "OnlineIncrementalMapper: failed to write empty pair "
                       << prev_id << "-" << image_id << ": "
                       << write_error.what();
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: matching session failed: "
               << e.what();
    return false;
  }

  return true;
}

}  // namespace colmap
