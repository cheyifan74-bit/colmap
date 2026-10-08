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

#include "colmap/online_sfm/online_feature_matcher.h"

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/index.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/cache.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <algorithm>
#include <exception>
#include <future>
#include <iomanip>
#include <sstream>

namespace colmap {
namespace {

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
  stats->AddMeta("cpu_brute_force",
                 options.matching.sift &&
                         options.matching.sift->cpu_brute_force_matcher
                     ? "1"
                     : "0");
  stats->AddMeta("gpu_index", options.matching.gpu_index);
  stats->AddMeta("overlap", std::to_string(options.overlap));
  stats->AddMeta("spatial_max_distance",
                 std::to_string(options.spatial_max_distance));
  stats->AddMeta("spatial_max_angle_deg",
                 std::to_string(options.spatial_max_angle_deg));
  stats->AddMeta("spatial_max_num_images",
                 std::to_string(options.spatial_max_num_images));
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

struct PairMatchResult {
  image_t prev_id = kInvalidImageId;
  FeatureMatches matches;
  TwoViewGeometry geometry;
  double match_ms = 0.0;
  double tvg_ms = 0.0;
  std::string error;
};

PairMatchResult MatchAndVerifyPair(
    FeatureMatcher* matcher,
    const bool e_only,
    const TwoViewGeometryOptions& geometry_options,
    const FeatureMatcher::Image& prev,
    const FeatureMatcher::Image& curr) {
  PairMatchResult result;
  result.prev_id = prev.image_id;
  Timer timer;
  try {
    timer.Restart();
    matcher->Match(prev, curr, &result.matches);
    result.match_ms = timer.ElapsedMicroSeconds() / 1000.0;

    if (result.matches.size() >=
        static_cast<size_t>(geometry_options.min_num_inliers)) {
      timer.Restart();
      const auto points_prev =
          FeatureKeypointsToPointsVector(*prev.keypoints);
      const auto points_curr =
          FeatureKeypointsToPointsVector(*curr.keypoints);
      if (e_only) {
        result.geometry = EstimateEssentialTwoViewGeometry(
            *prev.camera, points_prev, *curr.camera, points_curr,
            result.matches, geometry_options);
      } else {
        result.geometry = EstimateTwoViewGeometry(
            *prev.camera, points_prev, *curr.camera, points_curr,
            result.matches, geometry_options);
      }
      result.tvg_ms = timer.ElapsedMicroSeconds() / 1000.0;
    }
  } catch (const std::exception& e) {
    result.error = e.what();
  }
  return result;
}

void WritePairResult(Database& database,
                     TimingStats* match_timing,
                     const image_t image_id,
                     PairMatchResult result,
                     const int min_num_inliers) {
  if (!result.error.empty()) {
    LOG(WARNING) << "OnlineFeatureMatcher: pair " << result.prev_id << "-"
                 << image_id << " failed: " << result.error;
    try {
      if (!database.ExistsMatches(result.prev_id, image_id)) {
        database.WriteMatches(result.prev_id, image_id, FeatureMatches());
      }
      if (!database.ExistsTwoViewGeometry(result.prev_id, image_id)) {
        database.WriteTwoViewGeometry(
            result.prev_id, image_id, TwoViewGeometry());
      }
    } catch (const std::exception& write_error) {
      LOG(WARNING) << "OnlineFeatureMatcher: failed to write empty pair "
                   << result.prev_id << "-" << image_id << ": "
                   << write_error.what();
    }
    return;
  }

  match_timing->Record(
      std::to_string(image_id), result.match_ms + result.tvg_ms,
      {std::to_string(result.prev_id), FormatMs(result.match_ms),
       FormatMs(result.tvg_ms), std::to_string(result.matches.size()),
       std::to_string(result.geometry.inlier_matches.size()),
       std::to_string(result.geometry.config)});

  if (result.matches.size() < static_cast<size_t>(min_num_inliers)) {
    result.matches.clear();
  }
  if (result.geometry.inlier_matches.size() <
      static_cast<size_t>(min_num_inliers)) {
    result.geometry = TwoViewGeometry();
  }

  database.WriteMatches(result.prev_id, image_id, result.matches);
  database.WriteTwoViewGeometry(result.prev_id, image_id, result.geometry);

  LOG(INFO) << "OnlineFeatureMatcher: match " << result.prev_id << "-"
            << image_id << " matches=" << result.matches.size()
            << " inliers=" << result.geometry.inlier_matches.size()
            << " config=" << result.geometry.config;
}

}  // namespace

struct OnlineFeatureMatcher::CpuMatchContext {
  std::unique_ptr<ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>>
      index_cache;
  std::vector<std::unique_ptr<FeatureMatcher>> matchers;
  std::unique_ptr<ThreadPool> thread_pool;
};

OnlineFeatureMatcher::OnlineFeatureMatcher(std::string database_path,
                                           OnlineMatchingOptions options)
    : database_path_(std::move(database_path)),
      options_(std::move(options)),
      timing_(CreateMatchTiming(database_path_, options_)) {}

OnlineFeatureMatcher::~OnlineFeatureMatcher() = default;

void OnlineFeatureMatcher::CacheFeatures(
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
      static_cast<size_t>(std::max(options_.overlap, 0)) + 1;
  while (feature_cache_order_.size() > max_n) {
    const image_t old_id = feature_cache_order_.front();
    feature_cache_order_.pop_front();
    feature_cache_.erase(old_id);
    if (cpu_match_ && cpu_match_->index_cache) {
      cpu_match_->index_cache->Evict(old_id);
    }
  }
}

void OnlineFeatureMatcher::PutPosePrior(const image_t image_id,
                                        const Rigid3d& cam_from_world) {
  if (image_id == kInvalidImageId) {
    return;
  }
  pose_priors_[image_id] = cam_from_world;
}

bool OnlineFeatureMatcher::SpatialEnabled() const {
  return options_.spatial_max_distance > 0.0 &&
         options_.spatial_max_angle_deg > 0.0;
}

OnlineFeatureMatcher::FeatureCacheEntry OnlineFeatureMatcher::LoadFeatures(
    Database& database, const image_t image_id, const bool remember) {
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
  FeatureCacheEntry entry{keypoints, descriptors};
  if (remember) {
    CacheFeatures(image_id, std::move(keypoints), std::move(descriptors));
  }
  return entry;
}

std::vector<image_t> OnlineFeatureMatcher::SelectTemporalOverlapImages(
    const image_t image_id, const DatabaseCache& cache) const {
  std::vector<image_t> previous_ids;
  if (options_.overlap <= 0) {
    return previous_ids;
  }

  previous_ids.reserve(cache.NumImages());
  for (const auto& [id, image] : cache.Images()) {
    (void)image;
    if (id < image_id) {
      previous_ids.push_back(id);
    }
  }
  std::sort(previous_ids.begin(), previous_ids.end());
  if (previous_ids.size() > static_cast<size_t>(options_.overlap)) {
    previous_ids.erase(previous_ids.begin(),
                       previous_ids.end() - options_.overlap);
  }
  return previous_ids;
}

std::vector<image_t> OnlineFeatureMatcher::SelectSpatialImages(
    const image_t image_id,
    const DatabaseCache& cache,
    const std::unordered_set<image_t>& excluded) const {
  std::vector<image_t> spatial_ids;
  if (!SpatialEnabled()) {
    return spatial_ids;
  }
  const auto current_prior = pose_priors_.find(image_id);
  if (current_prior == pose_priors_.end()) {
    return spatial_ids;
  }

  const Eigen::Vector3d current_center = current_prior->second.TgtOriginInSrc();
  const Eigen::Vector3d current_fwd =
      (Inverse(current_prior->second).rotation() * Eigen::Vector3d::UnitZ())
          .normalized();
  const double max_distance = options_.spatial_max_distance;
  const double max_angle_rad = DegToRad(options_.spatial_max_angle_deg);

  struct Candidate {
    image_t id = kInvalidImageId;
    double dist = 0.0;
  };
  std::vector<Candidate> candidates;
  for (const auto& [id, image] : cache.Images()) {
    (void)image;
    if (id >= image_id || excluded.count(id) > 0) {
      continue;
    }
    const auto prior = pose_priors_.find(id);
    if (prior == pose_priors_.end()) {
      continue;
    }
    const Eigen::Vector3d center = prior->second.TgtOriginInSrc();
    const double dist = (center - current_center).norm();
    if (dist > max_distance) {
      continue;
    }
    const Eigen::Vector3d fwd =
        (Inverse(prior->second).rotation() * Eigen::Vector3d::UnitZ())
            .normalized();
    const double angle =
        std::acos(std::clamp(current_fwd.dot(fwd), -1.0, 1.0));
    if (angle > max_angle_rad) {
      continue;
    }
    candidates.push_back({id, dist});
  }
  std::sort(candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.dist != b.dist) {
                return a.dist < b.dist;
              }
              return a.id < b.id;
            });
  if (options_.spatial_max_num_images > 0 &&
      candidates.size() >
          static_cast<size_t>(options_.spatial_max_num_images)) {
    candidates.resize(static_cast<size_t>(options_.spatial_max_num_images));
  }
  spatial_ids.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    spatial_ids.push_back(candidate.id);
  }
  return spatial_ids;
}

bool OnlineFeatureMatcher::InitMatchers() {
  if (gpu_matcher_ || cpu_match_) {
    return true;
  }

  if (options_.matching.use_gpu) {
#if !defined(COLMAP_GPU_ENABLED)
    LOG(ERROR) << "OnlineFeatureMatcher: match use_gpu=true but COLMAP was "
                  "built without GPU. Rebuild with CUDA_ENABLED=ON.";
    return false;
#endif
  }

  options_.matching.guided_matching = false;

  // GPU SiftMatch allocates O(max_num_matches^2) device memory. COLMAP's
  // default 32768 would request a ~4GB score matrix and Create() returns null.
  int max_keypoints = options_.max_num_features_hint;
  try {
    DatabaseSession session(database_path_);
    max_keypoints = std::max(
        max_keypoints, static_cast<int>(session->MaxNumKeypoints()));
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineFeatureMatcher: MaxNumKeypoints failed: "
                 << e.what();
  }
  const int needed = std::max(max_keypoints * 2, 1024);
  if (options_.matching.max_num_matches > needed) {
    LOG(INFO) << "OnlineFeatureMatcher: cap max_num_matches "
              << options_.matching.max_num_matches << " -> " << needed;
    options_.matching.max_num_matches = needed;
  }

  if (!options_.matching.Check() || !options_.geometry.Check()) {
    LOG(ERROR) << "OnlineFeatureMatcher: invalid matching options type="
               << FeatureMatcherTypeToString(options_.matching.type)
               << " use_gpu=" << options_.matching.use_gpu
               << " gpu_index=" << options_.matching.gpu_index
               << " overlap=" << options_.overlap
               << " spatial_max_distance=" << options_.spatial_max_distance
               << " spatial_max_angle_deg=" << options_.spatial_max_angle_deg
               << " spatial_max_num_images="
               << options_.spatial_max_num_images;
    return false;
  }

  if (options_.matching.use_gpu) {
    options_.matching.num_threads = 1;
    try {
      gpu_matcher_ = FeatureMatcher::Create(options_.matching);
    } catch (const std::exception& e) {
      LOG(ERROR) << "OnlineFeatureMatcher: matcher Create threw: " << e.what();
      gpu_matcher_.reset();
      return false;
    }
    if (!gpu_matcher_) {
      LOG(ERROR) << "OnlineFeatureMatcher: failed to create GPU matcher "
                 << "type="
                 << FeatureMatcherTypeToString(options_.matching.type)
                 << " gpu_index=" << options_.matching.gpu_index
                 << " max_num_matches=" << options_.matching.max_num_matches;
      return false;
    }
    LOG(INFO) << "OnlineFeatureMatcher: matcher="
              << FeatureMatcherTypeToString(options_.matching.type)
              << " use_gpu=1 gpu_index=" << options_.matching.gpu_index
              << " max_num_matches=" << options_.matching.max_num_matches
              << " overlap=" << options_.overlap
              << " spatial_max_distance=" << options_.spatial_max_distance
              << " spatial_max_angle_deg=" << options_.spatial_max_angle_deg
              << " spatial_max_num_images="
              << options_.spatial_max_num_images;
    return true;
  }

  const int num_workers =
      std::max(1, GetEffectiveNumThreads(options_.matching.num_threads));
  cpu_match_ = std::make_unique<CpuMatchContext>();

  const bool brute_force = options_.matching.sift->cpu_brute_force_matcher;
  if (brute_force) {
    options_.matching.sift->cpu_descriptor_index_cache = nullptr;
  } else {
    const size_t cache_size =
        static_cast<size_t>(std::max(options_.overlap, 0)) + 1;
    cpu_match_->index_cache =
        std::make_unique<ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>>(
            cache_size, [this](const image_t image_id) {
              auto index = FeatureDescriptorIndex::Create(
                  FeatureDescriptorIndex::Type::FAISS, /*num_threads=*/1);
              const auto it = feature_cache_.find(image_id);
              if (it != feature_cache_.end() && it->second.descriptors) {
                index->Build(it->second.descriptors->ToFloat());
              }
              return index;
            });
    options_.matching.sift->cpu_descriptor_index_cache =
        cpu_match_->index_cache.get();
  }

  FeatureMatchingOptions worker_options = options_.matching;
  worker_options.num_threads = 1;
  cpu_match_->matchers.reserve(static_cast<size_t>(num_workers));
  try {
    for (int i = 0; i < num_workers; ++i) {
      auto matcher = FeatureMatcher::Create(worker_options);
      if (!matcher) {
        LOG(ERROR) << "OnlineFeatureMatcher: failed to create CPU matcher "
                   << i << "/" << num_workers;
        cpu_match_.reset();
        options_.matching.sift->cpu_descriptor_index_cache = nullptr;
        return false;
      }
      cpu_match_->matchers.push_back(std::move(matcher));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineFeatureMatcher: CPU matcher Create threw: "
               << e.what();
    cpu_match_.reset();
    options_.matching.sift->cpu_descriptor_index_cache = nullptr;
    return false;
  }
  cpu_match_->thread_pool = std::make_unique<ThreadPool>(num_workers);

  LOG(INFO) << "OnlineFeatureMatcher: matcher="
            << FeatureMatcherTypeToString(options_.matching.type)
            << " use_gpu=0 cpu_workers=" << num_workers
            << " brute_force="
            << options_.matching.sift->cpu_brute_force_matcher
            << " overlap=" << options_.overlap
            << " spatial_max_distance=" << options_.spatial_max_distance
            << " spatial_max_angle_deg=" << options_.spatial_max_angle_deg
            << " spatial_max_num_images="
            << options_.spatial_max_num_images;
  return true;
}

bool OnlineFeatureMatcher::MatchAndWrite(const image_t image_id,
                                         const DatabaseCache& cache) {
  last_temporal_ids_ = SelectTemporalOverlapImages(image_id, cache);
  const std::vector<image_t>& temporal_ids = last_temporal_ids_;
  std::unordered_set<image_t> temporal_set(temporal_ids.begin(),
                                           temporal_ids.end());
  last_spatial_ids_ = SelectSpatialImages(image_id, cache, temporal_set);
  const std::vector<image_t>& spatial_ids = last_spatial_ids_;
  std::vector<image_t> previous_ids = temporal_ids;
  previous_ids.insert(
      previous_ids.end(), spatial_ids.begin(), spatial_ids.end());
  if (previous_ids.empty()) {
    LOG(INFO) << "OnlineFeatureMatcher: skip match, image " << image_id
              << " has no temporal/spatial candidates";
    return true;
  }
  LOG(INFO) << "OnlineFeatureMatcher: image " << image_id
            << " candidates temporal=" << temporal_ids.size()
            << " spatial=" << spatial_ids.size();
  return MatchPairsAndWrite(
      image_id, previous_ids, cache, temporal_set, /*rematch_weak=*/false);
}

bool OnlineFeatureMatcher::MatchSpecificPairs(
    const image_t image_id,
    const std::vector<image_t>& other_ids,
    const DatabaseCache& cache) {
  if (other_ids.empty()) {
    return true;
  }
  LOG(INFO) << "OnlineFeatureMatcher: image " << image_id
            << " specific pairs=" << other_ids.size();
  return MatchPairsAndWrite(image_id,
                            other_ids,
                            cache,
                            /*remember_ids=*/{},
                            /*rematch_weak=*/true);
}

bool OnlineFeatureMatcher::MatchPairsAndWrite(
    const image_t image_id,
    const std::vector<image_t>& previous_ids,
    const DatabaseCache& cache,
    const std::unordered_set<image_t>& remember_ids,
    const bool rematch_weak) {
  if (previous_ids.empty()) {
    return true;
  }

  if (!InitMatchers()) {
    return false;
  }

  struct PairWork {
    image_t prev_id = kInvalidImageId;
    FeatureCacheEntry previous;
    const Camera* camera_prev = nullptr;
  };

  FeatureCacheEntry current;
  const Camera* camera_cur = nullptr;
  std::vector<PairWork> jobs;
  try {
    DatabaseSession session(database_path_);
    current = LoadFeatures(*session, image_id, /*remember=*/true);
    if (!current.keypoints || !current.descriptors ||
        current.keypoints->empty() || current.descriptors->data.rows() == 0) {
      LOG(WARNING) << "OnlineFeatureMatcher: skip match, image " << image_id
                   << " has no features";
      return true;
    }

    if (!cache.ExistsImage(image_id) ||
        !cache.Image(image_id).HasCameraId() ||
        !cache.ExistsCamera(cache.Image(image_id).CameraId())) {
      LOG(ERROR) << "OnlineFeatureMatcher: image " << image_id
                 << " missing camera for matching";
      return false;
    }
    camera_cur = &cache.Camera(cache.Image(image_id).CameraId());

    jobs.reserve(previous_ids.size());
    for (const image_t prev_id : previous_ids) {
      if (!session->ExistsKeypoints(prev_id) ||
          !session->ExistsDescriptors(prev_id)) {
        LOG(WARNING) << "OnlineFeatureMatcher: skip pair " << prev_id << "-"
                     << image_id << ", previous image has no features";
        continue;
      }
      if (!cache.ExistsImage(prev_id) ||
          !cache.Image(prev_id).HasCameraId() ||
          !cache.ExistsCamera(cache.Image(prev_id).CameraId())) {
        LOG(WARNING) << "OnlineFeatureMatcher: skip pair " << prev_id << "-"
                     << image_id << ", previous image missing camera";
        continue;
      }

      const bool exists_matches = session->ExistsMatches(prev_id, image_id);
      const bool exists_tvg = session->ExistsTwoViewGeometry(prev_id, image_id);
      bool existing_usable = false;
      if (exists_tvg) {
        const TwoViewGeometry existing =
            session->ReadTwoViewGeometry(prev_id, image_id);
        existing_usable =
            existing.config != TwoViewGeometry::ConfigurationType::DEGENERATE &&
            existing.inlier_matches.size() >=
                static_cast<size_t>(options_.geometry.min_num_inliers);
      }
      if (exists_matches && exists_tvg && (!rematch_weak || existing_usable)) {
        LOG(INFO) << "OnlineFeatureMatcher: skip pair " << prev_id << "-"
                  << image_id << ", already matched";
        continue;
      }
      if (exists_matches) {
        session->DeleteMatches(prev_id, image_id);
      }
      if (exists_tvg) {
        session->DeleteTwoViewGeometry(prev_id, image_id);
      }

      PairWork work;
      work.prev_id = prev_id;
      work.previous = LoadFeatures(
          *session, prev_id, /*remember=*/remember_ids.count(prev_id) > 0);
      if (!work.previous.keypoints || !work.previous.descriptors ||
          work.previous.keypoints->empty() ||
          work.previous.descriptors->data.rows() == 0) {
        LOG(WARNING) << "OnlineFeatureMatcher: skip pair " << prev_id << "-"
                     << image_id << ", failed to load previous features";
        continue;
      }
      work.camera_prev = &cache.Camera(cache.Image(prev_id).CameraId());
      jobs.push_back(std::move(work));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineFeatureMatcher: matching session failed: "
               << e.what();
    return false;
  }

  if (jobs.empty()) {
    return true;
  }

  for (const PairWork& job : jobs) {
    if (feature_cache_.find(job.prev_id) == feature_cache_.end()) {
      feature_cache_[job.prev_id] = job.previous;
    }
  }
  auto unpin_unremembered = [&]() {
    for (const image_t prev_id : previous_ids) {
      if (remember_ids.count(prev_id) > 0) {
        continue;
      }
      const bool kept =
          std::find(feature_cache_order_.begin(),
                    feature_cache_order_.end(),
                    prev_id) != feature_cache_order_.end();
      if (kept) {
        continue;
      }
      feature_cache_.erase(prev_id);
      if (cpu_match_ && cpu_match_->index_cache) {
        cpu_match_->index_cache->Evict(prev_id);
      }
    }
  };

  const FeatureMatcher::Image curr_image{
      image_id, camera_cur, current.keypoints, current.descriptors};

  std::vector<PairMatchResult> results;
  results.reserve(jobs.size());
  if (cpu_match_) {
    if (cpu_match_->index_cache) {
      cpu_match_->index_cache->Get(image_id);
    }
    std::vector<std::shared_future<PairMatchResult>> futures;
    futures.reserve(jobs.size());
    try {
      for (const PairWork& job : jobs) {
        const FeatureMatcher::Image prev_image{job.prev_id,
                                               job.camera_prev,
                                               job.previous.keypoints,
                                               job.previous.descriptors};
        futures.push_back(cpu_match_->thread_pool->AddTask(
            [this, prev_image, curr_image]() {
              const int idx = cpu_match_->thread_pool->GetThreadIndex();
              return MatchAndVerifyPair(cpu_match_->matchers[idx].get(),
                                        options_.e_only,
                                        options_.geometry,
                                        prev_image,
                                        curr_image);
            }));
      }
      cpu_match_->thread_pool->Wait();
      for (auto& future : futures) {
        results.push_back(future.get());
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "OnlineFeatureMatcher: CPU matching failed: " << e.what();
      unpin_unremembered();
      return false;
    }
  } else {
    for (const PairWork& job : jobs) {
      const FeatureMatcher::Image prev_image{job.prev_id,
                                             job.camera_prev,
                                             job.previous.keypoints,
                                             job.previous.descriptors};
      results.push_back(MatchAndVerifyPair(gpu_matcher_.get(),
                                           options_.e_only,
                                           options_.geometry,
                                           prev_image,
                                           curr_image));
    }
  }

  try {
    DatabaseSession session(database_path_);
    for (PairMatchResult& result : results) {
      WritePairResult(*session,
                      timing_.get(),
                      image_id,
                      std::move(result),
                      options_.geometry.min_num_inliers);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineFeatureMatcher: failed to write matches: "
               << e.what();
    unpin_unremembered();
    return false;
  }

  unpin_unremembered();
  return true;
}

}  // namespace colmap
