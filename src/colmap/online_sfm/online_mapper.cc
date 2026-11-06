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

#include "colmap/online_sfm/online_mapper.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/pose.h"
#include "colmap/estimators/triangulation.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/track.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/endian.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

namespace colmap {
namespace {

BundleAdjustmentOptions MakeLocalBaOptions(const int min_track_length,
                                           const int max_num_iterations = 10) {
  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.refine_sensor_from_rig = false;
  options.print_summary = false;
  options.min_track_length = min_track_length;
  if (options.ceres) {
    options.ceres->loss_function_type =
        CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
    options.ceres->loss_function_scale = 1.0;
    options.ceres->solver_options.max_num_iterations = max_num_iterations;
    options.ceres->solver_options.logging_type = ceres::LoggingType::SILENT;
  }
  return options;
}

std::string JoinImageIds(const std::vector<image_t>& ids) {
  if (ids.empty()) {
    return "-";
  }
  std::ostringstream oss;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << ids[i];
  }
  return oss.str();
}

std::vector<image_t> FirstLevelNeighbors(const image_t image_id,
                                         const std::vector<image_t>& local_ids) {
  std::vector<image_t> first_level;
  first_level.reserve(local_ids.size());
  for (const image_t id : local_ids) {
    if (id != image_id) {
      first_level.push_back(id);
    }
  }
  std::sort(first_level.begin(), first_level.end());
  return first_level;
}

void AppendPoseColumns(std::ostream& out, const Rigid3d& pose) {
  const Eigen::Quaterniond q = pose.rotation();
  const Eigen::Vector3d t = pose.translation();
  const Eigen::Vector3d c = pose.TgtOriginInSrc();
  out << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << t.x()
      << " " << t.y() << " " << t.z() << " " << c.x() << " " << c.y() << " "
      << c.z();
}

int DescriptorL2Sq(const FeatureDescriptors& a,
                   const point2D_t ia,
                   const FeatureDescriptors& b,
                   const point2D_t ib) {
  return (a.data.row(ia).cast<int>() - b.data.row(ib).cast<int>())
      .squaredNorm();
}

bool DescriptorsMatchImage(const FeatureDescriptors* desc, const Image& image) {
  return desc != nullptr && desc->data.cols() > 0 &&
         desc->data.rows() == static_cast<Eigen::Index>(image.NumPoints2D());
}

}  // namespace

OnlineMapper::OnlineMapper(std::shared_ptr<DatabaseCache> database_cache,
                           std::string sparse_path,
                           OnlineMapperOptions options)
    : database_cache_(std::move(database_cache)),
      sparse_path_(std::move(sparse_path)),
      options_(options),
      reconstruction_(std::make_shared<Reconstruction>()) {
  if (!sparse_path_.empty()) {
    const std::filesystem::path timing_path =
        std::filesystem::path(sparse_path_).parent_path() / "database" /
        "lba_timing.txt";
    lba_timing_ = std::make_unique<TimingStats>(
        timing_path.string(),
        "lba",
        std::vector<std::string>{
            "window", "variable", "residuals", "vio_priors"});
    lba_timing_->AddMeta(
        "ba_local_num_images",
        std::to_string(options_.ba_local_num_images));
    lba_timing_->AddMeta("max_num_iterations", "10");
    lba_timing_->AddMeta("scope", "AdjustLocalBundle::Solve");
  }
}

OnlineMapper::~OnlineMapper() {
  if (sparse_writes_since_flush_ > 0) {
    WriteSparse();
  }
}

bool OnlineMapper::InitReconstruction() {
  if (obs_manager_ && triangulator_) {
    return true;
  }
  if (!database_cache_ || database_cache_->NumImages() == 0) {
    LOG(ERROR) << "OnlineMapper: database cache has no images";
    return false;
  }
  reconstruction_->Load(*database_cache_);
  obs_manager_ = std::make_shared<ObservationManager>(
      *reconstruction_, database_cache_->CorrespondenceGraph());
  triangulator_ = std::make_shared<IncrementalTriangulator>(
      database_cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);
  LOG(INFO) << "OnlineMapper: reconstruction ready with "
            << reconstruction_->NumImages() << " image(s)";
  return true;
}

bool OnlineMapper::SyncReconstruction() {
  if (!database_cache_) {
    return false;
  }
  if (!obs_manager_) {
    return InitReconstruction();
  }

  const std::vector<image_t> new_image_ids =
      reconstruction_->Load(*database_cache_);
  for (const image_t image_id : new_image_ids) {
    obs_manager_->AddImage(image_id);
  }
  return true;
}

void OnlineMapper::NotifyNewImagePair(const image_t image_id1,
                                      const image_t image_id2) {
  if (!obs_manager_) {
    return;
  }
  // seed_point3D_counts: the pair was just added to the correspondence graph
  // while mapping may already be running, so any already-triangulated partner
  // point needs its per-point3D visibility counted for the new correspondences.
  obs_manager_->AddImagePair(image_id1, image_id2,
                             /*seed_point3D_counts=*/true);
}

void OnlineMapper::CacheDescriptors(
    const image_t image_id,
    std::shared_ptr<const FeatureDescriptors> descriptors) {
  if (descriptors) {
    descriptors_[image_id] = std::move(descriptors);
  }
}

void OnlineMapper::CacheKeypointScales(const image_t image_id,
                                       const FeatureKeypoints& keypoints) {
  if (options_.scale_check_max_ratio <= 1.0) {
    return;
  }
  std::vector<float>& scales = keypoint_scales_[image_id];
  scales.resize(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    scales[i] = keypoints[i].ComputeScale();
  }
}

size_t OnlineMapper::LocalMapWindowSize() const {
  return static_cast<size_t>(std::max(options_.local_map_num_images, 0));
}

std::vector<OnlineMapper::ProjectionMatch>
OnlineMapper::SearchLocalMapByProjection(
    const image_t image_id,
    const Rigid3d& cam_from_world,
    const double radius_px,
    const FlatHashSet<point2D_t>& skip_points2D,
    const FlatHashSet<point3D_t>& skip_points3D) {
  if (options_.local_map_num_images <= 0 || radius_px <= 0.0 ||
      !reconstruction_->ExistsImage(image_id)) {
    return {};
  }

  std::vector<image_t> local_ids;
  for (const image_t id : reconstruction_->RegImageIds()) {
    if (id < image_id && reconstruction_->Image(id).HasPose()) {
      local_ids.push_back(id);
    }
  }
  std::sort(local_ids.begin(), local_ids.end(), std::greater<image_t>());
  if (local_ids.size() > static_cast<size_t>(options_.local_map_num_images)) {
    local_ids.resize(options_.local_map_num_images);
  }
  const std::vector<image_t> spatial_ids =
      SpatialLocalMapImages(image_id, cam_from_world, local_ids);
  if (!spatial_ids.empty()) {
    // Projection needs a reference descriptor from the point's track.
    FlatHashSet<image_t> load_ids;
    for (const image_t id : spatial_ids) {
      load_ids.insert(id);
    }
    EnsureDescriptors(load_ids);
    local_ids.insert(local_ids.end(), spatial_ids.begin(), spatial_ids.end());
  }
  FlatHashSet<point3D_t> local_points;
  for (const image_t id : local_ids) {
    for (const Point2D& point2D : reconstruction_->Image(id).Points2D()) {
      if (point2D.HasPoint3D() && skip_points3D.count(point2D.point3D_id) == 0) {
        local_points.insert(point2D.point3D_id);
      }
    }
  }
  return SearchPointsByProjection(image_id,
                                  cam_from_world,
                                  radius_px,
                                  skip_points2D,
                                  local_points,
                                  /*visible_points=*/nullptr);
}

std::vector<image_t> OnlineMapper::SpatialLocalMapImages(
    const image_t image_id,
    const Rigid3d& cam_from_world,
    const std::vector<image_t>& exclude_ids) const {
  std::vector<image_t> ids;
  if (options_.local_map_spatial_num_images <= 0) {
    return ids;
  }
  const FlatHashSet<image_t> exclude(exclude_ids.begin(), exclude_ids.end());
  const Eigen::Vector3d center = cam_from_world.TgtOriginInSrc();
  const Eigen::Vector3d axis =
      cam_from_world.rotation().inverse() * Eigen::Vector3d::UnitZ();
  const double max_dist_sq = options_.local_map_spatial_max_distance_m *
                             options_.local_map_spatial_max_distance_m;
  const double min_cos =
      std::cos(DegToRad(options_.local_map_spatial_max_angle_deg));

  std::vector<std::pair<double, image_t>> nearby;
  for (const image_t id : reconstruction_->RegImageIds()) {
    // Only this session's frames carry a VIO prior.
    if (id == image_id || pose_priors_.count(id) > 0 || exclude.count(id) > 0) {
      continue;
    }
    const Image& other = reconstruction_->Image(id);
    if (!other.HasPose()) {
      continue;
    }
    const double dist_sq = (other.ProjectionCenter() - center).squaredNorm();
    if (dist_sq > max_dist_sq) {
      continue;
    }
    const Eigen::Vector3d other_axis =
        other.CamFromWorld().rotation().inverse() * Eigen::Vector3d::UnitZ();
    if (axis.dot(other_axis) < min_cos) {
      continue;
    }
    nearby.emplace_back(dist_sq, id);
  }
  std::sort(nearby.begin(), nearby.end());
  const size_t num =
      std::min(nearby.size(),
               static_cast<size_t>(options_.local_map_spatial_num_images));
  ids.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    ids.push_back(nearby[i].second);
  }
  return ids;
}

std::vector<OnlineMapper::ProjectionMatch>
OnlineMapper::SearchPointsByProjection(
    const image_t image_id,
    const Rigid3d& cam_from_world,
    const double radius_px,
    const FlatHashSet<point2D_t>& skip_points2D,
    const FlatHashSet<point3D_t>& candidate_points,
    std::vector<point3D_t>* visible_points,
    std::vector<point3D_t>* bare_points) const {
  std::vector<ProjectionMatch> matches;
  if (radius_px <= 0.0 || candidate_points.empty() ||
      !reconstruction_->ExistsImage(image_id)) {
    return matches;
  }
  const Image& image = reconstruction_->Image(image_id);
  const auto desc_it = descriptors_.find(image_id);
  if (desc_it == descriptors_.end() ||
      !DescriptorsMatchImage(desc_it->second.get(), image)) {
    return matches;
  }
  const FeatureDescriptors& desc = *desc_it->second;
  const Camera& camera = *image.CameraPtr();

  const double cell = std::max(radius_px, 8.0);
  const int grid_cols = static_cast<int>(std::ceil(camera.width / cell)) + 1;
  const int grid_rows = static_cast<int>(std::ceil(camera.height / cell)) + 1;
  std::vector<std::vector<point2D_t>> grid(grid_cols * grid_rows);
  for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
    if (skip_points2D.count(idx) > 0) {
      continue;
    }
    const Eigen::Vector2d& xy = image.Point2D(idx).xy;
    const int gx = std::clamp(static_cast<int>(xy.x() / cell), 0, grid_cols - 1);
    const int gy = std::clamp(static_cast<int>(xy.y() / cell), 0, grid_rows - 1);
    grid[gy * grid_cols + gx].push_back(idx);
  }

  const double max_l2 = options_.local_map_max_desc_distance * 512.0;
  const int max_l2_sq = static_cast<int>(max_l2 * max_l2);
  const double ratio_sq =
      options_.local_map_max_ratio * options_.local_map_max_ratio;
  const double radius_sq = radius_px * radius_px;
  const double min_cos_view = std::cos(DegToRad(60.0));
  const Eigen::Vector3d center = cam_from_world.TgtOriginInSrc();
  constexpr size_t kMaxRefDescriptors = 3;

  const double max_log_scale = options_.scale_check_max_ratio > 1.0
                                   ? std::log(options_.scale_check_max_ratio)
                                   : 0.0;
  const std::vector<float>* scales = nullptr;
  if (max_log_scale > 0.0) {
    const auto scale_it = keypoint_scales_.find(image_id);
    if (scale_it != keypoint_scales_.end() &&
        scale_it->second.size() == image.NumPoints2D()) {
      scales = &scale_it->second;
    }
  }
  const double focal = camera.MeanFocalLength();

  // Keypoint -> (point3D, descriptor distance); the closest point wins.
  FlatHashMap<point2D_t, std::pair<point3D_t, int>> claims;
  std::vector<std::pair<const FeatureDescriptors*, point2D_t>> refs;
  std::vector<double> ref_sizes;
  std::vector<TrackElement> track_elements;
  for (const point3D_t point3D_id : candidate_points) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const Point3D& point3D = reconstruction_->Point3D(point3D_id);
    const Eigen::Vector3d xyz_cam = cam_from_world * point3D.xyz;
    if (xyz_cam.z() <= std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const std::optional<Eigen::Vector2d> proj = camera.ImgFromCam(xyz_cam);
    if (!proj || proj->x() < 0.0 || proj->y() < 0.0 ||
        proj->x() >= camera.width || proj->y() >= camera.height) {
      continue;
    }

    Eigen::Vector3d mean_dir = Eigen::Vector3d::Zero();
    track_elements = point3D.track.Elements();
    std::sort(track_elements.begin(),
              track_elements.end(),
              [](const TrackElement& a, const TrackElement& b) {
                return a.image_id > b.image_id;
              });
    refs.clear();
    ref_sizes.clear();
    for (const TrackElement& el : track_elements) {
      const Image& track_image = reconstruction_->Image(el.image_id);
      if (!track_image.HasPose()) {
        continue;
      }
      const Eigen::Vector3d ray = point3D.xyz - track_image.ProjectionCenter();
      mean_dir += ray.normalized();
      if (refs.size() >= kMaxRefDescriptors) {
        continue;
      }
      const auto ref_it = descriptors_.find(el.image_id);
      if (ref_it != descriptors_.end() &&
          DescriptorsMatchImage(ref_it->second.get(), track_image)) {
        refs.emplace_back(ref_it->second.get(), el.point2D_idx);
        if (scales != nullptr) {
          const auto ref_scale_it = keypoint_scales_.find(el.image_id);
          if (ref_scale_it != keypoint_scales_.end() &&
              el.point2D_idx < ref_scale_it->second.size()) {
            // Range, not optical-axis depth: the fisheye scale follows range.
            ref_sizes.push_back(ref_scale_it->second[el.point2D_idx] *
                                ray.norm() /
                                track_image.CameraPtr()->MeanFocalLength());
          }
        }
      }
    }
    if (refs.empty() || mean_dir.squaredNorm() == 0.0 ||
        mean_dir.normalized().dot((point3D.xyz - center).normalized()) <
            min_cos_view) {
      continue;
    }
    double predicted_scale = 0.0;
    if (!ref_sizes.empty()) {
      std::nth_element(ref_sizes.begin(),
                       ref_sizes.begin() + ref_sizes.size() / 2,
                       ref_sizes.end());
      predicted_scale = ref_sizes[ref_sizes.size() / 2] * focal /
                        (point3D.xyz - center).norm();
    }
    const int gx0 = std::max(static_cast<int>((proj->x() - radius_px) / cell), 0);
    const int gx1 =
        std::min(static_cast<int>((proj->x() + radius_px) / cell), grid_cols - 1);
    const int gy0 = std::max(static_cast<int>((proj->y() - radius_px) / cell), 0);
    const int gy1 =
        std::min(static_cast<int>((proj->y() + radius_px) / cell), grid_rows - 1);
    int best = std::numeric_limits<int>::max();
    int second = std::numeric_limits<int>::max();
    point2D_t best_idx = kInvalidPoint2DIdx;
    bool any_near = false;
    for (int gy = gy0; gy <= gy1; ++gy) {
      for (int gx = gx0; gx <= gx1; ++gx) {
        for (const point2D_t idx : grid[gy * grid_cols + gx]) {
          if ((image.Point2D(idx).xy - *proj).squaredNorm() > radius_sq) {
            continue;
          }
          any_near = true;
          if (predicted_scale > 0.0 && (*scales)[idx] > 0.0f &&
              std::abs(std::log((*scales)[idx] / predicted_scale)) >
                  max_log_scale) {
            ++num_scale_rejected_;
            continue;
          }
          int l2 = std::numeric_limits<int>::max();
          for (const auto& [ref_desc, ref_idx] : refs) {
            l2 = std::min(l2, DescriptorL2Sq(*ref_desc, ref_idx, desc, idx));
          }
          if (l2 < best) {
            second = best;
            best = l2;
            best_idx = idx;
          } else if (l2 < second) {
            second = l2;
          }
        }
      }
    }
    // With no keypoint near the projection the point may be occluded or not
    // detectable from here, which is no evidence that it is gone.
    if (visible_points != nullptr && any_near) {
      visible_points->push_back(point3D_id);
    }
    if (bare_points != nullptr && !any_near) {
      bare_points->push_back(point3D_id);
    }
    if (best_idx == kInvalidPoint2DIdx || best > max_l2_sq ||
        (options_.local_map_max_ratio < 1.0 &&
         second != std::numeric_limits<int>::max() &&
         best > ratio_sq * second)) {
      continue;
    }
    const auto claim_it = claims.find(best_idx);
    if (claim_it == claims.end() || best < claim_it->second.second) {
      claims[best_idx] = {point3D_id, best};
    }
  }

  matches.reserve(claims.size());
  for (const auto& [point2D_idx, claim] : claims) {
    matches.push_back({point2D_idx, claim.first});
  }
  return matches;
}

size_t OnlineMapper::ExtendTracksByProjection(const image_t image_id,
                                              const char* stage) {
  if (options_.local_map_num_images <= 0 ||
      !reconstruction_->ExistsImage(image_id) ||
      !reconstruction_->Image(image_id).HasPose()) {
    return 0;
  }
  Image& image = reconstruction_->Image(image_id);
  FlatHashSet<point2D_t> skip_points2D;
  FlatHashSet<point3D_t> skip_points3D;
  for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
    const Point2D& point2D = image.Point2D(idx);
    if (point2D.HasPoint3D()) {
      skip_points2D.insert(idx);
      skip_points3D.insert(point2D.point3D_id);
    }
  }
  const Rigid3d cam_from_world = image.CamFromWorld();
  const size_t scale_rejected_before = num_scale_rejected_;
  const std::vector<ProjectionMatch> matches =
      SearchLocalMapByProjection(image_id,
                                 cam_from_world,
                                 options_.local_map_refine_radius_px,
                                 skip_points2D,
                                 skip_points3D);
  const Camera& camera = *image.CameraPtr();
  const double max_sq =
      options_.abs_pose_max_error * options_.abs_pose_max_error;
  size_t num_added = 0;
  for (const ProjectionMatch& match : matches) {
    if (image.Point2D(match.point2D_idx).HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(match.point3D_id)) {
      continue;
    }
    if (CalculateSquaredReprojectionError(
            image.Point2D(match.point2D_idx).xy,
            reconstruction_->Point3D(match.point3D_id).xyz,
            cam_from_world,
            camera) > max_sq) {
      continue;
    }
    obs_manager_->AddObservation(match.point3D_id,
                                 TrackElement(image_id, match.point2D_idx));
    triangulator_->AddModifiedPoint3D(match.point3D_id);
    ++num_added;
  }
  LOG(INFO) << "OnlineMapper: local-map extend (" << stage << ") image "
            << image_id
            << " candidates=" << matches.size() << " added=" << num_added
            << " scale_rejected=" << num_scale_rejected_ - scale_rejected_before;
  return num_added;
}

bool OnlineMapper::RegisterImageWithKnownPose(const image_t image_id,
                                              const Rigid3d& cam_from_world) {
  if (!reconstruction_->ExistsImage(image_id)) {
    LOG(ERROR) << "OnlineMapper: image " << image_id
               << " missing from reconstruction";
    return false;
  }
  Image& image = reconstruction_->Image(image_id);
  if (image.HasPose()) {
    return true;
  }
  if (!image.HasFramePtr()) {
    LOG(ERROR) << "OnlineMapper: image " << image_id << " has no frame";
    return false;
  }
  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  obs_manager_->RegisterFrame(image.FrameId());
  LOG(INFO) << "OnlineMapper: registered image " << image_id
            << " with known pose";
  return true;
}

image_t OnlineMapper::FindLastRegisteredWithPrior(
    const image_t image_id) const {
  image_t best_id = kInvalidImageId;
  for (const image_t id : reconstruction_->RegImageIds()) {
    if (id >= image_id || !reconstruction_->ExistsImage(id) ||
        !reconstruction_->Image(id).HasPose()) {
      continue;
    }
    if (pose_priors_.find(id) == pose_priors_.end()) {
      continue;
    }
    if (best_id == kInvalidImageId || id > best_id) {
      best_id = id;
    }
  }
  return best_id;
}

Rigid3d OnlineMapper::MapFrameCamFromWorld(const image_t image_id,
                                           const Rigid3d& vio_prior) const {
  const image_t last_id = FindLastRegisteredWithPrior(image_id);
  if (tracking_anchor_ && tracking_anchor_->image_id < image_id &&
      (last_id == kInvalidImageId || tracking_anchor_->image_id > last_id)) {
    return vio_prior * Inverse(tracking_anchor_->vio_cam_from_world) *
           tracking_anchor_->cam_from_world;
  }
  if (last_id == kInvalidImageId) {
    return vio_prior;
  }
  return vio_prior * Inverse(pose_priors_.at(last_id)) *
         reconstruction_->Image(last_id).CamFromWorld();
}

bool OnlineMapper::RegisterWithRelativeVio(const image_t image_id,
                                           const Rigid3d& vio_prior) {
  const image_t last_id = FindLastRegisteredWithPrior(image_id);
  const Rigid3d cam_from_world = MapFrameCamFromWorld(image_id, vio_prior);
  if (last_id == kInvalidImageId) {
    LOG(INFO) << "OnlineMapper: relative VIO unavailable for image " << image_id
              << ", using absolute VIO";
  } else {
    LOG(INFO) << "OnlineMapper: relative VIO image " << image_id << " from "
              << last_id;
  }
  return RegisterImageWithKnownPose(image_id, cam_from_world);
}

void OnlineMapper::UnregisterFrame(const image_t image_id) {
  if (!reconstruction_->ExistsImage(image_id)) {
    return;
  }
  Image& image = reconstruction_->Image(image_id);
  if (!image.HasFramePtr()) {
    return;
  }
  const frame_t frame_id = image.FrameId();
  if (image.HasPose()) {
    obs_manager_->DeRegisterFrame(frame_id);
  }
  reconstruction_->Frame(frame_id).ResetPose();
}

OnlineMapper::PriorRegisterResult OnlineMapper::RegisterNextImageWithPrior(
    const image_t image_id) {
  PriorRegisterResult result;
  if (reconstruction_->NumRegFrames() == 0) {
    return result;
  }
  Image& image = reconstruction_->Image(image_id);
  if (image.HasPose()) {
    result.success = true;
    result.cam_from_world = image.CamFromWorld();
    return result;
  }

  Camera& camera = *image.CameraPtr();
  const auto correspondence_graph = database_cache_->CorrespondenceGraph();
  if (!correspondence_graph) {
    return result;
  }

  const auto prior_it = pose_priors_.find(image_id);
  if (prior_it == pose_priors_.end()) {
    LOG(INFO) << "OnlineMapper: skip image " << image_id << ", no VIO prior";
    return result;
  }
  const bool use_local_map = options_.local_map_num_images > 0;
  if (!use_local_map &&
      obs_manager_->NumVisiblePoints3D(image_id) <
          static_cast<size_t>(options_.abs_pose_min_num_inliers)) {
    LOG(INFO) << "OnlineMapper: skip image " << image_id
              << ", visible_points3D="
              << obs_manager_->NumVisiblePoints3D(image_id);
    return result;
  }

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;
  FlatHashSet<point3D_t> corr_point3D_ids;

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    corr_point3D_ids.clear();
    const auto corr_range =
        correspondence_graph->FindCorrespondences(image_id, point2D_idx);
    for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
      if (!reconstruction_->ExistsImage(corr->image_id)) {
        continue;
      }
      const Image& corr_image = reconstruction_->Image(corr->image_id);
      if (!corr_image.HasPose()) {
        continue;
      }
      const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }
      if (corr_point3D_ids.count(corr_point2D.point3D_id) > 0) {
        continue;
      }
      corr_point3D_ids.insert(corr_point2D.point3D_id);
      tri_corrs.emplace_back(point2D_idx, corr_point2D.point3D_id);
      tri_points2D.push_back(point2D.xy);
      tri_points3D.push_back(
          reconstruction_->Point3D(corr_point2D.point3D_id).xyz);
    }
  }

  const size_t num_graph_corrs = tri_corrs.size();
  size_t num_local_map_corrs = 0;
  if (use_local_map) {
    FlatHashSet<point2D_t> skip_points2D;
    FlatHashSet<point3D_t> skip_points3D;
    for (const auto& [point2D_idx, point3D_id] : tri_corrs) {
      skip_points2D.insert(point2D_idx);
      skip_points3D.insert(point3D_id);
    }
    const std::vector<ProjectionMatch> local_matches =
        SearchLocalMapByProjection(
            image_id,
            MapFrameCamFromWorld(image_id, prior_it->second),
            options_.local_map_search_radius_px,
            skip_points2D,
            skip_points3D);
    for (const ProjectionMatch& match : local_matches) {
      tri_corrs.emplace_back(match.point2D_idx, match.point3D_id);
      tri_points2D.push_back(image.Point2D(match.point2D_idx).xy);
      tri_points3D.push_back(reconstruction_->Point3D(match.point3D_id).xyz);
    }
    num_local_map_corrs = local_matches.size();
  }

  if (tri_points2D.size() <
      static_cast<size_t>(options_.abs_pose_min_num_inliers)) {
    LOG(INFO) << "OnlineMapper: skip image " << image_id
              << ", 2D-3D=" << tri_points2D.size()
              << " graph=" << num_graph_corrs
              << " local_map=" << num_local_map_corrs;
    return result;
  }

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  abs_pose_refinement_options.refine_focal_length = false;
  abs_pose_refinement_options.refine_extra_params = false;

  const size_t min_num_inliers =
      static_cast<size_t>(options_.abs_pose_min_num_inliers);
  size_t num_inliers = 0;
  std::vector<char> inlier_mask(tri_points2D.size(), 0);
  Rigid3d cam_from_world = MapFrameCamFromWorld(image_id, prior_it->second);
  const double max_squared_error =
      options_.abs_pose_max_error * options_.abs_pose_max_error;
  for (size_t i = 0; i < tri_points2D.size(); ++i) {
    if (CalculateSquaredReprojectionError(tri_points2D[i],
                                          tri_points3D[i],
                                          cam_from_world,
                                          camera) <= max_squared_error) {
      inlier_mask[i] = 1;
      ++num_inliers;
    }
  }
  if (num_inliers < min_num_inliers ||
      !RefineAbsolutePose(abs_pose_refinement_options,
                          inlier_mask,
                          tri_points2D,
                          tri_points3D,
                          &cam_from_world,
                          &camera)) {
    LOG(INFO) << "OnlineMapper: prior init failed for image " << image_id
              << ", inliers=" << num_inliers;
    return result;
  }

  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  obs_manager_->RegisterFrame(image.FrameId());

  FlatHashSet<point3D_t> observed_point3D_ids;
  size_t num_local_map_inliers = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (!inlier_mask[i]) {
      continue;
    }
    if (i >= num_graph_corrs) {
      ++num_local_map_inliers;
    }
    const auto [point2D_idx, point3D_id] = tri_corrs[i];
    if (!image.Point2D(point2D_idx).HasPoint3D() &&
        observed_point3D_ids.insert(point3D_id).second) {
      obs_manager_->AddObservation(point3D_id,
                                   TrackElement(image_id, point2D_idx));
      triangulator_->AddModifiedPoint3D(point3D_id);
    }
  }

  result.success = true;
  result.num_inliers = num_inliers;
  result.cam_from_world = cam_from_world;
  LOG(INFO) << "OnlineMapper: prior-init registered image " << image_id
            << " inliers=" << num_inliers
            << " graph=" << num_graph_corrs
            << " local_map=" << num_local_map_corrs
            << " local_map_inliers=" << num_local_map_inliers;
  return result;
}

image_t OnlineMapper::FindInitPartner(const image_t image_id) const {
  if (!database_cache_->CorrespondenceGraph()) {
    return kInvalidImageId;
  }
  image_t best_id = kInvalidImageId;
  const auto graph = database_cache_->CorrespondenceGraph();
  for (const auto& [prev_id, image] : database_cache_->Images()) {
    (void)image;
    if (prev_id >= image_id) {
      continue;
    }
    if (pose_priors_.find(prev_id) == pose_priors_.end()) {
      continue;
    }
    if (graph->NumMatchesBetweenImages(prev_id, image_id) <
        static_cast<point2D_t>(options_.min_num_inliers)) {
      continue;
    }
    if (best_id == kInvalidImageId || prev_id > best_id) {
      best_id = prev_id;
    }
  }
  return best_id;
}

bool OnlineMapper::LoadSparseMap(
    std::vector<std::string>* registered_image_names) {
  if (registered_image_names == nullptr) {
    return false;
  }
  registered_image_names->clear();
  if (sparse_path_.empty()) {
    LOG(ERROR) << "OnlineMapper: empty sparse path, cannot resume";
    return false;
  }
  try {
    reconstruction_->Read(sparse_path_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineMapper: failed to read sparse map at " << sparse_path_
               << ": " << e.what();
    return false;
  }
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    registered_image_names->push_back(reconstruction_->Image(image_id).Name());
  }
  ReadPointStats();
  ReadVioPriors();
  LOG(INFO) << "OnlineMapper: loaded sparse map "
            << sparse_path_ << " frames=" << reconstruction_->NumRegImages()
            << " points=" << reconstruction_->NumPoints3D();
  return !registered_image_names->empty();
}

bool OnlineMapper::RebuildAfterRestore() {
  if (!database_cache_ || database_cache_->NumImages() == 0) {
    LOG(ERROR) << "OnlineMapper: cannot restore, database cache has no images";
    return false;
  }
  const size_t num_reg_before = reconstruction_->NumRegImages();
  try {
    // Only adds metadata for cache images; restored poses and points stay.
    reconstruction_->Load(*database_cache_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineMapper: existing map does not match the current "
                  "feature configuration: "
               << e.what();
    return false;
  }
  if (reconstruction_->NumRegImages() != num_reg_before) {
    LOG(WARNING) << "OnlineMapper: restore changed registered frames from "
                 << num_reg_before << " to " << reconstruction_->NumRegImages();
  }
  obs_manager_ = std::make_shared<ObservationManager>(
      *reconstruction_, database_cache_->CorrespondenceGraph());
  triangulator_ = std::make_shared<IncrementalTriangulator>(
      database_cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);
  initialized_ = reconstruction_->NumRegImages() > 0;
  LOG(INFO) << "OnlineMapper: resumed map frames="
            << reconstruction_->NumRegImages()
            << " points=" << reconstruction_->NumPoints3D()
            << " cache_images=" << database_cache_->NumImages()
            << " initialized=" << initialized_;
  return initialized_;
}

bool OnlineMapper::RegisterRelocalizedFrame(
    const image_t image_id,
    const Rigid3d& cam_from_world,
    const Rigid3d& vio_cam_from_world,
    const std::vector<std::pair<point2D_t, point3D_t>>& loop_observations) {
  if (!reconstruction_->ExistsImage(image_id)) {
    LOG(ERROR) << "OnlineMapper: relocalized image " << image_id
               << " missing from reconstruction";
    return false;
  }
  if (!RegisterImageWithKnownPose(image_id, cam_from_world)) {
    return false;
  }
  // VIO-frame prior only. Using the map pose here would poison the chain that
  // MapFrameCamFromWorld() uses for every following frame.
  pose_priors_[image_id] = vio_cam_from_world;
  reloc_anchor_ids_.insert(image_id);
  initialized_ = true;

  Image& image = reconstruction_->Image(image_id);
  size_t num_attached = 0;
  for (const auto& [point2D_idx, point3D_id] : loop_observations) {
    if (point2D_idx >= image.NumPoints2D() ||
        image.Point2D(point2D_idx).HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    try {
      obs_manager_->AddObservation(point3D_id,
                                   TrackElement(image_id, point2D_idx));
      ++num_attached;
    } catch (const std::exception& e) {
      LOG(WARNING) << "OnlineMapper: attach relocalization point " << point3D_id
                   << " to image " << image_id << " failed: " << e.what();
    }
  }

  ExtendTracksByProjection(image_id, "reloc");
  TriangulateImage(image_id);
  MergeAndFilterPoints(image_id);
  AdjustLocalBundle(image_id);
  LOG(INFO) << "OnlineMapper: relocalized image " << image_id
            << " attached_map_points=" << num_attached
            << " center=" << cam_from_world.TgtOriginInSrc().transpose();
  return true;
}

bool OnlineMapper::TryInitialize(const image_t image_id) {
  // A resumed map already has registered frames. Starting a second, disjoint
  // component inside the same reconstruction would corrupt the sparse model.
  if (reconstruction_->NumRegFrames() > 0) {
    initialized_ = true;
    LOG(WARNING) << "OnlineMapper: skip init for image " << image_id
                 << ", map already has " << reconstruction_->NumRegFrames()
                 << " registered frame(s)";
    return true;
  }
  const auto current_prior = pose_priors_.find(image_id);
  if (current_prior == pose_priors_.end()) {
    LOG(INFO) << "OnlineMapper: init wait, image " << image_id
              << " has no VIO prior";
    return true;
  }

  const image_t partner_id = FindInitPartner(image_id);
  if (partner_id == kInvalidImageId) {
    LOG(INFO) << "OnlineMapper: init wait, image " << image_id
              << " has no partner with VIO + inliers";
    return true;
  }

  if (!RegisterImageWithKnownPose(partner_id, pose_priors_.at(partner_id)) ||
      !RegisterImageWithKnownPose(image_id, current_prior->second)) {
    UnregisterFrame(partner_id);
    UnregisterFrame(image_id);
    return false;
  }

  const size_t num_tris = TriangulateImage(partner_id) +
                          TriangulateImage(image_id);
  if (reconstruction_->NumPoints3D() == 0) {
    LOG(WARNING) << "OnlineMapper: init triangulation produced 0 points, "
                    "pair="
                 << partner_id << "-" << image_id;
    UnregisterFrame(partner_id);
    UnregisterFrame(image_id);
    return true;
  }
  // AdjustLocalBundle(image_id);
  initialized_ = true;
  LOG(INFO) << "OnlineMapper: initialized with pair " << partner_id << "-"
            << image_id << " points3D=" << reconstruction_->NumPoints3D()
            << " tris=" << num_tris;
  WriteSparse();
  return true;
}

size_t OnlineMapper::TriangulateImage(const image_t image_id) {
  const size_t num_tris =
      triangulator_->TriangulateImage(options_.triangulation, image_id);
  LOG(INFO) << "OnlineMapper: triangulate image " << image_id
            << " added=" << num_tris
            << " points3D=" << reconstruction_->NumPoints3D();
  return num_tris;
}

size_t OnlineMapper::FilterPointsInImages(
    const FlatHashSet<image_t>& image_ids) {
  if (!obs_manager_) {
    return 0;
  }
  size_t num_filtered = obs_manager_->FilterPoints3DInImages(
      options_.filter_max_reproj_error,
      options_.filter_min_tri_angle,
      image_ids);
  if (options_.filter_two_view_min_tri_angle <= 0.0) {
    return num_filtered;
  }
  FlatHashSet<point3D_t> point_ids;
  for (const image_t image_id : image_ids) {
    if (!reconstruction_->ExistsImage(image_id)) {
      continue;
    }
    for (const Point2D& point2D : reconstruction_->Image(image_id).Points2D()) {
      if (point2D.HasPoint3D()) {
        point_ids.insert(point2D.point3D_id);
      }
    }
  }
  num_filtered += obs_manager_->FilterTwoViewPointsWithSmallTriangulationAngle(
      options_.filter_two_view_min_tri_angle, point_ids);
  return num_filtered;
}

void OnlineMapper::MergeAndFilterPoints(const image_t image_id) {
  if (!triangulator_ || !obs_manager_ ||
      !reconstruction_->ExistsImage(image_id)) {
    return;
  }

  const size_t num_completed_image =
      triangulator_->CompleteImage(options_.triangulation, image_id);

  FlatHashSet<point3D_t> point_ids = triangulator_->GetModifiedPoints3D();
  for (const Point2D& point2D : reconstruction_->Image(image_id).Points2D()) {
    if (point2D.HasPoint3D()) {
      point_ids.insert(point2D.point3D_id);
    }
  }

  size_t num_completed_tracks = 0;
  size_t num_merged = 0;
  if (!point_ids.empty()) {
    num_completed_tracks =
        triangulator_->CompleteTracks(options_.triangulation, point_ids);
    num_merged = triangulator_->MergeTracks(options_.triangulation, point_ids);
  }

  FlatHashSet<image_t> filter_ids;
  filter_ids.insert(image_id);
  const size_t num_filtered = FilterPointsInImages(filter_ids);
  triangulator_->ClearModifiedPoints3D();

  LOG(INFO) << "OnlineMapper: merge/filter image " << image_id
            << " merged=" << num_merged
            << " completed=" << (num_completed_tracks + num_completed_image)
            << " filtered=" << num_filtered
            << " points3D=" << reconstruction_->NumPoints3D();
}

bool OnlineMapper::CollectRegisteredImage(const image_t image_id,
                                          std::vector<image_t>* images) const {
  if (images == nullptr || !reconstruction_->ExistsImage(image_id)) {
    return false;
  }
  const Image& image = reconstruction_->Image(image_id);
  if (!image.HasPose() || !image.HasFramePtr()) {
    return false;
  }
  images->push_back(image_id);
  return true;
}

void OnlineMapper::BuildCovisibilityBundle(
    const image_t image_id,
    std::vector<image_t>* local_ids,
    std::vector<image_t>* constant_ids) const {
  local_ids->clear();
  constant_ids->clear();
  CollectRegisteredImage(image_id, local_ids);
  if (local_ids->empty()) {
    return;
  }

  const Image& image = reconstruction_->Image(image_id);
  FlatHashMap<image_t, int> shared_points;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    for (const TrackElement& el :
         reconstruction_->Point3D(point2D.point3D_id).track.Elements()) {
      if (el.image_id == image_id) {
        continue;
      }
      shared_points[el.image_id] += 1;
    }
  }

  std::vector<std::pair<int, image_t>> covisible;
  covisible.reserve(shared_points.size());
  int best_count = 0;
  image_t best_id = kInvalidImageId;
  for (const auto& [other_id, count] : shared_points) {
    if (count > best_count) {
      best_count = count;
      best_id = other_id;
    }
    if (count >= options_.ba_min_covisibility_points) {
      covisible.emplace_back(count, other_id);
    }
  }
  // ORB-SLAM3: if nobody reaches the threshold, keep the strongest edge.
  if (covisible.empty() && best_id != kInvalidImageId) {
    covisible.emplace_back(best_count, best_id);
  }
  std::sort(covisible.begin(),
            covisible.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  const size_t max_local =
      options_.ba_local_num_images > 0
          ? static_cast<size_t>(options_.ba_local_num_images)
          : std::numeric_limits<size_t>::max();
  FlatHashSet<image_t> local_set(local_ids->begin(), local_ids->end());
  for (size_t i = 0; i < covisible.size() && local_set.size() < max_local;
       ++i) {
    const image_t other_id = covisible[i].second;
    if (CollectRegisteredImage(other_id, local_ids)) {
      local_set.insert(other_id);
    }
  }
}

void OnlineMapper::AppendLbaPoseRecord(
    const image_t image_id,
    const std::vector<image_t>& first_level_ids,
    const std::vector<image_t>& second_level_ids,
    const Rigid3d& pose_before,
    const Rigid3d& pose_after,
    const bool usable,
    const bool jumped,
    const bool reverted) const {
  if (sparse_path_.empty()) {
    return;
  }
  std::filesystem::path out_path =
      std::filesystem::path(sparse_path_).parent_path() / "lba_poses.txt";
  if (out_path.parent_path().empty()) {
    out_path = std::filesystem::path(sparse_path_) / "lba_poses.txt";
  }

  try {
    std::filesystem::create_directories(out_path.parent_path());
    const bool write_header = !std::filesystem::exists(out_path) ||
                              std::filesystem::file_size(out_path) == 0;
    std::ofstream file(out_path, std::ios::app);
    if (!file.is_open()) {
      LOG(WARNING) << "OnlineMapper: failed to open " << out_path.string();
      return;
    }
    file << std::setprecision(9);
    if (write_header) {
      file << "# OnlineMapper LBA current-frame pose (COLMAP cam_from_world)\n"
           << "# q = qx qy qz qw, t = tx ty tz, c = camera center in world\n"
           << "# delta = after * Inverse(before); dcenter_m / dangle_deg "
              "are the pose change\n"
           << "# image_id usable jumped reverted first_level fixed "
              "qx_b qy_b qz_b qw_b tx_b ty_b tz_b cx_b cy_b cz_b "
              "qx_a qy_a qz_a qw_a tx_a ty_a tz_a cx_a cy_a cz_a "
              "qx_d qy_d qz_d qw_d tx_d ty_d tz_d dcenter_m dangle_deg\n";
    }
    const Rigid3d delta = pose_after * Inverse(pose_before);
    const double dcenter =
        (pose_after.TgtOriginInSrc() - pose_before.TgtOriginInSrc()).norm();
    const double dangle_deg = RadToDeg(
        Eigen::AngleAxisd(pose_after.rotation() * pose_before.rotation().inverse())
            .angle());
    file << image_id << " " << static_cast<int>(usable) << " "
         << static_cast<int>(jumped) << " " << static_cast<int>(reverted) << " "
         << JoinImageIds(first_level_ids) << " "
         << JoinImageIds(second_level_ids) << " ";
    AppendPoseColumns(file, pose_before);
    file << " ";
    AppendPoseColumns(file, pose_after);
    file << " " << delta.rotation().x() << " " << delta.rotation().y() << " "
         << delta.rotation().z() << " " << delta.rotation().w() << " "
         << delta.translation().x() << " " << delta.translation().y() << " "
         << delta.translation().z() << " " << dcenter << " " << dangle_deg
         << "\n";
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write LBA pose log: " << e.what();
  }
}

size_t OnlineMapper::AddVioPriorsToLocalBundle(
    const std::vector<image_t>& image_ids,
    Reconstruction& reconstruction,
    ceres::Problem* problem,
    ceres::LossFunction* loss,
    const FlatHashSet<image_t>* free_ids) const {
  if (options_.ba_vio_prior_rot_sigma_deg <= 0.0 || problem == nullptr) {
    return 0;
  }
  std::vector<image_t> ids;
  for (const image_t id : image_ids) {
    if (FindVioPrior(id, nullptr) != nullptr &&
        reconstruction.ExistsImage(id) && reconstruction.Image(id).HasPose() &&
        reconstruction.Image(id).IsRefInFrame()) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());

  const double rot_var = std::pow(DegToRad(options_.ba_vio_prior_rot_sigma_deg), 2);
  const size_t max_gap =
      static_cast<size_t>(std::max(options_.ba_vio_prior_max_gap, 1));
  size_t num_priors = 0;
  for (size_t k = 1; k < ids.size(); ++k) {
    const image_t id_i = ids[k - 1];
    const image_t id_j = ids[k];
    const std::optional<Rigid3d> i_from_j = VioRelativePose(id_i, id_j);
    if (!i_from_j) {
      continue;
    }
    const auto stored_i = stored_priors_.find(id_i);
    const size_t gap = stored_i != stored_priors_.end()
                           ? stored_priors_.at(id_j).rank - stored_i->second.rank
                           : static_cast<size_t>(KeyframeGap(id_i, id_j));
    if (gap > max_gap) {
      continue;
    }
    double* params_i =
        reconstruction.Image(id_i).FramePtr()->RigFromWorld().params.data();
    double* params_j =
        reconstruction.Image(id_j).FramePtr()->RigFromWorld().params.data();
    // Blocks missing from the problem belong to constant-pose frames, except
    // `free_ids` frames, which enter the problem here as variable poses.
    const bool has_i = problem->HasParameterBlock(params_i);
    const bool has_j = problem->HasParameterBlock(params_j);
    const bool free_i = !has_i && free_ids != nullptr && free_ids->count(id_i) > 0;
    const bool free_j = !has_j && free_ids != nullptr && free_ids->count(id_j) > 0;
    if (!has_i && !has_j && !free_i && !free_j) {
      continue;
    }
    if (free_i) {
      reconstruction.Image(id_i).FramePtr()->RigFromWorld().rotation().normalize();
    }
    if (free_j) {
      reconstruction.Image(id_j).FramePtr()->RigFromWorld().rotation().normalize();
    }
    const Rigid3d& i_from_j_prior = *i_from_j;
    const double trans_sigma =
        options_.ba_vio_prior_trans_sigma_m +
        options_.ba_vio_prior_trans_sigma_ratio *
            i_from_j_prior.translation().norm();
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov.diagonal().head<3>().setConstant(rot_var);
    cov.diagonal().tail<3>().setConstant(trans_sigma * trans_sigma);
    problem->AddResidualBlock(
        CovarianceWeightedCostFunctor<RelativePosePriorCostFunctor>::Create(
            cov, i_from_j_prior),
        loss,
        params_i,
        params_j);
    const auto SetNewBlock = [&](double* params, const bool is_free) {
      if (is_free) {
        SetManifold(problem,
                    params,
                    CreateProductManifold(CreateEigenQuaternionManifold(),
                                          CreateEuclideanManifold<3>()));
      } else {
        problem->SetParameterBlockConstant(params);
      }
    };
    if (!has_i) {
      SetNewBlock(params_i, free_i);
    }
    if (!has_j) {
      SetNewBlock(params_j, free_j);
    }
    ++num_priors;
  }
  return num_priors;
}

void OnlineMapper::AdjustLocalBundle(const image_t image_id) {
  if (!reconstruction_->ExistsImage(image_id) ||
      !reconstruction_->Image(image_id).HasPose()) {
    return;
  }

  std::vector<image_t> local_ids;
  std::vector<image_t> constant_ids;
  BuildCovisibilityBundle(image_id, &local_ids, &constant_ids);
  const std::vector<image_t> first_level_ids =
      FirstLevelNeighbors(image_id, local_ids);
  // Weak covisibility (e.g. VIO-fallback frames): pad with the most recent
  // registered frames, tied together by the VIO priors.
  size_t num_temporal = 0;
  if (local_ids.size() <
      static_cast<size_t>(std::max(options_.ba_min_window_images, 0))) {
    std::vector<image_t> recent_ids;
    for (const image_t id : reconstruction_->RegImageIds()) {
      if (id < image_id &&
          std::find(local_ids.begin(), local_ids.end(), id) == local_ids.end()) {
        recent_ids.push_back(id);
      }
    }
    std::sort(recent_ids.begin(), recent_ids.end(), std::greater<image_t>());
    for (const image_t id : recent_ids) {
      if (num_temporal >= static_cast<size_t>(
                              std::max(options_.ba_temporal_window_images, 0))) {
        break;
      }
      if (CollectRegisteredImage(id, &local_ids)) {
        ++num_temporal;
      }
    }
  }
  LOG(INFO) << "OnlineMapper: LBA window image " << image_id
            << " first=[" << JoinImageIds(first_level_ids) << "]"
            << " temporal=" << num_temporal;

  // Two registered frames are not enough to constrain a local BA. Mapping
  // already triangulated this image before AdjustLocalBundle.
  if (local_ids.size() <= 2) {
    LOG(INFO) << "OnlineMapper: skip LBA image " << image_id
              << ", frames=" << local_ids.size()
              << " first=[" << JoinImageIds(first_level_ids) << "]";
    return;
  }

  image_t oldest_id = local_ids.front();
  image_t oldest_strong_id = kInvalidImageId;
  for (const image_t id : local_ids) {
    if (id < oldest_id) {
      oldest_id = id;
    }
    if (id != image_id && options_.ba_fixed_min_points > 0 &&
        reconstruction_->Image(id).NumPoints3D() >=
            static_cast<size_t>(options_.ba_fixed_min_points) &&
        (oldest_strong_id == kInvalidImageId || id < oldest_strong_id)) {
      oldest_strong_id = id;
    }
  }
  if (oldest_strong_id != kInvalidImageId) {
    oldest_id = oldest_strong_id;
  }
  std::vector<image_t> variable_ids;
  variable_ids.reserve(local_ids.size() - 1);
  for (const image_t id : local_ids) {
    if (id != oldest_id) {
      variable_ids.push_back(id);
    }
  }
  constant_ids.clear();
  constant_ids.push_back(oldest_id);
  const std::vector<image_t> fixed_ids = {oldest_id};

  BundleAdjustmentConfig ba_config;
  for (const image_t id : variable_ids) {
    ba_config.AddImage(id);
  }
  for (const image_t id : constant_ids) {
    ba_config.AddImage(id);
    ba_config.SetConstantRigFromWorldPose(
        reconstruction_->Image(id).FrameId());
  }

  FlatHashSet<camera_t> camera_ids;
  for (const image_t ba_image_id : ba_config.Images()) {
    camera_ids.insert(reconstruction_->Image(ba_image_id).CameraId());
  }
  for (const camera_t camera_id : camera_ids) {
    ba_config.SetConstantCamIntrinsics(camera_id);
  }

  constexpr size_t kMaxTrackLength = 15;
  FlatHashSet<point3D_t> variable_point3D_ids;
  auto add_points_from_image = [&](const image_t ba_id) {
    for (const Point2D& point2D : reconstruction_->Image(ba_id).Points2D()) {
      if (!point2D.HasPoint3D() ||
          variable_point3D_ids.count(point2D.point3D_id) > 0) {
        continue;
      }
      const Point3D& point3D = reconstruction_->Point3D(point2D.point3D_id);
      if (!point3D.HasError() || point3D.track.Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point2D.point3D_id);
        variable_point3D_ids.insert(point2D.point3D_id);
      }
    }
  };
  for (const image_t ba_id : local_ids) {
    add_points_from_image(ba_id);
  }

  if (ba_config.NumImages() < 2 || ba_config.NumVariablePoints() == 0) {
    LOG(INFO) << "OnlineMapper: skip LBA, images=" << ba_config.NumImages()
              << " points=" << ba_config.NumVariablePoints();
    return;
  }

  const Rigid3d pose_before = reconstruction_->Image(image_id).CamFromWorld();

  // Problem does not own loss functions; must outlive the solve.
  const auto prior_loss =
      std::make_unique<ceres::HuberLoss>(std::sqrt(kChiSquare95ThreeDof * 2.0));
  auto bundle_adjuster = CreateDefaultCeresBundleAdjuster(
      MakeLocalBaOptions(0), ba_config, *reconstruction_);
  const size_t num_vio_priors = AddVioPriorsToLocalBundle(
      local_ids, *reconstruction_, bundle_adjuster->Problem().get(),
      prior_loss.get());
  if (lba_timing_) {
    lba_timing_->Start();
  }
  const auto summary = bundle_adjuster->Solve();
  if (lba_timing_) {
    lba_timing_->Record(
        std::to_string(image_id),
        {std::to_string(local_ids.size()),
         std::to_string(variable_ids.size()),
         std::to_string(summary ? summary->num_residuals : 0),
         std::to_string(num_vio_priors)});
  }
  const bool usable = summary && summary->IsSolutionUsable();
  const Rigid3d pose_after = reconstruction_->Image(image_id).CamFromWorld();
  AppendLbaPoseRecord(image_id,
                      first_level_ids,
                      fixed_ids,
                      pose_before,
                      pose_after,
                      usable,
                      /*jumped=*/false,
                      /*reverted=*/false);
  if (!summary) {
    LOG(WARNING) << "OnlineMapper: LBA failed image " << image_id;
    return;
  }
  if (!usable) {
    LOG(WARNING) << "OnlineMapper: LBA solution not usable image " << image_id
                 << " residuals=" << summary->num_residuals;
  }

  LOG(INFO) << "OnlineMapper: LBA image " << image_id
            << " first=[" << JoinImageIds(first_level_ids) << "]"
            << " fixed=" << oldest_id
            << " variable=" << variable_ids.size()
            << " temporal=" << num_temporal
            << " vio_priors=" << num_vio_priors
            << " residuals=" << summary->num_residuals;

  if (usable) {
    ExtendTracksByProjection(image_id, "lba");
    for (const point3D_t point3D_id : triangulator_->GetModifiedPoints3D()) {
      variable_point3D_ids.insert(point3D_id);
    }
  }
  triangulator_->MergeTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteImage(options_.triangulation, image_id);
  FlatHashSet<image_t> filter_ids(variable_ids.begin(), variable_ids.end());
  FilterPointsInImages(filter_ids);
}

void OnlineMapper::AdjustLoopBundle(const std::vector<image_t>& variable_ids,
                                    const std::vector<image_t>& anchor_ids) {
  BundleAdjustmentConfig ba_config;
  std::vector<image_t> used_variable;
  for (const image_t id : variable_ids) {
    if (reconstruction_->ExistsImage(id) &&
        reconstruction_->Image(id).HasPose()) {
      ba_config.AddImage(id);
      used_variable.push_back(id);
    }
  }
  size_t num_anchors = 0;
  for (const image_t id : anchor_ids) {
    if (!reconstruction_->ExistsImage(id) ||
        !reconstruction_->Image(id).HasPose() || ba_config.HasImage(id)) {
      continue;
    }
    ba_config.AddImage(id);
    ba_config.SetConstantRigFromWorldPose(
        reconstruction_->Image(id).FrameId());
    ++num_anchors;
  }
  if (used_variable.empty() || num_anchors == 0) {
    LOG(INFO) << "OnlineMapper: skip loop BA, variable=" << used_variable.size()
              << " anchors=" << num_anchors;
    return;
  }

  FlatHashSet<camera_t> camera_ids;
  for (const image_t ba_image_id : ba_config.Images()) {
    camera_ids.insert(reconstruction_->Image(ba_image_id).CameraId());
  }
  for (const camera_t camera_id : camera_ids) {
    ba_config.SetConstantCamIntrinsics(camera_id);
  }

  FlatHashSet<point3D_t> variable_point3D_ids;
  for (const image_t id : used_variable) {
    for (const Point2D& point2D : reconstruction_->Image(id).Points2D()) {
      if (point2D.HasPoint3D() &&
          variable_point3D_ids.insert(point2D.point3D_id).second) {
        ba_config.AddVariablePoint(point2D.point3D_id);
      }
    }
  }
  if (variable_point3D_ids.empty()) {
    return;
  }

  // Problem does not own loss functions; must outlive the solve.
  const auto prior_loss =
      std::make_unique<ceres::HuberLoss>(std::sqrt(kChiSquare95ThreeDof * 2.0));
  auto bundle_adjuster = CreateDefaultCeresBundleAdjuster(
      MakeLocalBaOptions(0, options_.ba_loop_max_num_iterations),
      ba_config,
      *reconstruction_);
  // Weak-texture frames have few or no observations; the VIO priors keep them
  // rigidly attached to their neighbours while the chain is refined.
  // Frames just before the chain stay fixed and tie its start to the old
  // trajectory, as the boundary node does in the pose graph.
  std::vector<image_t> prior_ids = used_variable;
  const image_t chain_front =
      *std::min_element(used_variable.begin(), used_variable.end());
  const image_t max_gap =
      static_cast<image_t>(std::max(options_.ba_vio_prior_max_gap, 1));
  image_t id = chain_front;
  for (image_t steps = 0; steps < max_gap && id > 1;) {
    --id;
    if (culled_ids_.count(id) > 0) {
      continue;
    }
    ++steps;
    if (reconstruction_->ExistsImage(id) &&
        reconstruction_->Image(id).HasPose()) {
      prior_ids.push_back(id);
    }
  }
  const FlatHashSet<image_t> free_ids(used_variable.begin(),
                                      used_variable.end());
  const size_t num_vio_priors =
      AddVioPriorsToLocalBundle(prior_ids,
                                *reconstruction_,
                                bundle_adjuster->Problem().get(),
                                prior_loss.get(),
                                &free_ids);
  const auto summary = bundle_adjuster->Solve();
  if (!summary || !summary->IsSolutionUsable()) {
    LOG(WARNING) << "OnlineMapper: loop BA failed, variable="
                 << JoinImageIds(used_variable)
                 << " vio_priors=" << num_vio_priors;
    return;
  }
  LOG(INFO) << "OnlineMapper: loop BA variable=[" << JoinImageIds(used_variable)
            << "] anchors=" << num_anchors
            << " points=" << variable_point3D_ids.size()
            << " vio_priors=" << num_vio_priors
            << " max_iterations=" << options_.ba_loop_max_num_iterations
            << " residuals=" << summary->num_residuals;

  triangulator_->MergeTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteTracks(options_.triangulation, variable_point3D_ids);
  FlatHashSet<image_t> filter_ids(used_variable.begin(), used_variable.end());
  FilterPointsInImages(filter_ids);
}

image_t OnlineMapper::KeyframeGap(const image_t id_i, const image_t id_j) const {
  const image_t lo = std::min(id_i, id_j);
  const image_t hi = std::max(id_i, id_j);
  const auto num_culled = std::distance(culled_ids_.lower_bound(lo),
                                        culled_ids_.lower_bound(hi));
  return hi - lo - static_cast<image_t>(num_culled);
}

double OnlineMapper::RedundantPointRatio(const image_t image_id,
                                         size_t* num_points) const {
  *num_points = 0;
  const Image& image = reconstruction_->Image(image_id);
  const Eigen::Vector3d center = image.ProjectionCenter();
  const double max_scale_sq =
      options_.cull_max_scale_ratio * options_.cull_max_scale_ratio;
  const int min_observers = std::max(options_.cull_min_other_observers, 1);
  std::unordered_map<image_t, Eigen::Vector3d> centers;
  const auto center_of = [&](const image_t id) -> const Eigen::Vector3d& {
    auto it = centers.find(id);
    if (it == centers.end()) {
      it = centers.emplace(id, reconstruction_->Image(id).ProjectionCenter())
               .first;
    }
    return it->second;
  };

  size_t num_redundant = 0;
  FlatHashSet<image_t> counted;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    ++(*num_points);
    const Point3D& point3D = reconstruction_->Point3D(point2D.point3D_id);
    const double own_dist_sq = (point3D.xyz - center).squaredNorm();
    counted.clear();
    int num_observers = 0;
    for (const TrackElement& el : point3D.track.Elements()) {
      if (el.image_id == image_id || !counted.insert(el.image_id).second) {
        continue;
      }
      const Image& other = reconstruction_->Image(el.image_id);
      if (!other.HasPose()) {
        continue;
      }
      if ((point3D.xyz - center_of(el.image_id)).squaredNorm() <=
          max_scale_sq * own_dist_sq) {
        if (++num_observers >= min_observers) {
          break;
        }
      }
    }
    if (num_observers >= min_observers) {
      ++num_redundant;
    }
  }
  return *num_points > 0
             ? static_cast<double>(num_redundant) / static_cast<double>(*num_points)
             : 0.0;
}

double OnlineMapper::ShortTrackRatio(const image_t image_id) const {
  const Image& image = reconstruction_->Image(image_id);
  size_t num_points = 0;
  size_t num_short = 0;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    ++num_points;
    if (reconstruction_->Point3D(point2D.point3D_id).track.Length() == 2) {
      ++num_short;
    }
  }
  return num_points > 0
             ? static_cast<double>(num_short) / static_cast<double>(num_points)
             : 0.0;
}

bool OnlineMapper::KeepsTriangulationParallax(const image_t image_id) const {
  const Image& image = reconstruction_->Image(image_id);
  const double min_multi_angle = options_.filter_min_tri_angle > 0.0
                                     ? DegToRad(options_.filter_min_tri_angle)
                                     : 0.0;
  const double min_two_view_angle =
      options_.filter_two_view_min_tri_angle > 0.0
          ? DegToRad(options_.filter_two_view_min_tri_angle)
          : 0.0;
  size_t num_points = 0;
  size_t num_failed = 0;
  std::vector<Eigen::Vector3d> centers;
  FlatHashSet<image_t> counted;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    ++num_points;
    const Point3D& point3D = reconstruction_->Point3D(point2D.point3D_id);
    centers.clear();
    counted.clear();
    for (const TrackElement& el : point3D.track.Elements()) {
      if (el.image_id == image_id || !counted.insert(el.image_id).second) {
        continue;
      }
      const Image& other = reconstruction_->Image(el.image_id);
      if (!other.HasPose()) {
        continue;
      }
      centers.push_back(other.ProjectionCenter());
    }
    const double min_angle =
        centers.size() >= 3 ? min_multi_angle : min_two_view_angle;
    if (centers.size() < 2 || min_angle <= 0.0) {
      if (centers.size() < 2) {
        ++num_failed;
      }
      continue;
    }
    double max_angle = 0.0;
    for (size_t i = 0; i < centers.size(); ++i) {
      for (size_t j = i + 1; j < centers.size(); ++j) {
        max_angle = std::max(
            max_angle,
            CalculateTriangulationAngle(centers[i], centers[j], point3D.xyz));
      }
    }
    if (max_angle <= min_angle) {
      ++num_failed;
    }
  }
  if (num_points == 0) {
    return false;
  }
  const double fail_ratio =
      static_cast<double>(num_failed) / static_cast<double>(num_points);
  return fail_ratio <= std::max(options_.cull_max_parallax_fail_ratio, 0.0);
}

void OnlineMapper::AppendCullRecord(const image_t image_id,
                                    const char* reason,
                                    const double score) const {
  if (sparse_path_.empty()) {
    return;
  }
  const std::filesystem::path out_path =
      std::filesystem::path(sparse_path_).parent_path() /
      "culled_keyframes.txt";
  try {
    const bool write_header = !std::filesystem::exists(out_path) ||
                              std::filesystem::file_size(out_path) == 0;
    std::ofstream file(out_path, std::ios::app);
    if (!file.is_open()) {
      return;
    }
    if (write_header) {
      file << "# image_id name reason score\n";
    }
    file << image_id << " " << reconstruction_->Image(image_id).Name() << " "
         << reason << " " << score << "\n";
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write cull record: " << e.what();
  }
}

bool OnlineMapper::RemoveKeyframe(const image_t image_id,
                                  const char* reason,
                                  const double score) {
  if (!obs_manager_ || !reconstruction_->ExistsImage(image_id) ||
      !reconstruction_->Image(image_id).HasPose()) {
    return false;
  }
  const size_t num_points_before = reconstruction_->NumPoints3D();
  UnregisterFrame(image_id);
  culled_ids_.insert(image_id);
  descriptors_.erase(image_id);
  keypoint_scales_.erase(image_id);
  pose_priors_.erase(image_id);
  stored_priors_.erase(image_id);
  ++sparse_writes_since_flush_;
  AppendCullRecord(image_id, reason, score);
  LOG(INFO) << "OnlineMapper: removed keyframe " << image_id << " (" << reason
            << " score=" << score << ") points3D=" << num_points_before
            << "->" << reconstruction_->NumPoints3D()
            << " frames=" << reconstruction_->NumRegFrames();
  return true;
}

FlatHashSet<image_t> OnlineMapper::RecentRegisteredIds() const {
  std::vector<image_t> reg_ids = reconstruction_->RegImageIds();
  std::sort(reg_ids.begin(), reg_ids.end(), std::greater<image_t>());
  const size_t num_recent = std::min(
      reg_ids.size(),
      static_cast<size_t>(std::max(options_.cull_protect_recent, 0)));
  return FlatHashSet<image_t>(reg_ids.begin(), reg_ids.begin() + num_recent);
}

bool OnlineMapper::IsRemovable(const image_t image_id,
                               const image_t current_id,
                               const FlatHashSet<image_t>& recent_ids,
                               const std::unordered_set<image_t>& protected_ids,
                               const bool allow_prior_session) const {
  if (image_id == current_id || recent_ids.count(image_id) > 0 ||
      protected_ids.count(image_id) > 0 ||
      reloc_anchor_ids_.count(image_id) > 0) {
    return false;
  }
  return pose_priors_.count(image_id) > 0 || allow_prior_session;
}

std::vector<image_t> OnlineMapper::CullRedundantKeyframes(
    const image_t image_id, const std::unordered_set<image_t>& protected_ids) {
  if (!options_.cull_enabled || !initialized_ || !obs_manager_ ||
      !reconstruction_->ExistsImage(image_id) ||
      !reconstruction_->Image(image_id).HasPose()) {
    return {};
  }
  std::vector<image_t> local_ids;
  std::vector<image_t> constant_ids;
  BuildCovisibilityBundle(image_id, &local_ids, &constant_ids);
  return CullRedundantKeyframes(local_ids, protected_ids);
}

std::vector<image_t> OnlineMapper::CullRedundantKeyframes(
    const std::vector<image_t>& candidate_ids,
    const std::unordered_set<image_t>& protected_ids) {
  std::vector<image_t> culled;
  if (!options_.cull_enabled || !initialized_ || !obs_manager_) {
    return culled;
  }

  const size_t min_points =
      static_cast<size_t>(std::max(options_.cull_min_num_points, 0));
  const double short_track_limit =
      std::max(options_.cull_protect_short_track_ratio, 0.0);
  const size_t max_culled =
      static_cast<size_t>(std::max(options_.cull_max_per_frame, 0));

  for (size_t num_removed = 0; num_removed < max_culled; ++num_removed) {
    const FlatHashSet<image_t> recent_ids = RecentRegisteredIds();
    struct Candidate {
      double unique_ratio = 0.0;
      image_t image_id = kInvalidImageId;
    };
    std::vector<Candidate> ranked;
    ranked.reserve(candidate_ids.size());
    for (const image_t id : candidate_ids) {
      if (!reconstruction_->ExistsImage(id) ||
          !reconstruction_->Image(id).HasPose() ||
          protected_ids.count(id) > 0 || reloc_anchor_ids_.count(id) > 0 ||
          (pose_priors_.count(id) == 0 && !options_.cull_prior_session)) {
        continue;
      }
      if (short_track_limit > 0.0 && recent_ids.count(id) > 0 &&
          ShortTrackRatio(id) >= short_track_limit) {
        continue;
      }
      size_t num_points = 0;
      const double redundant_ratio = RedundantPointRatio(id, &num_points);
      if (num_points < min_points ||
          redundant_ratio < options_.cull_redundant_ratio) {
        continue;
      }
      ranked.push_back({1.0 - redundant_ratio, id});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) {
      if (a.unique_ratio != b.unique_ratio) {
        return a.unique_ratio < b.unique_ratio;
      }
      return a.image_id > b.image_id;
    });

    bool removed = false;
    for (const Candidate& candidate : ranked) {
      if (!KeepsTriangulationParallax(candidate.image_id)) {
        LOG(INFO) << "OnlineMapper: keep keyframe " << candidate.image_id
                  << " unique_ratio=" << candidate.unique_ratio
                  << ", remaining parallax would collapse";
        continue;
      }
      if (RemoveKeyframe(candidate.image_id, "redundant", candidate.unique_ratio)) {
        LOG(INFO) << "OnlineMapper: culled redundant keyframe "
                  << candidate.image_id
                  << " unique_ratio=" << candidate.unique_ratio;
        culled.push_back(candidate.image_id);
        removed = true;
        break;
      }
    }
    if (!removed) {
      break;
    }
  }
  return culled;
}

std::vector<image_t> OnlineMapper::EnforceSpatialCapacity(
    const image_t image_id, const std::unordered_set<image_t>& protected_ids) {
  std::vector<image_t> evicted;
  if (!options_.capacity_enabled || !initialized_ || !obs_manager_ ||
      options_.capacity_voxel_m <= 0.0 || options_.capacity_max_per_cell < 1 ||
      !reconstruction_->ExistsImage(image_id) ||
      !reconstruction_->Image(image_id).HasPose()) {
    return evicted;
  }

  const double voxel_m = options_.capacity_voxel_m;
  const double boundary_m = std::max(options_.capacity_boundary_m, 0.0);
  const double cos_view =
      options_.capacity_view_angle_deg > 0.0
          ? std::cos(DegToRad(options_.capacity_view_angle_deg))
          : -1.0;
  const auto voxel_of = [&](const Eigen::Vector3d& center) {
    return std::array<int64_t, 3>{
        static_cast<int64_t>(std::floor(center.x() / voxel_m)),
        static_cast<int64_t>(std::floor(center.y() / voxel_m)),
        static_cast<int64_t>(std::floor(center.z() / voxel_m))};
  };
  const auto axis_of = [](const Image& image) {
    return image.CamFromWorld().rotation().inverse() * Eigen::Vector3d::UnitZ();
  };

  const Image& current = reconstruction_->Image(image_id);
  const Eigen::Vector3d current_center = current.ProjectionCenter();
  const Eigen::Vector3d current_axis = axis_of(current);
  const auto current_voxel = voxel_of(current_center);
  std::vector<std::array<int64_t, 3>> cells;
  cells.push_back(current_voxel);
  for (int axis = 0; axis < 3; ++axis) {
    const double origin = static_cast<double>(current_voxel[axis]) * voxel_m;
    const double local = current_center[axis] - origin;
    if (local < boundary_m) {
      auto neighbor = current_voxel;
      neighbor[axis] -= 1;
      cells.push_back(neighbor);
    }
    if (voxel_m - local < boundary_m) {
      auto neighbor = current_voxel;
      neighbor[axis] += 1;
      cells.push_back(neighbor);
    }
  }

  std::vector<image_t> members;
  for (const image_t id : reconstruction_->RegImageIds()) {
    const Image& image = reconstruction_->Image(id);
    if (!image.HasPose() ||
        axis_of(image).dot(current_axis) <= cos_view) {
      continue;
    }
    const auto voxel = voxel_of(image.ProjectionCenter());
    if (std::find(cells.begin(), cells.end(), voxel) != cells.end()) {
      members.push_back(id);
    }
  }
  const size_t max_per_cell =
      static_cast<size_t>(options_.capacity_max_per_cell);
  if (members.size() <= max_per_cell) {
    return evicted;
  }

  const auto score_of = [&](const image_t id) {
    return ClosestObserverRatio(id);
  };
  const double new_score = score_of(image_id);

  const FlatHashSet<image_t> recent_ids = RecentRegisteredIds();
  const bool allow_prior_session =
      session_mapped_frames_ >=
      static_cast<size_t>(std::max(options_.capacity_min_session_frames, 0));
  std::vector<std::pair<double, image_t>> scored;
  for (const image_t id : members) {
    if (id == image_id ||
        !IsRemovable(
            id, image_id, recent_ids, protected_ids, allow_prior_session)) {
      continue;
    }
    scored.emplace_back(score_of(id), id);
  }
  std::sort(scored.begin(), scored.end());

  // Existing keyframes are replaced only when clearly worse than the new one.
  const size_t excess = members.size() - max_per_cell;
  const double replace_below = options_.capacity_replace_ratio * new_score;
  for (const auto& [score, id] : scored) {
    if (evicted.size() >= excess || score >= replace_below) {
      break;
    }
    if (RemoveKeyframe(id, "capacity", score)) {
      LOG(INFO) << "OnlineMapper: evicted keyframe " << id << " from cell of "
                << image_id << " members=" << members.size()
                << " score=" << score << " new_score=" << new_score;
      evicted.push_back(id);
    }
  }
  if (evicted.size() >= excess) {
    consecutive_capacity_drops_ = 0;
    return evicted;
  }

  // Otherwise drop the new keyframe, as a non-keyframe. Bounded so a revisit
  // still refreshes the map now and then.
  if (protected_ids.count(image_id) > 0 ||
      reloc_anchor_ids_.count(image_id) > 0 ||
      consecutive_capacity_drops_ >=
          std::max(options_.insert_max_consecutive_skips, 0)) {
    consecutive_capacity_drops_ = 0;
    return evicted;
  }
  const auto prior_it = pose_priors_.find(image_id);
  if (prior_it != pose_priors_.end()) {
    tracking_anchor_ = TrackingAnchor{
        image_id, prior_it->second, reconstruction_->Image(image_id).CamFromWorld()};
  }
  if (RemoveKeyframe(image_id, "capacity_new", new_score)) {
    ++consecutive_capacity_drops_;
    LOG(INFO) << "OnlineMapper: dropped new keyframe " << image_id
              << ", cell already holds " << members.size() - 1
              << " keyframe(s) of score >= " << replace_below;
    evicted.push_back(image_id);
  }
  return evicted;
}

std::vector<image_t> OnlineMapper::EnforceSpatialCapacity(
    const std::vector<image_t>& image_ids,
    const std::unordered_set<image_t>& protected_ids) {
  std::vector<image_t> evicted;
  if (!options_.capacity_enabled || !initialized_ || !obs_manager_ ||
      options_.capacity_voxel_m <= 0.0 || options_.capacity_max_per_cell < 1 ||
      image_ids.empty()) {
    return evicted;
  }

  const double voxel_m = options_.capacity_voxel_m;
  const double boundary_m = std::max(options_.capacity_boundary_m, 0.0);
  const double cos_view =
      options_.capacity_view_angle_deg > 0.0
          ? std::cos(DegToRad(options_.capacity_view_angle_deg))
          : -1.0;
  const size_t max_per_cell =
      static_cast<size_t>(options_.capacity_max_per_cell);
  const auto voxel_of = [&](const Eigen::Vector3d& center) {
    return std::array<int64_t, 3>{
        static_cast<int64_t>(std::floor(center.x() / voxel_m)),
        static_cast<int64_t>(std::floor(center.y() / voxel_m)),
        static_cast<int64_t>(std::floor(center.z() / voxel_m))};
  };
  const auto axis_of = [](const Image& image) {
    return image.CamFromWorld().rotation().inverse() * Eigen::Vector3d::UnitZ();
  };
  const auto cells_of = [&](const Eigen::Vector3d& center,
                            const std::array<int64_t, 3>& voxel) {
    std::vector<std::array<int64_t, 3>> cells;
    cells.push_back(voxel);
    for (int axis = 0; axis < 3; ++axis) {
      const double local =
          center[axis] - static_cast<double>(voxel[axis]) * voxel_m;
      if (local < boundary_m) {
        auto neighbor = voxel;
        neighbor[axis] -= 1;
        cells.push_back(neighbor);
      }
      if (voxel_m - local < boundary_m) {
        auto neighbor = voxel;
        neighbor[axis] += 1;
        cells.push_back(neighbor);
      }
    }
    std::sort(cells.begin(), cells.end());
    return cells;
  };

  struct VoxelKey {
    std::array<int64_t, 3> voxel;
    bool operator==(const VoxelKey& other) const { return voxel == other.voxel; }
  };
  struct VoxelKeyHash {
    size_t operator()(const VoxelKey& key) const {
      return (static_cast<size_t>(key.voxel[0]) * 73856093u) ^
             (static_cast<size_t>(key.voxel[1]) * 19349663u) ^
             (static_cast<size_t>(key.voxel[2]) * 83492791u);
    }
  };
  struct PoseRecord {
    image_t id = kInvalidImageId;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    std::array<int64_t, 3> voxel{};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  std::vector<PoseRecord, Eigen::aligned_allocator<PoseRecord>> records;
  std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> buckets;
  records.reserve(reconstruction_->NumRegFrames());
  for (const image_t id : reconstruction_->RegImageIds()) {
    const Image& image = reconstruction_->Image(id);
    if (!image.HasPose()) {
      continue;
    }
    PoseRecord record;
    record.id = id;
    record.center = image.ProjectionCenter();
    record.axis = axis_of(image);
    record.voxel = voxel_of(record.center);
    buckets[VoxelKey{record.voxel}].push_back(records.size());
    records.push_back(record);
  }

  std::unordered_map<image_t, size_t> record_of;
  record_of.reserve(records.size());
  for (size_t i = 0; i < records.size(); ++i) {
    record_of.emplace(records[i].id, i);
  }

  struct Query {
    size_t record_index = 0;
    std::vector<std::array<int64_t, 3>> cells;
  };
  std::vector<Query> queries;
  FlatHashSet<image_t> queued;
  for (const image_t id : image_ids) {
    const auto it = record_of.find(id);
    if (it == record_of.end() || !queued.insert(id).second) {
      continue;
    }
    const PoseRecord& record = records[it->second];
    queries.push_back({it->second, cells_of(record.center, record.voxel)});
  }
  std::sort(queries.begin(), queries.end(), [](const Query& a, const Query& b) {
    return a.record_index < b.record_index;
  });

  std::vector<char> consumed(queries.size(), 0);
  const FlatHashSet<image_t> recent_ids = RecentRegisteredIds();
  const bool allow_prior_session =
      session_mapped_frames_ >=
      static_cast<size_t>(std::max(options_.capacity_min_session_frames, 0));
  FlatHashSet<image_t> gone;

  for (size_t i = 0; i < queries.size(); ++i) {
    if (consumed[i] || gone.count(records[queries[i].record_index].id) > 0) {
      continue;
    }
    const PoseRecord& seed = records[queries[i].record_index];
    const auto& cells = queries[i].cells;
    consumed[i] = 1;
    for (size_t j = i + 1; j < queries.size(); ++j) {
      if (consumed[j]) {
        continue;
      }
      const PoseRecord& other = records[queries[j].record_index];
      if (other.axis.dot(seed.axis) > cos_view && queries[j].cells == cells) {
        consumed[j] = 1;
      }
    }

    FlatHashSet<image_t> seen;
    std::vector<image_t> members;
    for (const auto& cell : cells) {
      const auto bucket = buckets.find(VoxelKey{cell});
      if (bucket == buckets.end()) {
        continue;
      }
      for (const size_t index : bucket->second) {
        const PoseRecord& record = records[index];
        if (gone.count(record.id) > 0 || record.axis.dot(seed.axis) <= cos_view ||
            !seen.insert(record.id).second) {
          continue;
        }
        members.push_back(record.id);
      }
    }
    if (members.size() <= max_per_cell) {
      continue;
    }

    std::vector<std::pair<double, image_t>> removable;
    size_t num_alive = 0;
    for (const image_t id : members) {
      if (gone.count(id) > 0 || !reconstruction_->ExistsImage(id) ||
          !reconstruction_->Image(id).HasPose()) {
        continue;
      }
      ++num_alive;
      if (!IsRemovable(id,
                       kInvalidImageId,
                       recent_ids,
                       protected_ids,
                       allow_prior_session)) {
        continue;
      }
      removable.emplace_back(ClosestObserverRatio(id), id);
    }
    if (num_alive <= max_per_cell) {
      continue;
    }
    std::sort(removable.begin(), removable.end());
    size_t excess = num_alive - max_per_cell;
    for (const auto& [score, id] : removable) {
      if (excess == 0) {
        break;
      }
      if (RemoveKeyframe(id, "capacity_loop", score)) {
        gone.insert(id);
        evicted.push_back(id);
        --excess;
        LOG(INFO) << "OnlineMapper: evicted keyframe " << id
                  << " after loop correction, score=" << score
                  << " members=" << num_alive;
      }
    }
  }
  return evicted;
}

std::vector<image_t> OnlineMapper::RemoveStaleKeyframes(
    const image_t current_id,
    const std::unordered_set<image_t>& protected_ids) {
  std::vector<image_t> removed;
  if (!options_.point_stats_enabled || !initialized_ || !obs_manager_ ||
      stale_lost_points_.empty()) {
    return removed;
  }
  const FlatHashSet<image_t> recent_ids = RecentRegisteredIds();
  const bool allow_prior_session =
      session_mapped_frames_ >=
      static_cast<size_t>(std::max(options_.capacity_min_session_frames, 0));
  const size_t min_points =
      static_cast<size_t>(std::max(options_.stale_frame_min_points, 0));
  std::vector<std::pair<double, image_t>> candidates;
  for (auto it = stale_lost_points_.begin(); it != stale_lost_points_.end();) {
    const image_t id = it->first;
    if (!reconstruction_->ExistsImage(id) ||
        !reconstruction_->Image(id).HasPose()) {
      it = stale_lost_points_.erase(it);
      continue;
    }
    const size_t remaining = reconstruction_->Image(id).NumPoints3D();
    const double lost_ratio =
        static_cast<double>(it->second) /
        static_cast<double>(it->second + remaining);
    // A weak frame that lost only a point or two is not judged by its size.
    const bool emptied = remaining < min_points && it->second >= remaining;
    if ((emptied || lost_ratio >= options_.stale_frame_max_lost_ratio) &&
        IsRemovable(
            id, current_id, recent_ids, protected_ids, allow_prior_session)) {
      candidates.emplace_back(lost_ratio, id);
    }
    ++it;
  }
  std::sort(candidates.begin(), candidates.end(), std::greater<>());
  for (const auto& [lost_ratio, id] : candidates) {
    const size_t lost = stale_lost_points_[id];
    if (RemoveKeyframe(id, "stale", lost_ratio)) {
      LOG(INFO) << "OnlineMapper: removed stale keyframe " << id
                << " lost=" << lost << " lost_ratio=" << lost_ratio;
      stale_lost_points_.erase(id);
      removed.push_back(id);
    }
  }
  return removed;
}

double OnlineMapper::ClosestObserverRatio(const image_t image_id) const {
  const Image& image = reconstruction_->Image(image_id);
  if (!image.HasPose()) {
    return 0.0;
  }
  const Eigen::Vector3d center = image.ProjectionCenter();
  std::unordered_map<image_t, Eigen::Vector3d> centers;
  const auto center_of = [&](const image_t id) -> const Eigen::Vector3d& {
    auto it = centers.find(id);
    if (it == centers.end()) {
      it = centers.emplace(id, reconstruction_->Image(id).ProjectionCenter())
               .first;
    }
    return it->second;
  };

  size_t num_points = 0;
  size_t num_closest = 0;
  FlatHashSet<image_t> counted;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    ++num_points;
    // Points the scene no longer shows keep the denominator but earn nothing.
    if (PointLooksGone(point2D.point3D_id)) {
      continue;
    }
    const Point3D& point3D = reconstruction_->Point3D(point2D.point3D_id);
    const double own_dist_sq = (point3D.xyz - center).squaredNorm();
    counted.clear();
    bool closest = true;
    for (const TrackElement& el : point3D.track.Elements()) {
      if (el.image_id == image_id || !counted.insert(el.image_id).second) {
        continue;
      }
      const Image& other = reconstruction_->Image(el.image_id);
      if (!other.HasPose()) {
        continue;
      }
      if ((point3D.xyz - center_of(el.image_id)).squaredNorm() < own_dist_sq) {
        closest = false;
        break;
      }
    }
    if (closest) {
      ++num_closest;
    }
  }
  return num_points > 0
             ? static_cast<double>(num_closest) / static_cast<double>(num_points)
             : 0.0;
}

double OnlineMapper::PointQuality(const point3D_t point3D_id) const {
  if (!options_.point_stats_enabled) {
    return 1.0;
  }
  const auto it = point_stats_.find(point3D_id);
  if (it == point_stats_.end()) {
    return 0.5;
  }
  return (it->second.num_found + 1.0) / (it->second.num_visible + 2.0);
}

bool OnlineMapper::PointLooksGone(const point3D_t point3D_id) const {
  if (!options_.point_stats_enabled) {
    return false;
  }
  const auto it = point_stats_.find(point3D_id);
  return it != point_stats_.end() && it->second.consecutive_misses >= 2;
}

void OnlineMapper::EnsureDescriptors(const FlatHashSet<image_t>& image_ids) {
  if (!descriptor_loader_) {
    return;
  }
  for (const image_t id : image_ids) {
    if (descriptors_.count(id) > 0) {
      continue;
    }
    // A failed load is remembered as null so it is not retried every frame.
    descriptors_[id] = descriptor_loader_(id);
    if (options_.scale_check_max_ratio > 1.0 && keypoint_loader_ &&
        keypoint_scales_.count(id) == 0) {
      const std::shared_ptr<const FeatureKeypoints> keypoints =
          keypoint_loader_(id);
      if (keypoints) {
        CacheKeypointScales(id, *keypoints);
      }
    }
  }
}

void OnlineMapper::UpdatePointStats(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  if (!image.HasPose()) {
    return;
  }

  FlatHashSet<point2D_t> skip_points2D;
  FlatHashSet<point3D_t> observed;
  FlatHashSet<image_t> covis_ids;
  for (point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
    const Point2D& point2D = image.Point2D(idx);
    if (!point2D.HasPoint3D()) {
      continue;
    }
    skip_points2D.insert(idx);
    observed.insert(point2D.point3D_id);
    for (const TrackElement& el :
         reconstruction_->Point3D(point2D.point3D_id).track.Elements()) {
      if (el.image_id != image_id) {
        covis_ids.insert(el.image_id);
      }
    }
  }
  const Eigen::Vector3d center = image.ProjectionCenter();
  // Where the scene changed the current frame shares few points with the old
  // keyframes, so their points are reached by position, not covisibility.
  if (options_.stale_spatial_num_images > 0 &&
      options_.stale_spatial_max_distance_m > 0.0) {
    const Eigen::Vector3d axis =
        image.CamFromWorld().rotation().inverse() * Eigen::Vector3d::UnitZ();
    const double max_dist_sq = options_.stale_spatial_max_distance_m *
                               options_.stale_spatial_max_distance_m;
    const double min_cos =
        std::cos(DegToRad(options_.stale_spatial_max_angle_deg));
    std::vector<std::pair<double, image_t>> nearby;
    for (const image_t id : reconstruction_->RegImageIds()) {
      if (id == image_id || covis_ids.count(id) > 0) {
        continue;
      }
      const Image& other = reconstruction_->Image(id);
      if (!other.HasPose()) {
        continue;
      }
      const double dist_sq = (other.ProjectionCenter() - center).squaredNorm();
      if (dist_sq > max_dist_sq ||
          (other.CamFromWorld().rotation().inverse() * Eigen::Vector3d::UnitZ())
                  .dot(axis) < min_cos) {
        continue;
      }
      nearby.emplace_back(dist_sq, id);
    }
    std::sort(nearby.begin(), nearby.end());
    const size_t num = std::min(
        nearby.size(), static_cast<size_t>(options_.stale_spatial_num_images));
    for (size_t i = 0; i < num; ++i) {
      covis_ids.insert(nearby[i].second);
    }
  }
  EnsureDescriptors(covis_ids);

  FlatHashSet<point3D_t> candidates;
  for (const image_t id : covis_ids) {
    for (const Point2D& point2D : reconstruction_->Image(id).Points2D()) {
      if (point2D.HasPoint3D() && observed.count(point2D.point3D_id) == 0) {
        candidates.insert(point2D.point3D_id);
      }
    }
  }
  std::vector<point3D_t> visible;
  std::vector<point3D_t> bare;
  const std::vector<ProjectionMatch> matches =
      SearchPointsByProjection(image_id,
                               image.CamFromWorld(),
                               options_.local_map_refine_radius_px,
                               skip_points2D,
                               candidates,
                               &visible,
                               &bare);
  FlatHashSet<point3D_t> matched;
  for (const ProjectionMatch& match : matches) {
    matched.insert(match.point3D_id);
  }

  for (const point3D_t id : observed) {
    PointStat& stat = point_stats_[id];
    stat.num_visible += 1.0f;
    stat.num_found += 1.0f;
    stat.last_seen_session = session_index_;
    stat.consecutive_misses = 0;
    stat.consecutive_unseen = 0;
  }

  const auto judged = [&](const Point3D& point3D) {
    return !options_.stale_prior_session_only ||
           std::none_of(point3D.track.Elements().begin(),
                        point3D.track.Elements().end(),
                        [&](const TrackElement& el) {
                          return pose_priors_.count(el.image_id) > 0;
                        });
  };
  const bool unseen_enabled = options_.stale_max_consecutive_unseen > 0;
  const double min_found_ratio = options_.stale_max_found_ratio;
  const float min_visible =
      static_cast<float>(std::max(options_.stale_min_visible, 1));
  size_t num_checked = 0;
  size_t num_matched = 0;
  size_t num_bare = 0;
  std::vector<point3D_t> stale;
  for (const point3D_t id : visible) {
    const Point3D& point3D = reconstruction_->Point3D(id);
    ++num_checked;
    PointStat& stat = point_stats_[id];
    stat.num_visible += 1.0f;
    if (matched.count(id) > 0) {
      ++num_matched;
      stat.num_found += 1.0f;
      stat.last_seen_session = session_index_;
      stat.consecutive_misses = 0;
      stat.consecutive_unseen = 0;
      continue;
    }
    ++stat.consecutive_misses;
    ++stat.consecutive_unseen;
    if (!judged(point3D)) {
      continue;
    }
    const bool missed_in_a_row =
        options_.stale_max_consecutive_misses > 0 &&
        stat.consecutive_misses >= options_.stale_max_consecutive_misses;
    const bool unseen_in_a_row =
        unseen_enabled &&
        stat.consecutive_unseen >= options_.stale_max_consecutive_unseen;
    if (missed_in_a_row || unseen_in_a_row ||
        (stat.num_visible >= min_visible &&
         stat.num_found < min_found_ratio * stat.num_visible)) {
      stale.push_back(id);
    }
  }
  // An object taken away often leaves a textureless spot, so a projection
  // with no keypoint nearby is weak evidence: it only feeds the longer streak.
  if (unseen_enabled) {
    for (const point3D_t id : bare) {
      const Point3D& point3D = reconstruction_->Point3D(id);
      ++num_bare;
      PointStat& stat = point_stats_[id];
      ++stat.consecutive_unseen;
      if (judged(point3D) &&
          stat.consecutive_unseen >= options_.stale_max_consecutive_unseen) {
        stale.push_back(id);
      }
    }
  }

  for (const point3D_t id : stale) {
    if (reconstruction_->ExistsPoint3D(id)) {
      for (const TrackElement& el :
           reconstruction_->Point3D(id).track.Elements()) {
        ++stale_lost_points_[el.image_id];
      }
      obs_manager_->DeletePoint3D(id);
    }
    point_stats_.erase(id);
  }
  LOG(INFO) << "OnlineMapper: point stats image " << image_id
            << " observed=" << observed.size() << " checked=" << num_checked
            << " matched=" << num_matched << " bare=" << num_bare
            << " stale_removed=" << stale.size()
            << " points3D=" << reconstruction_->NumPoints3D();
}

std::string OnlineMapper::PointStatsPath() const {
  return (std::filesystem::path(sparse_path_) / "point_stats.bin").string();
}

namespace {
constexpr uint32_t kPointStatsMagic = 0x50535431;  // "PST1"
}  // namespace

void OnlineMapper::WritePointStats() const {
  if (!options_.point_stats_enabled || sparse_path_.empty()) {
    return;
  }
  try {
    std::ofstream file(PointStatsPath(), std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return;
    }
    uint64_t count = 0;
    for (const auto& [id, stat] : point_stats_) {
      (void)stat;
      count += reconstruction_->ExistsPoint3D(id) ? 1 : 0;
    }
    WriteBinaryLittleEndian<uint32_t>(&file, kPointStatsMagic);
    WriteBinaryLittleEndian<uint32_t>(&file, session_index_);
    WriteBinaryLittleEndian<uint64_t>(&file, count);
    for (const auto& [id, stat] : point_stats_) {
      if (!reconstruction_->ExistsPoint3D(id)) {
        continue;
      }
      WriteBinaryLittleEndian<uint64_t>(&file, id);
      WriteBinaryLittleEndian<float>(&file, stat.num_visible);
      WriteBinaryLittleEndian<float>(&file, stat.num_found);
      WriteBinaryLittleEndian<uint32_t>(&file, stat.last_seen_session);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write point stats: " << e.what();
  }
}

void OnlineMapper::ReadPointStats() {
  point_stats_.clear();
  if (!options_.point_stats_enabled ||
      !std::filesystem::exists(PointStatsPath())) {
    return;
  }
  try {
    std::ifstream file(PointStatsPath(), std::ios::binary);
    if (ReadBinaryLittleEndian<uint32_t>(&file) != kPointStatsMagic) {
      LOG(WARNING) << "OnlineMapper: ignoring point stats with unknown format";
      return;
    }
    const uint32_t stored_session = ReadBinaryLittleEndian<uint32_t>(&file);
    const uint64_t count = ReadBinaryLittleEndian<uint64_t>(&file);
    const float decay = static_cast<float>(
        std::clamp(options_.stale_session_decay, 0.0, 1.0));
    for (uint64_t i = 0; i < count && file.good(); ++i) {
      const point3D_t id = ReadBinaryLittleEndian<uint64_t>(&file);
      PointStat stat;
      stat.num_visible = ReadBinaryLittleEndian<float>(&file) * decay;
      stat.num_found = ReadBinaryLittleEndian<float>(&file) * decay;
      stat.last_seen_session = ReadBinaryLittleEndian<uint32_t>(&file);
      if (reconstruction_->ExistsPoint3D(id)) {
        point_stats_.emplace(id, stat);
      }
    }
    session_index_ = std::max(session_index_, stored_session + 1);
    LOG(INFO) << "OnlineMapper: loaded point stats for " << point_stats_.size()
              << " point(s), session " << session_index_;
  } catch (const std::exception& e) {
    point_stats_.clear();
    LOG(WARNING) << "OnlineMapper: failed to read point stats: " << e.what();
  }
}

namespace {
constexpr uint32_t kVioPriorsMagic = 0x56505231;  // "VPR1"
}  // namespace

const Rigid3d* OnlineMapper::FindVioPrior(const image_t image_id,
                                          uint32_t* session) const {
  const auto current_it = pose_priors_.find(image_id);
  if (current_it != pose_priors_.end()) {
    if (session != nullptr) {
      *session = session_index_;
    }
    return &current_it->second;
  }
  const auto stored_it = stored_priors_.find(image_id);
  if (stored_it != stored_priors_.end()) {
    if (session != nullptr) {
      *session = stored_it->second.session;
    }
    return &stored_it->second.vio_cam_from_world;
  }
  return nullptr;
}

std::optional<Rigid3d> OnlineMapper::VioRelativePose(
    const image_t tgt_id, const image_t src_id) const {
  uint32_t tgt_session = 0;
  uint32_t src_session = 0;
  const Rigid3d* tgt = FindVioPrior(tgt_id, &tgt_session);
  const Rigid3d* src = FindVioPrior(src_id, &src_session);
  if (tgt == nullptr || src == nullptr || tgt_session != src_session) {
    return std::nullopt;
  }
  return *tgt * Inverse(*src);
}

std::string OnlineMapper::VioPriorsPath() const {
  return (std::filesystem::path(sparse_path_) / "vio_priors.bin").string();
}

void OnlineMapper::WriteVioPriors() const {
  if (sparse_path_.empty()) {
    return;
  }
  try {
    std::vector<std::pair<image_t, uint32_t>> entries;
    for (const image_t id : reconstruction_->RegImageIds()) {
      uint32_t session = 0;
      if (FindVioPrior(id, &session) != nullptr) {
        entries.emplace_back(id, session);
      }
    }
    std::ofstream file(VioPriorsPath(), std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      return;
    }
    WriteBinaryLittleEndian<uint32_t>(&file, kVioPriorsMagic);
    WriteBinaryLittleEndian<uint32_t>(&file, session_index_);
    WriteBinaryLittleEndian<uint64_t>(&file, entries.size());
    for (const auto& [id, session] : entries) {
      WriteBinaryLittleEndian<uint32_t>(&file, id);
      WriteBinaryLittleEndian<uint32_t>(&file, session);
      const Rigid3d& prior = *FindVioPrior(id, nullptr);
      for (int k = 0; k < 7; ++k) {
        WriteBinaryLittleEndian<double>(&file, prior.params[k]);
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write VIO priors: " << e.what();
  }
}

void OnlineMapper::ReadVioPriors() {
  stored_priors_.clear();
  if (!std::filesystem::exists(VioPriorsPath())) {
    return;
  }
  try {
    std::ifstream file(VioPriorsPath(), std::ios::binary);
    if (ReadBinaryLittleEndian<uint32_t>(&file) != kVioPriorsMagic) {
      LOG(WARNING) << "OnlineMapper: ignoring VIO priors with unknown format";
      return;
    }
    const uint32_t stored_session = ReadBinaryLittleEndian<uint32_t>(&file);
    const uint64_t count = ReadBinaryLittleEndian<uint64_t>(&file);
    std::unordered_map<uint32_t, std::vector<image_t>> session_ids;
    for (uint64_t i = 0; i < count && file.good(); ++i) {
      const image_t id = ReadBinaryLittleEndian<uint32_t>(&file);
      StoredPrior prior;
      prior.session = ReadBinaryLittleEndian<uint32_t>(&file);
      for (int k = 0; k < 7; ++k) {
        prior.vio_cam_from_world.params[k] =
            ReadBinaryLittleEndian<double>(&file);
      }
      if (!file.good() || !reconstruction_->ExistsImage(id) ||
          !reconstruction_->Image(id).HasPose()) {
        continue;
      }
      session_ids[prior.session].push_back(id);
      stored_priors_.emplace(id, prior);
    }
    for (auto& [session, ids] : session_ids) {
      std::sort(ids.begin(), ids.end());
      for (size_t rank = 0; rank < ids.size(); ++rank) {
        stored_priors_.at(ids[rank]).rank = rank;
      }
    }
    session_index_ = std::max(session_index_, stored_session + 1);
    LOG(INFO) << "OnlineMapper: loaded VIO priors of " << stored_priors_.size()
              << " keyframe(s) from " << session_ids.size()
              << " session(s), session " << session_index_;
  } catch (const std::exception& e) {
    stored_priors_.clear();
    LOG(WARNING) << "OnlineMapper: failed to read VIO priors: " << e.what();
  }
}

void OnlineMapper::WriteSparse() const {
  if (sparse_path_.empty() || reconstruction_->NumRegFrames() == 0) {
    return;
  }
  try {
    const std::filesystem::path out_dir(sparse_path_);
    std::filesystem::create_directories(out_dir);
    reconstruction_->Write(out_dir);
    WritePointStats();
    WriteVioPriors();
    sparse_writes_since_flush_ = 0;
    LOG(INFO) << "OnlineMapper: wrote sparse to " << out_dir.string()
              << " frames=" << reconstruction_->NumRegFrames()
              << " points3D=" << reconstruction_->NumPoints3D()
              << " every_n=" << options_.sparse_write_every_n;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write sparse: " << e.what();
  }
}

void OnlineMapper::MaybeWriteSparse() {
  const int every_n = std::max(options_.sparse_write_every_n, 1);
  ++sparse_writes_since_flush_;
  if (sparse_writes_since_flush_ < every_n) {
    LOG(INFO) << "OnlineMapper: defer sparse write "
              << sparse_writes_since_flush_ << "/" << every_n;
    return;
  }
  WriteSparse();
}

bool OnlineMapper::MappingCurrentImage(const image_t image_id,
                                       const std::optional<Rigid3d>& vio_prior) {
  last_frame_skipped_ = false;
  if (!SyncReconstruction()) {
    return false;
  }
  if (!reconstruction_->ExistsImage(image_id)) {
    LOG(ERROR) << "OnlineMapper: image " << image_id
               << " not in reconstruction after sync";
    return false;
  }
  if (vio_prior.has_value()) {
    pose_priors_[image_id] = *vio_prior;
  }

  if (reconstruction_->Image(image_id).HasPose()) {
    MaybeWriteSparse();
    return true;
  }

  if (!initialized_) {
    return TryInitialize(image_id);
  }

  const PriorRegisterResult prior_reg = RegisterNextImageWithPrior(image_id);
  bool registered = prior_reg.success;
  if (!registered && vio_prior.has_value()) {
    LOG(INFO) << "OnlineMapper: prior-init failed, fallback to map-frame VIO "
                 "for image "
              << image_id;
    registered = RegisterWithRelativeVio(image_id, *vio_prior);
  }
  if (!registered) {
    LOG(WARNING) << "OnlineMapper: skip mapping image " << image_id
                 << ", prior-init failed and no VIO prior";
    return true;
  }

  if (prior_reg.success) {
    ExtendTracksByProjection(image_id, "register");
  }
  // VIO-fallback frames vote too: PnP fails mostly where the scene changed.
  // Frames about to be skipped still vote: a revisit is mostly skipped frames.
  if (options_.point_stats_enabled) {
    UpdatePointStats(image_id);
  }
  if (ShouldSkipKeyframe(image_id, prior_reg.success)) {
    SkipFrame(image_id);
    return true;
  }
  consecutive_skips_ = 0;
  ++session_mapped_frames_;
  TriangulateImage(image_id);
  MergeAndFilterPoints(image_id);
  AdjustLocalBundle(image_id);
  MaybeWriteSparse();
  return true;
}

bool OnlineMapper::ShouldSkipKeyframe(const image_t image_id,
                                      const bool pose_from_pnp) const {
  if (!options_.insert_check_enabled ||
      consecutive_skips_ >= std::max(options_.insert_max_consecutive_skips, 0)) {
    return false;
  }
  const Image& image = reconstruction_->Image(image_id);
  const size_t num_tracked = image.NumPoints3D();
  if (num_tracked <
      static_cast<size_t>(std::max(options_.insert_min_tracked, 1))) {
    return false;
  }

  // A VIO-prior pose can leave features unassociated without any new structure,
  // so those two keep rules apply only to a PnP pose.
  const size_t num_features = image.NumPoints2D();
  if (pose_from_pnp && num_features > 0 &&
      options_.insert_min_unmatched_ratio > 0.0) {
    const size_t num_unmatched =
        num_features > num_tracked ? num_features - num_tracked : 0;
    const double unmatched_ratio = static_cast<double>(num_unmatched) /
                                   static_cast<double>(num_features);
    if (unmatched_ratio >= options_.insert_min_unmatched_ratio) {
      LOG(INFO) << "OnlineMapper: insert check keep image " << image_id
                << " unmatched_ratio=" << unmatched_ratio
                << " tracked=" << num_tracked << "/" << num_features;
      return false;
    }
  }
  if (pose_from_pnp && options_.insert_min_short_track_ratio > 0.0) {
    size_t num_short = 0;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
        continue;
      }
      if (reconstruction_->Point3D(point2D.point3D_id).track.Length() <= 2) {
        ++num_short;
      }
    }
    const double short_ratio = static_cast<double>(num_short) /
                               static_cast<double>(num_tracked);
    if (short_ratio >= options_.insert_min_short_track_ratio) {
      LOG(INFO) << "OnlineMapper: insert check keep image " << image_id
                << " short_track_ratio=" << short_ratio
                << " short=" << num_short << " tracked=" << num_tracked;
      return false;
    }
  }

  FlatHashMap<image_t, size_t> shared_points;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }
    for (const TrackElement& el :
         reconstruction_->Point3D(point2D.point3D_id).track.Elements()) {
      if (el.image_id != image_id) {
        ++shared_points[el.image_id];
      }
    }
  }
  // Reference keyframe: the nearby keyframe sharing the most points.
  const Eigen::Vector3d center = image.ProjectionCenter();
  const Eigen::Quaterniond rotation = image.CamFromWorld().rotation();
  const double max_angle = DegToRad(options_.insert_max_kf_angle_deg);
  image_t ref_id = kInvalidImageId;
  size_t ref_shared = 0;
  for (const auto& [other_id, count] : shared_points) {
    if (2 * count < num_tracked || count <= ref_shared) {
      continue;
    }
    const Image& other = reconstruction_->Image(other_id);
    if (!other.HasPose() ||
        (other.ProjectionCenter() - center).norm() >
            options_.insert_max_kf_distance_m ||
        Eigen::AngleAxisd(other.CamFromWorld().rotation() * rotation.inverse())
                .angle() > max_angle) {
      continue;
    }
    ref_id = other_id;
    ref_shared = count;
  }
  if (ref_id == kInvalidImageId) {
    return false;
  }
  const size_t ref_tracked = reconstruction_->Image(ref_id).NumPoints3D();
  if (static_cast<double>(num_tracked) <
      options_.insert_ref_tracked_ratio * static_cast<double>(ref_tracked)) {
    return false;
  }

  size_t num_points = 0;
  const double ratio = RedundantPointRatio(image_id, &num_points);
  LOG(INFO) << "OnlineMapper: insert check image " << image_id
            << " tracked=" << num_tracked << " ref_kf=" << ref_id
            << " ref_tracked=" << ref_tracked << " redundant_ratio=" << ratio;
  return ratio >= options_.insert_redundant_ratio;
}

void OnlineMapper::SkipFrame(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  const auto prior_it = pose_priors_.find(image_id);
  if (prior_it != pose_priors_.end()) {
    tracking_anchor_ =
        TrackingAnchor{image_id, prior_it->second, image.CamFromWorld()};
  }
  const size_t num_tracked = image.NumPoints3D();
  RemoveKeyframe(image_id, "skipped", static_cast<double>(num_tracked));
  ++consecutive_skips_;
  last_frame_skipped_ = true;
  LOG(INFO) << "OnlineMapper: dropped non-keyframe " << image_id
            << " tracked=" << num_tracked
            << " consecutive=" << consecutive_skips_;
}

}  // namespace colmap
