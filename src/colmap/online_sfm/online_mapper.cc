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
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/pose.h"
#include "colmap/estimators/triangulation.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/track.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

namespace colmap {
namespace {

BundleAdjustmentOptions MakeLocalBaOptions(const int min_track_length) {
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
    options.ceres->solver_options.max_num_iterations = 25;
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
      reconstruction_(std::make_shared<Reconstruction>()) {}

OnlineMapper::~OnlineMapper() = default;

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
  obs_manager_->AddImagePair(image_id1, image_id2);
}

void OnlineMapper::CacheDescriptors(
    const image_t image_id,
    std::shared_ptr<const FeatureDescriptors> descriptors) {
  if (descriptors) {
    descriptors_[image_id] = std::move(descriptors);
  }
}

std::vector<OnlineMapper::ProjectionMatch>
OnlineMapper::SearchLocalMapByProjection(
    const image_t image_id,
    const Rigid3d& cam_from_world,
    const double radius_px,
    const FlatHashSet<point2D_t>& skip_points2D,
    const FlatHashSet<point3D_t>& skip_points3D) const {
  std::vector<ProjectionMatch> matches;
  if (options_.local_map_num_images <= 0 || radius_px <= 0.0 ||
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
  FlatHashSet<point3D_t> local_points;
  for (const image_t id : local_ids) {
    for (const Point2D& point2D : reconstruction_->Image(id).Points2D()) {
      if (point2D.HasPoint3D() && skip_points3D.count(point2D.point3D_id) == 0) {
        local_points.insert(point2D.point3D_id);
      }
    }
  }
  if (local_points.empty()) {
    return matches;
  }

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

  // Keypoint -> (point3D, descriptor distance); the closest point wins.
  FlatHashMap<point2D_t, std::pair<point3D_t, int>> claims;
  std::vector<std::pair<const FeatureDescriptors*, point2D_t>> refs;
  std::vector<TrackElement> track_elements;
  for (const point3D_t point3D_id : local_points) {
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
    for (const TrackElement& el : track_elements) {
      const Image& track_image = reconstruction_->Image(el.image_id);
      if (!track_image.HasPose()) {
        continue;
      }
      mean_dir += (point3D.xyz - track_image.ProjectionCenter()).normalized();
      if (refs.size() >= kMaxRefDescriptors) {
        continue;
      }
      const auto ref_it = descriptors_.find(el.image_id);
      if (ref_it != descriptors_.end() &&
          DescriptorsMatchImage(ref_it->second.get(), track_image)) {
        refs.emplace_back(ref_it->second.get(), el.point2D_idx);
      }
    }
    if (refs.empty() || mean_dir.squaredNorm() == 0.0 ||
        mean_dir.normalized().dot((point3D.xyz - center).normalized()) <
            min_cos_view) {
      continue;
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
    for (int gy = gy0; gy <= gy1; ++gy) {
      for (int gx = gx0; gx <= gx1; ++gx) {
        for (const point2D_t idx : grid[gy * grid_cols + gx]) {
          if ((image.Point2D(idx).xy - *proj).squaredNorm() > radius_sq) {
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
            << " candidates=" << matches.size() << " added=" << num_added;
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

bool OnlineMapper::TryInitialize(const image_t image_id) {
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
  const size_t num_filtered = obs_manager_->FilterPoints3DInImages(
      options_.filter_max_reproj_error,
      options_.filter_min_tri_angle,
      filter_ids);
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
    ceres::LossFunction* loss) const {
  if (options_.ba_vio_prior_rot_sigma_deg <= 0.0 || problem == nullptr) {
    return 0;
  }
  std::vector<image_t> ids;
  for (const image_t id : image_ids) {
    if (pose_priors_.count(id) > 0 && reconstruction.ExistsImage(id) &&
        reconstruction.Image(id).HasPose() &&
        reconstruction.Image(id).IsRefInFrame()) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());

  const double rot_var = std::pow(DegToRad(options_.ba_vio_prior_rot_sigma_deg), 2);
  size_t num_priors = 0;
  for (size_t k = 1; k < ids.size(); ++k) {
    const image_t id_i = ids[k - 1];
    const image_t id_j = ids[k];
    if (id_j - id_i > static_cast<image_t>(std::max(options_.ba_vio_prior_max_gap, 1))) {
      continue;
    }
    double* params_i =
        reconstruction.Image(id_i).FramePtr()->RigFromWorld().params.data();
    double* params_j =
        reconstruction.Image(id_j).FramePtr()->RigFromWorld().params.data();
    // Blocks missing from the problem belong to constant-pose frames.
    const bool has_i = problem->HasParameterBlock(params_i);
    const bool has_j = problem->HasParameterBlock(params_j);
    if (!has_i && !has_j) {
      continue;
    }
    const Rigid3d i_from_j_prior =
        pose_priors_.at(id_i) * Inverse(pose_priors_.at(id_j));
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
    if (!has_i) {
      problem->SetParameterBlockConstant(params_i);
    }
    if (!has_j) {
      problem->SetParameterBlockConstant(params_j);
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
  for (const image_t id : local_ids) {
    if (id < oldest_id) {
      oldest_id = id;
    }
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
  const auto summary = bundle_adjuster->Solve();
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
  obs_manager_->FilterPoints3DInImages(options_.filter_max_reproj_error,
                                       options_.filter_min_tri_angle,
                                       filter_ids);
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

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      MakeLocalBaOptions(0), ba_config, *reconstruction_);
  const auto summary = bundle_adjuster->Solve();
  if (!summary || !summary->IsSolutionUsable()) {
    LOG(WARNING) << "OnlineMapper: loop BA failed, variable="
                 << JoinImageIds(used_variable);
    return;
  }
  LOG(INFO) << "OnlineMapper: loop BA variable=[" << JoinImageIds(used_variable)
            << "] anchors=" << num_anchors
            << " points=" << variable_point3D_ids.size()
            << " residuals=" << summary->num_residuals;

  triangulator_->MergeTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteTracks(options_.triangulation, variable_point3D_ids);
  FlatHashSet<image_t> filter_ids(used_variable.begin(), used_variable.end());
  obs_manager_->FilterPoints3DInImages(options_.filter_max_reproj_error,
                                       options_.filter_min_tri_angle,
                                       filter_ids);
}

void OnlineMapper::WriteSparse() const {
  if (sparse_path_.empty() || reconstruction_->NumRegFrames() == 0) {
    return;
  }
  try {
    const std::filesystem::path out_dir(sparse_path_);
    std::filesystem::create_directories(out_dir);
    reconstruction_->Write(out_dir);
    LOG(INFO) << "OnlineMapper: wrote sparse to " << out_dir.string()
              << " frames=" << reconstruction_->NumRegFrames()
              << " points3D=" << reconstruction_->NumPoints3D();
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineMapper: failed to write sparse: " << e.what();
  }
}

bool OnlineMapper::MappingCurrentImage(const image_t image_id,
                                       const std::optional<Rigid3d>& vio_prior) {
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
    WriteSparse();
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
  TriangulateImage(image_id);
  MergeAndFilterPoints(image_id);
  AdjustLocalBundle(image_id);
  WriteSparse();
  return true;
}

}  // namespace colmap
