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
#include <unordered_map>
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

struct PoseJumpStats {
  double center_m = 0.0;
  double angle_deg = 0.0;
  bool jumped = false;
};

PoseJumpStats ComputePoseJump(const Rigid3d& before,
                              const Rigid3d& after,
                              const double max_center_jump,
                              const double max_angle_deg) {
  PoseJumpStats stats;
  stats.center_m =
      (after.TgtOriginInSrc() - before.TgtOriginInSrc()).norm();
  stats.angle_deg = RadToDeg(
      Eigen::AngleAxisd(after.rotation() * before.rotation().inverse())
          .angle());
  stats.jumped =
      stats.center_m > max_center_jump || stats.angle_deg > max_angle_deg;
  return stats;
}

bool PoseJumpedTooFar(const Rigid3d& before,
                      const Rigid3d& after,
                      const double max_center_jump,
                      const double max_angle_deg) {
  return ComputePoseJump(before, after, max_center_jump, max_angle_deg).jumped;
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

}  // namespace

OnlineMapper::OnlineMapper(std::shared_ptr<DatabaseCache> cache,
                           std::string sparse_path,
                           OnlineMapperOptions options)
    : cache_(std::move(cache)),
      sparse_path_(std::move(sparse_path)),
      options_(std::move(options)),
      reconstruction_(std::make_shared<Reconstruction>()) {}

OnlineMapper::~OnlineMapper() = default;

bool OnlineMapper::EnsureReady() {
  if (obs_manager_ && triangulator_) {
    return true;
  }
  if (!cache_ || cache_->NumImages() == 0) {
    LOG(ERROR) << "OnlineMapper: cache has no images";
    return false;
  }
  reconstruction_->Load(*cache_);
  obs_manager_ = std::make_shared<ObservationManager>(
      *reconstruction_, cache_->CorrespondenceGraph());
  triangulator_ = std::make_shared<IncrementalTriangulator>(
      cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    (void)image;
    synced_images_.insert(image_id);
  }
  LOG(INFO) << "OnlineMapper: reconstruction ready with "
            << reconstruction_->NumImages() << " image(s)";
  return true;
}

bool OnlineMapper::SyncReconstruction() {
  if (!cache_) {
    return false;
  }
  if (!obs_manager_) {
    return EnsureReady();
  }

  reconstruction_->Load(*cache_);
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    (void)image;
    if (synced_images_.insert(image_id).second) {
      obs_manager_->AddImage(image_id);
    }
  }
  return true;
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

bool OnlineMapper::RegisterWithRelativeVio(const image_t image_id,
                                           const Rigid3d& vio_prior) {
  const image_t last_id = FindLastRegisteredWithPrior(image_id);
  if (last_id == kInvalidImageId) {
    LOG(INFO) << "OnlineMapper: relative VIO unavailable for image " << image_id
              << ", using absolute VIO";
    return RegisterImageWithKnownPose(image_id, vio_prior);
  }
  const Rigid3d cam_from_world =
      vio_prior * Inverse(pose_priors_.at(last_id)) *
      reconstruction_->Image(last_id).CamFromWorld();
  LOG(INFO) << "OnlineMapper: relative VIO image " << image_id << " from "
            << last_id;
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

OnlineMapper::P3PRegisterResult OnlineMapper::RegisterNextImageP3P(
    const image_t image_id) {
  P3PRegisterResult result;
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
  const auto correspondence_graph = cache_->CorrespondenceGraph();
  if (!correspondence_graph) {
    return result;
  }

  if (obs_manager_->NumVisiblePoints3D(image_id) <
      static_cast<size_t>(options_.abs_pose_min_num_inliers)) {
    LOG(INFO) << "OnlineMapper: P3P skip image " << image_id
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

  if (tri_points2D.size() <
      static_cast<size_t>(options_.abs_pose_min_num_inliers)) {
    LOG(INFO) << "OnlineMapper: P3P skip image " << image_id
              << ", 2D-3D=" << tri_points2D.size();
    return result;
  }

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.estimate_focal_length = false;
  abs_pose_options.ransac_options.max_error = options_.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options_.abs_pose_min_inlier_ratio;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  abs_pose_refinement_options.refine_focal_length = false;
  abs_pose_refinement_options.refine_extra_params = false;

  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Rigid3d cam_from_world;
  if (!EstimateAbsolutePose(abs_pose_options,
                            tri_points2D,
                            tri_points3D,
                            &cam_from_world,
                            &camera,
                            &num_inliers,
                            &inlier_mask)) {
    LOG(INFO) << "OnlineMapper: P3P estimate failed for image " << image_id;
    return result;
  }
  if (num_inliers < static_cast<size_t>(options_.abs_pose_min_num_inliers)) {
    LOG(INFO) << "OnlineMapper: P3P inliers=" << num_inliers << " for image "
              << image_id;
    return result;
  }
  if (!RefineAbsolutePose(abs_pose_refinement_options,
                          inlier_mask,
                          tri_points2D,
                          tri_points3D,
                          &cam_from_world,
                          &camera)) {
    LOG(INFO) << "OnlineMapper: P3P refine failed for image " << image_id;
    return result;
  }

  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  obs_manager_->RegisterFrame(image.FrameId());

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (!inlier_mask[i]) {
      continue;
    }
    const auto [point2D_idx, point3D_id] = tri_corrs[i];
    if (!image.Point2D(point2D_idx).HasPoint3D()) {
      obs_manager_->AddObservation(point3D_id,
                                   TrackElement(image_id, point2D_idx));
      triangulator_->AddModifiedPoint3D(point3D_id);
    }
  }

  result.success = true;
  result.num_inliers = num_inliers;
  result.cam_from_world = cam_from_world;
  LOG(INFO) << "OnlineMapper: P3P registered image " << image_id
            << " inliers=" << num_inliers;
  return result;
}

image_t OnlineMapper::FindInitPartner(const image_t image_id) const {
  if (!cache_->CorrespondenceGraph()) {
    return kInvalidImageId;
  }
  image_t best_id = kInvalidImageId;
  const auto graph = cache_->CorrespondenceGraph();
  for (const auto& [prev_id, image] : cache_->Images()) {
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
  FlatHashSet<image_t> constant_set;
  for (size_t i = 0; i < covisible.size(); ++i) {
    const image_t other_id = covisible[i].second;
    if (local_set.size() < max_local) {
      if (CollectRegisteredImage(other_id, local_ids)) {
        local_set.insert(other_id);
      }
    } else if (CollectRegisteredImage(other_id, constant_ids)) {
      constant_set.insert(other_id);
    }
  }
  for (const image_t local_id : *local_ids) {
    for (const Point2D& point2D : reconstruction_->Image(local_id).Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction_->ExistsPoint3D(point2D.point3D_id)) {
        continue;
      }
      for (const TrackElement& el :
           reconstruction_->Point3D(point2D.point3D_id).track.Elements()) {
        if (local_set.count(el.image_id) > 0 ||
            constant_set.count(el.image_id) > 0) {
          continue;
        }
        if (CollectRegisteredImage(el.image_id, constant_ids)) {
          constant_set.insert(el.image_id);
        }
      }
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
           << "# image_id usable jumped reverted first_level second_level "
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
  std::vector<image_t> second_level_ids = constant_ids;
  std::sort(second_level_ids.begin(), second_level_ids.end());
  LOG(INFO) << "OnlineMapper: LBA window image " << image_id
            << " first=[" << JoinImageIds(first_level_ids) << "]"
            << " second=[" << JoinImageIds(second_level_ids) << "]";

  std::vector<image_t> variable_ids = local_ids;
  const bool freeze_local_poses =
      static_cast<int>(first_level_ids.size()) <
      options_.ba_min_first_level_for_pose;
  if (freeze_local_poses) {
    // Too few first-level edges: do not move the current or neighbor poses.
    FlatHashSet<image_t> constant_set(constant_ids.begin(), constant_ids.end());
    for (const image_t id : local_ids) {
      if (constant_set.insert(id).second) {
        constant_ids.push_back(id);
      }
    }
    variable_ids.clear();
    LOG(INFO) << "OnlineMapper: freeze poses image " << image_id
              << " first_level=" << first_level_ids.size()
              << " < min=" << options_.ba_min_first_level_for_pose;
  } else if (constant_ids.empty() && variable_ids.size() >= 2) {
    // No outer covisible cameras: pin the oldest local KF so the window has a
    // gauge (ORB-SLAM3 aborts if zero fixed KFs).
    image_t oldest_id = variable_ids.front();
    for (const image_t id : variable_ids) {
      if (id < oldest_id) {
        oldest_id = id;
      }
    }
    constant_ids.push_back(oldest_id);
    variable_ids.erase(
        std::remove(variable_ids.begin(), variable_ids.end(), oldest_id),
        variable_ids.end());
  }

  if ((variable_ids.size() + constant_ids.size()) < 2) {
    LOG(INFO) << "OnlineMapper: skip LBA image " << image_id
              << " first=[" << JoinImageIds(first_level_ids) << "]"
              << " second=[" << JoinImageIds(second_level_ids) << "]"
              << " variable=" << variable_ids.size()
              << " constant=" << constant_ids.size();
    return;
  }

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

  FlatHashMap<point3D_t, Eigen::Vector3d> point_backup;
  for (const image_t ba_id : ba_config.Images()) {
    for (const Point2D& point2D : reconstruction_->Image(ba_id).Points2D()) {
      if (!point2D.HasPoint3D() ||
          point_backup.count(point2D.point3D_id) > 0) {
        continue;
      }
      point_backup.emplace(point2D.point3D_id,
                           reconstruction_->Point3D(point2D.point3D_id).xyz);
    }
  }
  FlatHashMap<image_t, Rigid3d> pose_backup;
  for (const image_t ba_id : variable_ids) {
    pose_backup.emplace(ba_id, reconstruction_->Image(ba_id).CamFromWorld());
  }
  const Rigid3d pose_before = reconstruction_->Image(image_id).CamFromWorld();

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      MakeLocalBaOptions(0), ba_config, *reconstruction_);
  const auto summary = bundle_adjuster->Solve();
  const bool usable = summary && summary->IsSolutionUsable();
  bool jumped = false;
  for (const auto& [ba_id, pose_before_id] : pose_backup) {
    if (PoseJumpedTooFar(pose_before_id,
                         reconstruction_->Image(ba_id).CamFromWorld(),
                         options_.ba_max_pose_center_jump,
                         options_.ba_max_pose_angle_deg)) {
      jumped = true;
      break;
    }
  }
  const Rigid3d pose_after = reconstruction_->Image(image_id).CamFromWorld();
  const bool reverted = !usable || jumped;
  AppendLbaPoseRecord(image_id,
                      first_level_ids,
                      second_level_ids,
                      pose_before,
                      pose_after,
                      usable,
                      jumped,
                      reverted);
  if (reverted) {
    LOG(WARNING) << "OnlineMapper: revert LBA image " << image_id
                 << " usable=" << usable << " jumped=" << jumped
                 << " residuals=" << (summary ? summary->num_residuals : 0)
                 << " first=[" << JoinImageIds(first_level_ids) << "]"
                 << " second=[" << JoinImageIds(second_level_ids) << "]";
    for (const auto& [ba_id, pose_before_id] : pose_backup) {
      Image& ba_image = reconstruction_->Image(ba_id);
      ba_image.FramePtr()->SetCamFromWorld(ba_image.CameraId(), pose_before_id);
    }
    for (const auto& [pt_id, xyz] : point_backup) {
      if (reconstruction_->ExistsPoint3D(pt_id)) {
        reconstruction_->Point3D(pt_id).xyz = xyz;
      }
    }
    return;
  }

  LOG(INFO) << "OnlineMapper: LBA image " << image_id
            << " first=[" << JoinImageIds(first_level_ids) << "]"
            << " second=[" << JoinImageIds(second_level_ids) << "]"
            << " variable=" << variable_ids.size()
            << " fixed=" << constant_ids.size()
            << " residuals=" << summary->num_residuals;

  triangulator_->MergeTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteTracks(options_.triangulation, variable_point3D_ids);
  triangulator_->CompleteImage(options_.triangulation, image_id);
  FlatHashSet<image_t> filter_ids(variable_ids.begin(), variable_ids.end());
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

  const P3PRegisterResult p3p = RegisterNextImageP3P(image_id);
  bool registered = p3p.success;
  if (!registered && vio_prior.has_value()) {
    LOG(INFO) << "OnlineMapper: P3P failed, fallback to map-frame VIO for image "
              << image_id;
    registered = RegisterWithRelativeVio(image_id, *vio_prior);
  }
  if (!registered) {
    LOG(WARNING) << "OnlineMapper: skip mapping image " << image_id
                 << ", no P3P and no VIO prior";
    return true;
  }

  TriangulateImage(image_id);
  MergeAndFilterPoints(image_id);
  AdjustLocalBundle(image_id);
  WriteSparse();
  return true;
}

}  // namespace colmap
