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

#include "colmap/online_sfm/online_loop_closer.h"
#include "mixvpr/mixvpr_encoder.h"

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/math.h"
#include "colmap/online_sfm/online_pose_graph.h"
#include "colmap/scene/projection.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/track.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Geometry>

namespace colmap {
namespace {

// Keypoint noise behind the loop PnP covariance.
constexpr double kLoopPixelSigma = 1.0;
// PnP covariance treats map points as exact; inflate the loop edge sigma.
constexpr double kLoopSigmaInflation = 2.0;
// Historical edges whose residual is farther than this multiple of the
// correction gate are treated as broken measurements and left out.
constexpr double kHistoricalResidualFactor = 2.0;
// Candidate + covisibles used as the loop-side map for PnP and fusion.
constexpr size_t kMaxLoopMapImages = 10;
// A corrected frame joins the fusion / loop-BA window when at least this many
// loop-side map points project inside it.
constexpr size_t kMinOverlapProjections = 50;
// Newest track observations used as fusion descriptor references. The minimum
// L2 among them is kept, same as local-map projection matching.
constexpr size_t kMaxFuseRefDescriptors = 3;

// Camera +Z in world. COLMAP x_cam = R x_world + t, so +Z_world = R^T e_z.
Eigen::Vector3d OpticalAxisWorld(const Image& image) {
  return image.CamFromWorld().rotation().toRotationMatrix().row(2).transpose();
}

bool GeometryUsable(const TwoViewGeometry& geometry, const int min_inliers) {
  return geometry.config != TwoViewGeometry::ConfigurationType::DEGENERATE &&
         geometry.inlier_matches.size() >= static_cast<size_t>(min_inliers);
}

// SIFT descriptors are L2-normalized to 512.
constexpr float kLoopDescMaxL2Sq = 512.0f * 512.0f * 0.7f * 0.7f;

// Current keypoint ↔ loop-side map point.
struct LoopPnPCorr {
  point2D_t idx_current = kInvalidPoint2DIdx;
  point3D_t point3D_id = kInvalidPoint3DId;
  Eigen::Vector2d xy_current = Eigen::Vector2d::Zero();
};

// Feature scale consistency (BAFS observation model s = f * S / range). The
// point size S is the median over its observations in the loaded images; a
// keypoint is consistent when its scale is within the ratio of the scale S
// predicts at the queried pose. Range is used instead of optical-axis depth
// because the fisheye feature scale follows range.
class LoopScaleCheck {
 public:
  explicit LoopScaleCheck(const double max_ratio)
      : max_log_(max_ratio > 1.0 ? std::log(max_ratio) : 0.0) {}

  bool Enabled() const { return max_log_ > 0.0; }

  void Load(const Database& database, const image_t image_id) {
    if (!Enabled() || scales_.count(image_id) > 0 ||
        !database.ExistsKeypoints(image_id)) {
      return;
    }
    const FeatureKeypoints keypoints = database.ReadKeypoints(image_id);
    std::vector<float>& scales = scales_[image_id];
    scales.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      scales[i] = keypoints[i].ComputeScale();
    }
  }

  // True also when there is nothing to compare against.
  bool Consistent(const Reconstruction& reconstruction,
                  const image_t image_id,
                  const point2D_t point2D_idx,
                  const point3D_t point3D_id,
                  const Rigid3d& cam_from_world,
                  const Camera& camera) const {
    if (!Enabled() || !reconstruction.ExistsPoint3D(point3D_id)) {
      return true;
    }
    const auto it = scales_.find(image_id);
    if (it == scales_.end() || point2D_idx >= it->second.size() ||
        it->second[point2D_idx] <= 0.0f) {
      return true;
    }
    const double size = PointSize(reconstruction, point3D_id, image_id);
    const double range = (reconstruction.Point3D(point3D_id).xyz -
                          cam_from_world.TgtOriginInSrc())
                             .norm();
    if (size <= 0.0 || range <= 0.0) {
      return true;
    }
    const double predicted = size * camera.MeanFocalLength() / range;
    return std::abs(std::log(it->second[point2D_idx] / predicted)) <= max_log_;
  }

 private:
  double PointSize(const Reconstruction& reconstruction,
                   const point3D_t point3D_id,
                   const image_t query_id) const {
    const auto cached = sizes_.find(point3D_id);
    if (cached != sizes_.end()) {
      return cached->second;
    }
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    std::vector<double> sizes;
    for (const TrackElement& el : point3D.track.Elements()) {
      const auto it = scales_.find(el.image_id);
      if (el.image_id == query_id || it == scales_.end() ||
          el.point2D_idx >= it->second.size() ||
          !reconstruction.ExistsImage(el.image_id)) {
        continue;
      }
      const Image& image = reconstruction.Image(el.image_id);
      if (!image.HasPose() || !image.HasCameraPtr()) {
        continue;
      }
      sizes.push_back(it->second[el.point2D_idx] *
                      (point3D.xyz - image.ProjectionCenter()).norm() /
                      image.CameraPtr()->MeanFocalLength());
    }
    double size = 0.0;
    if (!sizes.empty()) {
      std::nth_element(
          sizes.begin(), sizes.begin() + sizes.size() / 2, sizes.end());
      size = sizes[sizes.size() / 2];
    }
    sizes_.emplace(point3D_id, size);
    return size;
  }

  double max_log_ = 0.0;
  std::unordered_map<image_t, std::vector<float>> scales_;
  mutable std::unordered_map<point3D_t, double> sizes_;
};

std::optional<Eigen::Vector2d> ProjectCam(const Camera& camera,
                                          const Eigen::Vector3d& xyz_cam) {
  if (xyz_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return std::nullopt;
  }
  return camera.ImgFromCam(xyz_cam);
}

bool InsideImage(const Camera& camera, const Eigen::Vector2d& xy) {
  return xy.x() >= 0.0 && xy.y() >= 0.0 &&
         xy.x() < static_cast<double>(camera.width) &&
         xy.y() < static_cast<double>(camera.height);
}

size_t CountInliers(const std::vector<char>& inliers) {
  return static_cast<size_t>(std::count(inliers.begin(), inliers.end(), 1));
}

double RotationDiffDeg(const Eigen::Quaterniond& a,
                       const Eigen::Quaterniond& b) {
  return RadToDeg(Eigen::AngleAxisd(a * b.inverse()).angle());
}

float DescriptorL2Sq(const FeatureDescriptors& a,
                     const point2D_t ia,
                     const FeatureDescriptors& b,
                     const point2D_t ib) {
  if (static_cast<int>(ia) >= a.data.rows() ||
      static_cast<int>(ib) >= b.data.rows() || a.data.cols() != b.data.cols() ||
      a.data.cols() == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(
      (a.data.row(ia).cast<int>() - b.data.row(ib).cast<int>()).squaredNorm());
}

// Matches from ReadMatches(loop_image, current): idx1 = loop image, idx2 =
// current. Every loop-side keypoint with a 3D point gives one 2D–3D pair.
void CollectLoopPnPCorrs(const Image& current,
                         const Image& loop_image,
                         const Reconstruction& reconstruction,
                         const FeatureMatches& matches,
                         std::unordered_set<point2D_t>* used_current,
                         std::unordered_set<point3D_t>* used_points,
                         std::vector<LoopPnPCorr>* corrs) {
  for (const FeatureMatch& match : matches) {
    const point2D_t idx_current = match.point2D_idx2;
    const point2D_t idx_loop = match.point2D_idx1;
    if (idx_current >= current.NumPoints2D() ||
        idx_loop >= loop_image.NumPoints2D() ||
        used_current->count(idx_current) > 0) {
      continue;
    }
    const Point2D& p_loop = loop_image.Point2D(idx_loop);
    if (!p_loop.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
        used_points->count(p_loop.point3D_id) > 0) {
      continue;
    }
    LoopPnPCorr corr;
    corr.idx_current = idx_current;
    corr.point3D_id = p_loop.point3D_id;
    corr.xy_current = current.Point2D(idx_current).xy;
    corrs->push_back(corr);
    used_current->insert(idx_current);
    used_points->insert(p_loop.point3D_id);
  }
}

std::vector<char> ComputePnPInliers(const std::vector<LoopPnPCorr>& corrs,
                                    const Reconstruction& reconstruction,
                                    const Rigid3d& cam_from_world,
                                    const Camera& camera,
                                    const double max_error_px) {
  const double max_sq = max_error_px * max_error_px;
  std::vector<char> inliers(corrs.size(), 0);
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (!reconstruction.ExistsPoint3D(corrs[i].point3D_id)) {
      continue;
    }
    const double err = CalculateSquaredReprojectionError(
        corrs[i].xy_current,
        reconstruction.Point3D(corrs[i].point3D_id).xyz,
        cam_from_world,
        camera);
    inliers[i] = err <= max_sq ? 1 : 0;
  }
  return inliers;
}

// Isotropic 1-sigma of the PnP pose from the Gauss-Newton information of the
// inlier reprojections, with a left perturbation [w, v] of cam_from_world.
// The rotation sigma covers w; the translation sigma covers the camera center
// in any fixed frame, which only moves with v.
bool EstimatePnPSigmas(const std::vector<LoopPnPCorr>& corrs,
                       const std::vector<char>& inliers,
                       const Reconstruction& reconstruction,
                       const Rigid3d& cam_from_world,
                       const Camera& camera,
                       double* rot_sigma,
                       double* trans_sigma) {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  size_t num_used = 0;
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (!inliers[i] || !reconstruction.ExistsPoint3D(corrs[i].point3D_id)) {
      continue;
    }
    const Eigen::Vector3d xyz_cam =
        cam_from_world * reconstruction.Point3D(corrs[i].point3D_id).xyz;
    const auto proj = ProjectCam(camera, xyz_cam);
    if (!proj) {
      continue;
    }
    const double h = 1e-6 * std::max(1.0, xyz_cam.norm());
    Eigen::Matrix<double, 2, 3> J_proj;
    bool ok = true;
    for (int a = 0; a < 3 && ok; ++a) {
      Eigen::Vector3d xyz_h = xyz_cam;
      xyz_h[a] += h;
      const auto proj_h = ProjectCam(camera, xyz_h);
      ok = proj_h.has_value();
      if (ok) {
        J_proj.col(a) = (*proj_h - *proj) / h;
      }
    }
    if (!ok) {
      continue;
    }
    Eigen::Matrix<double, 3, 6> J_xyz;
    J_xyz.leftCols<3>() << 0.0, xyz_cam.z(), -xyz_cam.y(),
        -xyz_cam.z(), 0.0, xyz_cam.x(),
        xyz_cam.y(), -xyz_cam.x(), 0.0;
    J_xyz.rightCols<3>().setIdentity();
    const Eigen::Matrix<double, 2, 6> J = J_proj * J_xyz;
    H += J.transpose() * J;
    ++num_used;
  }
  if (num_used < 6) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig_h(H);
  if (eig_h.info() != Eigen::Success || !(eig_h.eigenvalues()(0) > 1e-9)) {
    return false;
  }
  const Eigen::Matrix<double, 6, 6> cov =
      kLoopPixelSigma * kLoopPixelSigma * H.inverse();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_rot(
      cov.topLeftCorner<3, 3>());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans(
      cov.bottomRightCorner<3, 3>());
  *rot_sigma = std::sqrt(std::max(eig_rot.eigenvalues()(2), 0.0));
  *trans_sigma = std::sqrt(std::max(eig_trans.eigenvalues()(2), 0.0));
  return true;
}

// Project loop-map points with the PnP pose into the current image and take
// the descriptor-closest keypoint inside `radius_px` (ORB SearchByProjection).
size_t SearchLoopByProjection(
    const Image& current,
    const Reconstruction& reconstruction,
    const Rigid3d& cam_from_world,
    const std::vector<image_t>& loop_ids,
    const std::unordered_map<image_t, FeatureDescriptors>& loop_desc,
    const FeatureDescriptors& desc_current,
    const double radius_px,
    const LoopScaleCheck& scale_check,
    std::unordered_set<point2D_t>* used_current,
    std::unordered_set<point3D_t>* used_points,
    std::vector<LoopPnPCorr>* corrs) {
  if (desc_current.data.rows() == 0 || !current.HasCameraPtr()) {
    return 0;
  }
  const Camera& camera = *current.CameraPtr();
  const double radius_sq = radius_px * radius_px;
  size_t added = 0;
  for (const image_t loop_id : loop_ids) {
    const auto desc_it = loop_desc.find(loop_id);
    if (desc_it == loop_desc.end() || !reconstruction.ExistsImage(loop_id)) {
      continue;
    }
    const Image& loop_image = reconstruction.Image(loop_id);
    for (point2D_t j = 0; j < loop_image.NumPoints2D(); ++j) {
      const Point2D& p_loop = loop_image.Point2D(j);
      if (!p_loop.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
          used_points->count(p_loop.point3D_id) > 0) {
        continue;
      }
      const auto proj = ProjectCam(
          camera,
          cam_from_world * reconstruction.Point3D(p_loop.point3D_id).xyz);
      if (!proj || !InsideImage(camera, *proj)) {
        continue;
      }
      float best_l2 = kLoopDescMaxL2Sq;
      point2D_t best_idx = kInvalidPoint2DIdx;
      for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
        if (used_current->count(i) > 0 ||
            (current.Point2D(i).xy - *proj).squaredNorm() >= radius_sq ||
            !scale_check.Consistent(reconstruction,
                                    current.ImageId(),
                                    i,
                                    p_loop.point3D_id,
                                    cam_from_world,
                                    camera)) {
          continue;
        }
        const float l2 = DescriptorL2Sq(desc_it->second, j, desc_current, i);
        if (l2 < best_l2) {
          best_l2 = l2;
          best_idx = i;
        }
      }
      if (best_idx == kInvalidPoint2DIdx) {
        continue;
      }
      LoopPnPCorr corr;
      corr.idx_current = best_idx;
      corr.point3D_id = p_loop.point3D_id;
      corr.xy_current = current.Point2D(best_idx).xy;
      corrs->push_back(corr);
      used_current->insert(best_idx);
      used_points->insert(p_loop.point3D_id);
      ++added;
    }
  }
  return added;
}

// Pose-free 2D-3D bootstrap against the loop-side island. Each current
// keypoint takes its nearest SIFT neighbour among island observations (Lowe
// ratio against the best observation of a different point) that lies within
// `max_epipolar_angle` of the epipolar plane E * x_candidate; each point keeps
// its closest keypoint. E maps candidate rays to current rays.
size_t SearchLoopIslandByDescriptor(
    const Image& current,
    const Image& candidate,
    const Reconstruction& reconstruction,
    const Eigen::Matrix3d& E,
    const std::vector<image_t>& loop_ids,
    const std::unordered_map<image_t, FeatureDescriptors>& loop_desc,
    const FeatureDescriptors& desc_current,
    const double max_epipolar_angle,
    std::unordered_set<point2D_t>* used_current,
    std::unordered_set<point3D_t>* used_points,
    std::vector<LoopPnPCorr>* corrs) {
  const int dim = static_cast<int>(desc_current.data.cols());
  if (dim == 0 || !current.HasCameraPtr() || !candidate.HasPose()) {
    return 0;
  }
  const Rigid3d& cand_from_world = candidate.CamFromWorld();

  std::vector<point3D_t> row_points;
  std::vector<Eigen::Vector3d> row_normals;
  std::vector<const uint8_t*> row_desc;
  std::unordered_map<point3D_t, size_t> rows_per_point;
  for (const image_t loop_id : loop_ids) {
    const auto desc_it = loop_desc.find(loop_id);
    if (desc_it == loop_desc.end() || !reconstruction.ExistsImage(loop_id) ||
        desc_it->second.data.cols() != dim) {
      continue;
    }
    const Image& loop_image = reconstruction.Image(loop_id);
    const FeatureDescriptorsData& data = desc_it->second.data;
    for (point2D_t j = 0; j < loop_image.NumPoints2D(); ++j) {
      const Point2D& p_loop = loop_image.Point2D(j);
      if (static_cast<int>(j) >= data.rows() || !p_loop.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(p_loop.point3D_id) ||
          used_points->count(p_loop.point3D_id) > 0) {
        continue;
      }
      size_t& num_rows = rows_per_point[p_loop.point3D_id];
      if (num_rows >= kMaxFuseRefDescriptors) {
        continue;
      }
      const Eigen::Vector3d normal =
          E * (cand_from_world * reconstruction.Point3D(p_loop.point3D_id).xyz);
      if (normal.norm() < 1e-12) {
        continue;
      }
      ++num_rows;
      row_points.push_back(p_loop.point3D_id);
      row_normals.push_back(normal.normalized());
      row_desc.push_back(data.row(j).data());
    }
  }

  std::vector<point2D_t> col_idx;
  std::vector<Eigen::Vector3d> col_rays;
  for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
    if (static_cast<int>(i) >= desc_current.data.rows() ||
        used_current->count(i) > 0) {
      continue;
    }
    const auto ray = current.CameraPtr()->CamFromImg(current.Point2D(i).xy);
    if (!ray) {
      continue;
    }
    col_idx.push_back(i);
    col_rays.push_back(ray->homogeneous().normalized());
  }
  if (row_points.empty() || col_idx.empty()) {
    return 0;
  }

  Eigen::MatrixXf rows(row_points.size(), dim);
  for (size_t r = 0; r < row_desc.size(); ++r) {
    rows.row(r) = Eigen::Map<const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>>(
                      row_desc[r], dim)
                      .cast<float>();
  }
  const Eigen::VectorXf row_sq = rows.rowwise().squaredNorm();
  const double max_sin = std::sin(max_epipolar_angle);
  constexpr float kMaxRatioSq = 0.8f * 0.8f;
  constexpr int kBlock = 256;

  // point -> (L2^2, current keypoint), mutual best.
  std::unordered_map<point3D_t, std::pair<float, point2D_t>> best_for_point;
  for (size_t c0 = 0; c0 < col_idx.size(); c0 += kBlock) {
    const int n = static_cast<int>(std::min<size_t>(kBlock, col_idx.size() - c0));
    Eigen::MatrixXf cols(n, dim);
    for (int c = 0; c < n; ++c) {
      cols.row(c) = desc_current.data.row(col_idx[c0 + c]).cast<float>();
    }
    const Eigen::VectorXf col_sq = cols.rowwise().squaredNorm();
    const Eigen::MatrixXf dots = rows * cols.transpose();
    for (int c = 0; c < n; ++c) {
      const Eigen::Vector3d& ray = col_rays[c0 + c];
      float best = std::numeric_limits<float>::infinity();
      float second = std::numeric_limits<float>::infinity();
      point3D_t best_point = kInvalidPoint3DId;
      for (size_t r = 0; r < row_points.size(); ++r) {
        if (std::abs(ray.dot(row_normals[r])) > max_sin) {
          continue;
        }
        const float d = row_sq(r) + col_sq(c) - 2.0f * dots(r, c);
        if (d < best) {
          if (row_points[r] != best_point) {
            second = best;
          }
          best = d;
          best_point = row_points[r];
        } else if (d < second && row_points[r] != best_point) {
          second = d;
        }
      }
      if (best_point == kInvalidPoint3DId || best > kLoopDescMaxL2Sq ||
          best > kMaxRatioSq * second) {
        continue;
      }
      const auto it = best_for_point.find(best_point);
      if (it == best_for_point.end() || best < it->second.first) {
        best_for_point[best_point] = {best, col_idx[c0 + c]};
      }
    }
  }

  size_t added = 0;
  for (const auto& [point3D_id, match] : best_for_point) {
    if (used_current->count(match.second) > 0) {
      continue;
    }
    LoopPnPCorr corr;
    corr.idx_current = match.second;
    corr.point3D_id = point3D_id;
    corr.xy_current = current.Point2D(match.second).xy;
    corrs->push_back(corr);
    used_current->insert(match.second);
    used_points->insert(point3D_id);
    ++added;
  }
  return added;
}

// Rotation angle between the PnP-implied relative pose and the essential
// matrix pose of the same pair (image1 = candidate, image2 = current).
// Returns -1 when the essential matrix is unavailable.
double RotationVsEssentialDeg(const TwoViewGeometry& geometry,
                              const Image& candidate,
                              const Image& current,
                              const Rigid3d& current_cam_from_world,
                              double* tdir_deg) {
  *tdir_deg = -1.0;
  if (!geometry.E || geometry.inlier_matches.size() < 8 ||
      !candidate.HasCameraPtr() || !current.HasCameraPtr()) {
    return -1.0;
  }
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  rays1.reserve(geometry.inlier_matches.size());
  rays2.reserve(geometry.inlier_matches.size());
  for (const FeatureMatch& m : geometry.inlier_matches) {
    if (m.point2D_idx1 >= candidate.NumPoints2D() ||
        m.point2D_idx2 >= current.NumPoints2D()) {
      continue;
    }
    const auto n1 =
        candidate.CameraPtr()->CamFromImg(candidate.Point2D(m.point2D_idx1).xy);
    const auto n2 =
        current.CameraPtr()->CamFromImg(current.Point2D(m.point2D_idx2).xy);
    if (!n1 || !n2) {
      continue;
    }
    rays1.push_back(n1->homogeneous().normalized());
    rays2.push_back(n2->homogeneous().normalized());
  }
  if (rays1.size() < 8) {
    return -1.0;
  }
  Rigid3d e_cur_from_cand;
  std::vector<int> valid;
  PoseFromEssentialMatrix(*geometry.E, rays1, rays2, &e_cur_from_cand, &valid);
  if (valid.empty()) {
    return -1.0;
  }
  const Rigid3d pnp_cur_from_cand =
      current_cam_from_world * Inverse(candidate.CamFromWorld());
  const Eigen::Vector3d t_pnp = pnp_cur_from_cand.translation();
  const Eigen::Vector3d t_e = e_cur_from_cand.translation();
  if (t_pnp.norm() > 1e-6 && t_e.norm() > 1e-6) {
    *tdir_deg = RadToDeg(std::acos(
        std::clamp(t_pnp.normalized().dot(t_e.normalized()), -1.0, 1.0)));
  }
  return RotationDiffDeg(pnp_cur_from_cand.rotation(),
                         e_cur_from_cand.rotation());
}

// RMSE between `before` and `after` camera centers once `after` is rigidly
// aligned onto `before`: yaw about +z plus translation, or full SE3.
double AlignedTrajectoryRmse(const std::vector<Eigen::Vector3d>& before,
                             const std::vector<Eigen::Vector3d>& after,
                             const bool yaw_only) {
  const size_t n = before.size();
  if (n == 0 || after.size() != n) {
    return 0.0;
  }
  Eigen::Matrix3Xd a(3, n);
  Eigen::Matrix3Xd b(3, n);
  for (size_t i = 0; i < n; ++i) {
    a.col(i) = before[i];
    b.col(i) = after[i];
  }
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  if (yaw_only) {
    const Eigen::Vector3d mean_a = a.rowwise().mean();
    const Eigen::Vector3d mean_b = b.rowwise().mean();
    const Eigen::Matrix3Xd ca = a.colwise() - mean_a;
    const Eigen::Matrix3Xd cb = b.colwise() - mean_b;
    const double s = (cb.row(0).cwiseProduct(ca.row(1)) -
                      cb.row(1).cwiseProduct(ca.row(0)))
                         .sum();
    const double c = (cb.row(0).cwiseProduct(ca.row(0)) +
                      cb.row(1).cwiseProduct(ca.row(1)))
                         .sum();
    R = Eigen::AngleAxisd(std::atan2(s, c), Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    t = mean_a - R * mean_b;
  } else if (n >= 3) {
    const Eigen::Matrix4d T = Eigen::umeyama(b, a, /*with_scaling=*/false);
    R = T.topLeftCorner<3, 3>();
    t = T.topRightCorner<3, 1>();
  } else {
    t = a.rowwise().mean() - b.rowwise().mean();
  }
  const Eigen::Matrix3Xd residual = a - ((R * b).colwise() + t);
  return std::sqrt(residual.colwise().squaredNorm().mean());
}

bool ImageIdsWithin(const image_t a, const image_t b, const int gap) {
  if (gap < 0) {
    return false;
  }
  const image_t delta = a > b ? a - b : b - a;
  return delta <= static_cast<image_t>(gap);
}

bool SetImagePose(Image& image, const Rigid3d& cam_from_world) {
  if (!image.HasFramePtr()) {
    return false;
  }
  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);
  return true;
}

}  // namespace

OnlineLoopCloser::OnlineLoopCloser(std::string database_path,
                                   OnlineLoopCloserOptions options)
    : database_path_(std::move(database_path)), options_(std::move(options)) {
  if (!options_.enabled) {
    LOG(INFO) << "OnlineLoopCloser: disabled by options";
    return;
  }
  timing_ = std::make_unique<TimingStats>(
      TimingStats::FileBesideDatabase(database_path_, "loop_pairs.txt"),
      "loop",
      std::vector<std::string>{"candidate",
                               "score",
                               "matches",
                               "inliers",
                               "n3d",
                               "n3d_inliers",
                               "correction_m",
                               "pnp",
                               "status",
                               "candidate_ids"});
  timing_->AddMeta("mixvpr_engine_path", options_.mixvpr_engine_path);
  timing_->AddMeta("mixvpr_use_gpu", options_.mixvpr_use_gpu ? "1" : "0");
  timing_->AddMeta("cooldown_num_images",
                   std::to_string(options_.cooldown_num_images));
  timing_->AddMeta("min_frame_gap", std::to_string(options_.min_frame_gap));
  timing_->AddMeta("max_distance", std::to_string(options_.max_distance));
  timing_->AddMeta("max_view_angle_deg",
                   std::to_string(options_.max_view_angle_deg));
  timing_->AddMeta("min_mixvpr_score",
                   std::to_string(options_.min_mixvpr_score));
  timing_->AddMeta("topk", std::to_string(options_.topk));
  timing_->AddMeta("min_num_matches",
                   std::to_string(options_.min_num_matches));
  timing_->AddMeta("min_num_verified_matches",
                   std::to_string(options_.min_num_verified_matches));
  timing_->AddMeta("reloc_topk", std::to_string(options_.reloc_topk));
  timing_->AddMeta("reloc_min_mixvpr_score",
                   std::to_string(options_.reloc_min_mixvpr_score));
  timing_->AddMeta("reloc_min_num_matches",
                   std::to_string(options_.reloc_min_num_matches));
  timing_->AddMeta("reloc_min_num_3d_correspondences",
                   std::to_string(options_.reloc_min_num_3d_correspondences));
  timing_->AddMeta("reloc_min_inliers",
                   std::to_string(options_.reloc_min_inliers));
  timing_->AddMeta("reloc_max_rot_vs_essential_deg",
                   std::to_string(options_.reloc_max_rot_vs_essential_deg));
}

OnlineLoopCloser::~OnlineLoopCloser() = default;

bool OnlineLoopCloser::EncodeQuery(const image_t image_id,
                                   const std::string& image_abs_path,
                                   const Bitmap* bitmap,
                                   Eigen::VectorXf* query) {
  if (query == nullptr) {
    return false;
  }
  if (!InitMixVprModel()) {
    return false;
  }
  const bool encoded = (bitmap != nullptr && !bitmap->IsEmpty())
                           ? mixvpr_->Encode(*bitmap, query)
                           : mixvpr_->Encode(image_abs_path, query);
  if (!encoded || query->size() != kMixVprDescDim) {
    LOG(WARNING) << "OnlineLoopCloser: MixVPR encode failed for image "
                 << image_id << " path=" << image_abs_path
                 << " desc_dim=" << query->size();
    return false;
  }
  return true;
}

OnlineLoopCloser::CandidateGate OnlineLoopCloser::LoopGate() const {
  CandidateGate gate;
  gate.min_matches = static_cast<size_t>(std::max(options_.min_num_matches, 0));
  gate.min_3d_corrs =
      static_cast<size_t>(std::max(options_.min_num_3d_correspondences, 0));
  gate.min_inliers =
      static_cast<size_t>(std::max(options_.min_num_verified_matches, 0));
  gate.max_rot_vs_essential_deg = options_.max_rot_vs_essential_deg;
  gate.require_current_pose = true;
  gate.check_correction = true;
  return gate;
}

OnlineLoopCloser::CandidateGate OnlineLoopCloser::RelocGate() const {
  CandidateGate gate;
  gate.min_matches =
      static_cast<size_t>(std::max(options_.reloc_min_num_matches, 0));
  gate.min_3d_corrs =
      static_cast<size_t>(std::max(options_.reloc_min_num_3d_correspondences, 0));
  gate.min_inliers = static_cast<size_t>(std::max(options_.reloc_min_inliers, 0));
  gate.max_rot_vs_essential_deg = options_.reloc_max_rot_vs_essential_deg;
  gate.require_current_pose = false;
  gate.check_correction = false;
  return gate;
}

bool OnlineLoopCloser::Process(const image_t image_id,
                               const std::string& image_abs_path,
                               const DatabaseCache& cache,
                               Reconstruction& reconstruction,
                               ObservationManager* obs_manager,
                               OnlineFeatureMatcher* matcher,
                               const Bitmap* bitmap) {
  if (!IsLoopClosureActive()) {
    return false;
  }
  if (image_id == kInvalidImageId ||
      (image_abs_path.empty() && (bitmap == nullptr || bitmap->IsEmpty()))) {
    return false;
  }

  Timer timer;
  timer.Start();
  if (!InitMixVprModel()) {
    return false;
  }

  Eigen::VectorXf query;
  if (!EncodeQuery(image_id, image_abs_path, bitmap, &query)) {
    return false;
  }

  const std::vector<std::pair<image_t, float>> candidates =
      DetectLoop(image_id, query, reconstruction);
  if (!StoreMixVprDescriptor(image_id, query)) {
    return false;
  }

  if (candidates.empty()) {
    timing_->Record(std::to_string(image_id),
                    timer.ElapsedMicroSeconds() / 1000.0,
                    {"-",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "0",
                     "no_candidate",
                     "-"});
    return false;
  }

  LOG(INFO) << "OnlineLoopCloser: image " << image_id << " has "
            << candidates.size() << " loop candidate(s)";

  const auto RecordPair = [&](const OnlineLoopPair& pair, const char* status) {
    timing_->Record(
        std::to_string(image_id),
        timer.ElapsedMicroSeconds() / 1000.0,
        {pair.image_id2 == kInvalidImageId ? "-"
                                          : std::to_string(pair.image_id2),
         std::to_string(pair.retrieval_score),
         std::to_string(pair.num_matches),
         std::to_string(pair.num_inliers),
         std::to_string(pair.num_3d_correspondences),
         std::to_string(pair.num_3d_inliers),
         std::to_string(pair.correction_m),
         pair.has_loop_pose ? "1" : "0",
         status,
         "-"});
  };
  const auto BetterAttempt = [](const OnlineLoopPair& a,
                                const OnlineLoopPair& b) {
    if (a.num_3d_inliers != b.num_3d_inliers) {
      return a.num_3d_inliers > b.num_3d_inliers;
    }
    if (a.num_3d_correspondences != b.num_3d_correspondences) {
      return a.num_3d_correspondences > b.num_3d_correspondences;
    }
    if (a.num_inliers != b.num_inliers) {
      return a.num_inliers > b.num_inliers;
    }
    return a.num_matches > b.num_matches;
  };

  bool confirmed = false;
  OnlineLoopPair confirmed_pair;
  OnlineLoopPair best_attempt;
  bool has_attempt = false;
  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(candidates.size());
  for (const auto& [candidate_id, score] : candidates) {
    candidate_ids.push_back(candidate_id);
  }
  // Drop weak stored pairs so the batch rematch uses the loop match threshold,
  // which is stricter than the matcher's own inlier gate.
  try {
    DatabaseSession session(database_path_);
    for (const image_t candidate_id : candidate_ids) {
      if (!session->ExistsTwoViewGeometry(candidate_id, image_id)) {
        continue;
      }
      const TwoViewGeometry existing =
          session->ReadTwoViewGeometry(candidate_id, image_id);
      if (GeometryUsable(existing, options_.min_num_matches)) {
        continue;
      }
      if (session->ExistsMatches(candidate_id, image_id)) {
        session->DeleteMatches(candidate_id, image_id);
      }
      session->DeleteTwoViewGeometry(candidate_id, image_id);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to clear weak pairs for image "
                 << image_id << ": " << e.what();
  }
  if (matcher == nullptr ||
      !matcher->MatchSpecificPairs(image_id, candidate_ids, cache)) {
    LOG(WARNING) << "OnlineLoopCloser: batch matching image " << image_id
                 << " against " << candidate_ids.size() << " candidate(s) failed";
    return false;
  }
  for (const auto& [candidate_id, score] : candidates) {
    OnlineLoopPair loop_pair;
    const bool ok = VerifyCandidate(image_id,
                                    candidate_id,
                                    score,
                                    reconstruction,
                                    LoopGate(),
                                    &loop_pair);
    if (loop_pair.image_id2 != kInvalidImageId) {
      if (!has_attempt || BetterAttempt(loop_pair, best_attempt)) {
        best_attempt = loop_pair;
        has_attempt = true;
      }
    }
    if (!ok) {
      continue;
    }
    confirmed_loops_.push_back(loop_pair);
    last_loop_image_id_ = image_id;
    confirmed = true;
    confirmed_pair = loop_pair;
    LOG(INFO) << "OnlineLoopCloser: CONFIRMED loop " << image_id << "-"
              << candidate_id << " matches=" << loop_pair.num_matches
              << " inliers=" << loop_pair.num_inliers
              << " n3d=" << loop_pair.num_3d_correspondences
              << " n3d_inliers=" << loop_pair.num_3d_inliers
              << " correction_m=" << loop_pair.correction_m
              << " correction_deg=" << loop_pair.correction_deg;
    RecordPair(loop_pair, "confirmed");
    break;
  }

  if (!confirmed) {
    if (has_attempt) {
      RecordPair(best_attempt, "verify_failed");
    } else {
      timing_->Record(std::to_string(image_id),
                      timer.ElapsedMicroSeconds() / 1000.0,
                      {"-",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "0",
                       "no_candidate",
                       "-"});
    }
    return false;
  }

  if (!options_.correct_local || obs_manager == nullptr ||
      !confirmed_pair.has_loop_pose) {
    return false;
  }

  const bool ok = CorrectLoop(reconstruction, obs_manager, confirmed_pair);
  if (ok) {
    confirmed_loops_.back() = confirmed_pair;
  }
  return ok;
}

bool OnlineLoopCloser::InitMixVprModel() {
  if (disabled_) {
    return false;
  }
  if (mixvpr_ready_) {
    return mixvpr_ != nullptr;
  }
  mixvpr_ready_ = true;

#ifndef VSLAM_MIXVPR_TENSORRT_ENABLED
  LOG(ERROR) << "OnlineLoopCloser: MixVPR requires TensorRT. Disabling "
                "loop detection";
  disabled_ = true;
  return false;
#else
  const std::filesystem::path model_path = options_.mixvpr_engine_path;
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    LOG(ERROR) << "OnlineLoopCloser: MixVPR TensorRT engine not found at "
               << (model_path.empty() ? std::string("<empty>")
                                      : model_path.string())
               << ". Set loop_mixvpr_engine_path to a valid .engine file";
    disabled_ = true;
    return false;
  }

  try {
    LOG(INFO) << "OnlineLoopCloser: loading MixVPR from " << model_path
              << " gpu=" << static_cast<int>(options_.mixvpr_use_gpu)
              << " gpu_index=" << options_.mixvpr_gpu_index;
    mixvpr_ = std::make_unique<MixVprEncoder>(
        model_path.string(),
        options_.mixvpr_use_gpu,
        options_.mixvpr_gpu_index);
    LOG(INFO) << "OnlineLoopCloser: MixVPR ready desc_dim=" << mixvpr_->DescDim()
              << " input=" << kMixVprInputSize << "x" << kMixVprInputSize
              << " used_gpu=" << static_cast<int>(mixvpr_->UsedGpu())
              << " backend=" << mixvpr_->Backend();
    timing_->AddMeta("mixvpr_desc_dim", std::to_string(mixvpr_->DescDim()));
    timing_->AddMeta("mixvpr_backend", mixvpr_->Backend());
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineLoopCloser: failed to load MixVPR: " << e.what()
               << ", disabling loop detection";
    mixvpr_.reset();
    disabled_ = true;
    return false;
  }
  return true;
#endif
}

bool OnlineLoopCloser::StoreMixVprDescriptor(const image_t image_id,
                                             const Eigen::VectorXf& desc) {
  if (mixvpr_ == nullptr) {
    return false;
  }
  if (desc.size() != kMixVprDescDim) {
    LOG(WARNING) << "OnlineLoopCloser: cannot store MixVPR descriptor for image "
                 << image_id << " desc_dim=" << desc.size() << " expected "
                 << kMixVprDescDim;
    return false;
  }
  try {
    DatabaseSession session(database_path_);
    if (session->ExistsMixVprDescriptor(image_id)) {
      return true;
    }
    session->WriteMixVprDescriptor(image_id, desc);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to write MixVPR descriptor for "
                 << image_id << ": " << e.what();
    return false;
  }
  return true;
}

std::vector<std::pair<image_t, float>> OnlineLoopCloser::DetectLoop(
    const image_t image_id,
    const Eigen::VectorXf& query,
    const Reconstruction& reconstruction) {
  std::vector<std::pair<image_t, float>> result;
  if (query.size() != kMixVprDescDim) {
    return result;
  }
  if (last_loop_image_id_ != kInvalidImageId &&
      image_id < last_loop_image_id_ +
                     static_cast<image_t>(options_.cooldown_num_images)) {
    return result;
  }
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.Image(image_id).HasPose()) {
    return result;
  }

  const Eigen::Vector3d current_center =
      reconstruction.Image(image_id).ProjectionCenter();
  const Eigen::Vector3d current_view =
      OpticalAxisWorld(reconstruction.Image(image_id));
  const double max_distance = options_.max_distance;
  const double max_distance_sq =
      max_distance > 0.0 ? max_distance * max_distance : -1.0;
  const double max_view_angle_deg = options_.max_view_angle_deg;
  const double min_view_cos =
      max_view_angle_deg > 0.0 && max_view_angle_deg < 180.0
          ? std::cos(max_view_angle_deg * 3.14159265358979323846 / 180.0)
          : -2.0;
  const std::unordered_set<image_t> covis =
      CovisibilityGroup(image_id, reconstruction);

  const auto usable = [&](const image_t other_id) {
    if (other_id == image_id) {
      return false;
    }
    if (other_id + static_cast<image_t>(std::max(options_.min_frame_gap, 0)) >
        image_id) {
      return false;
    }
    if (covis.count(other_id) > 0) {
      return false;
    }
    if (!reconstruction.ExistsImage(other_id) ||
        !reconstruction.Image(other_id).HasPose()) {
      return false;
    }
    const Image& other = reconstruction.Image(other_id);
    if (options_.min_candidate_points > 0 &&
        other.NumPoints3D() <
            static_cast<size_t>(options_.min_candidate_points)) {
      return false;
    }
    if (max_distance_sq >= 0.0) {
      if ((other.ProjectionCenter() - current_center).squaredNorm() >
          max_distance_sq) {
        return false;
      }
    }
    if (min_view_cos > -1.5) {
      const Eigen::Vector3d other_view = OpticalAxisWorld(other);
      const double na = current_view.norm();
      const double nb = other_view.norm();
      if (na < 1e-12 || nb < 1e-12 ||
          current_view.dot(other_view) < min_view_cos * na * nb) {
        return false;
      }
    }
    return true;
  };

  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(reconstruction.NumRegImages());
  for (const image_t other_id : reconstruction.RegImageIds()) {
    if (usable(other_id)) {
      candidate_ids.push_back(other_id);
    }
  }

  size_t num_mixvpr_images = 0;
  std::vector<std::pair<image_t, Eigen::VectorXf>> mixvpr_entries;
  try {
    DatabaseSession session(database_path_);
    num_mixvpr_images = session->NumMixVprDescriptors();
    if (num_mixvpr_images == 0 || candidate_ids.empty()) {
      if (candidate_ids.empty()) {
        LOG(INFO) << "OnlineLoopCloser: image " << image_id
                  << " mixvpr_images=" << num_mixvpr_images
                  << " no loop candidates (within " << max_distance
                  << "m, view<" << max_view_angle_deg << "deg, score>="
                  << options_.min_mixvpr_score
                  << ", excluding first-level covis)";
      }
      return result;
    }
    mixvpr_entries = session->ReadMixVprDescriptors(candidate_ids);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read MixVPR descriptors: "
                 << e.what();
    return result;
  }
  if (mixvpr_entries.empty()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " mixvpr_images=" << num_mixvpr_images
              << " geo_candidates=" << candidate_ids.size()
              << " no MixVPR rows for candidates";
    return result;
  }

  // Candidates already geometry-filtered; rank all loaded descriptors.
  std::vector<std::pair<image_t, float>> ranked =
      RankMixVprMatches(query, mixvpr_entries, /*usable=*/{}, 0);
  if (options_.min_mixvpr_score > 0.0) {
    const float min_score = static_cast<float>(options_.min_mixvpr_score);
    ranked.erase(std::remove_if(ranked.begin(),
                                ranked.end(),
                                [min_score](const auto& item) {
                                  return item.second < min_score;
                                }),
                 ranked.end());
  }
  if (options_.topk > 0 &&
      static_cast<int>(ranked.size()) > options_.topk) {
    ranked.resize(static_cast<size_t>(options_.topk));
  }
  if (ranked.empty()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " mixvpr_images=" << num_mixvpr_images
              << " geo_candidates=" << candidate_ids.size()
              << " loaded=" << mixvpr_entries.size()
              << " no loop candidates (score>=" << options_.min_mixvpr_score
              << ")";
    return result;
  }

  LOG(INFO) << "OnlineLoopCloser: image " << image_id
            << " mixvpr_images=" << num_mixvpr_images
            << " geo_candidates=" << candidate_ids.size()
            << " loaded=" << mixvpr_entries.size()
            << " topk=" << ranked.size() << " best=" << ranked.front().first
            << " score=" << ranked.front().second;
  return ranked;
}

bool OnlineLoopCloser::VerifyCandidate(const image_t image_id,
                                       const image_t candidate_id,
                                       const float retrieval_score,
                                       const Reconstruction& reconstruction,
                                       const CandidateGate& gate,
                                       OnlineLoopPair* loop_pair) {
  if (loop_pair == nullptr) {
    return false;
  }

  loop_pair->image_id1 = image_id;
  loop_pair->image_id2 = candidate_id;
  loop_pair->retrieval_score = retrieval_score;
  loop_pair->has_loop_pose = false;

  FeatureMatches matches;
  TwoViewGeometry geometry;
  FeatureDescriptors desc_current;
  FeatureDescriptors desc_candidate;
  LoopScaleCheck scale_check(options_.scale_check_max_ratio);
  try {
    DatabaseSession session(database_path_);
    if (session->ExistsMatches(candidate_id, image_id)) {
      matches = session->ReadMatches(candidate_id, image_id);
    }
    loop_pair->num_matches = matches.size();
    if (matches.size() < gate.min_matches) {
      LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-"
                << candidate_id << ", matches=" << matches.size();
      return false;
    }
    if (session->ExistsTwoViewGeometry(candidate_id, image_id)) {
      geometry = session->ReadTwoViewGeometry(candidate_id, image_id);
    }
    loop_pair->num_inliers = geometry.inlier_matches.size();
    if (!GeometryUsable(geometry, static_cast<int>(gate.min_matches))) {
      LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-"
                << candidate_id
                << ", essential inliers=" << geometry.inlier_matches.size();
      return false;
    }
    if (session->ExistsDescriptors(image_id)) {
      desc_current = session->ReadDescriptors(image_id);
    }
    if (session->ExistsDescriptors(candidate_id)) {
      desc_candidate = session->ReadDescriptors(candidate_id);
    }
    scale_check.Load(*session, image_id);
    scale_check.Load(*session, candidate_id);
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read pair " << candidate_id
                 << "-" << image_id << ": " << e.what();
    return false;
  }

  const FeatureMatches& verify_matches = geometry.inlier_matches.empty()
                                             ? matches
                                             : geometry.inlier_matches;
  const Image& current = reconstruction.Image(image_id);
  const Image& candidate = reconstruction.Image(candidate_id);
  if (!current.HasCameraPtr() || !candidate.HasCameraPtr() ||
      !candidate.HasPose() ||
      (gate.require_current_pose && !current.HasPose())) {
    return false;
  }

  std::vector<LoopPnPCorr> corrs;
  std::unordered_set<point2D_t> used_current;
  std::unordered_set<point3D_t> used_points;
  CollectLoopPnPCorrs(current,
                      candidate,
                      reconstruction,
                      verify_matches,
                      &used_current,
                      &used_points,
                      &corrs);
  const size_t num_candidate_corrs = corrs.size();

  std::vector<image_t> loop_ids;
  std::unordered_map<image_t, FeatureDescriptors> loop_desc;
  bool loop_map_loaded = false;
  const auto LoadLoopMap = [&]() {
    if (loop_map_loaded) {
      return;
    }
    loop_map_loaded = true;
    loop_ids = LoopMapImages(candidate_id, image_id, reconstruction);
    loop_desc.emplace(candidate_id, desc_candidate);
    try {
      DatabaseSession session(database_path_);
      for (const image_t id : loop_ids) {
        if (id != candidate_id && session->ExistsDescriptors(id)) {
          loop_desc.emplace(id, session->ReadDescriptors(id));
        }
        scale_check.Load(*session, id);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "OnlineLoopCloser: failed to read loop-map descriptors "
                   << "for " << image_id << "-" << candidate_id << ": "
                   << e.what();
    }
  };

  // The candidate may have triangulated few of its matched keypoints; borrow
  // the 3D points of its covisibility island instead of rejecting.
  size_t num_island = 0;
  const size_t island_target =
      2 * std::max(gate.min_3d_corrs,
                   static_cast<size_t>(std::max(options_.min_num_3d_inliers, 0)));
  if (options_.island_2d3d && corrs.size() < island_target && geometry.E) {
    LoadLoopMap();
    const double focal = std::max(current.CameraPtr()->MeanFocalLength(), 1.0);
    num_island = SearchLoopIslandByDescriptor(
        current,
        candidate,
        reconstruction,
        *geometry.E,
        loop_ids,
        loop_desc,
        desc_current,
        std::atan(std::max(options_.island_epipolar_px, 0.1) / focal),
        &used_current,
        &used_points,
        &corrs);
  }
  loop_pair->num_3d_correspondences = corrs.size();
  if (corrs.size() < gate.min_3d_corrs) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", 2D-3D=" << corrs.size() << " (candidate="
              << num_candidate_corrs << " island=" << num_island << ")"
              << " inliers=" << geometry.inlier_matches.size();
    return false;
  }

  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  const auto FillPoints = [&]() {
    points2D.clear();
    points3D.clear();
    points2D.reserve(corrs.size());
    points3D.reserve(corrs.size());
    for (const LoopPnPCorr& corr : corrs) {
      points2D.push_back(corr.xy_current);
      points3D.push_back(reconstruction.Point3D(corr.point3D_id).xyz);
    }
  };
  FillPoints();

  Camera camera = *current.CameraPtr();
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.estimate_focal_length = false;
  abs_pose_options.ransac_options.max_error = options_.pnp_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio = 0.1;
  AbsolutePoseRefinementOptions refine_options;
  refine_options.refine_focal_length = false;
  refine_options.refine_extra_params = false;

  Rigid3d cam_from_world;
  size_t num_ransac_inliers = 0;
  std::vector<char> inliers;
  const bool pnp_ok = EstimateAbsolutePose(abs_pose_options,
                                           points2D,
                                           points3D,
                                           &cam_from_world,
                                           &camera,
                                           &num_ransac_inliers,
                                           &inliers);
  loop_pair->num_3d_inliers = pnp_ok ? num_ransac_inliers : 0;
  if (!pnp_ok || num_ransac_inliers <
                     static_cast<size_t>(options_.min_num_3d_inliers)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp ransac failed 2D-3D=" << corrs.size()
              << " island=" << num_island
              << " inliers=" << loop_pair->num_3d_inliers;
    return false;
  }
  if (!RefineAbsolutePose(
          refine_options, inliers, points2D, points3D, &cam_from_world, &camera)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp refine failed";
    return false;
  }

  // Grow matches by projecting the whole loop-side local map.
  LoadLoopMap();
  const size_t num_guided = SearchLoopByProjection(current,
                                                   reconstruction,
                                                   cam_from_world,
                                                   loop_ids,
                                                   loop_desc,
                                                   desc_current,
                                                   options_.guided_radius_px,
                                                   scale_check,
                                                   &used_current,
                                                   &used_points,
                                                   &corrs);
  FillPoints();
  // Reprojection inliers whose keypoint scale disagrees with the loop-side
  // observations are dropped; returns how many.
  const auto RemoveScaleOutliers = [&]() {
    size_t num_removed = 0;
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (inliers[i] && !scale_check.Consistent(reconstruction,
                                                image_id,
                                                corrs[i].idx_current,
                                                corrs[i].point3D_id,
                                                cam_from_world,
                                                camera)) {
        inliers[i] = 0;
        ++num_removed;
      }
    }
    return num_removed;
  };
  inliers = ComputePnPInliers(
      corrs, reconstruction, cam_from_world, camera, options_.pnp_max_error);
  size_t num_scale_rejected = RemoveScaleOutliers();
  if (CountInliers(inliers) >=
          static_cast<size_t>(options_.min_num_3d_inliers) &&
      RefineAbsolutePose(refine_options,
                         inliers,
                         points2D,
                         points3D,
                         &cam_from_world,
                         &camera)) {
    inliers = ComputePnPInliers(
        corrs, reconstruction, cam_from_world, camera, options_.pnp_max_error);
    num_scale_rejected = RemoveScaleOutliers();
  }
  const size_t num_pnp_inliers = CountInliers(inliers);

  const bool has_pose_before = current.HasPose();
  if (has_pose_before) {
    const Rigid3d& pose_before = current.CamFromWorld();
    loop_pair->correction_m =
        (cam_from_world.TgtOriginInSrc() - pose_before.TgtOriginInSrc()).norm();
    loop_pair->correction_deg =
        RotationDiffDeg(cam_from_world.rotation(), pose_before.rotation());
  } else {
    loop_pair->correction_m = 0.0;
    loop_pair->correction_deg = 0.0;
  }
  double tdir_vs_essential_deg = -1.0;
  loop_pair->rot_vs_essential_deg = RotationVsEssentialDeg(
      geometry, candidate, current, cam_from_world, &tdir_vs_essential_deg);
  const double baseline_m =
      (cam_from_world.TgtOriginInSrc() -
       candidate.CamFromWorld().TgtOriginInSrc())
          .norm();
  LOG(INFO) << "OnlineLoopCloser: pnp " << image_id << "-" << candidate_id
            << " 2D-3D=" << corrs.size() << " island=" << num_island
            << " guided=" << num_guided
            << " ransac_inliers=" << num_ransac_inliers
            << " inliers=" << num_pnp_inliers
            << " scale_rejected=" << num_scale_rejected
            << " loop_map=" << loop_ids.size()
            << " baseline_m=" << baseline_m
            << " correction_m=" << loop_pair->correction_m
            << " correction_deg=" << loop_pair->correction_deg
            << " rot_vs_E_deg=" << loop_pair->rot_vs_essential_deg
            << " tdir_vs_E_deg=" << tdir_vs_essential_deg;

  if (num_pnp_inliers < gate.min_inliers) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp weak inliers=" << num_pnp_inliers;
    return false;
  }
  if (gate.max_rot_vs_essential_deg > 0.0 &&
      loop_pair->rot_vs_essential_deg > gate.max_rot_vs_essential_deg) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", pnp rotation disagrees with essential by "
              << loop_pair->rot_vs_essential_deg << " deg";
    return false;
  }
  if (gate.check_correction &&
      (loop_pair->correction_m > options_.max_correction_m ||
       loop_pair->correction_deg > options_.max_correction_deg)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", correction too large " << loop_pair->correction_m
              << " m / " << loop_pair->correction_deg << " deg";
    return false;
  }

  double pnp_rot_sigma = 0.0;
  double pnp_trans_sigma = 0.0;
  const bool has_pnp_sigma = EstimatePnPSigmas(corrs,
                                               inliers,
                                               reconstruction,
                                               cam_from_world,
                                               camera,
                                               &pnp_rot_sigma,
                                               &pnp_trans_sigma);
  const double min_rot_sigma =
      DegToRad(std::max(options_.pose_graph_loop_min_rot_sigma_deg, 1e-3));
  const double min_trans_sigma =
      std::max(options_.pose_graph_loop_min_trans_sigma_m, 1e-4);
  // Degenerate PnP information: keep the loop but trust it like odometry.
  loop_pair->rot_sigma =
      has_pnp_sigma
          ? std::max(kLoopSigmaInflation * pnp_rot_sigma, min_rot_sigma)
          : DegToRad(std::max(options_.pose_graph_odom_rot_sigma_deg, 1e-3));
  loop_pair->trans_sigma =
      has_pnp_sigma
          ? std::max(kLoopSigmaInflation * pnp_trans_sigma, min_trans_sigma)
          : std::max(options_.pose_graph_odom_trans_sigma_m, 1e-4) +
                options_.pose_graph_odom_trans_sigma_ratio * baseline_m;

  loop_pair->has_loop_pose = true;
  loop_pair->loop_cam_from_world = cam_from_world;
  loop_pair->loop_observations.clear();
  loop_pair->loop_observations.reserve(num_pnp_inliers);
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (inliers[i]) {
      loop_pair->loop_observations.emplace_back(corrs[i].idx_current,
                                                corrs[i].point3D_id);
    }
  }
  LOG(INFO) << "OnlineLoopCloser: accept " << image_id << "-" << candidate_id
            << " by pnp inliers=" << num_pnp_inliers
            << " correction_m=" << loop_pair->correction_m
            << " correction_deg=" << loop_pair->correction_deg
            << " pnp_sigma_deg=" << RadToDeg(pnp_rot_sigma)
            << " pnp_sigma_m=" << pnp_trans_sigma
            << " edge_sigma_deg=" << RadToDeg(loop_pair->rot_sigma)
            << " edge_sigma_m=" << loop_pair->trans_sigma;
  return true;
}

bool OnlineLoopCloser::Relocalize(const image_t image_id,
                                  const DatabaseCache& cache,
                                  const Reconstruction& reconstruction,
                                  OnlineFeatureMatcher* matcher,
                                  const std::string& image_abs_path,
                                  const Bitmap* bitmap,
                                  OnlineLoopPair* loop_pair) {
  if (loop_pair == nullptr || matcher == nullptr) {
    return false;
  }
  if (!IsLoopClosureActive()) {
    LOG(WARNING) << "OnlineLoopCloser: relocalization needs an active loop "
                    "closer (MixVPR)";
    return false;
  }
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.Image(image_id).HasCameraPtr()) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize image " << image_id
                 << " is not in the reconstruction";
    return false;
  }
  if (reconstruction.Image(image_id).HasPose()) {
    LOG(INFO) << "OnlineLoopCloser: image " << image_id
              << " already has a pose, skip relocalization";
    return false;
  }

  Eigen::VectorXf query;
  if (!EncodeQuery(image_id, image_abs_path, bitmap, &query)) {
    return false;
  }
  if (!StoreMixVprDescriptor(image_id, query)) {
    return false;
  }

  std::vector<std::pair<image_t, Eigen::VectorXf>> entries;
  try {
    DatabaseSession session(database_path_);
    entries = session->ReadAllMixVprDescriptors();
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize failed to read MixVPR "
                    "descriptors: "
                 << e.what();
    return false;
  }
  if (entries.empty()) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " has no stored map descriptors";
    return false;
  }

  // No pose yet, so retrieval cannot be gated by distance / view angle. Only
  // frames that carry a pose are valid PnP targets.
  const auto usable = [&](const image_t id) {
    return id != image_id && reconstruction.ExistsImage(id) &&
           reconstruction.Image(id).HasPose();
  };
  std::vector<std::pair<image_t, float>> ranked =
      RankMixVprMatches(query, entries, usable, options_.reloc_topk);
  if (options_.reloc_min_mixvpr_score > 0.0) {
    const float min_score =
        static_cast<float>(options_.reloc_min_mixvpr_score);
    ranked.erase(std::remove_if(ranked.begin(),
                                ranked.end(),
                                [min_score](const auto& item) {
                                  return item.second < min_score;
                                }),
                 ranked.end());
  }
  if (ranked.empty()) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " descriptors=" << entries.size()
              << " no candidate above score " << options_.reloc_min_mixvpr_score;
    return false;
  }

  std::vector<image_t> candidate_ids;
  candidate_ids.reserve(ranked.size());
  for (const auto& [id, score] : ranked) {
    candidate_ids.push_back(id);
  }
  if (!matcher->MatchSpecificPairs(image_id, candidate_ids, cache)) {
    LOG(WARNING) << "OnlineLoopCloser: relocalize image " << image_id
                 << " matching against " << candidate_ids.size()
                 << " candidate(s) failed";
    return false;
  }

  const CandidateGate gate = RelocGate();
  OnlineLoopPair best;
  bool has_best = false;
  for (const auto& [candidate_id, score] : ranked) {
    OnlineLoopPair attempt;
    const bool ok = VerifyCandidate(
        image_id, candidate_id, score, reconstruction, gate, &attempt);
    LOG(INFO) << "OnlineLoopCloser: relocalize " << image_id << "-"
              << candidate_id << " score=" << score
              << " matches=" << attempt.num_matches
              << " 2D-3D=" << attempt.num_3d_correspondences
              << " inliers=" << attempt.num_3d_inliers
              << " accepted=" << static_cast<int>(ok);
    if (!ok) {
      continue;
    }
    if (!has_best || attempt.num_3d_inliers > best.num_3d_inliers ||
        (attempt.num_3d_inliers == best.num_3d_inliers &&
         attempt.num_inliers > best.num_inliers)) {
      best = attempt;
      has_best = true;
    }
  }
  if (!has_best) {
    LOG(INFO) << "OnlineLoopCloser: relocalize image " << image_id
              << " failed, candidates=" << candidate_ids.size();
    return false;
  }

  *loop_pair = best;
  // A successful relocalization is not a loop: keep it out of the loop list so
  // the cooldown and the confirmed-loop log stay meaningful.
  last_loop_image_id_ = image_id;
  LOG(INFO) << "OnlineLoopCloser: RELOCALIZED " << image_id << "-"
            << loop_pair->image_id2 << " inliers=" << loop_pair->num_3d_inliers
            << " matches=" << loop_pair->num_matches
            << " center="
            << loop_pair->loop_cam_from_world.TgtOriginInSrc().transpose();
  return true;
}

point3D_t ResolveMergedPoint(
    const std::unordered_map<point3D_t, point3D_t>& remap, point3D_t id) {
  while (true) {
    const auto it = remap.find(id);
    if (it == remap.end()) {
      return id;
    }
    id = it->second;
  }
}

bool MergeKeepLoopPoint(ObservationManager* obs_manager,
                        Reconstruction& reconstruction,
                        std::unordered_map<point3D_t, point3D_t>* remap,
                        const point3D_t cur_pid,
                        const point3D_t loop_pid) {
  const Eigen::Vector3d loop_xyz = reconstruction.Point3D(loop_pid).xyz;
  try {
    const point3D_t merged = obs_manager->MergePoints3D(cur_pid, loop_pid);
    reconstruction.Point3D(merged).xyz = loop_xyz;
    (*remap)[cur_pid] = merged;
    (*remap)[loop_pid] = merged;
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: merge " << cur_pid << "+" << loop_pid
                 << " failed: " << e.what();
    return false;
  }
}

bool AttachLoopObservation(ObservationManager* obs_manager,
                           const point3D_t loop_pid,
                           const image_t image_id,
                           const point2D_t idx) {
  try {
    obs_manager->AddObservation(loop_pid, TrackElement(image_id, idx));
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: attach point " << loop_pid
                 << " to image " << image_id << " failed: " << e.what();
    return false;
  }
}

struct ProjectionFuseStats {
  size_t num_fused = 0;
  size_t num_attached = 0;
  size_t num_rejected_dist = 0;
  size_t num_rejected_desc = 0;
  size_t num_rejected_scale = 0;
};

// Project loop-side points into `image_ids` and merge or attach descriptor
// matches inside the fuse radius.
ProjectionFuseStats FuseLoopMapByProjection(
    const std::string& database_path,
    const OnlineLoopCloserOptions& options,
    Reconstruction& reconstruction,
    ObservationManager* obs_manager,
    const std::unordered_set<point3D_t>& loop_point_ids,
    const std::vector<image_t>& image_ids,
    std::unordered_map<point3D_t, point3D_t>* remap) {
  ProjectionFuseStats stats;
  if (obs_manager == nullptr || image_ids.empty() || loop_point_ids.empty()) {
    return stats;
  }

  std::unordered_map<image_t, FeatureDescriptors> descriptors;
  std::unordered_map<point3D_t, std::vector<TrackElement>> loop_refs;
  LoopScaleCheck scale_check(options.scale_check_max_ratio);
  try {
    DatabaseSession session(database_path);
    const auto Load = [&](const image_t id) {
      if (descriptors.count(id) == 0 && session->ExistsDescriptors(id)) {
        descriptors.emplace(id, session->ReadDescriptors(id));
      }
      scale_check.Load(*session, id);
    };
    for (const image_t id : image_ids) {
      Load(id);
    }
    std::vector<TrackElement> track_elements;
    for (const point3D_t point3D_id_raw : loop_point_ids) {
      const point3D_t point3D_id = ResolveMergedPoint(*remap, point3D_id_raw);
      if (!reconstruction.ExistsPoint3D(point3D_id) ||
          loop_refs.count(point3D_id_raw) > 0) {
        continue;
      }
      track_elements = reconstruction.Point3D(point3D_id).track.Elements();
      std::sort(track_elements.begin(),
                track_elements.end(),
                [](const TrackElement& a, const TrackElement& b) {
                  return a.image_id > b.image_id;
                });
      std::vector<TrackElement> refs;
      for (const TrackElement& el : track_elements) {
        if (refs.size() >= kMaxFuseRefDescriptors) {
          break;
        }
        if (!reconstruction.ExistsImage(el.image_id) ||
            !reconstruction.Image(el.image_id).HasPose()) {
          continue;
        }
        Load(el.image_id);
        const auto ref_it = descriptors.find(el.image_id);
        if (ref_it == descriptors.end() ||
            static_cast<int>(el.point2D_idx) >= ref_it->second.data.rows()) {
          continue;
        }
        refs.push_back(el);
      }
      if (!refs.empty()) {
        loop_refs.emplace(point3D_id_raw, std::move(refs));
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read fusion descriptors: "
                 << e.what();
    return stats;
  }

  const double fuse_radius_sq = options.fuse_radius_px * options.fuse_radius_px;
  for (const image_t id : image_ids) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    Image& image = reconstruction.Image(id);
    const auto desc_it = descriptors.find(id);
    if (!image.HasPose() || !image.HasCameraPtr() ||
        desc_it == descriptors.end()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();
    const double cell = std::max(1.0, options.fuse_radius_px);
    const int grid_w = static_cast<int>(camera.width / cell) + 1;
    const int grid_h = static_cast<int>(camera.height / cell) + 1;
    std::vector<std::vector<point2D_t>> grid(grid_w * grid_h);
    for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
      const Eigen::Vector2d& xy = image.Point2D(i).xy;
      const int gx = std::clamp(static_cast<int>(xy.x() / cell), 0, grid_w - 1);
      const int gy = std::clamp(static_cast<int>(xy.y() / cell), 0, grid_h - 1);
      grid[gy * grid_w + gx].push_back(i);
    }
    std::unordered_set<point2D_t> used;
    for (const auto& [loop_pid_raw, refs] : loop_refs) {
      const point3D_t loop_pid = ResolveMergedPoint(*remap, loop_pid_raw);
      if (!reconstruction.ExistsPoint3D(loop_pid) ||
          image.HasPoint3D(loop_pid) || refs.empty()) {
        continue;
      }
      const Eigen::Vector3d loop_xyz = reconstruction.Point3D(loop_pid).xyz;
      const auto proj = ProjectCam(camera, cam_from_world * loop_xyz);
      if (!proj || !InsideImage(camera, *proj)) {
        continue;
      }
      float best_l2 = kLoopDescMaxL2Sq;
      point2D_t best_idx = kInvalidPoint2DIdx;
      bool any_in_radius = false;
      bool any_scale_ok = false;
      const int pgx = static_cast<int>(proj->x() / cell);
      const int pgy = static_cast<int>(proj->y() / cell);
      for (int gy = std::max(0, pgy - 1); gy <= std::min(grid_h - 1, pgy + 1);
           ++gy) {
        for (int gx = std::max(0, pgx - 1); gx <= std::min(grid_w - 1, pgx + 1);
             ++gx) {
          for (const point2D_t i : grid[gy * grid_w + gx]) {
            if (used.count(i) > 0 ||
                (image.Point2D(i).xy - *proj).squaredNorm() >= fuse_radius_sq) {
              continue;
            }
            any_in_radius = true;
            if (!scale_check.Consistent(
                    reconstruction, id, i, loop_pid, cam_from_world, camera)) {
              continue;
            }
            any_scale_ok = true;
            float l2 = std::numeric_limits<float>::infinity();
            for (const TrackElement& ref : refs) {
              const auto ref_desc_it = descriptors.find(ref.image_id);
              if (ref_desc_it == descriptors.end()) {
                continue;
              }
              l2 = std::min(l2,
                            DescriptorL2Sq(ref_desc_it->second,
                                           ref.point2D_idx,
                                           desc_it->second,
                                           i));
            }
            if (l2 < best_l2) {
              best_l2 = l2;
              best_idx = i;
            }
          }
        }
      }
      if (best_idx == kInvalidPoint2DIdx) {
        if (any_scale_ok) {
          ++stats.num_rejected_desc;
        } else if (any_in_radius) {
          ++stats.num_rejected_scale;
        }
        continue;
      }
      const Point2D& point2D = image.Point2D(best_idx);
      if (point2D.HasPoint3D()) {
        const point3D_t cur_pid = point2D.point3D_id;
        if (cur_pid == loop_pid || !reconstruction.ExistsPoint3D(cur_pid)) {
          continue;
        }
        if ((reconstruction.Point3D(cur_pid).xyz - loop_xyz).norm() >
            options.fuse_max_point_distance) {
          ++stats.num_rejected_dist;
          continue;
        }
        if (MergeKeepLoopPoint(
                obs_manager, reconstruction, remap, cur_pid, loop_pid)) {
          used.insert(best_idx);
          ++stats.num_fused;
        }
      } else if (AttachLoopObservation(obs_manager, loop_pid, id, best_idx)) {
        used.insert(best_idx);
        ++stats.num_attached;
      }
    }
  }
  return stats;
}

bool OnlineLoopCloser::CorrectLoop(Reconstruction& reconstruction,
                                   ObservationManager* obs_manager,
                                   const OnlineLoopPair& loop_pair) {
  if (obs_manager == nullptr || !loop_pair.has_loop_pose) {
    return false;
  }
  const image_t current_id = loop_pair.image_id1;
  const image_t loop_id = loop_pair.image_id2;
  if (!reconstruction.ExistsImage(current_id) ||
      !reconstruction.ExistsImage(loop_id) ||
      !reconstruction.Image(current_id).HasPose() ||
      !reconstruction.Image(loop_id).HasPose()) {
    return false;
  }

  // Old map = loop candidate, its local map, and everything registered before.
  const std::vector<image_t> loop_map_ids =
      LoopMapImages(loop_id, current_id, reconstruction);
  std::unordered_set<image_t> anchor_set =
      CovisibilityGroup(loop_id, reconstruction);
  anchor_set.insert(loop_map_ids.begin(), loop_map_ids.end());
  anchor_set.erase(current_id);
  image_t anchor_max = loop_id;
  for (const image_t id : anchor_set) {
    anchor_max = std::max(anchor_max, id);
  }

  std::vector<image_t> registered = reconstruction.RegImageIds();
  std::sort(registered.begin(), registered.end());
  std::vector<image_t> variable_ids;
  image_t boundary_id = kInvalidImageId;
  for (const image_t id : registered) {
    if (id <= anchor_max || anchor_set.count(id) > 0) {
      if (variable_ids.empty()) {
        boundary_id = id;
      }
      continue;
    }
    if (id <= current_id && reconstruction.Image(id).HasFramePtr()) {
      variable_ids.push_back(id);
    }
  }
  const std::unordered_set<image_t> variable_set(variable_ids.begin(),
                                                 variable_ids.end());
  if (variable_set.count(current_id) == 0) {
    LOG(INFO) << "OnlineLoopCloser: CorrectLoop skip " << current_id << "-"
              << loop_id << ", current inside old map (anchor_max="
              << anchor_max << ")";
    return false;
  }

  // Stage A: pose graph (odometry + covisibility + loop edges).
  std::unordered_map<image_t, Rigid3d> poses_old;
  for (const image_t id : variable_ids) {
    poses_old[id] = reconstruction.Image(id).CamFromWorld();
  }
  poses_old[loop_id] = reconstruction.Image(loop_id).CamFromWorld();
  if (boundary_id != kInvalidImageId) {
    poses_old[boundary_id] = reconstruction.Image(boundary_id).CamFromWorld();
  }
  const auto Rel = [&](const image_t i, const image_t j) {
    return poses_old.at(j) * Inverse(poses_old.at(i));
  };

  std::vector<PoseGraphEdge> edges;
  size_t num_vio_odom_edges = 0;
  image_t prev_id = boundary_id;
  for (const image_t id : variable_ids) {
    if (prev_id != kInvalidImageId) {
      std::optional<Rigid3d> vio_rel;
      if (options_.pose_graph_vio_odometry && vio_relative_pose_) {
        vio_rel = vio_relative_pose_(id, prev_id);
      }
      if (vio_rel) {
        ++num_vio_odom_edges;
      }
      edges.push_back(MakeOdometryEdge(
          prev_id, id, vio_rel ? *vio_rel : Rel(prev_id, id), 1.0));
    }
    prev_id = id;
  }
  // Old-map frames covisible with corrected frames (revisited areas) join as
  // fixed nodes, so the shared points they keep stay consistent.
  std::unordered_set<image_t> fixed_ids = {loop_id};
  if (boundary_id != kInvalidImageId) {
    fixed_ids.insert(boundary_id);
  }
  size_t num_covis_edges = 0;
  size_t num_old_covis_edges = 0;
  for (const image_t id : variable_ids) {
    for (const auto& [other_id, count] :
         CountCovisibilityPoints(id, reconstruction)) {
      if (count < options_.pose_graph_min_covis) {
        continue;
      }
      // More shared points, tighter edge; at most 2x tighter than odometry.
      const double covis_scale = std::clamp(
          std::sqrt(static_cast<double>(std::max(options_.pose_graph_min_covis, 1)) /
                    static_cast<double>(count)),
          0.5,
          1.0);
      if (variable_set.count(other_id) > 0) {
        if (other_id <= id) {
          continue;
        }
        edges.push_back(
            MakeOdometryEdge(id, other_id, Rel(id, other_id), covis_scale));
        ++num_covis_edges;
        continue;
      }
      if (other_id > current_id || !reconstruction.ExistsImage(other_id) ||
          !reconstruction.Image(other_id).HasPose()) {
        continue;
      }
      if (poses_old.count(other_id) == 0) {
        poses_old[other_id] = reconstruction.Image(other_id).CamFromWorld();
      }
      fixed_ids.insert(other_id);
      edges.push_back(
          MakeOdometryEdge(other_id, id, Rel(other_id, id), covis_scale));
      ++num_old_covis_edges;
    }
  }
  // Freeze the PnP relative pose against the loop frame's pre-correction pose.
  // Later graphs replay this measurement; they do not rebuild it from poses.
  const LoopConstraint current_loop{
      loop_id,
      current_id,
      loop_pair.loop_cam_from_world * Inverse(poses_old[loop_id]),
      loop_pair.loop_observations.size(),
      loop_pair.rot_sigma,
      loop_pair.trans_sigma};
  const size_t loop_edge_begin = edges.size();
  std::vector<LoopConstraint> loop_edge_constraints;
  size_t num_hist_loop_edges = AppendLoopEdges(reconstruction,
                                               variable_set,
                                               current_loop,
                                               &poses_old,
                                               &fixed_ids,
                                               &edges,
                                               &loop_edge_constraints);

  // Solve, then drop the historical loop edge most inconsistent with the rest
  // (whitened chi-square above the 99% quantile) and solve again. If the
  // current loop is the outlier, the correction is rejected.
  const PoseGraphOptions graph_options = MakePoseGraphOptions();
  const double outlier_chi2 = PoseGraphEdgeChi2Quantile(graph_options, 0.99);
  std::unordered_map<image_t, Rigid3d> poses;
  PoseGraphSummary pg;
  size_t num_dropped_loop_edges = 0;
  std::vector<LoopConstraint> dropped_loop_constraints;
  double current_loop_chi2 = -1.0;
  while (true) {
    poses = poses_old;
    pg = OptimizePoseGraph(graph_options, edges, fixed_ids, &poses);
    if (!pg.success) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop pose graph failed "
                   << current_id << "-" << loop_id << " nodes=" << pg.num_nodes
                   << " edges=" << pg.num_edges;
      return false;
    }
    size_t worst = edges.size();
    double worst_chi2 = outlier_chi2;
    for (size_t k = loop_edge_begin; k < edges.size(); ++k) {
      const LoopConstraint& c = loop_edge_constraints[k - loop_edge_begin];
      if (c.map_id == current_loop.map_id &&
          c.query_id == current_loop.query_id) {
        current_loop_chi2 = pg.edge_chi2[k];
      }
      if (pg.edge_chi2[k] > worst_chi2) {
        worst_chi2 = pg.edge_chi2[k];
        worst = k;
      }
    }
    if (worst == edges.size()) {
      break;
    }
    const LoopConstraint outlier = loop_edge_constraints[worst - loop_edge_begin];
    if (outlier.map_id == current_loop.map_id &&
        outlier.query_id == current_loop.query_id) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop reject " << current_id
                   << "-" << loop_id << ", loop edge chi2=" << worst_chi2
                   << " > " << outlier_chi2
                   << " against odometry and historical loops";
      return false;
    }
    LOG(INFO) << "OnlineLoopCloser: drop historical loop " << outlier.query_id
              << "-" << outlier.map_id << " chi2=" << worst_chi2 << " > "
              << outlier_chi2;
    dropped_loop_constraints.push_back(outlier);
    edges.erase(edges.begin() + static_cast<std::ptrdiff_t>(worst));
    loop_edge_constraints.erase(
        loop_edge_constraints.begin() +
        static_cast<std::ptrdiff_t>(worst - loop_edge_begin));
    --num_hist_loop_edges;
    ++num_dropped_loop_edges;
  }

  // Trajectory prior: frames between the loop and the current frame that the
  // graph does not move keep their pose.
  double rover_score = -1.0;
  double rover_threshold = -1.0;
  double rover_path_m = 0.0;
  if (options_.rover_enabled) {
    std::vector<Eigen::Vector3d> centers_before;
    std::vector<Eigen::Vector3d> centers_after;
    for (const image_t id : registered) {
      if (id < loop_id || id > current_id ||
          !reconstruction.Image(id).HasPose()) {
        continue;
      }
      const auto it_old = poses_old.find(id);
      const Rigid3d& before = it_old != poses_old.end()
                                  ? it_old->second
                                  : reconstruction.Image(id).CamFromWorld();
      const auto it_new = poses.find(id);
      const Rigid3d& after = it_new != poses.end() ? it_new->second : before;
      if (!centers_before.empty()) {
        rover_path_m += (before.TgtOriginInSrc() - centers_before.back()).norm();
      }
      centers_before.push_back(before.TgtOriginInSrc());
      centers_after.push_back(after.TgtOriginInSrc());
    }
    rover_score = AlignedTrajectoryRmse(
        centers_before, centers_after, graph_options.four_dof);
    rover_threshold =
        std::max(options_.rover_min_threshold_m,
                 options_.rover_threshold_ratio * rover_path_m);
    if (rover_score > rover_threshold) {
      LOG(WARNING) << "OnlineLoopCloser: CorrectLoop reject " << current_id
                   << "-" << loop_id << ", trajectory deformation "
                   << rover_score << " m > " << rover_threshold
                   << " m (frames=" << centers_before.size()
                   << " path_m=" << rover_path_m << ")";
      return false;
    }
  }
  const Rigid3d& cur_after = poses.at(current_id);
  const double cur_vs_pnp_m = (cur_after.TgtOriginInSrc() -
                               loop_pair.loop_cam_from_world.TgtOriginInSrc())
                                  .norm();
  const double cur_vs_pnp_deg = RotationDiffDeg(
      cur_after.rotation(), loop_pair.loop_cam_from_world.rotation());

  // Stage B: move each point with its reference (first observing) frame, so
  // points first seen by the old map keep their position.
  std::unordered_set<point3D_t> visited;
  size_t num_points_moved = 0;
  size_t num_points_kept = 0;
  for (const image_t id : variable_ids) {
    for (const Point2D& point2D : reconstruction.Image(id).Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(point2D.point3D_id) ||
          !visited.insert(point2D.point3D_id).second) {
        continue;
      }
      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      image_t ref_id = kInvalidImageId;
      for (const TrackElement& el : point3D.track.Elements()) {
        if (reconstruction.ExistsImage(el.image_id) &&
            reconstruction.Image(el.image_id).HasPose() &&
            (ref_id == kInvalidImageId || el.image_id < ref_id)) {
          ref_id = el.image_id;
        }
      }
      if (variable_set.count(ref_id) == 0) {
        ++num_points_kept;
        continue;
      }
      point3D.xyz =
          Inverse(poses.at(ref_id)) * (poses_old.at(ref_id) * point3D.xyz);
      ++num_points_moved;
    }
  }
  size_t num_poses_corrected = 0;
  for (const image_t id : variable_ids) {
    if (SetImagePose(reconstruction.Image(id), poses.at(id))) {
      ++num_poses_corrected;
    }
  }

  // Loop-side map points. Descriptor references are chosen later from the
  // newest observations on each track.
  std::unordered_set<point3D_t> loop_point_ids;
  for (const image_t id : anchor_set) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    const Image& image = reconstruction.Image(id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_point_ids.insert(point2D.point3D_id);
      }
    }
  }

  // Overlap window: corrected frames onto which enough of the loop-side map
  // projects, strongest first, capped by max_correct_connected.
  std::vector<std::pair<size_t, image_t>> overlap;
  for (const image_t id : variable_ids) {
    const Image& image = reconstruction.Image(id);
    if (!image.HasCameraPtr()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();
    size_t num_visible = 0;
    for (const point3D_t loop_pid : loop_point_ids) {
      const auto proj = ProjectCam(
          camera, cam_from_world * reconstruction.Point3D(loop_pid).xyz);
      if (proj && InsideImage(camera, *proj)) {
        ++num_visible;
      }
    }
    if (id == current_id || num_visible >= kMinOverlapProjections) {
      overlap.emplace_back(id == current_id ? loop_point_ids.size() + 1
                                            : num_visible,
                           id);
    }
  }
  std::sort(overlap.begin(), overlap.end(), std::greater<>());
  std::vector<image_t> connected;
  for (const auto& [num_visible, id] : overlap) {
    if (static_cast<int>(connected.size()) >=
        std::max(1, options_.max_correct_connected)) {
      break;
    }
    connected.push_back(id);
  }

  std::unordered_map<point3D_t, point3D_t> remap;

  // Stage C1: PnP inliers of the current frame.
  size_t num_fused_direct = 0;
  size_t num_attached_direct = 0;
  {
    const Image& current = reconstruction.Image(current_id);
    for (const auto& [idx, loop_pid_raw] : loop_pair.loop_observations) {
      const point3D_t loop_pid = ResolveMergedPoint(remap, loop_pid_raw);
      if (idx >= current.NumPoints2D() ||
          !reconstruction.ExistsPoint3D(loop_pid)) {
        continue;
      }
      const Point2D& point2D = current.Point2D(idx);
      if (point2D.HasPoint3D()) {
        if (point2D.point3D_id != loop_pid &&
            MergeKeepLoopPoint(obs_manager,
                               reconstruction,
                               &remap,
                               point2D.point3D_id,
                               loop_pid)) {
          ++num_fused_direct;
        }
      } else if (!current.HasPoint3D(loop_pid) &&
                 AttachLoopObservation(obs_manager, loop_pid, current_id, idx)) {
        ++num_attached_direct;
      }
    }
  }

  // Stage C2: project the loop-side map into the overlap window.
  const ProjectionFuseStats proj = FuseLoopMapByProjection(database_path_,
                                                           options_,
                                                           reconstruction,
                                                           obs_manager,
                                                           loop_point_ids,
                                                           connected,
                                                           &remap);
  const size_t num_fused_proj = proj.num_fused;
  const size_t num_attached_proj = proj.num_attached;
  const size_t num_rejected_dist = proj.num_rejected_dist;
  const size_t num_rejected_desc = proj.num_rejected_desc;

  last_correction_.current_id = current_id;
  last_correction_.loop_id = loop_id;
  // Every pose-graph-moved frame is refined and filtered; `connected` only
  // limits fusion. The same window is projected again after loop BA.
  last_correction_.variable_ids = variable_ids;
  last_correction_.anchor_ids = loop_map_ids;
  last_correction_.fusion_ids = connected;
  last_correction_.loop_point_image_ids.assign(anchor_set.begin(),
                                               anchor_set.end());

  LOG(INFO) << "OnlineLoopCloser: CorrectLoop " << current_id << "-" << loop_id
            << " variable=[" << variable_ids.front() << ".."
            << variable_ids.back() << "](" << variable_ids.size() << ")"
            << " boundary=" << boundary_id << " anchor_max=" << anchor_max
            << " edges=" << pg.num_edges
            << " vio_odom_edges=" << num_vio_odom_edges
            << " covis_edges=" << num_covis_edges
            << " old_covis_edges=" << num_old_covis_edges
            << " hist_loop_edges=" << num_hist_loop_edges
            << " dropped_loop_edges=" << num_dropped_loop_edges
            << " loop_chi2=" << current_loop_chi2
            << " rover_score=" << rover_score << "/" << rover_threshold
            << " rover_path_m=" << rover_path_m
            << " dof=" << (graph_options.four_dof ? 4 : 6)
            << " fixed_nodes=" << fixed_ids.size()
            << " cost=" << pg.initial_cost << "->" << pg.final_cost
            << " cur_vs_pnp_m=" << cur_vs_pnp_m
            << " cur_vs_pnp_deg=" << cur_vs_pnp_deg
            << " poses=" << num_poses_corrected
            << " points_moved=" << num_points_moved
            << " points_kept=" << num_points_kept
            << " fuse_direct=" << num_fused_direct
            << " attach_direct=" << num_attached_direct
            << " fuse_proj=" << num_fused_proj
            << " attach_proj=" << num_attached_proj
            << " fuse_rejected_dist=" << num_rejected_dist
            << " fuse_rejected_desc=" << num_rejected_desc
            << " fuse_rejected_scale=" << proj.num_rejected_scale
            << " window=" << connected.size() << " ["
            << *std::min_element(connected.begin(), connected.end()) << ".."
            << *std::max_element(connected.begin(), connected.end()) << "]";
  if (num_poses_corrected == 0) {
    return false;
  }
  for (const LoopConstraint& dropped : dropped_loop_constraints) {
    ForgetLoopConstraint(dropped.map_id, dropped.query_id);
  }
  RememberLoopConstraint(current_loop);
  return true;
}

void OnlineLoopCloser::FuseAfterLoopBA(Reconstruction& reconstruction,
                                       ObservationManager* obs_manager) {
  if (obs_manager == nullptr || last_correction_.fusion_ids.empty() ||
      last_correction_.loop_point_image_ids.empty()) {
    return;
  }

  std::unordered_set<point3D_t> loop_point_ids;
  for (const image_t id : last_correction_.loop_point_image_ids) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    for (const Point2D& point2D : reconstruction.Image(id).Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_point_ids.insert(point2D.point3D_id);
      }
    }
  }

  std::unordered_map<point3D_t, point3D_t> remap;
  const ProjectionFuseStats proj =
      FuseLoopMapByProjection(database_path_,
                              options_,
                              reconstruction,
                              obs_manager,
                              loop_point_ids,
                              last_correction_.fusion_ids,
                              &remap);
  LOG(INFO) << "OnlineLoopCloser: post-BA fuse " << last_correction_.current_id
            << "-" << last_correction_.loop_id
            << " window=" << last_correction_.fusion_ids.size()
            << " points=" << loop_point_ids.size()
            << " fuse_proj=" << proj.num_fused
            << " attach_proj=" << proj.num_attached
            << " fuse_rejected_dist=" << proj.num_rejected_dist
            << " fuse_rejected_desc=" << proj.num_rejected_desc
            << " fuse_rejected_scale=" << proj.num_rejected_scale;
}

PoseGraphOptions OnlineLoopCloser::MakePoseGraphOptions() const {
  PoseGraphOptions graph_options;
  graph_options.four_dof = options_.pose_graph_4dof;
  return graph_options;
}

PoseGraphEdge OnlineLoopCloser::MakeOdometryEdge(const image_t i,
                                                 const image_t j,
                                                 const Rigid3d& j_from_i,
                                                 const double sigma_scale) const {
  PoseGraphEdge edge;
  edge.image_id_i = i;
  edge.image_id_j = j;
  edge.j_from_i = j_from_i;
  edge.rot_sigma =
      sigma_scale *
      DegToRad(std::max(options_.pose_graph_odom_rot_sigma_deg, 1e-3));
  edge.trans_sigma =
      sigma_scale * (std::max(options_.pose_graph_odom_trans_sigma_m, 1e-4) +
                     options_.pose_graph_odom_trans_sigma_ratio *
                         j_from_i.translation().norm());
  return edge;
}

bool OnlineLoopCloser::SameLoopCluster(
    const image_t query_a,
    const image_t map_a,
    const image_t query_b,
    const image_t map_b,
    const Reconstruction& reconstruction) const {
  if (!ImageIdsWithin(query_a, query_b, options_.cooldown_num_images)) {
    return false;
  }
  if (map_a == map_b) {
    return true;
  }
  return CovisibilityGroup(map_a, reconstruction).count(map_b) > 0;
}

bool OnlineLoopCloser::HistoricalResidualPlausible(
    const Rigid3d& measured, const Rigid3d& current_rel) const {
  const Rigid3d error = Inverse(measured) * current_rel;
  const double trans_m = error.translation().norm();
  const double rot_deg =
      RotationDiffDeg(measured.rotation(), current_rel.rotation());
  if (options_.max_correction_m > 0.0 &&
      trans_m > kHistoricalResidualFactor * options_.max_correction_m) {
    return false;
  }
  if (options_.max_correction_deg > 0.0 &&
      rot_deg > kHistoricalResidualFactor * options_.max_correction_deg) {
    return false;
  }
  return true;
}

size_t OnlineLoopCloser::AppendLoopEdges(
    const Reconstruction& reconstruction,
    const std::unordered_set<image_t>& variable_set,
    const LoopConstraint& current,
    std::unordered_map<image_t, Rigid3d>* poses_old,
    std::unordered_set<image_t>* fixed_ids,
    std::vector<PoseGraphEdge>* edges,
    std::vector<LoopConstraint>* edge_constraints) {
  std::vector<LoopConstraint> replay;
  replay.reserve(loop_constraints_.size());
  for (const LoopConstraint& stored : loop_constraints_) {
    if (stored.map_id == current.map_id &&
        stored.query_id == current.query_id) {
      continue;
    }
    if (variable_set.count(stored.query_id) == 0 ||
        stored.map_id == stored.query_id ||
        poses_old->count(stored.query_id) == 0) {
      continue;
    }
    if (!reconstruction.ExistsImage(stored.map_id) ||
        !reconstruction.ExistsImage(stored.query_id) ||
        !reconstruction.Image(stored.map_id).HasPose() ||
        !reconstruction.Image(stored.query_id).HasPose()) {
      continue;
    }
    if (poses_old->count(stored.map_id) == 0) {
      (*poses_old)[stored.map_id] =
          reconstruction.Image(stored.map_id).CamFromWorld();
    }
    const Rigid3d current_rel = poses_old->at(stored.query_id) *
                                Inverse(poses_old->at(stored.map_id));
    if (!HistoricalResidualPlausible(stored.query_from_map, current_rel)) {
      LOG(INFO) << "OnlineLoopCloser: skip historical loop " << stored.query_id
                << "-" << stored.map_id << ", residual exceeds "
                << kHistoricalResidualFactor << "x correction gate";
      continue;
    }
    replay.push_back(stored);
  }
  std::sort(replay.begin(),
            replay.end(),
            [](const LoopConstraint& a, const LoopConstraint& b) {
              if (a.num_inliers != b.num_inliers) {
                return a.num_inliers > b.num_inliers;
              }
              return a.query_id < b.query_id;
            });

  // Current edge is kept even when an older edge has more inliers, so this
  // correction is not replaced by a previous measurement of the same place.
  std::vector<LoopConstraint> selected;
  selected.push_back(current);
  size_t num_hist = 0;
  for (const LoopConstraint& candidate : replay) {
    bool clustered = false;
    for (const LoopConstraint& kept : selected) {
      if (!SameLoopCluster(candidate.query_id,
                           candidate.map_id,
                           kept.query_id,
                           kept.map_id,
                           reconstruction)) {
        continue;
      }
      clustered = true;
      LOG(INFO) << "OnlineLoopCloser: drop clustered loop "
                << candidate.query_id << "-" << candidate.map_id
                << " inliers=" << candidate.num_inliers << ", keep "
                << kept.query_id << "-" << kept.map_id
                << " inliers=" << kept.num_inliers;
      break;
    }
    if (clustered) {
      continue;
    }
    selected.push_back(candidate);
    ++num_hist;
  }

  for (const LoopConstraint& edge : selected) {
    if (poses_old->count(edge.map_id) == 0 &&
        reconstruction.ExistsImage(edge.map_id) &&
        reconstruction.Image(edge.map_id).HasPose()) {
      (*poses_old)[edge.map_id] =
          reconstruction.Image(edge.map_id).CamFromWorld();
    }
    if (poses_old->count(edge.map_id) == 0 ||
        poses_old->count(edge.query_id) == 0) {
      continue;
    }
    if (variable_set.count(edge.map_id) == 0) {
      fixed_ids->insert(edge.map_id);
    }
    PoseGraphEdge loop_edge =
        MakeOdometryEdge(edge.map_id, edge.query_id, edge.query_from_map, 1.0);
    if (edge.rot_sigma > 0.0 && edge.trans_sigma > 0.0) {
      loop_edge.rot_sigma = edge.rot_sigma;
      loop_edge.trans_sigma = edge.trans_sigma;
    }
    loop_edge.robust = true;
    edges->push_back(loop_edge);
    edge_constraints->push_back(edge);
  }
  return num_hist;
}

void OnlineLoopCloser::CollectProtectedImageIds(
    std::unordered_set<image_t>* ids) const {
  for (const LoopConstraint& constraint : loop_constraints_) {
    ids->insert(constraint.map_id);
    ids->insert(constraint.query_id);
  }
  for (const OnlineLoopPair& pair : confirmed_loops_) {
    ids->insert(pair.image_id1);
    ids->insert(pair.image_id2);
  }
  if (last_correction_.current_id != kInvalidImageId) {
    ids->insert(last_correction_.current_id);
    ids->insert(last_correction_.loop_id);
  }
}

void OnlineLoopCloser::ForgetLoopConstraint(const image_t map_id,
                                            const image_t query_id) {
  loop_constraints_.erase(
      std::remove_if(loop_constraints_.begin(),
                     loop_constraints_.end(),
                     [&](const LoopConstraint& other) {
                       return other.map_id == map_id &&
                              other.query_id == query_id;
                     }),
      loop_constraints_.end());
}

void OnlineLoopCloser::RememberLoopConstraint(const LoopConstraint& constraint) {
  loop_constraints_.erase(
      std::remove_if(loop_constraints_.begin(),
                     loop_constraints_.end(),
                     [&](const LoopConstraint& other) {
                       return other.map_id == constraint.map_id &&
                              other.query_id == constraint.query_id;
                     }),
      loop_constraints_.end());
  loop_constraints_.push_back(constraint);
}

std::vector<image_t> OnlineLoopCloser::LoopMapImages(
    const image_t candidate_id,
    const image_t current_id,
    const Reconstruction& reconstruction) const {
  std::vector<image_t> ids = {candidate_id};
  const std::unordered_set<image_t> group =
      CovisibilityGroup(candidate_id, reconstruction);
  const std::unordered_set<image_t> current_group =
      CovisibilityGroup(current_id, reconstruction);
  for (const image_t id : RankedCovisible(candidate_id, reconstruction)) {
    if (ids.size() >= kMaxLoopMapImages) {
      break;
    }
    if (id == current_id || group.count(id) == 0 ||
        current_group.count(id) > 0 || !reconstruction.ExistsImage(id) ||
        !reconstruction.Image(id).HasPose()) {
      continue;
    }
    ids.push_back(id);
  }
  return ids;
}

std::unordered_map<image_t, int> OnlineLoopCloser::CountCovisibilityPoints(
    const image_t image_id, const Reconstruction& reconstruction) const {
  std::unordered_map<image_t, int> shared;
  if (!reconstruction.ExistsImage(image_id)) {
    return shared;
  }
  const Image& image = reconstruction.Image(image_id);
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    for (const TrackElement& el :
         reconstruction.Point3D(point2D.point3D_id).track.Elements()) {
      if (el.image_id == image_id) {
        continue;
      }
      shared[el.image_id] += 1;
    }
  }
  return shared;
}

std::unordered_set<image_t> OnlineLoopCloser::CovisibilityGroup(
    const image_t image_id, const Reconstruction& reconstruction) const {
  std::unordered_set<image_t> group;
  group.insert(image_id);
  const auto shared = CountCovisibilityPoints(image_id, reconstruction);
  int best_count = 0;
  image_t best_id = kInvalidImageId;
  for (const auto& [other_id, count] : shared) {
    if (count > best_count) {
      best_count = count;
      best_id = other_id;
    }
    if (count >= options_.min_covisibility_points) {
      group.insert(other_id);
    }
  }
  if (group.size() == 1 && best_id != kInvalidImageId) {
    group.insert(best_id);
  }
  return group;
}

std::vector<image_t> OnlineLoopCloser::RankedCovisible(
    const image_t image_id, const Reconstruction& reconstruction) const {
  const auto shared = CountCovisibilityPoints(image_id, reconstruction);
  std::vector<std::pair<int, image_t>> ranked;
  ranked.reserve(shared.size());
  for (const auto& [other_id, count] : shared) {
    ranked.emplace_back(count, other_id);
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  });
  std::vector<image_t> ids;
  ids.reserve(ranked.size());
  for (const auto& item : ranked) {
    ids.push_back(item.second);
  }
  return ids;
}

}  // namespace colmap
