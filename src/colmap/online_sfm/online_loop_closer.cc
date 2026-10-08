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
#include "colmap/geometry/sim3.h"
#include "colmap/online_sfm/online_sim3_solver.h"
#include "colmap/online_sfm/timing_stats.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/database_session.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/scene/track.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace colmap {
namespace {

std::string ScoreForFilename(const float score) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << score;
  std::string text = oss.str();
  for (char& ch : text) {
    if (ch == '.') {
      ch = 'p';
    } else if (ch == '-') {
      ch = 'm';
    } else if (ch == '+') {
      ch = 'p';
    }
  }
  return text;
}

std::filesystem::path KeyframeImagePath(const image_t image_id,
                                        const std::string& query_abs_path,
                                        const image_t query_id,
                                        const DatabaseCache& cache,
                                        const std::string& database_path) {
  if (image_id == query_id && !query_abs_path.empty() &&
      std::filesystem::exists(query_abs_path)) {
    return query_abs_path;
  }
  if (!cache.ExistsImage(image_id)) {
    return {};
  }
  const std::string& name = cache.Image(image_id).Name();
  if (name.empty()) {
    return {};
  }
  if (cache.ExistsImage(query_id) && !query_abs_path.empty()) {
    const std::string& query_name = cache.Image(query_id).Name();
    const std::string query_str =
        std::filesystem::path(query_abs_path).generic_string();
    if (!query_name.empty() && query_str.size() >= query_name.size() &&
        query_str.compare(query_str.size() - query_name.size(),
                          query_name.size(),
                          query_name) == 0) {
      const std::string root =
          query_str.substr(0, query_str.size() - query_name.size());
      return std::filesystem::path(root) / name;
    }
  }
  const auto map_dir =
      std::filesystem::path(database_path).parent_path().parent_path();
  return map_dir / "images" / name;
}

bool CopyKeyframeImage(const std::filesystem::path& src,
                       const std::filesystem::path& dst) {
  if (src.empty() || !std::filesystem::exists(src)) {
    return false;
  }
  std::error_code ec;
  std::filesystem::copy_file(
      src, dst, std::filesystem::copy_options::overwrite_existing, ec);
  if (!ec) {
    return true;
  }
  Bitmap bitmap;
  if (!bitmap.Read(src, /*as_rgb=*/true)) {
    return false;
  }
  return bitmap.Write(dst);
}

// Camera +Z in world. COLMAP x_cam = R x_world + t, so +Z_world = R^T e_z.
Eigen::Vector3d OpticalAxisWorld(const Image& image) {
  return image.CamFromWorld().rotation().toRotationMatrix().row(2).transpose();
}

bool GeometryUsable(const TwoViewGeometry& geometry, const int min_inliers) {
  return geometry.config != TwoViewGeometry::ConfigurationType::DEGENERATE &&
         geometry.inlier_matches.size() >= static_cast<size_t>(min_inliers);
}

Sim3d ToSim3(const Rigid3d& rigid) {
  return Sim3d(1.0, rigid.rotation(), rigid.translation());
}

struct LoopSim3Corr {
  Eigen::Vector3d xyz_cam_current = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_cam_candidate = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_cam_other = Eigen::Vector3d::Zero();
  Eigen::Vector2d xy_current = Eigen::Vector2d::Zero();
  Eigen::Vector2d xy_other = Eigen::Vector2d::Zero();
  Sim3d other_from_cand;
  const Camera* cam_other = nullptr;
  image_t other_id = kInvalidImageId;
  point2D_t idx_current = kInvalidPoint2DIdx;
  point2D_t idx_other = kInvalidPoint2DIdx;
  point3D_t current_point3D_id = kInvalidPoint3DId;
  point3D_t other_point3D_id = kInvalidPoint3DId;
};

std::optional<Eigen::Vector2d> ProjectCam(const Camera& camera,
                                          const Eigen::Vector3d& xyz_cam) {
  if (xyz_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return std::nullopt;
  }
  return camera.ImgFromCam(xyz_cam);
}

size_t CountInliers(const std::vector<char>& inliers) {
  return static_cast<size_t>(std::count(inliers.begin(), inliers.end(), 1));
}

std::vector<OnlineSim3Correspondence> ToOnlineSim3Corrs(
    const std::vector<LoopSim3Corr>& corrs,
    const image_t candidate_id,
    const Camera& cam_candidate) {
  std::vector<OnlineSim3Correspondence> out;
  out.reserve(corrs.size());
  for (const LoopSim3Corr& corr : corrs) {
    OnlineSim3Correspondence o;
    o.xyz_cam_current = corr.xyz_cam_current;
    o.xyz_cam_candidate = corr.xyz_cam_candidate;
    o.xy_current = corr.xy_current;
    if (corr.other_id == candidate_id) {
      o.xy_candidate = corr.xy_other;
    } else if (const auto proj =
                   ProjectCam(cam_candidate, corr.xyz_cam_candidate)) {
      o.xy_candidate = *proj;
    } else {
      o.xy_candidate = corr.xy_other;
    }
    out.push_back(o);
  }
  return out;
}

std::vector<char> ComputeReprojInliers(const Sim3d& cand_from_cur,
                                       const Camera& cam_current,
                                       const Camera& cam_candidate,
                                       const image_t candidate_id,
                                       const std::vector<LoopSim3Corr>& corrs,
                                       const double max_reproj_px) {
  return ComputeSim3ReprojInliers(
      cand_from_cur,
      cam_current,
      cam_candidate,
      ToOnlineSim3Corrs(corrs, candidate_id, cam_candidate),
      max_reproj_px);
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

bool FindNearestTriangulated(const Image& image,
                             const Reconstruction& reconstruction,
                             const Eigen::Vector2d& xy,
                             const double radius_sq,
                             const std::unordered_set<point2D_t>& used,
                             const point3D_t skip_point3D,
                             point2D_t* best_idx) {
  double best = radius_sq;
  bool found = false;
  for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
    if (used.count(i) > 0) {
      continue;
    }
    const Point2D& point2D = image.Point2D(i);
    if (!point2D.HasPoint3D() || point2D.point3D_id == skip_point3D ||
        !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    const double d2 = (point2D.xy - xy).squaredNorm();
    if (d2 < best) {
      best = d2;
      *best_idx = i;
      found = true;
    }
  }
  return found;
}

bool FindBestTriangulatedByDesc(const Image& image,
                                const Reconstruction& reconstruction,
                                const Eigen::Vector2d& xy,
                                const double radius_sq,
                                const float desc_max_l2_sq,
                                const std::unordered_set<point2D_t>& used,
                                const point3D_t skip_point3D,
                                const FeatureDescriptors& src_desc,
                                const point2D_t src_idx,
                                const FeatureDescriptors& dst_desc,
                                point2D_t* best_idx) {
  float best_l2 = desc_max_l2_sq;
  bool found = false;
  for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
    if (used.count(i) > 0) {
      continue;
    }
    const Point2D& point2D = image.Point2D(i);
    if (!point2D.HasPoint3D() || point2D.point3D_id == skip_point3D ||
        !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
      continue;
    }
    if ((point2D.xy - xy).squaredNorm() >= radius_sq) {
      continue;
    }
    const float l2 = DescriptorL2Sq(src_desc, src_idx, dst_desc, i);
    if (l2 < best_l2) {
      best_l2 = l2;
      *best_idx = i;
      found = true;
    }
  }
  return found;
}

void AppendSim3Corr(const Image& current,
                    const Image& other,
                    const Image& candidate,
                    const Reconstruction& reconstruction,
                    const point2D_t idx_current,
                    const point2D_t idx_other,
                    std::unordered_set<point2D_t>* used_current,
                    std::vector<LoopSim3Corr>* corrs) {
  if (idx_current >= current.NumPoints2D() ||
      idx_other >= other.NumPoints2D()) {
    return;
  }
  if (used_current != nullptr && used_current->count(idx_current) > 0) {
    return;
  }
  const Point2D& p_current = current.Point2D(idx_current);
  const Point2D& p_other = other.Point2D(idx_other);
  if (!p_current.HasPoint3D() || !p_other.HasPoint3D() ||
      p_current.point3D_id == p_other.point3D_id ||
      !reconstruction.ExistsPoint3D(p_current.point3D_id) ||
      !reconstruction.ExistsPoint3D(p_other.point3D_id) ||
      !other.HasCameraPtr()) {
    return;
  }
  const Eigen::Vector3d xyz_cam_current =
      current.CamFromWorld() * reconstruction.Point3D(p_current.point3D_id).xyz;
  const Eigen::Vector3d xyz_cam_other =
      other.CamFromWorld() * reconstruction.Point3D(p_other.point3D_id).xyz;
  const Eigen::Vector3d xyz_cam_candidate =
      candidate.CamFromWorld() * reconstruction.Point3D(p_other.point3D_id).xyz;
  if (xyz_cam_current.z() <= 0.0 || xyz_cam_other.z() <= 0.0) {
    return;
  }
  LoopSim3Corr corr;
  corr.xyz_cam_current = xyz_cam_current;
  corr.xyz_cam_candidate = xyz_cam_candidate;
  corr.xyz_cam_other = xyz_cam_other;
  corr.xy_current = p_current.xy;
  corr.xy_other = p_other.xy;
  if (other.ImageId() == candidate.ImageId()) {
    corr.other_from_cand = Sim3d();
  } else {
    corr.other_from_cand = ToSim3(other.CamFromWorld() *
                                  Inverse(candidate.CamFromWorld()));
  }
  corr.cam_other = other.CameraPtr();
  corr.other_id = other.ImageId();
  corr.idx_current = idx_current;
  corr.idx_other = idx_other;
  corr.current_point3D_id = p_current.point3D_id;
  corr.other_point3D_id = p_other.point3D_id;
  corrs->push_back(corr);
  if (used_current != nullptr) {
    used_current->insert(idx_current);
  }
}

bool CollectLoopSim3Corrs(const image_t image_id,
                          const image_t other_id,
                          const image_t candidate_id,
                          const FeatureMatches& matches,
                          const Reconstruction& reconstruction,
                          std::unordered_set<point2D_t>* used_current,
                          std::vector<LoopSim3Corr>* corrs) {
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.ExistsImage(other_id) ||
      !reconstruction.ExistsImage(candidate_id)) {
    return false;
  }
  const Image& current = reconstruction.Image(image_id);
  const Image& other = reconstruction.Image(other_id);
  const Image& candidate = reconstruction.Image(candidate_id);
  if (!current.HasPose() || !other.HasPose() || !candidate.HasPose() ||
      !current.HasCameraPtr() || !other.HasCameraPtr() ||
      !candidate.HasCameraPtr()) {
    return false;
  }
  corrs->reserve(corrs->size() + matches.size());
  for (const FeatureMatch& match : matches) {
    AppendSim3Corr(current,
                   other,
                   candidate,
                   reconstruction,
                   match.point2D_idx2,
                   match.point2D_idx1,
                   used_current,
                   corrs);
  }
  return true;
}

void SearchBySim3(const Image& current, const Image& candidate,
                  const Reconstruction& reconstruction,
                  const Sim3d& cand_from_cur,
                  const FeatureDescriptors& desc_current,
                  const FeatureDescriptors& desc_candidate,
                  const OnlineSim3SolverOptions& sim3_options,
                  std::vector<LoopSim3Corr>* corrs) {
  if (desc_current.data.rows() == 0 || desc_candidate.data.rows() == 0) {
    return;
  }
  std::unordered_set<point2D_t> used_current;
  std::unordered_set<point2D_t> used_candidate;
  used_current.reserve(corrs->size());
  used_candidate.reserve(corrs->size());
  for (const LoopSim3Corr& corr : *corrs) {
    used_current.insert(corr.idx_current);
    if (corr.other_id == candidate.ImageId()) {
      used_candidate.insert(corr.idx_other);
    }
  }
  const double radius_sq =
      sim3_options.guided_radius_px * sim3_options.guided_radius_px;
  const Sim3d cur_from_cand = Inverse(cand_from_cur);

  std::vector<int> match_cur_to_cand(current.NumPoints2D(), -1);
  std::vector<int> match_cand_to_cur(candidate.NumPoints2D(), -1);

  for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
    if (used_current.count(i) > 0) {
      continue;
    }
    const Point2D& p = current.Point2D(i);
    if (!p.HasPoint3D() || !reconstruction.ExistsPoint3D(p.point3D_id)) {
      continue;
    }
    const Eigen::Vector3d xyz_cam =
        current.CamFromWorld() * reconstruction.Point3D(p.point3D_id).xyz;
    const auto proj =
        ProjectCam(*candidate.CameraPtr(), cand_from_cur * xyz_cam);
    if (!proj) {
      continue;
    }
    point2D_t j = kInvalidPoint2DIdx;
    if (FindBestTriangulatedByDesc(candidate,
                                   reconstruction,
                                   *proj,
                                   radius_sq,
                                   sim3_options.desc_max_l2_sq,
                                   used_candidate,
                                   p.point3D_id,
                                   desc_current,
                                   i,
                                   desc_candidate,
                                   &j)) {
      match_cur_to_cand[i] = static_cast<int>(j);
    }
  }

  for (point2D_t j = 0; j < candidate.NumPoints2D(); ++j) {
    if (used_candidate.count(j) > 0) {
      continue;
    }
    const Point2D& p = candidate.Point2D(j);
    if (!p.HasPoint3D() || !reconstruction.ExistsPoint3D(p.point3D_id)) {
      continue;
    }
    const Eigen::Vector3d xyz_cam =
        candidate.CamFromWorld() * reconstruction.Point3D(p.point3D_id).xyz;
    const auto proj =
        ProjectCam(*current.CameraPtr(), cur_from_cand * xyz_cam);
    if (!proj) {
      continue;
    }
    point2D_t i = kInvalidPoint2DIdx;
    if (FindBestTriangulatedByDesc(current,
                                   reconstruction,
                                   *proj,
                                   radius_sq,
                                   sim3_options.desc_max_l2_sq,
                                   used_current,
                                   p.point3D_id,
                                   desc_candidate,
                                   j,
                                   desc_current,
                                   &i)) {
      match_cand_to_cur[j] = static_cast<int>(i);
    }
  }

  for (point2D_t i = 0; i < current.NumPoints2D(); ++i) {
    const int j = match_cur_to_cand[i];
    if (j < 0) {
      continue;
    }
    if (match_cand_to_cur[j] != static_cast<int>(i)) {
      continue;
    }
    if (used_current.count(i) > 0 ||
        used_candidate.count(static_cast<point2D_t>(j)) > 0) {
      continue;
    }
    AppendSim3Corr(current,
                   candidate,
                   candidate,
                   reconstruction,
                   i,
                   static_cast<point2D_t>(j),
                   &used_current,
                   corrs);
    used_current.insert(i);
    used_candidate.insert(static_cast<point2D_t>(j));
  }
}

size_t CountProjectedLoopMatches(const Image& current,
                                 const Image& candidate,
                                 const Reconstruction& reconstruction,
                                 const std::vector<image_t>& loop_image_ids,
                                 const Sim3d& cand_from_cur,
                                 const OnlineSim3SolverOptions& sim3_options,
                                 const std::vector<LoopSim3Corr>& corrs) {
  std::unordered_set<point3D_t> already;
  already.reserve(corrs.size());
  for (const LoopSim3Corr& corr : corrs) {
    if (corr.other_point3D_id != kInvalidPoint3DId) {
      already.insert(corr.other_point3D_id);
    }
  }

  std::unordered_set<point3D_t> loop_points;
  for (const image_t id : loop_image_ids) {
    if (!reconstruction.ExistsImage(id) ||
        !reconstruction.Image(id).HasPose()) {
      continue;
    }
    const Image& image = reconstruction.Image(id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_points.insert(point2D.point3D_id);
      }
    }
  }

  const double radius_sq =
      sim3_options.projection_radius_px * sim3_options.projection_radius_px;
  const Sim3d cur_from_cand = Inverse(cand_from_cur);
  const std::unordered_set<point2D_t> none_used;
  size_t extra = 0;
  for (const point3D_t point3D_id : loop_points) {
    if (already.count(point3D_id) > 0) {
      continue;
    }
    const Eigen::Vector3d xyz_cam_cand =
        candidate.CamFromWorld() * reconstruction.Point3D(point3D_id).xyz;
    const auto proj =
        ProjectCam(*current.CameraPtr(), cur_from_cand * xyz_cam_cand);
    if (!proj) {
      continue;
    }
    point2D_t idx = kInvalidPoint2DIdx;
    if (FindNearestTriangulated(current,
                                reconstruction,
                                *proj,
                                radius_sq,
                                none_used,
                                point3D_id,
                                &idx)) {
      ++extra;
    }
  }
  return extra;
}

void DrawPixel(Bitmap* image,
               const int x,
               const int y,
               const BitmapColor<uint8_t>& color) {
  if (image == nullptr) {
    return;
  }
  image->SetPixel(x, y, color);
}

void DrawCross(Bitmap* image,
               const int x,
               const int y,
               const int radius,
               const BitmapColor<uint8_t>& color) {
  for (int d = -radius; d <= radius; ++d) {
    DrawPixel(image, x + d, y, color);
    DrawPixel(image, x, y + d, color);
  }
}

void DrawLine(Bitmap* image,
              int x0,
              int y0,
              int x1,
              int y1,
              const BitmapColor<uint8_t>& color) {
  if (image == nullptr) {
    return;
  }
  const int dx = std::abs(x1 - x0);
  const int dy = std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;
  while (true) {
    DrawPixel(image, x0, y0, color);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0 += sy;
    }
  }
}

Bitmap MakeSideBySideMatchImage(
    const Bitmap& left,
    const Bitmap& right,
    const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& pairs,
    const BitmapColor<uint8_t>& color) {
  const int width = left.Width() + right.Width();
  const int height = std::max(left.Height(), right.Height());
  Bitmap canvas(width, height, /*as_rgb=*/true);
  canvas.Fill(BitmapColor<uint8_t>(0, 0, 0));

  for (int y = 0; y < left.Height(); ++y) {
    for (int x = 0; x < left.Width(); ++x) {
      const auto px = left.GetPixel(x, y);
      if (px) {
        canvas.SetPixel(x, y, *px);
      }
    }
  }
  for (int y = 0; y < right.Height(); ++y) {
    for (int x = 0; x < right.Width(); ++x) {
      const auto px = right.GetPixel(x, y);
      if (px) {
        canvas.SetPixel(x + left.Width(), y, *px);
      }
    }
  }

  for (const auto& [xy_left, xy_right] : pairs) {
    const int x0 = static_cast<int>(std::lround(xy_left.x()));
    const int y0 = static_cast<int>(std::lround(xy_left.y()));
    const int x1 =
        static_cast<int>(std::lround(xy_right.x())) + left.Width();
    const int y1 = static_cast<int>(std::lround(xy_right.y()));
    DrawLine(&canvas, x0, y0, x1, y1, color);
    DrawCross(&canvas, x0, y0, 3, color);
    DrawCross(&canvas, x1, y1, 3, color);
  }
  return canvas;
}

bool LoadKeyframeBitmap(const image_t image_id,
                        const std::string& query_abs_path,
                        const image_t query_id,
                        const DatabaseCache& cache,
                        const std::string& database_path,
                        Bitmap* out) {
  if (out == nullptr) {
    return false;
  }
  const auto path = KeyframeImagePath(
      image_id, query_abs_path, query_id, cache, database_path);
  if (path.empty()) {
    return false;
  }
  return out->Read(path, /*as_rgb=*/true);
}

// Per query folder: query_XXXXXX/
//   qA_cB_01_matches.png          feature / essential matches
//   qA_cB_02_n3d.png              3D-3D corrs (different Point3D ids)
//   qA_cB_03_ransac_inliers.png   EstimateSim3Orb inliers
void DumpLoopVerifyVisuals(
    const std::string& loop_verify_dir,
    const std::string& database_path,
    const image_t query_id,
    const image_t candidate_id,
    const std::string& query_abs_path,
    const DatabaseCache& cache,
    const Reconstruction& reconstruction,
    const FeatureMatches& matches,
    const std::vector<LoopSim3Corr>& corrs,
    const std::vector<char>& ransac_inliers) {
  if (loop_verify_dir.empty()) {
    return;
  }
  if (!reconstruction.ExistsImage(query_id) ||
      !reconstruction.ExistsImage(candidate_id)) {
    return;
  }

  std::ostringstream folder;
  folder << "query_" << std::setfill('0') << std::setw(6) << query_id;
  const std::filesystem::path query_dir =
      std::filesystem::path(loop_verify_dir) / folder.str();
  std::error_code ec;
  std::filesystem::create_directories(query_dir, ec);
  if (ec) {
    LOG(WARNING) << "OnlineLoopCloser: failed to create " << query_dir.string()
                 << " : " << ec.message();
    return;
  }

  Bitmap bmp_query;
  Bitmap bmp_cand;
  if (!LoadKeyframeBitmap(query_id,
                          query_abs_path,
                          query_id,
                          cache,
                          database_path,
                          &bmp_query) ||
      !LoadKeyframeBitmap(candidate_id,
                          query_abs_path,
                          query_id,
                          cache,
                          database_path,
                          &bmp_cand)) {
    LOG(WARNING) << "OnlineLoopCloser: skip verify dump " << query_id << "-"
                 << candidate_id << ", failed to load images";
    return;
  }

  const Image& current = reconstruction.Image(query_id);
  const Image& candidate = reconstruction.Image(candidate_id);

  const auto SavePairs =
      [&](const std::string& suffix,
          const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>&
              pairs,
          const BitmapColor<uint8_t>& color) {
        if (pairs.empty()) {
          return;
        }
        Bitmap canvas =
            MakeSideBySideMatchImage(bmp_query, bmp_cand, pairs, color);
        std::ostringstream name;
        name << "q" << query_id << "_c" << candidate_id << "_" << suffix
             << "_n" << pairs.size() << ".png";
        const std::filesystem::path out_path = query_dir / name.str();
        if (!canvas.Write(out_path)) {
          LOG(WARNING) << "OnlineLoopCloser: failed to write "
                       << out_path.string();
        }
      };

  // Matches from ReadMatches(candidate, query): idx1=candidate, idx2=query.
  {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs;
    pairs.reserve(matches.size());
    for (const FeatureMatch& m : matches) {
      if (m.point2D_idx2 >= current.NumPoints2D() ||
          m.point2D_idx1 >= candidate.NumPoints2D()) {
        continue;
      }
      pairs.emplace_back(current.Point2D(m.point2D_idx2).xy,
                         candidate.Point2D(m.point2D_idx1).xy);
    }
    SavePairs("01_matches", pairs, BitmapColor<uint8_t>(0, 255, 0));
  }

  {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs;
    pairs.reserve(corrs.size());
    for (const LoopSim3Corr& corr : corrs) {
      pairs.emplace_back(corr.xy_current, corr.xy_other);
    }
    SavePairs("02_n3d", pairs, BitmapColor<uint8_t>(0, 200, 255));
  }

  {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs;
    pairs.reserve(corrs.size());
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (i >= ransac_inliers.size() || !ransac_inliers[i]) {
        continue;
      }
      pairs.emplace_back(corrs[i].xy_current, corrs[i].xy_other);
    }
    SavePairs("03_ransac_inliers", pairs, BitmapColor<uint8_t>(255, 64, 64));
  }
}

bool FindNearestKeypoint(const Image& image,
                         const Eigen::Vector2d& xy,
                         const double radius_sq,
                         const std::unordered_set<point2D_t>& used,
                         point2D_t* best_idx) {
  double best = radius_sq;
  bool found = false;
  for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
    if (used.count(i) > 0) {
      continue;
    }
    const double d2 = (image.Point2D(i).xy - xy).squaredNorm();
    if (d2 < best) {
      best = d2;
      *best_idx = i;
      found = true;
    }
  }
  return found;
}

Rigid3d Sim3ToSE3(const Sim3d& sim3) {
  return Rigid3d(sim3.rotation(), sim3.translation() / sim3.scale());
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
  mixvpr_topk_dir_ =
      TimingStats::FileBesideDatabase(database_path_, "mixvpr_topk");
  loop_verify_dir_ =
      TimingStats::FileBesideDatabase(database_path_, "loop_verify");
  timing_ = std::make_unique<TimingStats>(
      TimingStats::FileBesideDatabase(database_path_, "loop_pairs.txt"),
      "loop",
      std::vector<std::string>{"candidate",
                               "score",
                               "matches",
                               "inliers",
                               "n3d",
                               "n3d_inliers",
                               "scale",
                               "sim3",
                               "status",
                               "candidate_ids"});
  timing_->AddMeta("mixvpr_engine_path", options_.mixvpr_engine_path);
  timing_->AddMeta("mixvpr_use_gpu", options_.mixvpr_use_gpu ? "1" : "0");
  timing_->AddMeta("cooldown_num_images",
                   std::to_string(options_.cooldown_num_images));
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

  std::error_code ec;
  std::filesystem::create_directories(mixvpr_topk_dir_, ec);
  if (ec) {
    LOG(WARNING) << "OnlineLoopCloser: failed to create MixVPR image dump dir "
                 << mixvpr_topk_dir_ << " : " << ec.message();
  } else {
    LOG(INFO) << "OnlineLoopCloser: writing MixVPR top-k images to "
              << mixvpr_topk_dir_;
  }
  ec.clear();
  std::filesystem::create_directories(loop_verify_dir_, ec);
  if (ec) {
    LOG(WARNING) << "OnlineLoopCloser: failed to create loop verify dump dir "
                 << loop_verify_dir_ << " : " << ec.message();
  } else {
    LOG(INFO) << "OnlineLoopCloser: writing loop verify images to "
              << loop_verify_dir_;
  }
}

OnlineLoopCloser::~OnlineLoopCloser() = default;

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
  const bool encoded =
      (bitmap != nullptr && !bitmap->IsEmpty())
          ? mixvpr_->Encode(*bitmap, &query)
          : mixvpr_->Encode(image_abs_path, &query);
  if (!encoded || query.size() != kMixVprDescDim) {
    LOG(WARNING) << "OnlineLoopCloser: MixVPR encode failed for image "
                 << image_id << " path=" << image_abs_path
                 << " desc_dim=" << query.size();
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
                     "1",
                     "0",
                     "no_candidate",
                     "-"});
    return false;
  }

  LOG(INFO) << "OnlineLoopCloser: image " << image_id << " has "
            << candidates.size() << " loop candidate(s)";
  DumpMixVprTopkImages(image_id, image_abs_path, cache, candidates);

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
         std::to_string(pair.scale),
         pair.used_sim3 ? "1" : "0",
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
  for (const auto& [candidate_id, score] : candidates) {
    OnlineLoopPair loop_pair;
    const bool ok = VerifyCandidate(image_id,
                                    candidate_id,
                                    score,
                                    image_abs_path,
                                    cache,
                                    reconstruction,
                                    matcher,
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
              << " scale=" << loop_pair.scale
              << " sim3=" << static_cast<int>(loop_pair.used_sim3);
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
                       "1",
                       "0",
                       "no_candidate",
                       "-"});
    }
    return false;
  }

  if (!options_.correct_local || obs_manager == nullptr ||
      !confirmed_pair.used_sim3) {
    return false;
  }

  const bool ok = CorrectLocalLoop(reconstruction, obs_manager, confirmed_pair);
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
    if (covis.count(other_id) > 0) {
      return false;
    }
    if (!reconstruction.ExistsImage(other_id) ||
        !reconstruction.Image(other_id).HasPose()) {
      return false;
    }
    const Image& other = reconstruction.Image(other_id);
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
                                       const std::string& query_abs_path,
                                       const DatabaseCache& cache,
                                       const Reconstruction& reconstruction,
                                       OnlineFeatureMatcher* matcher,
                                       OnlineLoopPair* loop_pair) {
  if (matcher == nullptr || loop_pair == nullptr) {
    return false;
  }

  loop_pair->image_id1 = image_id;
  loop_pair->image_id2 = candidate_id;
  loop_pair->retrieval_score = retrieval_score;
  loop_pair->scale = 1.0;
  loop_pair->used_sim3 = false;

  try {
    DatabaseSession session(database_path_);
    if (session->ExistsTwoViewGeometry(candidate_id, image_id)) {
      const TwoViewGeometry existing =
          session->ReadTwoViewGeometry(candidate_id, image_id);
      if (!GeometryUsable(existing, options_.min_num_matches)) {
        if (session->ExistsMatches(candidate_id, image_id)) {
          session->DeleteMatches(candidate_id, image_id);
        }
        session->DeleteTwoViewGeometry(candidate_id, image_id);
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to inspect existing pair "
                 << candidate_id << "-" << image_id << ": " << e.what();
  }

  if (!matcher->MatchSpecificPairs(image_id, {candidate_id}, cache)) {
    LOG(WARNING) << "OnlineLoopCloser: matching " << candidate_id << "-"
                 << image_id << " failed";
    return false;
  }

  FeatureMatches matches;
  TwoViewGeometry geometry;
  FeatureDescriptors desc_current;
  FeatureDescriptors desc_candidate;
  try {
    DatabaseSession session(database_path_);
    if (session->ExistsMatches(candidate_id, image_id)) {
      matches = session->ReadMatches(candidate_id, image_id);
    }
    if (session->ExistsTwoViewGeometry(candidate_id, image_id)) {
      geometry = session->ReadTwoViewGeometry(candidate_id, image_id);
    }
    if (session->ExistsDescriptors(image_id)) {
      desc_current = session->ReadDescriptors(image_id);
    }
    if (session->ExistsDescriptors(candidate_id)) {
      desc_candidate = session->ReadDescriptors(candidate_id);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "OnlineLoopCloser: failed to read pair " << candidate_id
                 << "-" << image_id << ": " << e.what();
    return false;
  }

  loop_pair->num_matches = matches.size();
  loop_pair->num_inliers = geometry.inlier_matches.size();

  if (matches.size() < static_cast<size_t>(options_.min_num_matches)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", matches=" << matches.size();
    return false;
  }
  if (!GeometryUsable(geometry, options_.min_num_matches)) {
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", essential inliers=" << geometry.inlier_matches.size();
    return false;
  }

  const FeatureMatches& verify_matches = geometry.inlier_matches.empty()
                                             ? matches
                                             : geometry.inlier_matches;
  std::vector<LoopSim3Corr> corrs;
  std::unordered_set<point2D_t> used_current;
  if (!CollectLoopSim3Corrs(image_id,
                            candidate_id,
                            candidate_id,
                            verify_matches,
                            reconstruction,
                            &used_current,
                            &corrs)) {
    return false;
  }
  loop_pair->num_3d_correspondences = corrs.size();
  LOG(INFO) << "OnlineLoopCloser: n3d " << image_id << "-" << candidate_id
            << " direct=" << corrs.size();

  const Image& current = reconstruction.Image(image_id);
  const Image& candidate = reconstruction.Image(candidate_id);
  const bool can_sim3 =
      current.HasPose() && candidate.HasPose() && current.HasCameraPtr() &&
      candidate.HasCameraPtr() &&
      corrs.size() >= static_cast<size_t>(options_.min_num_3d_correspondences);

  if (!can_sim3) {
    DumpLoopVerifyVisuals(loop_verify_dir_,
                          database_path_,
                          image_id,
                          candidate_id,
                          query_abs_path,
                          cache,
                          reconstruction,
                          verify_matches,
                          corrs,
                          /*ransac_inliers=*/{});
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", n3d=" << corrs.size()
              << " inliers=" << geometry.inlier_matches.size();
    return false;
  }

  Sim3d cand_from_cur;
  std::vector<char> inliers;
  int ransac_best_n = 0;
  OnlineSim3SolverOptions sim3_options;
  sim3_options.fix_scale = options_.fix_scale;
  sim3_options.min_inliers = options_.min_num_3d_inliers;
  const bool ransac_ok =
      EstimateSim3Orb(ToOnlineSim3Corrs(corrs, candidate_id, *candidate.CameraPtr()),
                      *current.CameraPtr(),
                      *candidate.CameraPtr(),
                      sim3_options,
                      &cand_from_cur,
                      &inliers,
                      &ransac_best_n);
  DumpLoopVerifyVisuals(loop_verify_dir_,
                        database_path_,
                        image_id,
                        candidate_id,
                        query_abs_path,
                        cache,
                        reconstruction,
                        verify_matches,
                        corrs,
                        inliers);
  if (!ransac_ok) {
    loop_pair->num_3d_inliers = static_cast<size_t>(std::max(ransac_best_n, 0));
    LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
              << ", sim3 ransac failed n3d=" << corrs.size()
              << " best_inliers=" << ransac_best_n;
    return false;
  }

  loop_pair->num_3d_inliers = CountInliers(inliers);

  SearchBySim3(current,
               candidate,
               reconstruction,
               cand_from_cur,
               desc_current,
               desc_candidate,
               sim3_options,
               &corrs);
  // ORB: guided matching → OptimizeSim3 again.
  {
    auto online_corrs =
        ToOnlineSim3Corrs(corrs, candidate_id, *candidate.CameraPtr());
    inliers = ComputeSim3ReprojInliers(cand_from_cur,
                                       *current.CameraPtr(),
                                       *candidate.CameraPtr(),
                                       online_corrs,
                                       sim3_options.max_reproj_error_px);
    OptimizeSim3Orb(online_corrs,
                    *current.CameraPtr(),
                    *candidate.CameraPtr(),
                    sim3_options,
                    &cand_from_cur,
                    &inliers);
  }

  std::vector<image_t> covis_ids;
  const std::unordered_set<image_t> covis_group =
      CovisibilityGroup(candidate_id, reconstruction);
  const std::vector<image_t> ranked =
      RankedCovisible(candidate_id, reconstruction);
  covis_ids.reserve(sim3_options.max_loop_covis_images);
  for (const image_t covis_id : ranked) {
    if (covis_id == image_id || covis_id == candidate_id) {
      continue;
    }
    if (covis_group.count(covis_id) == 0) {
      continue;
    }
    if (!reconstruction.ExistsImage(covis_id) ||
        !reconstruction.Image(covis_id).HasPose() ||
        !reconstruction.Image(covis_id).HasCameraPtr()) {
      continue;
    }
    covis_ids.push_back(covis_id);
    if (covis_ids.size() >=
        static_cast<size_t>(sim3_options.max_loop_covis_images)) {
      break;
    }
  }
  const size_t n3d_after_sim3 = corrs.size();
  if (!covis_ids.empty()) {
    if (!matcher->MatchSpecificPairs(image_id, covis_ids, cache)) {
      LOG(WARNING) << "OnlineLoopCloser: covis matching " << image_id << "-"
                   << candidate_id << " failed";
    } else {
      used_current.clear();
      for (const LoopSim3Corr& corr : corrs) {
        used_current.insert(corr.idx_current);
      }
      try {
        DatabaseSession session(database_path_);
        for (const image_t covis_id : covis_ids) {
          FeatureMatches covis_matches;
          TwoViewGeometry covis_geometry;
          if (session->ExistsMatches(covis_id, image_id)) {
            covis_matches = session->ReadMatches(covis_id, image_id);
          }
          if (session->ExistsTwoViewGeometry(covis_id, image_id)) {
            covis_geometry = session->ReadTwoViewGeometry(covis_id, image_id);
          }
          if (covis_matches.size() <
              static_cast<size_t>(options_.min_num_matches)) {
            continue;
          }
          const FeatureMatches& covis_verify =
              covis_geometry.inlier_matches.empty()
                  ? covis_matches
                  : covis_geometry.inlier_matches;
          CollectLoopSim3Corrs(image_id,
                               covis_id,
                               candidate_id,
                               covis_verify,
                               reconstruction,
                               &used_current,
                               &corrs);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "OnlineLoopCloser: failed to read covis pairs for "
                     << image_id << "-" << candidate_id << ": " << e.what();
      }
      inliers = ComputeReprojInliers(cand_from_cur,
                                     *current.CameraPtr(),
                                     *candidate.CameraPtr(),
                                     candidate_id,
                                     corrs,
                                     sim3_options.max_reproj_error_px);
    }
  }
  LOG(INFO) << "OnlineLoopCloser: n3d " << image_id << "-" << candidate_id
            << " after_sim3=" << n3d_after_sim3
            << " covis_frames=" << covis_ids.size()
            << " total=" << corrs.size();
  loop_pair->num_3d_correspondences = corrs.size();
  loop_pair->num_3d_inliers = CountInliers(inliers);
  std::vector<image_t> loop_ids = RankedCovisible(candidate_id, reconstruction);
  loop_ids.push_back(candidate_id);
  const size_t extra = CountProjectedLoopMatches(current,
                                                 candidate,
                                                 reconstruction,
                                                 loop_ids,
                                                 cand_from_cur,
                                                 sim3_options,
                                                 corrs);
  const size_t total = loop_pair->num_3d_inliers + extra;

  if (loop_pair->num_3d_inliers >=
          static_cast<size_t>(options_.min_num_3d_inliers) &&
      total >= static_cast<size_t>(options_.min_num_verified_matches)) {
    loop_pair->scale = cand_from_cur.scale();
    loop_pair->used_sim3 = true;
    loop_pair->cand_from_cur = cand_from_cur;
    loop_pair->matched_point3Ds.clear();
    loop_pair->matched_point3Ds.reserve(loop_pair->num_3d_inliers);
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (i >= inliers.size() || !inliers[i]) {
        continue;
      }
      const LoopSim3Corr& corr = corrs[i];
      if (corr.current_point3D_id == kInvalidPoint3DId ||
          corr.other_point3D_id == kInvalidPoint3DId ||
          corr.current_point3D_id == corr.other_point3D_id) {
        continue;
      }
      loop_pair->matched_point3Ds.emplace_back(corr.current_point3D_id,
                                               corr.other_point3D_id);
    }
    LOG(INFO) << "OnlineLoopCloser: accept " << image_id << "-" << candidate_id
              << " by sim3 inliers=" << loop_pair->num_3d_inliers
              << " projected=" << extra << " total=" << total
              << " fix_scale=" << options_.fix_scale
              << " scale=" << loop_pair->scale
              << " t=[" << cand_from_cur.translation().transpose() << "]"
              << " fuse_pairs=" << loop_pair->matched_point3Ds.size();
    return true;
  }

  LOG(INFO) << "OnlineLoopCloser: reject " << image_id << "-" << candidate_id
            << ", sim3 weak inliers=" << loop_pair->num_3d_inliers
            << " projected=" << extra << " total=" << total
            << " ransac_best=" << ransac_best_n;
  return false;
}

bool OnlineLoopCloser::CorrectLocalLoop(Reconstruction& reconstruction,
                                        ObservationManager* obs_manager,
                                        const OnlineLoopPair& loop_pair) {
  if (obs_manager == nullptr) {
    return false;
  }
  const image_t id1 = loop_pair.image_id1;
  const image_t id2 = loop_pair.image_id2;
  if (!reconstruction.ExistsImage(id1) || !reconstruction.ExistsImage(id2)) {
    return false;
  }

  // Older map side has the smaller strong-covisible image id → anchor.
  const image_t min1 = MinCovisibilityImageId(id1, reconstruction);
  const image_t min2 = MinCovisibilityImageId(id2, reconstruction);
  // Default: current=id1, loop=id2 (matches VerifyCandidate orientation).
  const bool id1_is_newer =
      (min1 > min2) || (min1 == min2 && id1 > id2);

  image_t current_id = id1;
  image_t loop_id = id2;
  Sim3d loop_from_current = loop_pair.cand_from_cur;
  std::vector<std::pair<point3D_t, point3D_t>> matched_point3Ds =
      loop_pair.matched_point3Ds;
  if (!id1_is_newer) {
    std::swap(current_id, loop_id);
    loop_from_current = Inverse(loop_from_current);
    for (auto& pair : matched_point3Ds) {
      std::swap(pair.first, pair.second);
    }
  }
  LOG(INFO) << "OnlineLoopCloser: CorrectLocalLoop sides"
            << " id1=" << id1 << "(min_covis=" << min1 << ")"
            << " id2=" << id2 << "(min_covis=" << min2 << ")"
            << " → current=" << current_id << " loop_anchor=" << loop_id
            << (id1_is_newer ? " (no swap)" : " (swapped)");

  Image& current = reconstruction.Image(current_id);
  Image& loop_image = reconstruction.Image(loop_id);
  if (!current.HasPose() || !loop_image.HasPose() || !current.HasFramePtr()) {
    return false;
  }

  // Connected window: current + its strong covisibles, but never frames that
  // already belong to the loop-anchor local map (e.g. 288 covisible with 2
  // while 2 is also in 1's neighborhood — those stay fixed).
  std::vector<image_t> connected;
  connected.push_back(current_id);
  const std::unordered_set<image_t> loop_side =
      CovisibilityGroup(loop_id, reconstruction);
  const auto shared = CountCovisibilityPoints(current_id, reconstruction);
  const std::vector<image_t> ranked =
      RankedCovisible(current_id, reconstruction);
  const int max_connected =
      std::max(1, options_.max_correct_connected);
  size_t num_skipped_loop_side = 0;
  for (const image_t id : ranked) {
    if (id == current_id || id == loop_id) {
      continue;
    }
    if (loop_side.count(id) > 0) {
      ++num_skipped_loop_side;
      continue;
    }
    if (!reconstruction.ExistsImage(id) ||
        !reconstruction.Image(id).HasPose() ||
        !reconstruction.Image(id).HasFramePtr()) {
      continue;
    }
    const auto it = shared.find(id);
    if (it == shared.end()) {
      continue;
    }
    if (it->second < options_.min_covisibility_points &&
        connected.size() > 1) {
      continue;
    }
    connected.push_back(id);
    if (static_cast<int>(connected.size()) >= max_connected) {
      break;
    }
  }
  if (num_skipped_loop_side > 0) {
    LOG(INFO) << "OnlineLoopCloser: CorrectLocalLoop skip " << num_skipped_loop_side
              << " loop-side covisible(s) of anchor " << loop_id;
  }

  // Scw_corr = Inverse(loop_from_current) * Smw  (ORB: mg2oScw = gScm * gSmw)
  const Sim3d Scw_corr =
      Inverse(loop_from_current) * ToSim3(loop_image.CamFromWorld());
  const Sim3d Swc_old = Inverse(ToSim3(current.CamFromWorld()));

  std::unordered_map<image_t, Sim3d> corrected_sim3;
  std::unordered_map<image_t, Sim3d> non_corrected_sim3;
  corrected_sim3.reserve(connected.size());
  non_corrected_sim3.reserve(connected.size());

  for (const image_t id : connected) {
    Image& image = reconstruction.Image(id);
    const Sim3d Siw_old = ToSim3(image.CamFromWorld());
    non_corrected_sim3[id] = Siw_old;
    if (id == current_id) {
      corrected_sim3[id] = Scw_corr;
    } else {
      const Sim3d Sic = Siw_old * Swc_old;
      corrected_sim3[id] = Sic * Scw_corr;
    }
  }

  // Stage B: correct map points then poses in the connected window.
  std::unordered_set<point3D_t> corrected_points;
  size_t num_points_corrected = 0;
  size_t num_poses_corrected = 0;
  for (const image_t id : connected) {
    Image& image = reconstruction.Image(id);
    const Sim3d& Siw_old = non_corrected_sim3[id];
    const Sim3d& Siw_corr = corrected_sim3[id];
    const Sim3d Swi_corr = Inverse(Siw_corr);

    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        continue;
      }
      const point3D_t point3D_id = point2D.point3D_id;
      if (!corrected_points.insert(point3D_id).second) {
        continue;
      }
      Point3D& point3D = reconstruction.Point3D(point3D_id);
      point3D.xyz = Swi_corr * (Siw_old * point3D.xyz);
      ++num_points_corrected;
    }

    if (SetImagePose(image, Sim3ToSE3(Siw_corr))) {
      ++num_poses_corrected;
    }
  }

  // Stage C1: fuse Sim3 inlier 3D–3D pairs.
  std::unordered_map<point3D_t, point3D_t> remap;
  const auto Resolve = [&](point3D_t id) {
    while (true) {
      const auto it = remap.find(id);
      if (it == remap.end()) {
        return id;
      }
      id = it->second;
    }
  };
  size_t num_fused_direct = 0;
  for (const auto& [cur_id_raw, loop_id_raw] : matched_point3Ds) {
    const point3D_t cur_id = Resolve(cur_id_raw);
    const point3D_t loop_pid = Resolve(loop_id_raw);
    if (cur_id == loop_pid || !reconstruction.ExistsPoint3D(cur_id) ||
        !reconstruction.ExistsPoint3D(loop_pid)) {
      continue;
    }
    try {
      const point3D_t merged =
          obs_manager->MergePoints3D(cur_id, loop_pid);
      remap[cur_id] = merged;
      remap[loop_pid] = merged;
      ++num_fused_direct;
    } catch (const std::exception& e) {
      LOG(WARNING) << "OnlineLoopCloser: merge " << cur_id << "+" << loop_pid
                   << " failed: " << e.what();
    }
  }

  // Stage C2: SearchAndFuse — project loop-neighborhood points into the
  // corrected current window and merge / attach observations.
  std::unordered_set<point3D_t> loop_points;
  const std::unordered_set<image_t> loop_group =
      CovisibilityGroup(loop_id, reconstruction);
  for (const image_t id : loop_group) {
    if (!reconstruction.ExistsImage(id)) {
      continue;
    }
    for (const Point2D& point2D : reconstruction.Image(id).Points2D()) {
      if (point2D.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        loop_points.insert(Resolve(point2D.point3D_id));
      }
    }
  }

  const OnlineSim3SolverOptions sim3_defaults;
  const double fuse_radius_sq =
      sim3_defaults.fuse_radius_px * sim3_defaults.fuse_radius_px;
  size_t num_fused_proj = 0;
  size_t num_attached = 0;
  std::unordered_set<point2D_t> empty_used;
  for (const image_t id : connected) {
    Image& image = reconstruction.Image(id);
    if (!image.HasPose() || !image.HasCameraPtr()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();

    for (point3D_t loop_pid_raw : loop_points) {
      const point3D_t loop_pid = Resolve(loop_pid_raw);
      if (!reconstruction.ExistsPoint3D(loop_pid)) {
        continue;
      }
      if (image.HasPoint3D(loop_pid)) {
        continue;
      }
      const Eigen::Vector3d xyz_cam =
          cam_from_world * reconstruction.Point3D(loop_pid).xyz;
      const auto proj = ProjectCam(camera, xyz_cam);
      if (!proj) {
        continue;
      }
      point2D_t best_idx = kInvalidPoint2DIdx;
      if (!FindNearestKeypoint(
              image, *proj, fuse_radius_sq, empty_used, &best_idx)) {
        continue;
      }
      Point2D& point2D = image.Point2D(best_idx);
      if (point2D.HasPoint3D()) {
        const point3D_t cur_pid = Resolve(point2D.point3D_id);
        if (cur_pid == loop_pid || !reconstruction.ExistsPoint3D(cur_pid) ||
            !reconstruction.ExistsPoint3D(loop_pid)) {
          continue;
        }
        try {
          const point3D_t merged =
              obs_manager->MergePoints3D(cur_pid, loop_pid);
          remap[cur_pid] = merged;
          remap[loop_pid] = merged;
          ++num_fused_proj;
        } catch (const std::exception& e) {
          LOG(WARNING) << "OnlineLoopCloser: fuse-merge " << cur_pid << "+"
                       << loop_pid << " failed: " << e.what();
        }
      } else {
        try {
          obs_manager->AddObservation(loop_pid,
                                      TrackElement(id, best_idx));
          ++num_attached;
        } catch (const std::exception& e) {
          LOG(WARNING) << "OnlineLoopCloser: attach point " << loop_pid
                       << " to image " << id << " failed: " << e.what();
        }
      }
    }
  }

  LOG(INFO) << "OnlineLoopCloser: CorrectLocalLoop " << current_id << "-"
            << loop_id << " connected=" << connected.size()
            << " poses=" << num_poses_corrected
            << " points=" << num_points_corrected
            << " fuse_direct=" << num_fused_direct
            << " fuse_proj=" << num_fused_proj
            << " attached=" << num_attached
            << " scale=" << loop_from_current.scale()
            << " t=[" << loop_from_current.translation().transpose() << "]";
  return num_poses_corrected > 0;
}

image_t OnlineLoopCloser::MinCovisibilityImageId(
    const image_t image_id, const Reconstruction& reconstruction) const {
  image_t min_id = image_id;
  const auto shared = CountCovisibilityPoints(image_id, reconstruction);
  for (const auto& [other_id, count] : shared) {
    if (count < options_.min_covisibility_points) {
      continue;
    }
    if (!reconstruction.ExistsImage(other_id) ||
        !reconstruction.Image(other_id).HasPose()) {
      continue;
    }
    min_id = std::min(min_id, other_id);
  }
  return min_id;
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

void OnlineLoopCloser::DumpMixVprTopkImages(
    const image_t query_id,
    const std::string& query_abs_path,
    const DatabaseCache& cache,
    const std::vector<std::pair<image_t, float>>& ranked) const {
  if (mixvpr_topk_dir_.empty() || ranked.empty()) {
    return;
  }

  std::ostringstream folder;
  folder << "query_" << std::setfill('0') << std::setw(6) << query_id;
  const std::filesystem::path query_dir =
      std::filesystem::path(mixvpr_topk_dir_) / folder.str();
  std::error_code ec;
  std::filesystem::create_directories(query_dir, ec);
  if (ec) {
    LOG(WARNING) << "OnlineLoopCloser: failed to create " << query_dir.string()
                 << " : " << ec.message();
    return;
  }

  const auto query_src = KeyframeImagePath(
      query_id, query_abs_path, query_id, cache, database_path_);
  std::string query_ext = query_src.extension().string();
  if (query_ext.empty()) {
    query_ext = ".png";
  }
  const std::filesystem::path query_dst =
      query_dir / ("id" + std::to_string(query_id) + "_query" +
                   std::to_string(query_id) + "_cand-_score-" + query_ext);
  if (!CopyKeyframeImage(query_src, query_dst)) {
    LOG(WARNING) << "OnlineLoopCloser: failed to save query image " << query_id
                 << " from " << query_src.string();
  }

  for (size_t i = 0; i < ranked.size(); ++i) {
    const image_t cand_id = ranked[i].first;
    const auto cand_src = KeyframeImagePath(
        cand_id, query_abs_path, query_id, cache, database_path_);
    std::string ext = cand_src.extension().string();
    if (ext.empty()) {
      ext = query_ext;
    }
    std::ostringstream name;
    name << "id" << cand_id << "_query" << query_id << "_cand" << cand_id
         << "_score" << ScoreForFilename(ranked[i].second) << ext;
    const std::filesystem::path cand_dst = query_dir / name.str();
    if (!CopyKeyframeImage(cand_src, cand_dst)) {
      LOG(WARNING) << "OnlineLoopCloser: failed to save MixVPR candidate "
                   << query_id << "-" << cand_id << " from "
                   << cand_src.string();
    }
  }
}

}  // namespace colmap
