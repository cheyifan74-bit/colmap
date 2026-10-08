// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// ORB-SLAM2-style Sim3 estimation for online loop closing:
//   1) Horn closed-form Sim3 (optional fixed scale) inside RANSAC
//   2) Bidirectional image-plane scoring (not 3D Euclidean)
//   3) Ceres OptimizeSim3 with bidirectional reprojection (g2o substitute)

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

struct OnlineSim3Correspondence {
  // 3D points in current / candidate camera frames.
  Eigen::Vector3d xyz_cam_current = Eigen::Vector3d::Zero();
  Eigen::Vector3d xyz_cam_candidate = Eigen::Vector3d::Zero();
  // Keypoint observations (used by OptimizeSim3).
  Eigen::Vector2d xy_current = Eigen::Vector2d::Zero();
  Eigen::Vector2d xy_candidate = Eigen::Vector2d::Zero();
};

struct OnlineSim3SolverOptions {
  // Defaults match ORB-SLAM2 LoopClosing::ComputeSim3 / CorrectLoop.

  // false: estimate scale (monocular / scale-drift); true: fix s=1 (stereo).
  bool fix_scale = false;
  // Sim3Solver::SetRansacParameters(0.99, 20, 300); also nInliers>=20.
  int min_inliers = 20;
  int max_iterations = 300;
  double confidence = 0.99;
  // Sim3Solver CheckInliers: 9.210*sigma^2 at octave 0 → ~3.035 px.
  // SIFT has no octave pyramid; use the octave-0 equivalent.
  double max_reproj_error_px = 3.035;
  // Optimizer::OptimizeSim3(..., th2=10, ...): chi2 gate on keypoint residuals.
  double optimize_chi2_th = 10.0;
  // OptimizeSim3: first optimize(5); if outliers then optimize(10) else (5).
  int optimize_iterations = 5;
  int optimize_iterations_with_outliers = 10;
  // Huber delta = sqrt(th2).
  double huber_delta_px = 3.16227766;

  // ORBmatcher::SearchBySim3(..., th=7.5) at octave 0.
  double guided_radius_px = 7.5;
  // ORBmatcher::SearchByProjection(..., th=10) at octave 0.
  double projection_radius_px = 10.0;
  // ORBmatcher::Fuse(..., th=4).
  double fuse_radius_px = 4.0;
  // Extra covis frames matched after Sim3 (no direct ORB knob; keep modest).
  int max_loop_covis_images = 10;
  // SIFT L2 stand-in for ORB TH_HIGH=100 (Hamming); COLMAP max_distance=0.7.
  float desc_max_l2_sq = static_cast<float>(512 * 512) * 0.7f * 0.7f;

  // Local RANSAC PRNG seed (does not touch the global COLMAP PRNG).
  // Same correspondences + same seed → same Sim3.
  unsigned int ransac_seed = 0;
  // ORB early-exits on first model with enough inliers (unstable across runs).
  // Default false: finish all trials and keep the best inlier count.
  bool ransac_early_exit = false;
};

// Estimate cand_from_cur (Xc_cand ≈ s R Xc_cur + t) via ORB-SLAM2 Sim3Solver
// RANSAC + OptimizeSim3. Returns false if fewer than min_inliers remain.
bool EstimateSim3Orb(const std::vector<OnlineSim3Correspondence>& corrs,
                     const Camera& cam_current,
                     const Camera& cam_candidate,
                     const OnlineSim3SolverOptions& options,
                     Sim3d* cand_from_cur,
                     std::vector<char>* inliers,
                     int* num_inliers);

// Bidirectional image-plane inliers for a fixed Sim3 (ORB CheckInliers).
std::vector<char> ComputeSim3ReprojInliers(
    const Sim3d& cand_from_cur,
    const Camera& cam_current,
    const Camera& cam_candidate,
    const std::vector<OnlineSim3Correspondence>& corrs,
    double max_reproj_error_px);

// Refine cand_from_cur with bidirectional reprojection (ORB OptimizeSim3).
// Updates inliers in-place. Returns remaining inlier count.
int OptimizeSim3Orb(const std::vector<OnlineSim3Correspondence>& corrs,
                    const Camera& cam_current,
                    const Camera& cam_candidate,
                    const OnlineSim3SolverOptions& options,
                    Sim3d* cand_from_cur,
                    std::vector<char>* inliers);

}  // namespace colmap
