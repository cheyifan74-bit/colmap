// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Port of ORB-SLAM2 Sim3Solver + OptimizeSim3 (Ceres instead of g2o).

#include "colmap/online_sfm/online_sim3_solver.h"

#include "colmap/math/random.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {
namespace {

std::optional<Eigen::Vector2d> ProjectCam(const Camera& camera,
                                          const Eigen::Vector3d& xyz_cam) {
  if (xyz_cam.z() <= std::numeric_limits<double>::epsilon()) {
    return std::nullopt;
  }
  return camera.ImgFromCam(xyz_cam);
}

// Horn 1987 closed-form: X_tgt ≈ s R X_src + t  (ORB ComputeSim3 with roles
// swapped so result is cand_from_cur when src=current, tgt=candidate).
bool ComputeSim3Horn(const Eigen::Vector3d& src0,
                     const Eigen::Vector3d& src1,
                     const Eigen::Vector3d& src2,
                     const Eigen::Vector3d& tgt0,
                     const Eigen::Vector3d& tgt1,
                     const Eigen::Vector3d& tgt2,
                     const bool fix_scale,
                     Sim3d* tgt_from_src) {
  Eigen::Matrix3d P_src, P_tgt;
  P_src.col(0) = src0;
  P_src.col(1) = src1;
  P_src.col(2) = src2;
  P_tgt.col(0) = tgt0;
  P_tgt.col(1) = tgt1;
  P_tgt.col(2) = tgt2;

  const Eigen::Vector3d O_src = P_src.rowwise().mean();
  const Eigen::Vector3d O_tgt = P_tgt.rowwise().mean();
  Eigen::Matrix3d Pr_src = P_src.colwise() - O_src;
  Eigen::Matrix3d Pr_tgt = P_tgt.colwise() - O_tgt;

  // ORB: M = Pr2 * Pr1^T with X1 = s R X2 + t.
  // Here X_tgt = s R X_src + t ⇒ M = Pr_src * Pr_tgt^T in ORB naming where
  // Pr1=Pr_tgt, Pr2=Pr_src ⇒ M = Pr_src * Pr_tgt^T.
  const Eigen::Matrix3d M = Pr_src * Pr_tgt.transpose();

  const double N11 = M(0, 0) + M(1, 1) + M(2, 2);
  const double N12 = M(1, 2) - M(2, 1);
  const double N13 = M(2, 0) - M(0, 2);
  const double N14 = M(0, 1) - M(1, 0);
  const double N22 = M(0, 0) - M(1, 1) - M(2, 2);
  const double N23 = M(0, 1) + M(1, 0);
  const double N24 = M(2, 0) + M(0, 2);
  const double N33 = -M(0, 0) + M(1, 1) - M(2, 2);
  const double N34 = M(1, 2) + M(2, 1);
  const double N44 = -M(0, 0) - M(1, 1) + M(2, 2);

  Eigen::Matrix4d N;
  N << N11, N12, N13, N14, N12, N22, N23, N24, N13, N23, N33, N34, N14, N24,
      N34, N44;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(N);
  if (es.info() != Eigen::Success) {
    return false;
  }
  // Largest eigenvalue is last for SelfAdjointEigenSolver.
  const Eigen::Vector4d ev = es.eigenvectors().col(3);
  Eigen::Quaterniond q(ev(0), ev(1), ev(2), ev(3));
  q.normalize();
  const Eigen::Matrix3d R = q.toRotationMatrix();

  // ORB: P3 = R * Pr2, scale from Pr1·P3 / ||P3||^2 with Pr1=tgt, Pr2=src.
  const Eigen::Matrix3d P3 = R * Pr_src;
  double scale = 1.0;
  if (!fix_scale) {
    const double nom = (Pr_tgt.array() * P3.array()).sum();
    const double den = P3.squaredNorm();
    if (den < 1e-12) {
      return false;
    }
    scale = nom / den;
    if (!std::isfinite(scale) || scale <= 1e-6) {
      return false;
    }
  }

  const Eigen::Vector3d t = O_tgt - scale * R * O_src;
  if (!t.allFinite()) {
    return false;
  }
  *tgt_from_src = Sim3d(scale, Eigen::Quaterniond(R), t);
  return true;
}

bool IsReprojInlier(const Sim3d& cand_from_cur,
                    const Camera& cam_current,
                    const Camera& cam_candidate,
                    const OnlineSim3Correspondence& corr,
                    const double max_error_sq) {
  const Eigen::Vector3d xyz_cand_from_cur =
      cand_from_cur * corr.xyz_cam_current;
  const Eigen::Vector3d xyz_cur_from_cand =
      Inverse(cand_from_cur) * corr.xyz_cam_candidate;

  const auto proj_cur_self =
      ProjectCam(cam_current, corr.xyz_cam_current);
  const auto proj_cur_from_cand =
      ProjectCam(cam_current, xyz_cur_from_cand);
  if (!proj_cur_self || !proj_cur_from_cand ||
      (*proj_cur_self - *proj_cur_from_cand).squaredNorm() > max_error_sq) {
    return false;
  }
  const auto proj_cand_self =
      ProjectCam(cam_candidate, corr.xyz_cam_candidate);
  const auto proj_cand_from_cur =
      ProjectCam(cam_candidate, xyz_cand_from_cur);
  if (!proj_cand_self || !proj_cand_from_cur ||
      (*proj_cand_self - *proj_cand_from_cur).squaredNorm() > max_error_sq) {
    return false;
  }
  return true;
}

int CountInliers(const std::vector<char>& inliers) {
  return static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
}

struct Sim3ReprojCost {
  Sim3ReprojCost(const Eigen::Vector3d& xyz_src,
                 const Eigen::Vector2d& xy_obs,
                 const double fx,
                 const double fy,
                 const double cx,
                 const double cy,
                 const bool invert)
      : xyz_src(xyz_src),
        xy_obs(xy_obs),
        fx(fx),
        fy(fy),
        cx(cx),
        cy(cy),
        invert(invert) {}

  template <typename T>
  bool operator()(const T* const angle_axis,
                  const T* const translation,
                  const T* const scale,
                  T* residuals) const {
    T pt[3] = {T(xyz_src.x()), T(xyz_src.y()), T(xyz_src.z())};
    T R_pt[3];
    ceres::AngleAxisRotatePoint(angle_axis, pt, R_pt);

    T xyz[3];
    if (!invert) {
      // X_tgt = s R X_src + t
      xyz[0] = scale[0] * R_pt[0] + translation[0];
      xyz[1] = scale[0] * R_pt[1] + translation[1];
      xyz[2] = scale[0] * R_pt[2] + translation[2];
    } else {
      // X_src = (1/s) R^T (X_tgt - t)
      T diff[3] = {pt[0] - translation[0],
                   pt[1] - translation[1],
                   pt[2] - translation[2]};
      T R_t[3];
      // Rotate by -angle_axis == R^T
      T neg_aa[3] = {-angle_axis[0], -angle_axis[1], -angle_axis[2]};
      ceres::AngleAxisRotatePoint(neg_aa, diff, R_t);
      const T inv_s = T(1.0) / scale[0];
      xyz[0] = inv_s * R_t[0];
      xyz[1] = inv_s * R_t[1];
      xyz[2] = inv_s * R_t[2];
    }

    if (xyz[2] <= T(1e-6)) {
      residuals[0] = T(0);
      residuals[1] = T(0);
      return true;
    }
    const T inv_z = T(1.0) / xyz[2];
    const T u = T(fx) * xyz[0] * inv_z + T(cx);
    const T v = T(fy) * xyz[1] * inv_z + T(cy);
    residuals[0] = u - T(xy_obs.x());
    residuals[1] = v - T(xy_obs.y());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& xyz_src,
                                     const Eigen::Vector2d& xy_obs,
                                     const double fx,
                                     const double fy,
                                     const double cx,
                                     const double cy,
                                     const bool invert) {
    return new ceres::AutoDiffCostFunction<Sim3ReprojCost, 2, 3, 3, 1>(
        new Sim3ReprojCost(xyz_src, xy_obs, fx, fy, cx, cy, invert));
  }

  const Eigen::Vector3d xyz_src;
  const Eigen::Vector2d xy_obs;
  const double fx;
  const double fy;
  const double cx;
  const double cy;
  const bool invert;
};

void Sim3ToAngleAxis(const Sim3d& sim3,
                     double angle_axis[3],
                     double translation[3],
                     double* scale) {
  const Eigen::AngleAxisd aa(sim3.rotation());
  Eigen::Vector3d v = aa.axis() * aa.angle();
  angle_axis[0] = v.x();
  angle_axis[1] = v.y();
  angle_axis[2] = v.z();
  translation[0] = sim3.translation().x();
  translation[1] = sim3.translation().y();
  translation[2] = sim3.translation().z();
  *scale = sim3.scale();
}

Sim3d AngleAxisToSim3(const double angle_axis[3],
                      const double translation[3],
                      const double scale) {
  Eigen::Matrix3d R;
  ceres::AngleAxisToRotationMatrix(
      angle_axis, ceres::ColumnMajorAdapter3x3(R.data()));
  return Sim3d(scale,
               Eigen::Quaterniond(R).normalized(),
               Eigen::Vector3d(translation[0], translation[1], translation[2]));
}

}  // namespace

std::vector<char> ComputeSim3ReprojInliers(
    const Sim3d& cand_from_cur,
    const Camera& cam_current,
    const Camera& cam_candidate,
    const std::vector<OnlineSim3Correspondence>& corrs,
    const double max_reproj_error_px) {
  const double max_error_sq = max_reproj_error_px * max_reproj_error_px;
  std::vector<char> inliers(corrs.size(), 0);
  for (size_t i = 0; i < corrs.size(); ++i) {
    if (IsReprojInlier(
            cand_from_cur, cam_current, cam_candidate, corrs[i], max_error_sq)) {
      inliers[i] = 1;
    }
  }
  return inliers;
}

int OptimizeSim3Orb(const std::vector<OnlineSim3Correspondence>& corrs,
                    const Camera& cam_current,
                    const Camera& cam_candidate,
                    const OnlineSim3SolverOptions& options,
                    Sim3d* cand_from_cur,
                    std::vector<char>* inliers) {
  THROW_CHECK_NOTNULL(cand_from_cur);
  THROW_CHECK_NOTNULL(inliers);
  if (inliers->size() != corrs.size()) {
    *inliers = ComputeSim3ReprojInliers(*cand_from_cur,
                                        cam_current,
                                        cam_candidate,
                                        corrs,
                                        options.max_reproj_error_px);
  }

  const double fx1 = cam_current.FocalLengthX();
  const double fy1 = cam_current.FocalLengthY();
  const double cx1 = cam_current.PrincipalPointX();
  const double cy1 = cam_current.PrincipalPointY();
  const double fx2 = cam_candidate.FocalLengthX();
  const double fy2 = cam_candidate.FocalLengthY();
  const double cx2 = cam_candidate.PrincipalPointX();
  const double cy2 = cam_candidate.PrincipalPointY();

  // ORB OptimizeSim3: optimize(5) → drop chi2>th2 → optimize(5 or 10).
  const double chi2_th = options.optimize_chi2_th;
  const double huber_delta =
      options.huber_delta_px > 0.0 ? options.huber_delta_px
                                   : std::sqrt(std::max(chi2_th, 1e-6));

  auto RunOptimize = [&](const int max_iters) {
    double aa[3];
    double t[3];
    double scale = 1.0;
    Sim3ToAngleAxis(*cand_from_cur, aa, t, &scale);
    if (options.fix_scale) {
      scale = 1.0;
    }

    ceres::Problem problem;
    problem.AddParameterBlock(aa, 3);
    problem.AddParameterBlock(t, 3);
    problem.AddParameterBlock(&scale, 1);
    if (options.fix_scale) {
      problem.SetParameterBlockConstant(&scale);
    }

    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);

    int num_edges = 0;
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (!(*inliers)[i]) {
        continue;
      }
      const auto& c = corrs[i];
      // Forward: project S * Xc_cur into candidate image.
      problem.AddResidualBlock(
          Sim3ReprojCost::Create(
              c.xyz_cam_current, c.xy_candidate, fx2, fy2, cx2, cy2, false),
          loss,
          aa,
          t,
          &scale);
      // Inverse: project S^{-1} * Xc_cand into current image.
      problem.AddResidualBlock(
          Sim3ReprojCost::Create(
              c.xyz_cam_candidate, c.xy_current, fx1, fy1, cx1, cy1, true),
          loss,
          aa,
          t,
          &scale);
      num_edges += 2;
    }
    if (num_edges < 6) {
      return 0;
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = max_iters;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    *cand_from_cur = AngleAxisToSim3(aa, t, options.fix_scale ? 1.0 : scale);
    return num_edges;
  };

  auto RejectChi2Outliers = [&]() -> int {
    int n_bad = 0;
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (!(*inliers)[i]) {
        continue;
      }
      const auto& c = corrs[i];
      const Eigen::Vector3d xyz_cand = (*cand_from_cur) * c.xyz_cam_current;
      const Eigen::Vector3d xyz_cur =
          Inverse(*cand_from_cur) * c.xyz_cam_candidate;
      const auto p_cand = ProjectCam(cam_candidate, xyz_cand);
      const auto p_cur = ProjectCam(cam_current, xyz_cur);
      // ORB: reject if either edge chi2 > th2 (squared pixel error).
      if (!p_cand || !p_cur ||
          (*p_cand - c.xy_candidate).squaredNorm() > chi2_th ||
          (*p_cur - c.xy_current).squaredNorm() > chi2_th) {
        (*inliers)[i] = 0;
        ++n_bad;
      }
    }
    return n_bad;
  };

  if (RunOptimize(options.optimize_iterations) <= 0) {
    return CountInliers(*inliers);
  }
  const int n_bad = RejectChi2Outliers();
  // ORB: if nCorrespondences-nBad < 10 → fail (return 0).
  if (CountInliers(*inliers) < 10) {
    return 0;
  }
  const int second_iters = n_bad > 0 ? options.optimize_iterations_with_outliers
                                     : options.optimize_iterations;
  if (RunOptimize(second_iters) <= 0) {
    return CountInliers(*inliers);
  }
  RejectChi2Outliers();

  return CountInliers(*inliers);
}

bool EstimateSim3Orb(const std::vector<OnlineSim3Correspondence>& corrs,
                     const Camera& cam_current,
                     const Camera& cam_candidate,
                     const OnlineSim3SolverOptions& options,
                     Sim3d* cand_from_cur,
                     std::vector<char>* inliers,
                     int* num_inliers) {
  THROW_CHECK_NOTNULL(cand_from_cur);
  if (num_inliers != nullptr) {
    *num_inliers = 0;
  }
  const int N = static_cast<int>(corrs.size());
  if (N < std::max(options.min_inliers, 3)) {
    if (inliers != nullptr) {
      inliers->assign(corrs.size(), 0);
    }
    return false;
  }

  const double max_error_sq =
      options.max_reproj_error_px * options.max_reproj_error_px;

  // ORB SetRansacParameters iteration count.
  int max_its = options.max_iterations;
  if (options.min_inliers < N) {
    const double eps =
        static_cast<double>(options.min_inliers) / static_cast<double>(N);
    const double denom = std::log(1.0 - std::pow(eps, 3.0));
    if (std::isfinite(denom) && denom < 0.0) {
      const int n_its = static_cast<int>(
          std::ceil(std::log(1.0 - options.confidence) / denom));
      max_its = std::max(1, std::min(n_its, options.max_iterations));
    }
  } else {
    max_its = 1;
  }

  Sim3d best_model;
  std::vector<char> best_inliers(corrs.size(), 0);
  int best_n = 0;

  for (int it = 0; it < max_its; ++it) {
    // Sample 3 unique correspondences.
    const int i0 = RandomUniformInteger<int>(0, N - 1);
    int i1 = RandomUniformInteger<int>(0, N - 1);
    int i2 = RandomUniformInteger<int>(0, N - 1);
    if (i1 == i0) {
      i1 = (i0 + 1) % N;
    }
    if (i2 == i0 || i2 == i1) {
      i2 = (i0 + 2) % N;
    }

    Sim3d model;
    if (!ComputeSim3Horn(corrs[i0].xyz_cam_current,
                         corrs[i1].xyz_cam_current,
                         corrs[i2].xyz_cam_current,
                         corrs[i0].xyz_cam_candidate,
                         corrs[i1].xyz_cam_candidate,
                         corrs[i2].xyz_cam_candidate,
                         options.fix_scale,
                         &model)) {
      continue;
    }

    std::vector<char> cur_inliers(corrs.size(), 0);
    int n_inl = 0;
    for (size_t i = 0; i < corrs.size(); ++i) {
      if (IsReprojInlier(
              model, cam_current, cam_candidate, corrs[i], max_error_sq)) {
        cur_inliers[i] = 1;
        ++n_inl;
      }
    }
    if (n_inl > best_n) {
      best_n = n_inl;
      best_model = model;
      best_inliers = std::move(cur_inliers);
      if (best_n >= options.min_inliers) {
        // ORB early exit once a good model is found.
        break;
      }
    }
  }

  if (best_n < options.min_inliers) {
    if (inliers != nullptr) {
      *inliers = std::move(best_inliers);
    }
    if (num_inliers != nullptr) {
      *num_inliers = best_n;
    }
    return false;
  }

  *cand_from_cur = best_model;
  std::vector<char> opt_inliers = best_inliers;
  const int n_opt = OptimizeSim3Orb(corrs,
                                    cam_current,
                                    cam_candidate,
                                    options,
                                    cand_from_cur,
                                    &opt_inliers);
  if (n_opt < options.min_inliers) {
    // Fall back to RANSAC model if optimization collapses inliers.
    *cand_from_cur = best_model;
    if (inliers != nullptr) {
      *inliers = std::move(best_inliers);
    }
    if (num_inliers != nullptr) {
      *num_inliers = best_n;
    }
    return true;
  }

  if (inliers != nullptr) {
    *inliers = std::move(opt_inliers);
  }
  if (num_inliers != nullptr) {
    *num_inliers = n_opt;
  }
  return true;
}

}  // namespace colmap
