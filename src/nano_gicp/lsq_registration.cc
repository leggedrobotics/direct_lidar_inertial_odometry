/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

/***********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, SMRT-AIST
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "dlio/dlio.h"
#include "nano_gicp/lsq_registration.h"

template class nano_gicp::LsqRegistration<PointType, PointType>;

namespace nano_gicp {

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::LsqRegistration() {
  this->reg_name_ = "LsqRegistration";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  lm_debug_print_ = false;
  lm_max_iterations_ = 15;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
  freeze_trial_correspondences_ = true;

  final_hessian_.setIdentity();
  final_error_ = 0.;
}

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::~LsqRegistration() {}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setTransformationEpsilon(double eps) {
  transformation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setMaximumIterations(int iter) {
  max_iterations_ = iter;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setFreezeTrialCorrespondences(bool freeze) {
  freeze_trial_correspondences_ = freeze;
}

template <typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6>& LsqRegistration<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::getFinalError() const {
  return final_error_;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  lm_lambda_ = -1.0;
  converged_ = false;

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  for (int i = 0; i < max_iterations_ && !converged_; i++) {
    nr_iterations_ = i;

    Eigen::Isometry3d delta;
    if (!step_optimize(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::computeLinearizationAtGuess(
    const Matrix4& guess,
    Eigen::Matrix<double, 6, 6>& H,
    Eigen::Matrix<double, 6, 1>& b,
    double& error) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  error = linearize(x0, &H, &b);
  H = 0.5 * (H + H.transpose());

  return std::isfinite(error) && H.allFinite() && b.allFinite();
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  double accum = 0.0;
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return step_lm(x0, delta);
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return step_gn(x0, delta);
  }

  return step_lm(x0, delta);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = linearize(x0, &H, &b);

  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
  Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

  delta.setIdentity();
  delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  x0 = delta * x0;
  final_hessian_ = H;
  final_error_ = y0;

  return true;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm(Eigen::Isometry3d& x0,
                                                        Eigen::Isometry3d& delta) {
  // Linearize at current x0
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  const double y0 = linearize(x0, &H, &b);

  // Symmetrize (safer rho prediction)
  H = 0.5 * (H + H.transpose());

  // Column scaling: S = diag(1 / sqrt(max(|Hii|, eps)))
  constexpr double kEps = 1e-12;
  Eigen::Matrix<double, 6, 1> s = H.diagonal().cwiseAbs().cwiseMax(kEps).cwiseSqrt();
  Eigen::Matrix<double, 6, 6> S = s.cwiseInverse().asDiagonal();

  // Scaled system
  Eigen::Matrix<double, 6, 6> Hs = S * H * S;
  Eigen::Matrix<double, 6, 1> bs = S * b;

  // Init LM lambda (scaled space)
  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * Hs.diagonal().cwiseAbs().maxCoeff();
  }

  double nu = 2.0;
  constexpr double kLambdaMin = 1e-18;
  constexpr double kLambdaMax = 1e30;

  // Trust-region caps (tune to your scene scale)
  constexpr double kMaxRotStep   = 0.35;  // rad
  constexpr double kMaxTransStep = 0.30;  // m

  // Diagonal damping in scaled space
  const Eigen::Matrix<double, 6, 6> diagHs = Hs.diagonal().asDiagonal();

  for (int i = 0; i < lm_max_iterations_; ++i) {
    // (Hs + λ·diag(Hs)) ds = -bs
    const Eigen::Matrix<double, 6, 6> A = Hs + lm_lambda_ * diagHs;

    Eigen::Matrix<double, 6, 1> ds;
    bool solved = false;

    // Try LDLT first
    {
      Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(A);
      if (ldlt.info() == Eigen::Success) {
        ds = ldlt.solve(-bs);
        solved = (ldlt.info() == Eigen::Success) && ds.allFinite();
      }
    }

    // Fallback: SVD (robust for near-singular A)
    if (!solved) {
      Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const auto& sing = svd.singularValues();
      Eigen::Matrix<double, 6, 6> S_inv = Eigen::Matrix<double, 6, 6>::Zero();
      // Threshold relative to max singular value
      const double rel = 1e-12 * std::max(1.0, sing(0));
      for (int k = 0; k < 6; ++k) if (sing(k) > rel) S_inv(k, k) = 1.0 / sing(k);
      ds = svd.matrixV() * S_inv * svd.matrixU().transpose() * (-bs);
      solved = ds.allFinite();
    }

    if (!solved) {
      // Increase λ and retry
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // Undo scaling: d = S * ds
    Eigen::Matrix<double, 6, 1> d = S * ds;

    // Trust-region cap (on original variables)
    const double rn = d.head<3>().norm();
    const double tn = d.tail<3>().norm();
    double fac = 1.0;
    if (rn > kMaxRotStep)   fac = std::min(fac, kMaxRotStep   / rn);
    if (tn > kMaxTransStep) fac = std::min(fac, kMaxTransStep / tn);
    if (fac < 1.0) d *= fac;

    // Build δ ∈ SE(3) and evaluate trial
    delta.setIdentity();
    delta.linear()      = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    const Eigen::Isometry3d xi = delta * x0;
    // Fixed mode makes the trial objective consistent with the linearization;
    // rematching mode retains the original ICP behavior for weak geometry.
    const double yi = freeze_trial_correspondences_
        ? compute_error_frozen(xi)
        : compute_error(xi);
    if (!std::isfinite(yi)) {
      lm_lambda_ = std::min(kLambdaMax, std::max(2.0 * lm_lambda_, lm_lambda_ * nu));
      nu = std::min(2.0 * nu, 1e6);
      continue;
    }

    // Predicted reduction of the cost sum(e'Me) with b = J'Me and H = J'MJ: pred = -(2 dᵀ b + dᵀ H d).
    // (The former -dᵀ b - 0.5 dᵀ H d was half of it, so the gain ratio rho came out 2x too large.)
    const double Hd = (H * d).dot(d);
    const double pred = -2.0 * d.dot(b) - Hd;

    double rho = -1.0;
    if (std::isfinite(pred) && pred > 0.0) {
      rho = (y0 - yi) / pred;
    }

    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format(
          "--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n")
          % "i" % "y0" % "yi" % "rho" % "lambda" % "|d|" % "dec";
      }
      const char dec = (rho > 0.0) ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c")
        % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho > 0.0 && yi < y0) {
      // Accept
      x0 = xi;
      // Nielsen update
      lm_lambda_ = std::max(kLambdaMin,
                            lm_lambda_ * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3)));
      nu = 2.0;

      final_hessian_ = H;
      final_error_   = yi;
      return true;
    }

    // Reject: increase λ and retry
    lm_lambda_ = std::min(kLambdaMax, lm_lambda_ * nu);
    nu = std::min(2.0 * nu, 1e6);

    if (!d.allFinite() || d.norm() < 1e-12) {
      delta.setIdentity();
      final_hessian_ = H;
      final_error_   = y0;
      return true;
    }
  }

  delta.setIdentity();
  return false;
}

}  // namespace nano_gicp
