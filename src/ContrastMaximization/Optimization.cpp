#include "ContrastMaximization/Optimization.h"

#include <opencv2/opencv.hpp>

void optimize(std::vector<double>& params, const std::vector<double>& xs, const std::vector<double>& ys,
              const std::vector<double>& ts, const std::vector<double>& ps, const double& t_ref, const int& width,
              const int& height) {
  ceres::Problem problem;
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ContrastMaximizationCost, 1, 2>(
      new ContrastMaximizationCost(xs, ys, ts, ps, t_ref, height, width));
  problem.AddResidualBlock(cost_function, nullptr, params.data());

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 20;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

cv::Mat getIWE(JetImage<double>& iwe, const LinearWarpFunction& warp_function, std::vector<double>& params,
               const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts,
               const std::vector<double>& ps, const double& t_ref, const int width, const int height) {
  // Generate IWE with optimized params
  generateIWE_usingWarpFunction(xs, ys, ts, ps, params.data(), t_ref, warp_function, iwe);
  // Convert IWE to cv::Mat
  cv::Mat iwe_cv(height, width, CV_64FC1);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) iwe_cv.at<double>(y, x) = iwe(y, x);

  return iwe_cv;
}
