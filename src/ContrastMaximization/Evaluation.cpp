#include "ContrastMaximization/Evaluation.h"

#include <limits>
#include <opencv2/opencv.hpp>

double computeVariance(const cv::Mat& image) {
  CV_Assert(image.type() == CV_64FC1);

  cv::Scalar mean, stddev;
  cv::meanStdDev(image, mean, stddev);
  return stddev[0] * stddev[0];  // variance = (stddev)^2
}

bool isNearAngle(double angle_deg, double target_deg, double tol_deg) {
  double diff = std::fabs(angle_deg - target_deg);
  if (diff > 180.0) {
    diff = 360.0 - diff;
  }
  return diff <= tol_deg;
}

bool isOptimizationResultValid(const cv::Mat iwe_cv, const std::vector<double>& xs, const std::vector<double>& ys,
                               const std::vector<double>& ts, const std::vector<double>& ps, const int height,
                               const int width) {
  cv::Mat no_warp_image = accumulateEventsToImage(xs, ys, ps, height, width);
  double var_iwe = computeVariance(iwe_cv);
  double var_unwarp = computeVariance(no_warp_image);

  return var_iwe > var_unwarp;
}