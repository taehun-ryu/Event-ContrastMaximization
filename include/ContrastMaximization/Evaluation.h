#include <opencv2/opencv.hpp>

#include "EventUtils/Processing.h"

/// @brief Compute variance of a CV_64FC1 image.
/// @param image Input OpenCV image (must be CV_64FC1).
/// @return Variance of the pixel values.
double computeVariance(const cv::Mat& image);

/// @brief Check whether a given angle (deg) is close to a target angle (deg) within a tolerance.
/// @param angle_deg  Angle to check [0, 360)
/// @param target_deg Target reference angle [0, 360)
/// @param tol_deg    Tolerance in degrees (default: 10)
/// @return true if angle_deg is within tol_deg of target_deg
bool isNearAngle(double angle_deg, double target_deg, double tol_deg = 10.0);

bool isOptimizationResultValid(const cv::Mat iwe_cv, const std::vector<double>& xs, const std::vector<double>& ys,
                               const std::vector<double>& ts, const std::vector<double>& ps, const int height,
                               const int width);