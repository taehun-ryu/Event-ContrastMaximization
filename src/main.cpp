#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "ContrastMaximization/Optimization.h"
#include "EventUtils/IO.h"
#include "EventUtils/Processing.h"

/// @brief Create output directory with current datetime in format YYYYMMDD_HHMMSS
/// @return Full path to the created directory
std::string createOutputDirectory() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm tm_local = *std::localtime(&now_c);

  std::ostringstream oss;
  oss << "../output/" << std::put_time(&tm_local, "%Y%m%d_%H%M%S");
  std::string dir_path = oss.str();

  if (!std::filesystem::exists(dir_path)) {
    std::filesystem::create_directories(dir_path);
  }

  return dir_path;
}

/// @brief Compute variance of a CV_64FC1 image.
/// @param image Input OpenCV image (must be CV_64FC1).
/// @return Variance of the pixel values.
double computeVariance(const cv::Mat& image) {
  CV_Assert(image.type() == CV_64FC1);

  cv::Scalar mean, stddev;
  cv::meanStdDev(image, mean, stddev);
  return stddev[0] * stddev[0];  // variance = (stddev)^2
}

/// @brief Check whether a given angle (deg) is close to a target angle (deg) within a tolerance.
/// @param angle_deg  Angle to check [0, 360)
/// @param target_deg Target reference angle [0, 360)
/// @param tol_deg    Tolerance in degrees (default: 10)
/// @return true if angle_deg is within tol_deg of target_deg
bool isNearAngle(double angle_deg, double target_deg, double tol_deg = 10.0) {
  double diff = std::fabs(angle_deg - target_deg);
  if (diff > 180.0) {
    diff = 360.0 - diff;
  }
  return diff <= tol_deg;
}

int main() {
  // Load configuration from YAML file
  YAML::Node config = YAML::LoadFile("../config/davis346.yaml");  // Assume the command is run in build directory.
  std::string h5_path = config["dataset"]["h5_path"].as<std::string>();
  int width = config["dataset"]["width"].as<int>();
  int height = config["dataset"]["height"].as<int>();
  double bin_width = config["dataset"]["bin_width"].as<double>();

  // Load events from HDF5 file
  std::vector<double> xs, ys, ts, ps;
  readEventsAsComponentVectors(h5_path, xs, ys, ts, ps);
  std::cout << "[Main] Loaded " << ts.size() << " events from: " << h5_path << std::endl;

  // Split events into time bins
  std::vector<std::vector<double>> xs_bins, ys_bins, ts_bins, ps_bins;
  splitComponentVectorsByTime(xs, ys, ts, ps, bin_width, xs_bins, ys_bins, ts_bins, ps_bins);
  std::cout << "[Main] Split into " << xs_bins.size() << " time bins with width " << bin_width << " seconds\n";

  // Logging config
  std::string out_dir = createOutputDirectory();  // save
  int valid_count = 0;                            // acceptance

  for (size_t i = 0; i < xs_bins.size(); ++i) {
    const auto& xs_i = xs_bins[i];
    const auto& ys_i = ys_bins[i];
    const auto& ts_i = ts_bins[i];
    const auto& ps_i = ps_bins[i];
    if (ts_i.empty()) continue;

    double t_ref = ts_i[0];
    double params[2] = {1.0, 1.0};  // THINK Always start at (1, 1). Is it bset way?

    // Setup Ceres optimization
    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ContrastMaximizationCost, 1, 2>(
        new ContrastMaximizationCost(xs_i, ys_i, ts_i, ps_i, t_ref, height, width));
    problem.AddResidualBlock(cost_function, nullptr, params);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "[Bin " << (i + 1) << "/" << xs_bins.size() << "] vx = " << params[0] << ", vy = " << params[1]
              << "\n";

    // Generate IWE with optimized params
    JetImage<double> iwe(height, width);
    LinearWarpFunction warp_fn;
    generateIWE_usingWarpFunction(xs_i, ys_i, ts_i, ps_i, params, t_ref, warp_fn, iwe);

    // Convert IWE to cv::Mat
    cv::Mat iwe_cv(height, width, CV_64FC1);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) iwe_cv.at<double>(y, x) = iwe(y, x);
    // No-warp image
    cv::Mat no_warp_img = accumulateEventsToImage(xs_i, ys_i, ps_i, height, width);
    // Filtering
    // 1. Based on image variance
    double var_iwe = computeVariance(iwe_cv);
    double var_image = computeVariance(no_warp_img);
    if (var_iwe <= var_image) {
      std::cout << "  [Skip] On Bin " << (i + 1) << ", Image Variance is unreliable." << "\n";
      continue;
    }
    // 2. Based on \theta
    double r = std::sqrt(params[0] * params[0] + params[1] * params[1]);
    double theta_rad = std::atan2(params[1], params[0]);
    double theta_deg = theta_rad * 180.0 / M_PI;
    if (theta_deg < 0.0) theta_deg += 360.0;
    std::cout << "  r = " << r << ", theta = " << theta_deg << " deg\n";

    if (isNearAngle(theta_deg, 0.0) || isNearAngle(theta_deg, 90.0) || isNearAngle(theta_deg, 180.0) ||
        isNearAngle(theta_deg, 270.0)) {
      std::cout << "  [Skip] Angle is near 0, 90, 180, or 270 degrees.\n";
      continue;
    }
    valid_count++;

    // TODO Corner detection or Intensity recon(e.g. QR-generateion)?

    // Narmalize and convert fot visualization
    cv::Mat iwe_vis;
    cv::normalize(iwe_cv, iwe_vis, 0, 255, cv::NORM_MINMAX);
    iwe_vis.convertTo(iwe_vis, CV_8UC1);
    cv::Mat no_warp_img_vis;
    cv::normalize(no_warp_img, no_warp_img_vis, 0, 255, cv::NORM_MINMAX);
    no_warp_img_vis.convertTo(no_warp_img_vis, CV_8UC1);
    // // Visualize
    // cv::imshow("IWE", iwe_vis);
    // cv::imshow("No Warping", no_warp_img_vis);
    // cv::waitKey(500);
    // Save
    std::string out_path = out_dir + "/iwe_" + valid_count + "_bin" + std::to_string(i + 1) + ".png";
    cv::imwrite(out_path, iwe_vis);
    std::cout << "  [Saved] IWE image " << valid_count << " to: " << out_path << "\n";
  }
  double acceptance_rate;
  acceptance_rate = (valid_count / xs_bins.size()) * 100;
  std::cout << "[Finish] IWE acceptance rate is " << acceptance_rate << "%";

  return 0;
}