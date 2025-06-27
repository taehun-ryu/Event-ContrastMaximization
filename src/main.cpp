#include <yaml-cpp/yaml.h>

#include <iomanip>
#include <limits>
#include <opencv2/opencv.hpp>

#include "ContrastMaximization/Optimization.h"
#include "EventUtils/IO.h"
#include "EventUtils/Processing.h"

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

  for (size_t i = 0; i < xs_bins.size(); ++i) {
    const auto& xs_i = xs_bins[i];
    const auto& ys_i = ys_bins[i];
    const auto& ts_i = ts_bins[i];
    const auto& ps_i = ps_bins[i];
    if (ts_i.empty()) continue;

    double t_ref = ts_i[0];
    double params[2] = {1.0, 1.0};

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

    std::cout << "[Bin " << i << "] vx = " << params[0] << ", vy = " << params[1] << "\n";

    // Generate IWE with optimized params
    JetImage<double> iwe(height, width);
    LinearWarpFunction warp_fn;
    generateIWE_usingWarpFunction(xs_i, ys_i, ts_i, ps_i, params, t_ref, warp_fn, iwe);

    // Convert IWE to cv::Mat
    cv::Mat iwe_img(height, width, CV_64FC1);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) iwe_img.at<double>(y, x) = iwe(y, x);

    // Normalize
    cv::Mat vis_img;
    cv::normalize(iwe_img, vis_img, 0, 255, cv::NORM_MINMAX);
    vis_img.convertTo(vis_img, CV_8UC1);

    // No-warp image
    cv::Mat no_warp_img = accumulateEventsToImage(xs_i, ys_i, ps_i, height, width);
    cv::normalize(no_warp_img, no_warp_img, 0, 255, cv::NORM_MINMAX);
    no_warp_img.convertTo(no_warp_img, CV_8UC1);

    // Visualize
    cv::imshow("IWE", vis_img);
    cv::imshow("No Warping", no_warp_img);
    cv::waitKey(500);
  }

  return 0;
}