#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "ContrastMaximization/Evaluation.h"
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
  int board_h = config["checkerboard"]["board_h"].as<int>();
  int board_w = config["checkerboard"]["board_w"].as<int>();
  double square_size = config["checkerboard"]["square_size"].as<double>();

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

  // Test
  std::vector<double> params = {0.1, 0.1};

  for (size_t i = 0; i < xs_bins.size(); ++i) {
    const auto& xs_i = xs_bins[i];
    const auto& ys_i = ys_bins[i];
    const auto& ts_i = ts_bins[i];
    const auto& ps_i = ps_bins[i];
    if (ts_i.empty()) continue;

    double t_ref = ts_i[0];

    // Constrast Maximization
    optimize(params, xs_i, ys_i, ts_i, ps_i, t_ref, width, height);
    std::cout << "[Bin " << (i + 1) << "/" << xs_bins.size() << "] vx = " << params[0] << ", vy = " << params[1]
              << "\n";
    // Get sharp IWE
    JetImage<double> iwe_jet(height, width);
    LinearWarpFunction warp_fn;
    cv::Mat iwe_cv = getIWE(iwe_jet, warp_fn, params, xs_i, ys_i, ts_i, ps_i, t_ref, width, height);

    // Filtering
    if (!isOptimizationResultValid(iwe_cv, xs_i, ys_i, ts_i, ps_i, height, width)) continue;

    valid_count++;
    std::cout << "  [Accepted] IWE is valid! #accpeted is " << valid_count << "\n";

    // TODO Corner detection or Intensity recon(e.g. QR-generateion)?

    //***** Just eval */
    cv::Mat iwe_vis;
    cv::normalize(iwe_cv, iwe_vis, 0, 255, cv::NORM_MINMAX);
    iwe_vis.convertTo(iwe_vis, CV_8UC1);
    // Save
    std::string out_path = out_dir + "/iwe_bin" + std::to_string(i + 1) + ".png";
    cv::imwrite(out_path, iwe_vis);
    std::cout << "  [Saved] IWE image " << valid_count << " to: " << out_path << "\n";
  }
  double acceptance_rate;
  acceptance_rate = (static_cast<double>(valid_count) / xs_bins.size()) * 100.0;
  std::cout << "[Finish] IWE acceptance rate is " << acceptance_rate << "%" << "\n";

  return 0;
}