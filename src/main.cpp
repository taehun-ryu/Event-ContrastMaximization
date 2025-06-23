#include <iomanip>
#include <limits>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "EventData/EventArray.h"
#include "EventData/H5Reader.h"

int main() {
  // Load configuration from YAML file
  YAML::Node config = YAML::LoadFile("../config/davis346.yaml");  // Assume the command is run in /IBEC3/build directory.
  std::string path = config["dataset"]["h5_path"].as<std::string>();
  int width = config["dataset"]["width"].as<int>();
  int height = config["dataset"]["height"].as<int>();
  double bin_width = config["dataset"]["bin_width"].as<double>();

  H5Reader reader(path);
  auto full_events = reader.readEvents();

  full_events->sortByTime();
  full_events->normalizeTimestampsToStart();

  // Tencode
  auto bins = full_events->splitByTimeBin(bin_width);
  std::cout << "Split into " << bins.size() << " time bins." << std::endl;

  return 0;
}