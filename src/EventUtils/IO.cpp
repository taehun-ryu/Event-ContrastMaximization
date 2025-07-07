#include "EventUtils/IO.h"

#include <H5Cpp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>

namespace {

struct EventBuffers {
  std::vector<uint16_t> xs, ys;
  std::vector<double> ts;
  std::vector<uint8_t> ps;
};

EventBuffers readEventsFromHDF5(const std::string& file_path, size_t& num_events) {
  H5::H5File file(file_path, H5F_ACC_RDONLY);
  H5::Group grp = file.openGroup("/events");

  H5::DataSet ds_x = grp.openDataSet("xs");
  H5::DataSpace space = ds_x.getSpace();
  hsize_t n;
  space.getSimpleExtentDims(&n, nullptr);
  num_events = static_cast<size_t>(n);

  EventBuffers buf;
  buf.xs.resize(num_events);
  buf.ys.resize(num_events);
  buf.ts.resize(num_events);
  buf.ps.resize(num_events);

  ds_x.read(buf.xs.data(), H5::PredType::NATIVE_UINT16);
  grp.openDataSet("ys").read(buf.ys.data(), H5::PredType::NATIVE_UINT16);
  grp.openDataSet("ts").read(buf.ts.data(), H5::PredType::NATIVE_DOUBLE);
  grp.openDataSet("ps").read(buf.ps.data(), H5::PredType::NATIVE_UINT8);

  return buf;
}

}  // anonymous namespace

void readEventsAsComponentVectors(const std::string& file_path, std::vector<double>& xs, std::vector<double>& ys,
                                  std::vector<double>& ts, std::vector<double>& ps) {
  size_t N;
  EventBuffers buf = readEventsFromHDF5(file_path, N);

  xs.resize(N);
  ys.resize(N);
  ts.resize(N);
  ps.resize(N);

  for (size_t i = 0; i < N; ++i) {
    xs[i] = static_cast<double>(buf.xs[i]);
    ys[i] = static_cast<double>(buf.ys[i]);
    ts[i] = buf.ts[i];
    ps[i] = buf.ps[i] ? +1.0 : -1.0;
  }
}

Eigen::MatrixXd readEventsAsMatrix(const std::string& file_path) {
  size_t N;
  EventBuffers buf = readEventsFromHDF5(file_path, N);

  Eigen::MatrixXd events(4, N);
  for (size_t i = 0; i < N; ++i) {
    events(0, i) = buf.xs[i];
    events(1, i) = buf.ys[i];
    events(2, i) = buf.ts[i];
    events(3, i) = buf.ps[i] ? +1.0 : -1.0;
  }

  return events;
}

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