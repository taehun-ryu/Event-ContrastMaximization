#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

/// @brief Reads event data from an HDF5 file and fills component vectors.
/// @param file_path Path to the HDF5 file.
/// @param[out] xs x-coordinates (double).
/// @param[out] ys y-coordinates (double).
/// @param[out] ts timestamps (double).
/// @param[out] ps polarities (+1.0 for ON, -1.0 for OFF).
void readEventsAsComponentVectors(const std::string& file_path, std::vector<double>& xs, std::vector<double>& ys,
                                  std::vector<double>& ts, std::vector<double>& ps);

/// @brief Reads event data from an HDF5 file and returns it as a 4xN matrix.
/// @details The matrix rows represent [x; y; t; p] respectively.
/// @param file_path Path to the HDF5 file.
/// @return A 4xN Eigen::MatrixXd containing [x; y; t; p] for each event.
Eigen::MatrixXd readEventsAsMatrix(const std::string& file_path);