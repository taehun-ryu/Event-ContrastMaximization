#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

/// @brief Split event component vectors into temporal bins of fixed duration.
/// @param xs         Input X coordinates of events.
/// @param ys         Input Y coordinates of events.
/// @param ts         Input timestamps of events (must be sorted).
/// @param ps         Input polarities of events (+1.0 or -1.0).
/// @param bin_width  Duration of each time bin.
/// @param[out] xs_bins  Binned X coordinates.
/// @param[out] ys_bins  Binned Y coordinates.
/// @param[out] ts_bins  Binned timestamps.
/// @param[out] ps_bins  Binned polarities.
void splitComponentVectorsByTime(const std::vector<double>& xs, const std::vector<double>& ys,
                                 const std::vector<double>& ts, const std::vector<double>& ps, double bin_width,
                                 std::vector<std::vector<double>>& xs_bins, std::vector<std::vector<double>>& ys_bins,
                                 std::vector<std::vector<double>>& ts_bins, std::vector<std::vector<double>>& ps_bins);

/// @brief Accumulate events into a 2D image by summing polarities at each pixel.
/// @param xs      X coordinates of events.
/// @param ys      Y coordinates of events.
/// @param ps      Polarity values of events (+1.0 or -1.0).
/// @param height  Image height (number of rows).
/// @param width   Image width (number of columns).
/// @return        An OpenCV image representing accumulated polarities.
cv::Mat accumulateEventsToImage(const std::vector<double>& xs, const std::vector<double>& ys,
                                const std::vector<double>& ps, int height, int width);