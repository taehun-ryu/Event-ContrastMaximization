#include "EventUtils/Processing.h"

void splitComponentVectorsByTime(const std::vector<double>& xs, const std::vector<double>& ys,
                                 const std::vector<double>& ts, const std::vector<double>& ps, double bin_width,
                                 std::vector<std::vector<double>>& xs_bins, std::vector<std::vector<double>>& ys_bins,
                                 std::vector<std::vector<double>>& ts_bins, std::vector<std::vector<double>>& ps_bins) {
  const size_t N = ts.size();
  if (N == 0) return;

  double t_curr = ts[0];
  double t_next = t_curr + bin_width;

  std::vector<double> xb, yb, tb, pb;

  for (size_t i = 0; i < N; ++i) {
    double t = ts[i];

    while (t >= t_next) {
      if (!xb.empty()) {
        xs_bins.push_back(std::move(xb));
        ys_bins.push_back(std::move(yb));
        ts_bins.push_back(std::move(tb));
        ps_bins.push_back(std::move(pb));
      }
      xb.clear();
      yb.clear();
      tb.clear();
      pb.clear();
      t_curr = t_next;
      t_next = t_curr + bin_width;
    }

    xb.push_back(xs[i]);
    yb.push_back(ys[i]);
    tb.push_back(ts[i]);
    pb.push_back(ps[i]);
  }
  if (!xb.empty()) {
    xs_bins.push_back(std::move(xb));
    ys_bins.push_back(std::move(yb));
    ts_bins.push_back(std::move(tb));
    ps_bins.push_back(std::move(pb));
  }
}

cv::Mat accumulateEventsToImage(const std::vector<double>& xs, const std::vector<double>& ys,
                                const std::vector<double>& ps, int height, int width) {
  cv::Mat img(height, width, CV_64FC1, cv::Scalar(0.0));
  for (size_t i = 0; i < xs.size(); ++i) {
    int x = static_cast<int>(xs[i]);
    int y = static_cast<int>(ys[i]);
    if (x >= 0 && x < width && y >= 0 && y < height) {
      img.at<double>(y, x) += ps[i];  // accumulate polarity
    }
  }
  return img;
}
