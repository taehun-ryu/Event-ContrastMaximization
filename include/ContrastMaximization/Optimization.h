#pragma once
#include <ceres/ceres.h>
#include <ceres/internal/port.h>
#include <ceres/jet.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <vector>

template <typename T>
struct is_jet : std::false_type {};

template <typename T, int N>
struct is_jet<ceres::Jet<T, N>> : std::true_type {};

template <typename T>
inline int floorToInt(const T& x) {
  if constexpr (is_jet<T>::value) {
    return static_cast<int>(ceres::floor(x).a);
  } else {
    return static_cast<int>(std::floor(x));
  }
}

class LinearWarpFunction {
 public:
  virtual ~LinearWarpFunction() = default;

  /// @brief Warp events given parameters.
  /// @tparam T Ceres Jet type (for autodiff)
  /// @param xs Input x coordinates (N,)
  /// @param ys Input y coordinates (N,)
  /// @param ts Timestamps of events (N,)
  /// @param t0 Reference time
  /// @param params Motion parameters (vx, vy)
  /// @param[out] x_warped Output warped x coordinates
  /// @param[out] y_warped Output warped y coordinates
  template <typename T>
  void warp(const std::vector<T>& xs, const std::vector<T>& ys, const std::vector<T>& ts, const T& t0, const T* params,
            std::vector<T>& x_warped, std::vector<T>& y_warped) const {
    size_t N = xs.size();
    x_warped.resize(N);
    y_warped.resize(N);

    const T vx = params[0];
    const T vy = params[1];

    for (size_t i = 0; i < N; ++i) {
      T dt = ts[i] - t0;
      x_warped[i] = xs[i] - dt * vx;
      y_warped[i] = ys[i] - dt * vy;
    }
  }
};

/// @brief JetImage class for accumulating warped events into an image.
/// @tparam T
template <typename T>
class JetImage {
 public:
  JetImage(int height, int width) : H_(height), W_(width), data_(height * width, T(0)) {}

  T& operator()(int y, int x) { return data_[y * W_ + x]; }

  const T& operator()(int y, int x) const { return data_[y * W_ + x]; }

  int height() const { return H_; }
  int width() const { return W_; }

 private:
  int H_, W_;
  std::vector<T> data_;
};

/// @brief Accumulate warped events into an IWE image using bilinear interpolation.
/// @param x_warped Warped x positions
/// @param y_warped Warped y positions
/// @param ps Event polarities (or weights)
/// @param[out] iwe Image to accumulate into
template <typename T>
void accumulateToIWE(const std::vector<T>& x_warped, const std::vector<T>& y_warped, const std::vector<T>& ps,
                     JetImage<T>& iwe) {
  int H = iwe.height();
  int W = iwe.width();

  for (size_t i = 0; i < x_warped.size(); ++i) {
    T x = x_warped[i];
    T y = y_warped[i];

    if (x < T(0) || x >= T(W - 1) || y < T(0) || y >= T(H - 1)) {
      continue;
    }

    int x0 = floorToInt(x);
    int y0 = floorToInt(y);
    if (x0 < 0 || x0 + 1 >= W || y0 < 0 || y0 + 1 >= H) continue;

    T dx = x - T(x0);
    T dy = y - T(y0);

    for (int dy_i = 0; dy_i <= 1; ++dy_i) {
      for (int dx_i = 0; dx_i <= 1; ++dx_i) {
        int xi = x0 + dx_i;
        int yi = y0 + dy_i;

        if (xi >= 0 && xi < W && yi >= 0 && yi < H) {
          T wx = (dx_i == 0) ? (T(1) - dx) : dx;
          T wy = (dy_i == 0) ? (T(1) - dy) : dy;
          T w = wx * wy;
          iwe(yi, xi) += ps[i] * w;
        }
      }
    }
  }
}

/// @brief Generate an Image of Warped Events (IWE) using a warp function and bilinear interpolation.
/// @param xs Original x positions of events
/// @param ys Original y positions of events
/// @param ts Timestamps of events
/// @param ps Event polarities (or weights)
/// @param params Warp parameters (e.g., vx, vy)
/// @param t_ref Reference time
/// @param warp_fn Warp function object
/// @param[out] iwe JetImage to be filled
template <typename T>
void generateIWE_usingWarpFunction(const std::vector<T>& xs, const std::vector<T>& ys, const std::vector<T>& ts,
                                   const std::vector<T>& ps, const T* params, const T& t_ref,
                                   const LinearWarpFunction& warp_fn, JetImage<T>& iwe) {
  std::vector<T> x_warped, y_warped;
  warp_fn.warp(xs, ys, ts, t_ref, params, x_warped, y_warped);
  accumulateToIWE(x_warped, y_warped, ps, iwe);
}

/// @brief Compute gradient-magnitude of an IWE.
/// @param iwe Input IWE image (already filled)
/// @return Sum of squared gradient magnitudes
template <typename T>
T computeIweGradientMagnitude(const JetImage<T>& iwe) {  // TODO Check whether this impl; is right.
  int H = iwe.height();
  int W = iwe.width();
  T total = T(0);

  for (int y = 1; y < H - 1; ++y) {
    for (int x = 1; x < W - 1; ++x) {
      T dx = (iwe(y, x + 1) - iwe(y, x - 1)) * T(0.5);
      T dy = (iwe(y + 1, x) - iwe(y - 1, x)) * T(0.5);
      total += dx * dx + dy * dy;
    }
  }

  return total;
}

/// @brief Compute variance of IWE
/// @param iwe Input IWE image (already filled)
/// @return Variance
template <typename T>
T computeIweVariance(const JetImage<T>& iwe) {
  int H = iwe.height();
  int W = iwe.width();
  T sum = T(0);
  T sum_sq = T(0);
  int N = H * W;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      T val = iwe(y, x);
      sum += val;
      sum_sq += val * val;
    }
  }

  T mean = sum / T(N);
  T variance = (sum_sq / T(N)) - (mean * mean);

  return variance;
}

/// @brief Compute Sum of Squares (SoS) of the IWE image.
/// @tparam T Ceres Jet type or double.
/// @param iwe Input IWE image (JetImage<T>)
/// @return Sum of squared pixel values.
template <typename T>
T computeIweSumOfSquares(const JetImage<T>& iwe) {
  int H = iwe.height();
  int W = iwe.width();
  T sum_sq = T(0);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      T val = iwe(y, x);
      sum_sq += val * val;
    }
  }

  return sum_sq;
}

/// @brief Ceres CostFunctor for optimizing contrast via IWE loss.
struct ContrastMaximizationCost {
  ContrastMaximizationCost(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts,
                           const std::vector<double>& ps, double t_ref, int height, int width)
      : xs_(xs), ys_(ys), ts_(ts), ps_(ps), t_ref_(t_ref), H_(height), W_(width) {}

  template <typename T>
  bool operator()(const T* const params, T* residual) const {
    std::vector<T> xs_j(xs_.begin(), xs_.end());
    std::vector<T> ys_j(ys_.begin(), ys_.end());
    std::vector<T> ts_j(ts_.begin(), ts_.end());
    std::vector<T> ps_j(ps_.begin(), ps_.end());

    JetImage<T> iwe(H_, W_);
    for (int y = 0; y < H_; ++y) {
      for (int x = 0; x < W_; ++x) {
        iwe(y, x) = T(0);
      }
    }
    LinearWarpFunction warp_fn;
    generateIWE_usingWarpFunction(xs_j, ys_j, ts_j, ps_j, params, T(t_ref_), warp_fn, iwe);

    // Compute the loss of the IWE
    T loss = computeIweVariance(iwe);
    residual[0] = T(1.0) / sqrt(loss + T(1e-6));  // THINK Is it best way to maximize contrast?
    return true;
  }

  const std::vector<double>& xs_;
  const std::vector<double>& ys_;
  const std::vector<double>& ts_;
  const std::vector<double>& ps_;
  double t_ref_;
  int H_, W_;
};

void optimize(std::vector<double>& params, const std::vector<double>& xs, const std::vector<double>& ys,
              const std::vector<double>& ts, const std::vector<double>& ps, const double& t_ref, const int& width,
              const int& height);

cv::Mat getIWE(JetImage<double>& iwe, const LinearWarpFunction& warp_function, std::vector<double>& params,
               const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts,
               const std::vector<double>& ps, const double& t_ref, const int width, const int height);