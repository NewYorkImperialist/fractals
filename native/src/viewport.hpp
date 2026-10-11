#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Host mirror of the WGSL `Viewport` uniform. Every member is a 4-byte
// scalar, so the C++ and WGSL layouts match with no implicit padding.
struct ViewportUniform {
  float x_min, x_max, y_min, y_max;
  float zoom;
  uint32_t iteration_limit;
  uint32_t width, height;
  float offset_x_hi, offset_x_lo, offset_y_hi, offset_y_lo;
  float julia_x_hi, julia_x_lo, julia_y_hi, julia_y_lo;
  uint32_t flags;
  float color_phase;
  uint32_t zero_bits = 0; // must stay 0: shader's fast-math barrier
  uint32_t _pad = 0;
};
static_assert(sizeof(ViewportUniform) == 80, "must match WGSL Viewport");

inline constexpr uint32_t kFlagJulia = 1u;
inline constexpr uint32_t kFlagDouble = 2u;
inline constexpr uint32_t kFlagOutline = 4u;

// Split a double into an unevaluated float sum hi + lo (~48-bit mantissa).
struct DoubleSingle {
  float hi, lo;
  explicit DoubleSingle(double value)
      : hi(static_cast<float>(value)),
        lo(static_cast<float>(value - static_cast<double>(hi))) {}
};

// A view of the complex plane. All interactive math happens here in double
// precision; the GPU only ever sees the (hi, lo) split of the centre.
struct Camera {
  static constexpr double kHalfHeight = 1.25; // visible half-height at zoom 1
  static constexpr double kMaxZoom = 1e13;    // double-single precision limit

  double center_x = 0.0, center_y = 0.0;
  double zoom = 1.0;

  // Extent at zoom 1 relative to the centre, corrected for aspect ratio.
  struct Extent { double x_min, x_max, y_min, y_max; };
  static Extent extent(uint32_t width, uint32_t height) {
    const double aspect = double(width) / double(std::max(height, 1u));
    return {-kHalfHeight * aspect, kHalfHeight * aspect, -kHalfHeight, kHalfHeight};
  }

  // Size of one pixel in complex-plane units.
  double pixel_size(uint32_t height) const {
    return 2.0 * kHalfHeight / (double(std::max(height, 1u)) * zoom);
  }

  // Complex coordinate under the given pixel (matches the shader's mapping).
  void to_complex(double px, double py, uint32_t width, uint32_t height,
                  double &re, double &im) const {
    const Extent e = extent(width, height);
    const double u = px / width, v = py / height;
    re = center_x + (e.x_min + (e.x_max - e.x_min) * u) / zoom;
    im = center_y + (e.y_max + (e.y_min - e.y_max) * v) / zoom;
  }

  // Inverse of to_complex: where a complex coordinate lands on screen.
  void to_pixel(double re, double im, uint32_t width, uint32_t height,
                double &px, double &py) const {
    const Extent e = extent(width, height);
    px = ((re - center_x) * zoom - e.x_min) / (e.x_max - e.x_min) * width;
    py = ((im - center_y) * zoom - e.y_max) / (e.y_min - e.y_max) * height;
  }

  // Zoom by `factor`, keeping the point under the cursor fixed on screen.
  void zoom_at(double px, double py, double factor, uint32_t width, uint32_t height) {
    double before_re, before_im, after_re, after_im;
    to_complex(px, py, width, height, before_re, before_im);
    zoom = std::clamp(zoom * factor, 0.25, kMaxZoom);
    to_complex(px, py, width, height, after_re, after_im);
    center_x += before_re - after_re;
    center_y += before_im - after_im;
  }

  // Drag the plane by a pixel delta (content follows the cursor).
  void pan_pixels(double dx, double dy, uint32_t height) {
    const double step = pixel_size(height);
    center_x -= dx * step;
    center_y += dy * step;
  }
};
