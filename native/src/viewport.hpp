#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "bigfloat.hpp"

// Host mirror of the WGSL `Viewport` uniform. Every member is a 4-byte
// scalar, so the C++ and WGSL layouts match with no implicit padding.
// "Fx" fields are a float mantissa with a separate int exponent (see
// fractal_shader.hpp): value = mantissa * 2^exp.
//
// The uniform is also the pane renderer's cache key (pane_renderer.hpp):
// everything that changes the image must be in it, which is why a
// perturbation view carries the id of its reference orbit.
struct ViewportUniform {
  float x_min, x_max, y_min, y_max;
  float zoom;
  uint32_t iteration_limit;
  uint32_t width, height;
  // Direct f32 path: the grid anchor as double-single (hi, lo) pairs, and
  // the Julia constant.
  float center_x_hi, center_x_lo, center_y_hi, center_y_lo;
  float julia_x, julia_y;
  uint32_t flags;
  float color_phase;
  // Perturbation: 1 / zoom (Fx).
  float inv_zoom = 0.0f;
  int32_t inv_zoom_exp = 0;
  // Grid anchor minus the reference point (c for Mandelbrot, z0 for Julia),
  // and the view's Julia c minus the reference's (Fx, shared exponent).
  float ref_offset_x = 0.0f, ref_offset_y = 0.0f;
  int32_t ref_offset_exp = 0;
  float julia_dc_x = 0.0f, julia_dc_y = 0.0f;
  int32_t julia_dc_exp = 0;
  // Orbit segments in the reference buffers: pixels start on the primary
  // orbit at index 0; rebasing jumps to the orbit that starts at z = 0.
  uint32_t orbit_length = 0;
  uint32_t rebase_base = 0, rebase_length = 0, rebase_bla_base = 0;
  uint32_t reference_id = 0; // which orbit / BLA table the buffers must hold
  uint32_t zero_bits = 0;    // must stay 0: shader's fast-math barrier
  uint32_t _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(ViewportUniform) == 128, "must match WGSL Viewport");

inline constexpr uint32_t kFlagJulia = 1u;
inline constexpr uint32_t kFlagPerturb = 2u;   // delta iteration against a reference
inline constexpr uint32_t kFlagOutline = 4u;
inline constexpr uint32_t kFlagFloatExp = 8u;  // perturbation with extended exponents
inline constexpr uint32_t kFlagBla = 16u;      // skip iterations with the BLA table
inline constexpr uint32_t kFlagInterior = 32u; // f32 cardioid/bulb test is accurate

// Split a double into an unevaluated float sum hi + lo (~48-bit mantissa).
struct DoubleSingle {
  float hi, lo;
  explicit DoubleSingle(double value)
      : hi(static_cast<float>(value)),
        lo(static_cast<float>(value - static_cast<double>(hi))) {}
};

// A view of the complex plane. The centre is an arbitrary-precision number
// whose precision grows with zoom, and zoom is kept as log2 so it can pass
// double range; the GPU sees either f32 coordinates (shallow) or offsets
// from a reference orbit (perturbation).
//
// Drags are kept apart from the centre as a whole number of render pixels
// (pan_x, pan_y), so every panned view lies on the same pixel grid as the
// unpanned one: pixel (px, py) of the view is grid pixel (px + pan_x,
// py + pan_y) of a render centred on (center_x, center_y). That lets the
// renderer shift the previous image instead of recomputing it, with results
// identical to a full render. The sub-pixel rest of each drag is carried in
// pan_rest_* so the motion stays exact.
struct Camera {
  static constexpr double kHalfHeight = 1.25;  // visible half-height at zoom 1
  // ~1e3010. Not a precision limit (Fx exponents reach ~2^±2^23): it only
  // bounds the MPFR cost of the reference orbit. Verified to 1e3001.
  static constexpr double kMaxLog2Zoom = 10000;
  static constexpr int32_t kMaxPan = 1 << 14; // fold into the centre beyond this

  BigFloat center_x = 0.0, center_y = 0.0;
  double log2_zoom = 0.0;
  int32_t pan_x = 0, pan_y = 0; // render pixels at height pan_height
  uint32_t pan_height = 0;
  double pan_rest_x = 0.0, pan_rest_y = 0.0;

  static Camera at(double x, double y, double zoom) {
    Camera camera{.center_x = x, .center_y = y, .log2_zoom = std::log2(zoom)};
    camera.update_precision();
    return camera;
  }

  // Zoom as a double; inf beyond double range (display only).
  double zoom() const { return std::exp2(log2_zoom); }
  FloatExp zoom_fx() const {
    const double whole = std::floor(log2_zoom);
    return FloatExp::make(std::exp2(log2_zoom - whole), int64_t(whole));
  }
  FloatExp inv_zoom() const {
    const double whole = std::floor(log2_zoom);
    return FloatExp::make(std::exp2(whole - log2_zoom), -int64_t(whole));
  }

  // Bits the centre needs so sub-pixel offsets survive: the pixel size is
  // ~2^-(log2_zoom + 11) for screens up to ~4k pixels, plus guard bits.
  static long bits_for(double log2_zoom) {
    const long bits = long(std::max(log2_zoom, 0.0)) + 64;
    return (bits + 63) / 64 * 64;
  }
  void update_precision() {
    center_x.ensure_precision(bits_for(log2_zoom));
    center_y.ensure_precision(bits_for(log2_zoom));
  }

  // Extent at zoom 1 relative to the centre, corrected for aspect ratio.
  struct Extent { double x_min, x_max, y_min, y_max; };
  static Extent extent(uint32_t width, uint32_t height) {
    const double aspect = double(width) / double(std::max(height, 1u));
    return {-kHalfHeight * aspect, kHalfHeight * aspect, -kHalfHeight, kHalfHeight};
  }

  // Size of one pixel at zoom 1.
  static double plane_pixel(uint32_t height) {
    return 2.0 * kHalfHeight / double(std::max(height, 1u));
  }
  // log2 of the size of one pixel in complex-plane units.
  double log2_pixel_size(uint32_t height) const {
    return std::log2(plane_pixel(height)) - log2_zoom;
  }

  // The view's actual centre (centre plus pan).
  BigFloat view_x() const {
    BigFloat x = center_x;
    if (pan_x) x.add_scaled(pan_x * plane_pixel(pan_height), -log2_zoom);
    return x;
  }
  BigFloat view_y() const {
    BigFloat y = center_y;
    if (pan_y) y.add_scaled(-pan_y * plane_pixel(pan_height), -log2_zoom);
    return y;
  }

  // Pixel grid a render of the given height uses: a render centred on (x, y)
  // whose pixel (0, 0) is grid pixel (origin_x, origin_y).
  struct Grid { BigFloat x, y; int32_t origin_x = 0, origin_y = 0; };
  Grid grid(uint32_t height) const {
    if (height == pan_height) return {center_x, center_y, pan_x, pan_y};
    return {view_x(), view_y(), 0, 0};
  }

  // Pixel -> offset from the centre at zoom 1 (matches the shader's mapping).
  static void plane_offset(double px, double py, uint32_t width, uint32_t height,
                           double &x, double &y) {
    const Extent e = extent(width, height);
    x = e.x_min + (e.x_max - e.x_min) * (px / width);
    y = e.y_max + (e.y_min - e.y_max) * (py / height);
  }

  // Complex coordinate under the given pixel.
  void to_complex(double px, double py, uint32_t width, uint32_t height,
                  BigFloat &re, BigFloat &im) const {
    double x, y;
    plane_offset(px, py, width, height, x, y);
    re = view_x();
    im = view_y();
    re.add_scaled(x, -log2_zoom);
    im.add_scaled(y, -log2_zoom);
  }

  // Inverse of to_complex: where a complex coordinate lands on screen.
  void to_pixel(const BigFloat &re, const BigFloat &im, uint32_t width,
                uint32_t height, double &px, double &py) const {
    const Extent e = extent(width, height);
    const double x = (difference(re, view_x()) * zoom_fx()).to_double();
    const double y = (difference(im, view_y()) * zoom_fx()).to_double();
    px = (x - e.x_min) / (e.x_max - e.x_min) * width;
    py = (y - e.y_max) / (e.y_min - e.y_max) * height;
  }

  // Moves the pan into the centre (the next render starts a new grid).
  void fold_pan() {
    if (pan_x == 0 && pan_y == 0) return;
    BigFloat x = view_x(), y = view_y();
    center_x = std::move(x);
    center_y = std::move(y);
    pan_x = pan_y = 0;
  }

  // Zoom by `factor`, keeping the point under the cursor fixed on screen.
  void zoom_at(double px, double py, double factor, uint32_t width, uint32_t height) {
    fold_pan();
    double x, y;
    plane_offset(px, py, width, height, x, y);
    const double before = log2_zoom;
    log2_zoom = std::clamp(log2_zoom + std::log2(factor), -2.0, kMaxLog2Zoom);
    update_precision();
    // centre += plane * (1/zoom_before - 1/zoom_after)
    const double shrink = 1.0 - std::exp2(before - log2_zoom);
    center_x.add_scaled(x * shrink, -before);
    center_y.add_scaled(y * shrink, -before);
  }

  // Drag the plane by a pixel delta (content follows the cursor). Moves in
  // whole pixels of a `height`-pixel render; the rest carries over.
  void pan_pixels(double dx, double dy, uint32_t height) {
    if (height != pan_height) {
      fold_pan();
      pan_height = height;
    }
    pan_rest_x -= dx;
    pan_rest_y -= dy;
    const double steps_x = std::round(pan_rest_x), steps_y = std::round(pan_rest_y);
    pan_rest_x -= steps_x;
    pan_rest_y -= steps_y;
    pan_x += int32_t(steps_x);
    pan_y += int32_t(steps_y);
    if (std::abs(pan_x) > kMaxPan || std::abs(pan_y) > kMaxPan) fold_pan();
  }
};
