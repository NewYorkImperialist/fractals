#pragma once

// Perturbation support for the fractal shader: the arbitrary-precision
// reference orbit, its BLA (bilinear approximation) table, their GPU
// layouts, a cache that recomputes them on a background thread, and the
// uniform builder shared by the app and the benchmark.
//
// Reference orbit: Z_{n+1} = Z_n^2 + C in MPFR at ~log2(zoom) + 96 bits,
// from Z_0 = 0 (Mandelbrot, C = reference point) or Z_0 = reference point
// (Julia, C = Julia constant). Julia also needs the critical orbit (from 0)
// to rebase onto; for Mandelbrot that is the primary orbit itself.
//
// BLA (Zhuoran's construction, as in Imagina / Fraktaler 3): one step
// dz -> 2 Z_m dz + dz^2 + dc is linear, dz -> A dz + B dc with A = 2 Z_m,
// B = 1, while dz^2 is negligible: |dz| < r = eps (|Z_m| - h) / (|A| + 1),
// h bounding |dc| over the view. Merging x then y gives A = Ay Ax,
// B = Ay Bx + By, r = min(rx, (ry - |Bx| h) / |Ax|). Level k of the table
// merges pairs from level k - 1 and skips 2^k iterations.

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include "gpu.hpp"

#include "bigfloat.hpp"
#include "viewport.hpp"

// Host mirrors of the shader's OrbitPoint and Bla.
struct GpuOrbitPoint {
  float x, y;
  int32_t e;
  float bla_r = 0.0f; // level-1 BLA radius from here (log2 of it in Fx mode)
};
static_assert(sizeof(GpuOrbitPoint) == 16, "must match WGSL OrbitPoint");

struct GpuBla {
  float ax, ay, bx, by;
  int32_t a_exp, b_exp;
  float r;
  int32_t r_exp;
};
static_assert(sizeof(GpuBla) == 32, "must match WGSL Bla");

namespace perturbation {

inline constexpr int32_t kFxZeroExp = -16777216; // shader FX_ZERO
inline constexpr double kBailoutSq = 256.0;      // shader BAILOUT_SQ
// BLA relative error per step. f32 rounding is 2^-24; 2^-20 is within f32
// noise of an MPFR render, 2^-18 and looser visibly drift from it (but do
// make skips longer).
inline constexpr double kBlaEpsilon = 1.0 / (1 << 20);
// Past this zoom (~1.2e24) the GPU uses Fx deltas instead of f32: pixel
// offsets and their products would soon flush to zero (f32 min ~1e-38).
// f32 deltas are ~1.5x faster where both work.
inline constexpr double kFloatExpLog2Zoom = 80.0;
// Below ~1e9 BLA skips are short and the lookups cost more than they save
// (measured: 10-15% slower at 1e5-1e7, even at 1e8-1e9, 3-13x faster from
// 1e20 on), so it is only switched on past this zoom.
inline constexpr double kBlaLog2Zoom = 30.0;

// Everything a view needs from the renderer's point of view. `camera` has
// the pan folded in (its centre is the view's actual centre: references are
// placed and judged there); `grid` is the pixel grid the image is rendered
// on (Camera::grid), which the uniform's offsets are relative to, so that a
// pan by whole pixels leaves the uniform -- and the image -- unchanged.
struct View {
  bool julia = false;
  Camera camera;
  Camera::Grid grid;
  BigFloat julia_re, julia_im;
  uint32_t iterations = 0, width = 0, height = 0;
};

inline View make_view(const Camera &camera, bool julia, const BigFloat &julia_re,
                      const BigFloat &julia_im, uint32_t iterations, uint32_t width,
                      uint32_t height) {
  View view{.julia = julia, .camera = camera, .grid = camera.grid(height),
            .julia_re = julia_re, .julia_im = julia_im, .iterations = iterations,
            .width = width, .height = height};
  view.camera.fold_pan();
  return view;
}

// What a reference was computed for; also describes in-flight requests.
struct ReferenceKey {
  bool julia = false;
  BigFloat x, y;               // reference point: c, or z0 for Julia
  BigFloat julia_re, julia_im; // Julia constant of the orbits
  long bits = 0;               // MPFR precision of the orbit
  double log2_zoom = 0.0;      // zoom it was requested at
  uint32_t iterations = 0;     // iteration limit of the orbit
  bool floatexp = false;       // GPU data format
  FloatExp bla_radius;         // BLA tables assume |dc| <= this
};

struct Reference {
  ReferenceKey key;
  uint32_t id = 0;      // unique per computed reference (goes into the uniform)
  bool escaped = false; // primary orbit escaped before the iteration limit
  std::vector<GpuOrbitPoint> orbit;
  std::vector<GpuBla> bla;
  uint32_t orbit_length = 0;
  uint32_t rebase_base = 0, rebase_length = 0, rebase_bla_base = 0;
  double seconds = 0.0; // CPU time to compute (orbit + BLA + packing)
};

// ---------------------------------------------------------------------------
// Reference orbit and BLA table.
// ---------------------------------------------------------------------------

// Iterates z <- z^2 + c from z0 at `bits` of precision. Stores Z_0 up to the
// first point beyond the bailout (at least two points) or limit + 1 points.
// Returns an empty orbit if cancelled.
inline std::vector<ComplexExp> compute_orbit(const BigFloat &z0x, const BigFloat &z0y,
                                             const BigFloat &cx, const BigFloat &cy,
                                             long bits, uint32_t limit, bool &escaped,
                                             const std::atomic<bool> *cancel) {
  mpfr_t x, y, x2, y2, xy;
  for (mpfr_ptr v : {x, y, x2, y2, xy}) mpfr_init2(v, bits);
  mpfr_set(x, z0x.get(), MPFR_RNDN);
  mpfr_set(y, z0y.get(), MPFR_RNDN);
  auto get = [](mpfr_srcptr v) {
    long exponent = 0;
    const double m = mpfr_get_d_2exp(&exponent, v, MPFR_RNDN);
    return FloatExp::make(m, exponent);
  };

  std::vector<ComplexExp> orbit;
  escaped = false;
  for (uint32_t n = 0;; ++n) {
    const ComplexExp z{get(x), get(y)};
    orbit.push_back(z);
    if (n >= 1 && z.norm().to_double() > kBailoutSq) {
      escaped = true;
      break;
    }
    if (n >= limit) break;
    if ((n & 1023) == 0 && cancel && cancel->load(std::memory_order_relaxed)) {
      orbit.clear();
      break;
    }
    mpfr_sqr(x2, x, MPFR_RNDN);
    mpfr_sqr(y2, y, MPFR_RNDN);
    mpfr_mul(xy, x, y, MPFR_RNDN);
    mpfr_mul_2ui(xy, xy, 1, MPFR_RNDN);
    mpfr_add(y, xy, cy.get(), MPFR_RNDN);
    mpfr_sub(x, x2, y2, MPFR_RNDN);
    mpfr_add(x, x, cx.get(), MPFR_RNDN);
  }
  for (mpfr_ptr v : {x, y, x2, y2, xy}) mpfr_clear(v);
  return orbit;
}

struct BlaStep {
  ComplexExp a, b;
  FloatExp r;
};

// Levels 1, 2, ... of the BLA table for `orbit` (level 0, single steps, is
// never stored: a single step costs the same as a perturbation step).
inline std::vector<BlaStep> build_bla(const std::vector<ComplexExp> &orbit, FloatExp h) {
  std::vector<BlaStep> table;
  if (orbit.size() < 4) return table;
  const size_t steps = orbit.size() - 2; // m = 1 .. size - 2
  const FloatExp eps(kBlaEpsilon), zero, one(1.0), two(2.0);

  auto single = [&](size_t s) {
    const ComplexExp &z = orbit[s + 1];
    const ComplexExp a = two * z;
    const FloatExp r = max(zero, (z.abs() - h) / (a.abs() + one)) * eps;
    return BlaStep{a, {one, zero}, r};
  };
  auto merge = [&](const BlaStep &x, const BlaStep &y) {
    const FloatExp ax = x.a.abs();
    const FloatExp slack = y.r - x.b.abs() * h;
    FloatExp r = x.r;
    if (!ax.is_zero()) r = min(r, max(zero, slack / ax));
    else if (!(slack > zero)) r = zero;
    return BlaStep{y.a * x.a, y.a * x.b + y.b, r};
  };

  std::vector<BlaStep> level((steps + 1) / 2);
  for (size_t j = 0; j < level.size(); ++j)
    level[j] = 2 * j + 1 < steps ? merge(single(2 * j), single(2 * j + 1)) : single(2 * j);
  table.insert(table.end(), level.begin(), level.end());
  while (level.size() > 1) {
    std::vector<BlaStep> next((level.size() + 1) / 2);
    for (size_t j = 0; j < next.size(); ++j)
      next[j] = 2 * j + 1 < level.size() ? merge(level[2 * j], level[2 * j + 1])
                                         : level[2 * j];
    table.insert(table.end(), next.begin(), next.end());
    level = std::move(next);
  }
  return table;
}

// ---------------------------------------------------------------------------
// GPU packing: f32 values, or Fx (f32 mantissa with max |component| in
// [1, 2), shared i32 exponent).
// ---------------------------------------------------------------------------

inline void pack_fx(const ComplexExp &z, float &x, float &y, int32_t &e) {
  if (z.re.is_zero() && z.im.is_zero()) {
    x = y = 0.0f;
    e = kFxZeroExp;
    return;
  }
  const int64_t top = std::max(z.re.is_zero() ? FloatExp::kZeroExp : z.re.e,
                               z.im.is_zero() ? FloatExp::kZeroExp : z.im.e) - 1;
  auto part = [&](const FloatExp &v) {
    if (v.is_zero() || v.e - top < -200) return 0.0f;
    return float(std::ldexp(v.m, int(v.e - top)));
  };
  x = part(z.re);
  y = part(z.im);
  e = top < kFxZeroExp / 2 ? kFxZeroExp : int32_t(std::min<int64_t>(top, 1 << 23));
}

inline void pack_fx(const FloatExp &v, float &x, int32_t &e) {
  float unused;
  pack_fx(ComplexExp{v, {}}, x, unused, e);
}

inline GpuOrbitPoint pack_orbit_point(const ComplexExp &z, bool floatexp) {
  GpuOrbitPoint p{};
  if (floatexp) pack_fx(z, p.x, p.y, p.e);
  else p = {float(z.re.to_double()), float(z.im.to_double()), 0};
  return p;
}

inline GpuBla pack_bla(const BlaStep &step, bool floatexp) {
  GpuBla g{};
  if (floatexp) {
    pack_fx(step.a, g.ax, g.ay, g.a_exp);
    pack_fx(step.b, g.bx, g.by, g.b_exp);
    pack_fx(step.r, g.r, g.r_exp);
    // Exponents this large would overflow the shader's i32 sums.
    if (g.a_exp > (1 << 22) || g.b_exp > (1 << 22)) g.r = 0.0f;
    return g;
  }
  // f32: keep coefficients well inside range so products cannot overflow
  // or flush; an entry that does not fit is simply never used.
  auto fits = [](const FloatExp &v) { return v.is_zero() || (v.e > -100 && v.e < 100); };
  const bool ok = fits(step.a.re) && fits(step.a.im) && fits(step.b.re) &&
                  fits(step.b.im) && step.r.e > -120 && !step.r.is_zero();
  g = {float(step.a.re.to_double()), float(step.a.im.to_double()),
       float(step.b.re.to_double()), float(step.b.im.to_double()),
       0, 0, ok ? float(std::min(step.r.to_double(), 1e30)) : 0.0f, 0};
  return g;
}

inline long bits_for_reference(double log2_zoom) {
  return Camera::bits_for(log2_zoom + 32);
}

// Computes the orbits and tables for `key` (bits, format etc. already set).
// Returns null if cancelled.
inline std::shared_ptr<Reference> compute_reference(const ReferenceKey &key,
                                                    const std::atomic<bool> *cancel = nullptr) {
  const auto start = std::chrono::steady_clock::now();
  auto ref = std::make_shared<Reference>();
  ref->key = key;
  const BigFloat zero(0.0, 64);
  bool escaped = false;
  std::vector<ComplexExp> primary, critical;
  if (key.julia) {
    primary = compute_orbit(key.x, key.y, key.julia_re, key.julia_im, key.bits,
                            key.iterations, ref->escaped, cancel);
    critical = compute_orbit(zero, zero, key.julia_re, key.julia_im, key.bits,
                             key.iterations, escaped, cancel);
    if (critical.empty()) return nullptr;
  } else {
    primary = compute_orbit(zero, zero, key.x, key.y, key.bits, key.iterations,
                            ref->escaped, cancel);
  }
  if (primary.empty()) return nullptr;

  // Julia's dc is 0 (or the stale-c offset, for which BLA is switched off).
  const FloatExp h = key.julia ? FloatExp() : key.bla_radius;
  auto append = [&](const std::vector<ComplexExp> &orbit) {
    const size_t base = ref->orbit.size(), bla_base = ref->bla.size();
    for (const ComplexExp &z : orbit) ref->orbit.push_back(pack_orbit_point(z, key.floatexp));
    for (const BlaStep &step : build_bla(orbit, h))
      ref->bla.push_back(pack_bla(step, key.floatexp));
    // Copy each level-1 radius into the orbit point it starts from (index
    // 1 + 2j), so the shader can rule out a skip without loading the table.
    const float none = key.floatexp ? -1e30f : 0.0f;
    for (size_t n = base; n < ref->orbit.size(); ++n) ref->orbit[n].bla_r = none;
    const size_t level1 = orbit.size() >= 4 ? (orbit.size() - 1) / 2 : 0;
    for (size_t j = 0; j < level1; ++j) {
      const GpuBla &g = ref->bla[bla_base + j];
      ref->orbit[base + 1 + 2 * j].bla_r =
          !key.floatexp ? g.r : g.r > 0.0f ? float(std::log2(g.r) + g.r_exp) : none;
    }
  };
  append(primary);
  ref->orbit_length = uint32_t(primary.size());
  if (key.julia) {
    ref->rebase_base = uint32_t(ref->orbit.size());
    ref->rebase_bla_base = uint32_t(ref->bla.size());
    ref->rebase_length = uint32_t(critical.size());
    append(critical);
  } else {
    ref->rebase_length = ref->orbit_length;
  }
  if (ref->bla.empty()) ref->bla.push_back(GpuBla{}); // r = 0: never valid
  if (cancel && cancel->load()) return nullptr;
  static std::atomic<uint32_t> next_id{1};
  ref->id = next_id.fetch_add(1, std::memory_order_relaxed);
  ref->seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return ref;
}

// ---------------------------------------------------------------------------
// Views and references.
// ---------------------------------------------------------------------------

// Whether direct f32 iteration has run out of precision for this camera:
// once a pixel shrinks to a few f32 ulps of the coordinates. (Below that,
// direct f32 is also the faster path; see bench.)
inline bool needed(const Camera &camera, uint32_t height) {
  const double magnitude = std::max({std::abs(camera.center_x.to_double()),
                                     std::abs(camera.center_y.to_double()), 1.0});
  return camera.log2_pixel_size(height) < std::log2(magnitude * 4e-6);
}

// Largest |dc| over the view for a reference at offset `offset` (Mandelbrot).
inline FloatExp view_radius(const View &view, const ComplexExp &offset) {
  const Camera::Extent e = Camera::extent(view.width, view.height);
  const double corner = std::hypot(e.x_max, e.y_max);
  return offset.abs() + FloatExp(corner) * view.camera.inv_zoom();
}

inline ComplexExp offset_from(const View &view, const ReferenceKey &key) {
  return {difference(view.camera.center_x, key.x), difference(view.camera.center_y, key.y)};
}

inline bool same_julia(const View &view, const ReferenceKey &key) {
  return view.julia_re == key.julia_re && view.julia_im == key.julia_im;
}

// The key a fresh reference for `view` would have.
inline ReferenceKey key_for(const View &view) {
  ReferenceKey key;
  key.julia = view.julia;
  key.x = view.camera.center_x;
  key.y = view.camera.center_y;
  if (view.julia) {
    key.julia_re = view.julia_re;
    key.julia_im = view.julia_im;
  }
  key.log2_zoom = view.camera.log2_zoom;
  key.bits = bits_for_reference(key.log2_zoom);
  key.iterations = view.iterations;
  key.floatexp = key.log2_zoom > kFloatExpLog2Zoom;
  // Twice the view's radius: stays valid through zooming out 2x or panning.
  if (!view.julia) key.bla_radius = FloatExp(2.0) * view_radius(view, {});
  return key;
}

enum class Fit { Fresh, Stale, Unusable };

// Whether a reference with `key` can render `view`. Stale references give
// correct images (perhaps slower, or without BLA) but should be replaced.
inline Fit assess(const ReferenceKey &key, bool escaped, const View &view) {
  const Camera &cam = view.camera;
  if (key.julia != view.julia) return Fit::Unusable;
  // The orbit is the exact orbit of a point within ~2^-bits of the
  // reference; that shift must stay far below a pixel (~2^-(zoom + 11)).
  const double spare_bits = double(key.bits) - cam.log2_zoom;
  if (spare_bits < 24) return Fit::Unusable;
  if (!key.floatexp && cam.log2_zoom > kFloatExpLog2Zoom + 16) return Fit::Unusable;
  // Reference offset in zoom-1 plane units; f32 pixel offsets are added to
  // it, so it must stay small for them to keep their precision.
  const ComplexExp offset = offset_from(view, key);
  const double plane = (offset.abs() * cam.zoom_fx()).to_double();
  if (plane > 64.0) return Fit::Unusable;

  Fit fit = Fit::Fresh;
  if (spare_bits < 48 || plane > 0.5) fit = Fit::Stale;
  if (key.floatexp != (cam.log2_zoom > kFloatExpLog2Zoom)) {
    // Hysteresis: f32 data is fine a little past the switch, Fx anywhere.
    if (!key.floatexp || cam.log2_zoom < kFloatExpLog2Zoom - 8) fit = Fit::Stale;
  }
  if (!escaped && key.iterations < view.iterations) fit = Fit::Stale;
  if (view.julia ? !same_julia(view, key)
                 : view_radius(view, offset) > key.bla_radius)
    fit = Fit::Stale;
  return fit;
}

// Keeps one pane's reference, recomputing it on a background thread when
// the view drifts out of its valid range. The previous reference keeps
// rendering (it is still correct) until the new one is ready.
class ReferenceCache {
public:
  ReferenceCache() = default;
  ReferenceCache(const ReferenceCache &) = delete;
  ReferenceCache &operator=(const ReferenceCache &) = delete;
  ~ReferenceCache() {
    if (cancel_) cancel_->store(true);
  }

  // Newest reference that can render `view`, or null if none can yet.
  std::shared_ptr<const Reference> get(const View &view) {
    collect(false);
    const Fit fit = current_ ? assess(current_->key, current_->escaped, view) : Fit::Unusable;
    if (fit != Fit::Fresh) {
      // A job for a view we have since left is not worth finishing.
      if (pending_.valid() && assess(pending_key_, true, view) != Fit::Fresh)
        cancel_->store(true);
      if (!pending_.valid()) launch(view);
    }
    return fit == Fit::Unusable ? nullptr : current_;
  }

  // Like get(), but blocks until a usable reference exists.
  std::shared_ptr<const Reference> wait(const View &view) {
    for (;;) {
      if (auto ref = get(view)) return ref;
      collect(true);
    }
  }

  bool busy() const { return pending_.valid(); }

private:
  void launch(const View &view) {
    pending_key_ = key_for(view);
    cancel_ = std::make_shared<std::atomic<bool>>(false);
    pending_ = std::async(std::launch::async, [key = pending_key_, cancel = cancel_] {
      return compute_reference(key, cancel.get());
    });
  }

  void collect(bool block) {
    if (!pending_.valid()) return;
    if (!block && pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      return;
    if (auto ref = pending_.get()) current_ = std::move(ref);
  }

  std::shared_ptr<const Reference> current_;
  std::future<std::shared_ptr<Reference>> pending_;
  ReferenceKey pending_key_;
  std::shared_ptr<std::atomic<bool>> cancel_;
};

// GPU storage buffers for a reference's orbit and BLA table. They only grow
// (powers of two); kernels bound to them must be rebuilt when they do.
class ReferenceBuffers {
public:
  explicit ReferenceBuffers(gpu::Context &context)
      : context_(context), orbit_(allocate(64 * sizeof(GpuOrbitPoint))),
        bla_(allocate(64 * sizeof(GpuBla))) {}

  const gpu::Tensor &orbit() const { return orbit_; }
  const gpu::Tensor &bla() const { return bla_; }

  // Uploads `ref` unless it is already resident. True if the buffers were
  // reallocated.
  bool upload(const std::shared_ptr<const Reference> &ref) {
    if (!ref || ref == resident_) return false;
    resident_ = ref;
    bool grown = reserve(orbit_, ref->orbit.size() * sizeof(GpuOrbitPoint));
    grown |= reserve(bla_, ref->bla.size() * sizeof(GpuBla));
    context_.queue.WriteBuffer(orbit_.buffer, 0, ref->orbit.data(),
                               ref->orbit.size() * sizeof(GpuOrbitPoint));
    context_.queue.WriteBuffer(bla_.buffer, 0, ref->bla.data(),
                               ref->bla.size() * sizeof(GpuBla));
    context_.check();
    return grown;
  }

private:
  gpu::Tensor allocate(size_t bytes) {
    // kf32 only fixes the 4-byte element size; the shader sees structs.
    return gpu::createTensor(context_, gpu::Shape{bytes / 4}, gpu::kf32);
  }
  bool reserve(gpu::Tensor &tensor, size_t bytes) {
    if (bytes <= tensor.bytes) return false;
    size_t capacity = tensor.bytes;
    while (capacity < bytes) capacity *= 2;
    tensor = allocate(capacity);
    return true;
  }

  gpu::Context &context_;
  gpu::Tensor orbit_, bla_;
  std::shared_ptr<const Reference> resident_;
};

} // namespace perturbation

// Uniform for `view`; perturbation iff `ref` is non-null. Positions are
// relative to the grid anchor (View::grid), so the uniform stays the same
// while the view pans by whole pixels on that grid.
inline ViewportUniform make_uniform(const perturbation::View &view,
                                    const perturbation::Reference *ref, bool outline,
                                    float color_phase, bool use_bla = true) {
  using namespace perturbation;
  const Camera &cam = view.camera;
  const Camera::Extent e = Camera::extent(view.width, view.height);
  const DoubleSingle ox(view.grid.x.to_double()), oy(view.grid.y.to_double());
  // The Mandelbrot image does not depend on the Julia point; leaving it out
  // keeps that image valid while the point moves.
  const double jx = view.julia ? view.julia_re.to_double() : 0.0;
  const double jy = view.julia ? view.julia_im.to_double() : 0.0;
  uint32_t flags = 0;
  if (view.julia) flags |= kFlagJulia;
  if (outline) flags |= kFlagOutline;
  ViewportUniform u{
      .x_min = float(e.x_min), .x_max = float(e.x_max),
      .y_min = float(e.y_min), .y_max = float(e.y_max),
      .zoom = float(std::min(cam.zoom(), 1e38)),
      .iteration_limit = view.iterations,
      .width = view.width, .height = view.height,
      .center_x_hi = ox.hi, .center_x_lo = ox.lo,
      .center_y_hi = oy.hi, .center_y_lo = oy.lo,
      .julia_x = float(jx), .julia_y = float(jy),
      .flags = 0,
      .color_phase = color_phase,
  };
  pack_fx(cam.inv_zoom(), u.inv_zoom, u.inv_zoom_exp);
  if (ref) {
    const ReferenceKey &key = ref->key;
    flags |= kFlagPerturb;
    if (key.floatexp) flags |= kFlagFloatExp;
    // Pixel offsets in the shader are from the grid anchor.
    const ComplexExp anchor{difference(view.grid.x, key.x), difference(view.grid.y, key.y)};
    pack_fx(anchor, u.ref_offset_x, u.ref_offset_y, u.ref_offset_exp);
    // The pixels computed for the view lie around its actual centre.
    const ComplexExp offset = offset_from(view, key);
    bool bla = use_bla && cam.log2_zoom > kBlaLog2Zoom;
    if (view.julia) {
      const ComplexExp dc{difference(view.julia_re, key.julia_re),
                          difference(view.julia_im, key.julia_im)};
      pack_fx(dc, u.julia_dc_x, u.julia_dc_y, u.julia_dc_exp);
      bla &= dc.re.is_zero() && dc.im.is_zero();
    } else {
      pack_fx(ComplexExp{}, u.julia_dc_x, u.julia_dc_y, u.julia_dc_exp);
      bla &= !(view_radius(view, offset) > key.bla_radius);
    }
    if (bla) flags |= kFlagBla;
    // The f32 cardioid / bulb test is only exact enough while f32 resolves
    // pixels (the same criterion the direct path needs).
    if (cam.log2_pixel_size(view.height) > std::log2(4e-6)) flags |= kFlagInterior;
    u.orbit_length = ref->orbit_length;
    u.rebase_base = ref->rebase_base;
    u.rebase_length = ref->rebase_length;
    u.rebase_bla_base = ref->rebase_bla_base;
    u.reference_id = ref->id;
  }
  u.flags = flags;
  return u;
}
