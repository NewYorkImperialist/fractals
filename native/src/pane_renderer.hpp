#pragma once

// GPU frame pipeline for the explorer, independent of SDL so tools/ can drive
// it headlessly:
//
//   FractalKernel  the compute step: fills a tile of a pane image for a view.
//                  Swap what happens inside it (perturbation, block filling,
//                  ...) without touching the rest; see its comment.
//   PaneRenderer   per pane: keeps up to three images, decides what to
//                  compute for each new view (nothing, shift + strips, or a
//                  fresh progressive render), time-slices the work and picks
//                  which image to show.
//   Compositor     render pass that draws each pane's shown image (resampled
//                  while a newer view is being computed), the pane divider and
//                  the Julia crosshair straight into the presented texture.
//
// Pane images are storage buffers of packed RGBA8, row-major, top row first;
// nothing is ever read back to the CPU.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu.hpp"

#include "bigfloat.hpp"
#include "fractal_shader.hpp"
#include "perturbation.hpp"
#include "viewport.hpp"

// Half-open pixel rectangle; empty rectangles are stored as all zeros.
struct PixelRect {
  int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;

  bool empty() const { return x1 <= x0 || y1 <= y0; }
  int64_t area() const { return empty() ? 0 : int64_t(x1 - x0) * (y1 - y0); }
  bool operator==(const PixelRect &) const = default;

  PixelRect normalized() const { return empty() ? PixelRect{} : *this; }
  PixelRect shifted(int32_t dx, int32_t dy) const {
    return empty() ? PixelRect{} : PixelRect{x0 + dx, y0 + dy, x1 + dx, y1 + dy};
  }
  PixelRect intersect(const PixelRect &o) const {
    return PixelRect{std::max(x0, o.x0), std::max(y0, o.y0), std::min(x1, o.x1),
                     std::min(y1, o.y1)}
        .normalized();
  }
  // Bounding box (exact when the union is a rectangle, as the planner keeps it).
  PixelRect unite(const PixelRect &o) const {
    if (empty()) return o.normalized();
    if (o.empty()) return *this;
    return {std::min(x0, o.x0), std::min(y0, o.y0), std::max(x1, o.x1),
            std::max(y1, o.y1)};
  }
};

namespace detail_pane {

inline wgpu::ShaderModule module(gpu::Context &context, const char *code,
                                 const char *label) {
  gpu::WGSL wgsl(code, gpu::Shape{16, 16, 1});
  wgsl.label = label;
  return gpu::detail::createShaderModule(context, wgsl);
}

inline wgpu::Buffer uniform_buffer(gpu::Context &context, size_t bytes,
                                   const char *label) {
  wgpu::BufferDescriptor descriptor{};
  descriptor.label = label;
  descriptor.size = (bytes + 15) & ~size_t(15);
  descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  return context.device.CreateBuffer(&descriptor);
}

inline wgpu::BindGroupLayoutEntry buffer_entry(uint32_t binding,
                                               wgpu::ShaderStage stage,
                                               wgpu::BufferBindingType type,
                                               uint64_t min_size,
                                               bool dynamic = false) {
  wgpu::BindGroupLayoutEntry entry{};
  entry.binding = binding;
  entry.visibility = stage;
  entry.buffer.type = type;
  entry.buffer.hasDynamicOffset = dynamic;
  entry.buffer.minBindingSize = min_size;
  return entry;
}

inline wgpu::BindGroupEntry group_entry(uint32_t binding, const wgpu::Buffer &buffer,
                                        uint64_t size = wgpu::kWholeSize) {
  wgpu::BindGroupEntry entry{};
  entry.binding = binding;
  entry.buffer = buffer;
  entry.size = size;
  return entry;
}

inline wgpu::BindGroupLayout layout(gpu::Context &context,
                                    const std::vector<wgpu::BindGroupLayoutEntry> &entries) {
  wgpu::BindGroupLayoutDescriptor descriptor{};
  descriptor.entryCount = entries.size();
  descriptor.entries = entries.data();
  return context.device.CreateBindGroupLayout(&descriptor);
}

inline wgpu::BindGroup group(gpu::Context &context, const wgpu::BindGroupLayout &layout,
                             const std::vector<wgpu::BindGroupEntry> &entries) {
  wgpu::BindGroupDescriptor descriptor{};
  descriptor.layout = layout;
  descriptor.entryCount = entries.size();
  descriptor.entries = entries.data();
  return context.device.CreateBindGroup(&descriptor);
}

inline wgpu::ComputePipeline compute_pipeline(gpu::Context &context,
                                              const wgpu::BindGroupLayout &layout,
                                              const wgpu::ShaderModule &module) {
  wgpu::PipelineLayoutDescriptor layout_descriptor{};
  layout_descriptor.bindGroupLayoutCount = 1;
  layout_descriptor.bindGroupLayouts = &layout;
  wgpu::ComputePipelineDescriptor descriptor{};
  descriptor.layout = context.device.CreatePipelineLayout(&layout_descriptor);
  descriptor.compute.module = module;
  descriptor.compute.entryPoint = "main";
  return context.device.CreateComputePipeline(&descriptor);
}

inline uint32_t ceil_div(int64_t value, int64_t divisor) {
  return value <= 0 ? 0u : uint32_t((value + divisor - 1) / divisor);
}

} // namespace detail_pane

// Submits a command buffer; the future completes when the GPU has finished it
// (gpu::poll / gpu::wait drive it, as for gpu.cpp's own dispatches).
inline gpu::Future submit(gpu::Context &context, const wgpu::CommandBuffer &commands) {
  context.queue.Submit(1, &commands);
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  auto event = context.queue.OnSubmittedWorkDone(
      gpu::detail::completionMode(),
      [promise](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
        if (status == wgpu::QueueWorkDoneStatus::Success)
          promise->set_value();
        else
          promise->set_exception(std::make_exception_ptr(std::runtime_error(
              "WebGPU submission failed: " + gpu::detail::string(message))));
      });
  return gpu::Future(std::move(future), event);
}

// ---------------------------------------------------------------------------
// FractalKernel: the pluggable compute step.
//
// Contract the rest of the pipeline relies on (keep it when replacing the
// inside, e.g. with perturbation or interior block filling):
//   * begin(uniform, reference) is called once per frame before that frame's
//     encode()s; everything that determines the image must be in the uniform
//     (or in state keyed by it, like the reference orbit: the uniform carries
//     its id), because PaneRenderer compares uniforms bytewise to decide
//     whether an image can be kept or shifted.
//   * encode() writes exactly the pixels of the tile (region x levels) into
//     `target`, and the value of a pixel depends only on the uniform and its
//     grid position (local pixel + tile origin) -- not on the tile, the
//     dispatch order or the image's other contents. Block-filling schemes
//     that read neighbours must therefore stay inside one tile.
//   * Several encode()s per frame share one submission; tile parameters go
//     through a dynamic-offset uniform ring so each dispatch sees its own.
// ---------------------------------------------------------------------------

class FractalKernel {
public:
  static constexpr int kLevels = 3; // see fractal_shader.hpp

  struct Tile {
    int32_t origin_x = 0, origin_y = 0; // grid position of image pixel (0, 0)
    PixelRect rect;                     // image pixels to fill
    uint32_t level_lo = 0, level_hi = kLevels - 1;
  };

  explicit FractalKernel(gpu::Context &context) : context_(context), reference_(context) {
    using namespace detail_pane;
    layout_ = layout(context, {
        buffer_entry(0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage, 4),
        buffer_entry(1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform,
                     sizeof(ViewportUniform)),
        buffer_entry(2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform,
                     sizeof(TileUniform), true),
        buffer_entry(3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage,
                     sizeof(GpuOrbitPoint)),
        buffer_entry(4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage,
                     sizeof(GpuBla)),
    });
    pipeline_ = compute_pipeline(context, layout_, module(context, shaders::kFractal, "fractal"));
    view_ = uniform_buffer(context, sizeof(ViewportUniform), "fractal view");
    tiles_ = uniform_buffer(context, kTileSlots * kSlotBytes, "fractal tiles");
    context.check();
  }

  // `reference` is required iff the uniform selects perturbation; it is
  // uploaded unless already resident (queue order keeps work submitted
  // earlier reading the previous one).
  void begin(const ViewportUniform &uniform,
             const std::shared_ptr<const perturbation::Reference> &reference = nullptr) {
    if (uniform.flags & kFlagPerturb) {
      if (!reference || reference->id != uniform.reference_id)
        throw std::logic_error("perturbation uniform without its reference");
      if (reference_.upload(reference)) groups_.clear(); // buffers reallocated
    }
    context_.queue.WriteBuffer(view_, 0, &uniform, sizeof(uniform));
    next_slot_ = 0;
  }

  // False when this frame's tile slots are used up.
  bool encode(wgpu::ComputePassEncoder &pass, const gpu::Tensor &target, const Tile &tile) {
    if (tile.rect.empty()) return true;
    if (next_slot_ == kTileSlots) return false;
    const TileUniform uniform{tile.origin_x, tile.origin_y, tile.rect.x0, tile.rect.y0,
                              tile.rect.x1, tile.rect.y1, tile.level_lo, tile.level_hi};
    const uint32_t offset = next_slot_++ * kSlotBytes;
    context_.queue.WriteBuffer(tiles_, offset, &uniform, sizeof(uniform));
    const auto [threads_x, threads_y] = threads(tile);
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bind_group(target), 1, &offset);
    pass.DispatchWorkgroups(detail_pane::ceil_div(threads_x, 16),
                            detail_pane::ceil_div(threads_y, 16), 1);
    return true;
  }

  // Rough cost of a pixel of this view relative to a plain f32 one, so the
  // frame budget adapts at once when a view gets much slower (e.g. switching
  // to perturbation) instead of after one overlong frame. Measured frame
  // times correct the absolute scale; only the jumps matter here. Ratios
  // from tools/bench.cpp on M1: perturbation 2.8x direct f32 per iteration,
  // Fx 4.5x; BLA skips cut deep views ~3.5x (deep_1e20).
  static double relative_cost(const ViewportUniform &uniform) {
    if (!(uniform.flags & kFlagPerturb)) return 1.0;
    const double cost = (uniform.flags & kFlagFloatExp) ? 4.5 : 2.8;
    return (uniform.flags & kFlagBla) ? 0.3 * cost : cost;
  }

  // Drops cached state for a buffer that is about to be released.
  void forget(const gpu::Tensor &target) {
    std::erase_if(groups_, [&](const auto &entry) {
      return entry.first.Get() == target.buffer.Get();
    });
  }

  // Invocation grid of a tile; mirrors tile_pixel() in the shader.
  static std::pair<uint32_t, uint32_t> threads(const Tile &tile) {
    const int32_t stride = 4 >> tile.level_hi;
    const bool packed = tile.level_lo == tile.level_hi && tile.level_lo > 0;
    const int32_t step = packed ? 2 * stride : stride;
    auto base = [&](int32_t local, int32_t origin) {
      const int32_t g = local + origin;
      return g - (g & (step - 1)) - origin;
    };
    const uint32_t nx = detail_pane::ceil_div(tile.rect.x1 - base(tile.rect.x0, tile.origin_x), step);
    const uint32_t ny = detail_pane::ceil_div(tile.rect.y1 - base(tile.rect.y0, tile.origin_y), step);
    return {packed ? 3 * nx : nx, ny};
  }

private:
  struct TileUniform {
    int32_t origin_x, origin_y, x0, y0, x1, y1;
    uint32_t level_lo, level_hi;
  };
  static_assert(sizeof(TileUniform) == 32, "must match WGSL Tile");
  static constexpr uint32_t kTileSlots = 64, kSlotBytes = 256; // min uniform offset alignment

  const wgpu::BindGroup &bind_group(const gpu::Tensor &target) {
    for (const auto &[buffer, group] : groups_)
      if (buffer.Get() == target.buffer.Get()) return group;
    using namespace detail_pane;
    groups_.emplace_back(target.buffer,
                         group(context_, layout_,
                               {group_entry(0, target.buffer, target.bytes),
                                group_entry(1, view_, sizeof(ViewportUniform)),
                                group_entry(2, tiles_, sizeof(TileUniform)),
                                group_entry(3, reference_.orbit().buffer, reference_.orbit().bytes),
                                group_entry(4, reference_.bla().buffer, reference_.bla().bytes)}));
    return groups_.back().second;
  }

  gpu::Context &context_;
  perturbation::ReferenceBuffers reference_;
  wgpu::BindGroupLayout layout_;
  wgpu::ComputePipeline pipeline_;
  wgpu::Buffer view_, tiles_;
  uint32_t next_slot_ = 0;
  std::vector<std::pair<wgpu::Buffer, wgpu::BindGroup>> groups_;
};

// ---------------------------------------------------------------------------
// ShiftKernel: dst(x, y) = src(x + dx, y + dy) where that is inside src.
// ---------------------------------------------------------------------------

namespace shaders {

inline constexpr const char *kShift = R"WGSL(
struct Shift {
  width: u32,
  height: u32,
  dx: i32,
  dy: i32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> shift: Shift;

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shift.width || gid.y >= shift.height) {
    return;
  }
  let s = vec2<i32>(gid.xy) + vec2<i32>(shift.dx, shift.dy);
  if (any(s < vec2<i32>(0)) || s.x >= i32(shift.width) || s.y >= i32(shift.height)) {
    return;
  }
  dst[gid.y * shift.width + gid.x] = src[u32(s.y) * shift.width + u32(s.x)];
}
)WGSL";

} // namespace shaders

class ShiftKernel {
public:
  explicit ShiftKernel(gpu::Context &context) : context_(context) {
    using namespace detail_pane;
    layout_ = layout(context, {
        buffer_entry(0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage, 4),
        buffer_entry(1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage, 4),
        buffer_entry(2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform, 16),
    });
    pipeline_ = compute_pipeline(context, layout_, module(context, shaders::kShift, "shift"));
    uniform_ = uniform_buffer(context, 16, "shift");
    context.check();
  }

  // One shift per frame (single uniform buffer).
  void encode(wgpu::ComputePassEncoder &pass, const gpu::Tensor &src,
              const gpu::Tensor &dst, uint32_t width, uint32_t height, int32_t dx,
              int32_t dy) {
    const int32_t params[4] = {int32_t(width), int32_t(height), dx, dy};
    context_.queue.WriteBuffer(uniform_, 0, params, sizeof(params));
    using namespace detail_pane;
    const wgpu::BindGroup bind = group(context_, layout_,
                                       {group_entry(0, src.buffer, src.bytes),
                                        group_entry(1, dst.buffer, dst.bytes),
                                        group_entry(2, uniform_, 16)});
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bind);
    pass.DispatchWorkgroups(ceil_div(width, 16), ceil_div(height, 16), 1);
  }

private:
  gpu::Context &context_;
  wgpu::BindGroupLayout layout_;
  wgpu::ComputePipeline pipeline_;
  wgpu::Buffer uniform_;
};

// ---------------------------------------------------------------------------
// PaneRenderer: what to compute for a pane, and which image to show.
//
// Each image records, per refinement level, the rectangle of pixels already
// computed. A new view is served, in order of preference, by
//   1. an image that already holds it (nothing to do),
//   2. an image of the same uniform on the same pixel grid at another integer
//      origin (a pan): shift it on the GPU, compute only the exposed strips,
//   3. a fresh image, computed coarse-to-fine (levels 0, 1, 2) when it does
//      not fit the frame budget; otherwise all levels at once.
// The shown image stays on screen (resampled onto the new view by the
// Compositor) until the new one is at least as sharp, or for at most
// kStaleLimit while input keeps changing the view.
// ---------------------------------------------------------------------------

class PaneRenderer {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr int kLevels = FractalKernel::kLevels;
  static constexpr auto kStaleLimit = std::chrono::milliseconds(100);
  static constexpr int32_t kMinChunk = 16; // rows/columns per tile, at least

  struct Request {
    ViewportUniform uniform;   // the view's pixel grid (see FractalKernel)
    // The uniform's reference orbit (perturbation only).
    std::shared_ptr<const perturbation::Reference> reference = nullptr;
    uint32_t width = 0, height = 0;
    int32_t origin_x = 0, origin_y = 0; // grid position of view pixel (0, 0)
    // Actual view geometry, only used to resample previews.
    BigFloat center_x, center_y;
    double log2_pixel_size = 0.0;
  };

  // How the shown image maps onto the current view: image pixel =
  // floor(view pixel * scale + offset), with view pixels continuous.
  struct Shown {
    wgpu::Buffer buffer;
    uint32_t width = 0, height = 0;
    int32_t origin_x = 0, origin_y = 0;
    uint32_t stride = 1; // sharpest complete lattice: 4, 2 or 1
    float scale = 1.0f, offset_x = 0.0f, offset_y = 0.0f;
  };

  explicit PaneRenderer(gpu::Context &context)
      : context_(context), kernel_(context), shift_(context) {}

  // Estimated GPU cost of one pixel of the current request.
  double ns_per_pixel() const { return ns_per_unit_ * cost_; }
  // Scales the estimate by measured / predicted time (see calibrate()).
  void correct_cost(double ratio) {
    ns_per_unit_ = std::clamp(ns_per_unit_ * ratio, 0.01, 1e6);
  }

  void request(const Request &request, Clock::time_point now) {
    request_ = request;
    cost_ = FractalKernel::relative_cost(request.uniform);
    has_request_ = true;
    pending_ = false;
    drop_shift_job();
    const bool shown_current = holds(images_[shown_], request);
    if (shown_current) stale_since_.reset();
    else if (!stale_since_) stale_since_ = now;

    if (holds(images_[work_], request)) return;
    if (shown_current) {
      work_ = shown_;
      return;
    }

    // A pan: shift whichever compatible image saves the most work.
    int source = -1;
    double best = 0.0;
    for (int candidate : {work_, shown_}) {
      const Image &image = images_[candidate];
      if (!same_grid(image, request)) continue;
      const double saved = coverage(shifted_done(image, request));
      if (saved > best) {
        best = saved;
        source = candidate;
      }
    }
    if (source >= 0) {
      const int target = spare(source);
      Image &dst = images_[target];
      allocate(dst, request.width, request.height);
      dst.done = shifted_done(images_[source], request);
      dst.view = request;
      dst.has_view = true;
      shift_job_ = ShiftJob{source, target,
                            request.origin_x - images_[source].view.origin_x,
                            request.origin_y - images_[source].view.origin_y};
      work_ = target;
      return;
    }

    const int target = spare(shown_);
    Image &dst = images_[target];
    allocate(dst, request.width, request.height);
    dst.done = {};
    dst.view = request;
    dst.has_view = true;
    work_ = target;
  }

  // A view that cannot be computed yet (its reference orbit is still being
  // computed): the shown image stays on screen, resampled onto `geometry`
  // (whose uniform is ignored), and nothing is computed until the next
  // request().
  void wait_for(const Request &geometry, Clock::time_point now) {
    request_ = geometry;
    has_request_ = true;
    pending_ = true;
    drop_shift_job();
    if (!stale_since_) stale_since_ = now;
  }

  // Pixels still to compute for the current request.
  double remaining() const {
    if (!has_request_ || pending_) return 0.0;
    const Image &image = images_[work_];
    const int64_t total = int64_t(image.width) * image.height;
    double pixels = 0.0;
    for (int level = 0; level < kLevels; ++level)
      pixels += double(total - image.done[level].area()) * kFraction[level];
    return pixels;
  }

  bool busy() const { return pending_ || remaining() > 0.0 || work_ != shown_; }

  // Encodes up to `budget` pixels of work (at least one tile, so slow views
  // still progress) and returns the pixels encoded.
  double encode(wgpu::ComputePassEncoder &pass, double budget) {
    if (!has_request_ || pending_) return 0.0;
    Image &image = images_[work_];
    if (shift_job_) {
      shift_.encode(pass, images_[shift_job_->source].pixels, image.pixels, image.width,
                    image.height, shift_job_->dx, shift_job_->dy);
      shift_job_.reset();
    }
    const PixelRect full{0, 0, int32_t(image.width), int32_t(image.height)};
    double encoded = 0.0;
    bool begun = false;
    for (;;) {
      int lo = 0;
      while (lo < kLevels && image.done[lo] == full) ++lo;
      if (lo == kLevels) break;
      const double left = budget - encoded;
      if (begun && left <= 0.0) break;
      // Levels that share lo's progress can be computed together -- but only
      // if all of them fit, so slow views still go coarse to fine.
      int hi = lo;
      while (hi + 1 < kLevels && image.done[hi + 1] == image.done[lo]) ++hi;
      if (double(full.area() - image.done[lo].area()) * fraction(lo, hi) > left) hi = lo;
      const PixelRect piece = next_piece(image.done[lo], full, fraction(lo, hi), left);
      if (!begun) kernel_.begin(image.view.uniform, image.view.reference);
      begun = true;
      const FractalKernel::Tile tile{image.view.origin_x, image.view.origin_y, piece,
                                     uint32_t(lo), uint32_t(hi)};
      if (!kernel_.encode(pass, image.pixels, tile)) break;
      for (int level = lo; level <= hi; ++level)
        image.done[level] = image.done[level].unite(piece);
      encoded += double(piece.area()) * fraction(lo, hi);
    }
    return encoded;
  }

  // Call after encode(): shows the work image once it beats the shown one.
  void settle(Clock::time_point now) {
    if (work_ == shown_ || pending_) return;
    const Image &work = images_[work_];
    const int level = work.level();
    if (level < 0) return;
    const Image &shown = images_[shown_];
    bool swap = !shown.has_view || shown.level() < 0 || level == kLevels - 1 ||
                (stale_since_ && now - *stale_since_ >= kStaleLimit);
    if (!swap) {
      // Shown pixels are magnified by the zoom since it was rendered.
      const double magnified =
          stride(shown.level()) *
          std::exp2(shown.view.log2_pixel_size - request_.log2_pixel_size);
      swap = stride(level) <= magnified * (1.0 + 1e-9);
    }
    if (!swap) return;
    shown_ = work_;
    if (holds(images_[shown_], request_)) stale_since_.reset();
  }

  // The image to draw and its mapping onto the current request.
  std::optional<Shown> shown() const {
    const Image &image = images_[shown_];
    if (!has_request_ || !image.has_view || image.level() < 0) return std::nullopt;
    Shown out{.buffer = image.pixels.buffer, .width = image.width, .height = image.height,
              .origin_x = image.view.origin_x, .origin_y = image.view.origin_y,
              .stride = uint32_t(stride(image.level()))};
    const Request &r = request_, &v = image.view;
    if (same_grid(image, r)) {
      out.offset_x = float(r.origin_x - v.origin_x);
      out.offset_y = float(r.origin_y - v.origin_y);
    } else {
      const double scale = std::exp2(r.log2_pixel_size - v.log2_pixel_size);
      // Centre offset in image pixels, at any zoom depth (clamped far off).
      const double whole = std::floor(-v.log2_pixel_size);
      const FloatExp per_pixel =
          FloatExp::make(std::exp2(-v.log2_pixel_size - whole), int64_t(whole));
      auto pixels = [&](const BigFloat &a, const BigFloat &b) {
        return std::clamp((difference(a, b) * per_pixel).to_double(), -1e7, 1e7);
      };
      out.scale = float(std::min(scale, 1e7));
      out.offset_x = float(-0.5 * r.width * scale + pixels(r.center_x, v.center_x) +
                           0.5 * image.width);
      out.offset_y = float(-0.5 * r.height * scale - pixels(r.center_y, v.center_y) +
                           0.5 * image.height);
    }
    return out;
  }

private:
  // Share of an image's pixels on each level (every 4th, rest of every 2nd, rest).
  static constexpr double kFraction[kLevels] = {1.0 / 16, 3.0 / 16, 12.0 / 16};
  static double fraction(int lo, int hi) {
    double sum = 0.0;
    for (int level = lo; level <= hi; ++level) sum += kFraction[level];
    return sum;
  }
  static int stride(int level) { return 4 >> std::max(level, 0); }

  struct Image {
    gpu::Tensor pixels;
    uint32_t width = 0, height = 0;
    bool has_view = false;
    Request view;
    std::array<PixelRect, kLevels> done{};

    // Highest level complete over the whole image (-1: none).
    int level() const {
      const PixelRect full{0, 0, int32_t(width), int32_t(height)};
      int level = -1;
      while (level + 1 < kLevels && done[level + 1] == full) ++level;
      return level;
    }
  };

  struct ShiftJob {
    int source, target;
    int32_t dx, dy;
  };

  void drop_shift_job() {
    if (shift_job_) { // planned but never encoded: its target holds nothing
      images_[shift_job_->target].has_view = false;
      shift_job_.reset();
    }
  }

  static bool same_grid(const Image &image, const Request &request) {
    return image.has_view && image.width == request.width &&
           image.height == request.height &&
           std::memcmp(&image.view.uniform, &request.uniform, sizeof(request.uniform)) == 0;
  }
  static bool holds(const Image &image, const Request &request) {
    return same_grid(image, request) && image.view.origin_x == request.origin_x &&
           image.view.origin_y == request.origin_y;
  }

  // An image's progress moved onto the request's origin.
  static std::array<PixelRect, kLevels> shifted_done(const Image &image,
                                                     const Request &request) {
    const int32_t dx = request.origin_x - image.view.origin_x;
    const int32_t dy = request.origin_y - image.view.origin_y;
    const PixelRect full{0, 0, int32_t(image.width), int32_t(image.height)};
    std::array<PixelRect, kLevels> done;
    for (int level = 0; level < kLevels; ++level)
      done[level] = image.done[level].shifted(-dx, -dy).intersect(full);
    return done;
  }
  static double coverage(const std::array<PixelRect, kLevels> &done) {
    double pixels = 0.0;
    for (int level = 0; level < kLevels; ++level)
      pixels += double(done[level].area()) * kFraction[level];
    return pixels;
  }

  // Next tile for a level whose done rectangle is `done`: always adjacent to
  // it so done stays a rectangle -- side strips first, then rows.
  static PixelRect next_piece(const PixelRect &done, const PixelRect &full,
                              double fraction, double budget) {
    auto count = [&](int32_t span, int32_t available) {
      const double fit = budget / (std::max(span, 1) * fraction);
      return std::clamp<int32_t>(int32_t(std::min(fit, 1e9)), std::min(kMinChunk, available),
                                 available);
    };
    if (done.empty()) return {0, 0, full.x1, count(full.x1, full.y1)};
    const int32_t rows = done.y1 - done.y0;
    if (done.x0 > 0) {
      const int32_t n = count(rows, done.x0);
      return {done.x0 - n, done.y0, done.x0, done.y1};
    }
    if (done.x1 < full.x1) {
      const int32_t n = count(rows, full.x1 - done.x1);
      return {done.x1, done.y0, done.x1 + n, done.y1};
    }
    if (done.y0 > 0) {
      const int32_t n = count(full.x1, done.y0);
      return {0, done.y0 - n, full.x1, done.y0};
    }
    const int32_t n = count(full.x1, full.y1 - done.y1);
    return {0, done.y1, full.x1, done.y1 + n};
  }

  // An image index other than `a` and the shown one (the old work image if
  // possible, so the third stays as it was).
  int spare(int a) const {
    if (work_ != a && work_ != shown_) return work_;
    for (int i = 0; i < int(images_.size()); ++i)
      if (i != a && i != shown_) return i;
    throw std::logic_error("no spare pane image");
  }

  void allocate(Image &image, uint32_t width, uint32_t height) {
    if (image.width == width && image.height == height && image.pixels.buffer) return;
    if (image.pixels.buffer) kernel_.forget(image.pixels);
    image.pixels = gpu::createTensor(context_, gpu::Shape{size_t(width) * height}, gpu::kf32);
    image.width = width;
    image.height = height;
    image.has_view = false;
    image.done = {};
  }

  gpu::Context &context_;
  FractalKernel kernel_;
  ShiftKernel shift_;
  std::array<Image, 3> images_;
  int shown_ = 0, work_ = 0;
  Request request_{};
  bool has_request_ = false;
  bool pending_ = false; // see wait_for()
  std::optional<ShiftJob> shift_job_;
  std::optional<Clock::time_point> stale_since_;
  double ns_per_unit_ = 2.0, cost_ = 1.0; // ~ M1, 1440x1650 home view: 3.8 ms
};

// ---------------------------------------------------------------------------
// Frame budget: how much each pane computes per frame, learnt from measured
// frame times (submit to completion, so it includes everything else that
// shares the GPU).
// ---------------------------------------------------------------------------

inline constexpr double kFrameBudgetMs = 12.0;  // compute per frame, all panes
inline constexpr double kFrameOverheadMs = 0.5; // composition, submit latency

// Pixels each pane may compute this frame: panes needing less than an equal
// share get what they need, the others split the rest.
inline std::vector<double> share_budget(const std::vector<PaneRenderer *> &panes,
                                        double budget_ms = kFrameBudgetMs) {
  auto need_ms = [](const PaneRenderer *p) { return p->remaining() * p->ns_per_pixel() * 1e-6; };
  std::vector<size_t> order(panes.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return need_ms(panes[a]) < need_ms(panes[b]); });
  std::vector<double> pixels(panes.size());
  for (size_t k = 0; k < order.size(); ++k) {
    const PaneRenderer *p = panes[order[k]];
    const double ms = std::min(need_ms(p), budget_ms / double(order.size() - k));
    budget_ms -= ms;
    pixels[order[k]] = ms * 1e6 / p->ns_per_pixel() * (1.0 + 1e-9);
  }
  return pixels;
}

// Corrects the panes' cost estimates after a frame that computed `encoded`
// pixels per pane took `measured_ms`.
inline void calibrate(const std::vector<PaneRenderer *> &panes,
                      const std::vector<double> &encoded, double measured_ms) {
  double predicted = 0.0;
  for (size_t i = 0; i < panes.size(); ++i)
    predicted += encoded[i] * panes[i]->ns_per_pixel() * 1e-6;
  const double work_ms = measured_ms - kFrameOverheadMs;
  // Too little work to tell apart from the fixed overhead.
  if (predicted <= 0.0 || (predicted < 1.0 && work_ms < 2.0)) return;
  const double ratio = std::clamp(std::max(work_ms, 0.05) / predicted, 0.1, 30.0);
  for (size_t i = 0; i < panes.size(); ++i)
    if (encoded[i] > 0.0) panes[i]->correct_cost(ratio);
}

// ---------------------------------------------------------------------------
// Compositor: draws panes into a colour target in one render pass.
// ---------------------------------------------------------------------------

namespace shaders {

inline constexpr const char *kComposite = R"WGSL(
struct Pane {
  rect_x: i32, rect_y: i32, rect_w: i32, rect_h: i32,  // output pixels
  image_w: i32, image_h: i32, origin_x: i32, origin_y: i32,
  stride: i32, valid: i32, divider: i32, marker: i32,   // divider: 1 left, 2 right column
  marker_x: i32, marker_y: i32, marker_radius: i32, _pad0: i32,
  scale: f32, offset_x: f32, offset_y: f32, inv_divisor: f32,
  marker_r: f32, marker_g: f32, marker_b: f32, _pad1: f32,
};

@group(0) @binding(0) var<storage, read> image: array<u32>;
@group(0) @binding(1) var<uniform> pane: Pane;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn sample(local: vec2<i32>) -> vec3<f32> {
  if (pane.valid == 0) {
    return vec3<f32>(0.0);
  }
  // Output pixel centre -> render pixel -> shown image pixel.
  let r = (vec2<f32>(local) + 0.5) * pane.inv_divisor;
  let f = r * pane.scale + vec2<f32>(pane.offset_x, pane.offset_y);
  let size = vec2<i32>(pane.image_w, pane.image_h);
  let i = vec2<i32>(floor(f));
  if (any(i < vec2<i32>(0)) || any(i >= size)) {
    return vec3<f32>(0.0);
  }
  // Nearest computed point of the sharpest complete lattice (on the grid).
  let origin = vec2<i32>(pane.origin_x, pane.origin_y);
  let g = i + origin;
  var p = g - (g & vec2<i32>(pane.stride - 1)) - origin;
  p = select(p, p + vec2<i32>(pane.stride), p < vec2<i32>(0));
  if (any(p >= size)) {
    return vec3<f32>(0.0);
  }
  return unpack4x8unorm(image[u32(p.y * pane.image_w + p.x)]).rgb;
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(floor(position.xy));
  let local = pixel - vec2<i32>(pane.rect_x, pane.rect_y);
  var color = sample(local);
  if (pane.marker != 0) {  // 3 px thick crosshair
    let d = abs(pixel - vec2<i32>(pane.marker_x, pane.marker_y));
    if ((d.y <= 1 && d.x <= pane.marker_radius) || (d.x <= 1 && d.y <= pane.marker_radius)) {
      color = vec3<f32>(pane.marker_r, pane.marker_g, pane.marker_b);
    }
  }
  if (((pane.divider & 1) != 0 && local.x == 0) ||
      ((pane.divider & 2) != 0 && local.x == pane.rect_w - 1)) {
    color = vec3<f32>(20.0 / 255.0);
  }
  return vec4<f32>(color, 1.0);
}
)WGSL";

} // namespace shaders

class Compositor {
public:
  struct Pane {
    int32_t x = 0, y = 0, w = 0, h = 0; // output pixels
    uint32_t divisor = 1;               // output pixels per render pixel
    std::optional<PaneRenderer::Shown> shown;
    bool divider_left = false, divider_right = false;
    bool marker = false;
    int32_t marker_x = 0, marker_y = 0, marker_radius = 0; // output pixels
    uint8_t marker_rgb[3] = {255, 255, 255};
  };

  Compositor(gpu::Context &context, wgpu::TextureFormat format) : context_(context) {
    using namespace detail_pane;
    layout_ = layout(context, {
        buffer_entry(0, wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage, 4),
        buffer_entry(1, wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::Uniform,
                     sizeof(PaneUniform)),
    });
    const wgpu::ShaderModule shader = module(context, shaders::kComposite, "composite");
    wgpu::PipelineLayoutDescriptor layout_descriptor{};
    layout_descriptor.bindGroupLayoutCount = 1;
    layout_descriptor.bindGroupLayouts = &layout_;
    wgpu::ColorTargetState target{};
    target.format = format;
    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "fs";
    fragment.targetCount = 1;
    fragment.targets = &target;
    wgpu::RenderPipelineDescriptor descriptor{};
    descriptor.label = "composite";
    descriptor.layout = context.device.CreatePipelineLayout(&layout_descriptor);
    descriptor.vertex.module = shader;
    descriptor.vertex.entryPoint = "vs";
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.fragment = &fragment;
    pipeline_ = context.device.CreateRenderPipeline(&descriptor);

    wgpu::BufferDescriptor empty{};
    empty.label = "composite placeholder";
    empty.size = 16;
    empty.usage = wgpu::BufferUsage::Storage;
    placeholder_ = context.device.CreateBuffer(&empty);
    context.check();
  }

  // True if both would draw exactly the same pixels.
  static bool same(const Pane &a, const Pane &b) {
    const PaneUniform ua = to_uniform(a), ub = to_uniform(b);
    auto buffer = [](const Pane &p) { return p.shown ? p.shown->buffer.Get() : nullptr; };
    return std::memcmp(&ua, &ub, sizeof(ua)) == 0 && buffer(a) == buffer(b);
  }

  void encode(wgpu::CommandEncoder &encoder, const wgpu::TextureView &target,
              uint32_t target_w, uint32_t target_h, const std::vector<Pane> &panes) {
    wgpu::RenderPassColorAttachment color{};
    color.view = target;
    color.loadOp = wgpu::LoadOp::Clear;
    color.storeOp = wgpu::StoreOp::Store;
    color.clearValue = {0.0, 0.0, 0.0, 1.0};
    wgpu::RenderPassDescriptor descriptor{};
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &color;
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&descriptor);
    pass.SetPipeline(pipeline_);
    while (uniforms_.size() < panes.size())
      uniforms_.push_back(detail_pane::uniform_buffer(context_, sizeof(PaneUniform), "composite pane"));
    for (size_t i = 0; i < panes.size(); ++i) {
      const Pane &p = panes[i];
      const int32_t x0 = std::clamp(p.x, 0, int32_t(target_w));
      const int32_t y0 = std::clamp(p.y, 0, int32_t(target_h));
      const int32_t x1 = std::clamp(p.x + p.w, 0, int32_t(target_w));
      const int32_t y1 = std::clamp(p.y + p.h, 0, int32_t(target_h));
      if (x1 <= x0 || y1 <= y0) continue;
      const PaneUniform uniform = to_uniform(p);
      context_.queue.WriteBuffer(uniforms_[i], 0, &uniform, sizeof(uniform));
      using namespace detail_pane;
      const wgpu::BindGroup bind = group(
          context_, layout_,
          {group_entry(0, p.shown ? p.shown->buffer : placeholder_),
           group_entry(1, uniforms_[i], sizeof(PaneUniform))});
      pass.SetBindGroup(0, bind);
      pass.SetScissorRect(uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0));
      pass.Draw(3);
    }
    pass.End();
  }

private:
  struct PaneUniform {
    int32_t rect_x, rect_y, rect_w, rect_h;
    int32_t image_w, image_h, origin_x, origin_y;
    int32_t stride, valid, divider, marker;
    int32_t marker_x, marker_y, marker_radius, pad0;
    float scale, offset_x, offset_y, inv_divisor;
    float marker_r, marker_g, marker_b, pad1;
  };
  static_assert(sizeof(PaneUniform) == 96, "must match WGSL Pane");

  static PaneUniform to_uniform(const Pane &p) {
    PaneUniform u{};
    u.rect_x = p.x;
    u.rect_y = p.y;
    u.rect_w = p.w;
    u.rect_h = p.h;
    if (p.shown) {
      u.image_w = int32_t(p.shown->width);
      u.image_h = int32_t(p.shown->height);
      u.origin_x = p.shown->origin_x;
      u.origin_y = p.shown->origin_y;
      u.stride = int32_t(p.shown->stride);
      u.valid = 1;
      u.scale = p.shown->scale;
      u.offset_x = p.shown->offset_x;
      u.offset_y = p.shown->offset_y;
    }
    u.inv_divisor = 1.0f / float(std::max(p.divisor, 1u));
    u.divider = (p.divider_left ? 1 : 0) | (p.divider_right ? 2 : 0);
    u.marker = p.marker ? 1 : 0;
    u.marker_x = p.marker_x;
    u.marker_y = p.marker_y;
    u.marker_radius = p.marker_radius;
    u.marker_r = p.marker_rgb[0] / 255.0f;
    u.marker_g = p.marker_rgb[1] / 255.0f;
    u.marker_b = p.marker_rgb[2] / 255.0f;
    return u;
  }

  gpu::Context &context_;
  wgpu::BindGroupLayout layout_;
  wgpu::RenderPipeline pipeline_;
  wgpu::Buffer placeholder_;
  std::vector<wgpu::Buffer> uniforms_;
};
