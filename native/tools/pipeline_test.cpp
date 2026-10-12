// Headless checks for the incremental frame pipeline (src/pane_renderer.hpp).
//
//   make pipeline_test && ./pipeline_test           all checks
//   ./pipeline_test --timing                         also time deep-zoom panning
//
// Every check drives PaneRenderer the way the app does (request, encode with a
// budget, settle, submit) and compares the image it shows against one
// full-image FractalKernel dispatch of the same view -- the reference
// renderer bench.cpp uses (itself bit-identical to the pre-pipeline kernel):
//   * progressive refinement under tiny budgets (many tiles, coarse to fine),
//   * pan reuse: shift + strips, also mid-refinement and chained pans,
//   * the compositor: presented pixels equal the image (1x and half-res),
//   * waiting for a reference orbit: nothing is computed, the old image
//     stays shown, and the view then renders as usual.
// Perturbation views (to 1e100) run the same checks with one reference
// orbit computed up front, as the app's ReferenceCache would provide it.
// Exit status is non-zero if any comparison differs.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu.hpp"

#include "../src/pane_renderer.hpp"
#include "../src/perturbation.hpp"
#include "../src/viewport.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// flags: kFlagJulia, kFlagOutline; kFlagPerturb renders with a reference
// orbit computed at the initial view (see with_reference).
struct View {
  Camera camera;
  uint32_t iterations = 512;
  uint32_t flags = 0;
  BigFloat julia_re = 0.0, julia_im = 0.0;
  std::shared_ptr<const perturbation::Reference> reference = nullptr;
};

constexpr const char *kDeepX = "-0.743643887037158704752191506114774";
constexpr const char *kDeepY = "0.131825904205311970493132056385139";
// Misiurewicz point M(4,1) (see bench.cpp), to 120 digits.
constexpr const char *kMisiurewiczX =
    "-0.10109636384562216102578544573862256546380544282625348387693117766078084074047058427482"
    "1219810516779033404531908556741";
constexpr const char *kMisiurewiczY =
    "0.956286510809141500771096057729977435809833336510529170034314321500524659065716732526978"
    "4107873398072043444724926469284";

Camera deep_camera(const char *x, const char *y, double log10_zoom) {
  const double log2_zoom = log10_zoom * std::log2(10.0);
  const long bits = Camera::bits_for(log2_zoom);
  Camera camera{.center_x = BigFloat(x, bits), .center_y = BigFloat(y, bits),
                .log2_zoom = log2_zoom};
  camera.update_precision();
  return camera;
}

perturbation::View pview(const View &v, uint32_t width, uint32_t height) {
  return perturbation::make_view(v.camera, (v.flags & kFlagJulia) != 0, v.julia_re,
                                 v.julia_im, v.iterations, width, height);
}

// Computes the reference orbit the app would use for the view as it is now.
View with_reference(View v, uint32_t width, uint32_t height) {
  if (v.flags & kFlagPerturb)
    v.reference = perturbation::compute_reference(perturbation::key_for(pview(v, width, height)));
  return v;
}

// Same construction as App::request in main.cpp.
PaneRenderer::Request request_for(const View &v, uint32_t width, uint32_t height) {
  const perturbation::View view = pview(v, width, height);
  return PaneRenderer::Request{
      .uniform = make_uniform(view, v.reference.get(), (v.flags & kFlagOutline) != 0, 0.0f),
      .reference = v.reference,
      .width = width, .height = height,
      .origin_x = view.grid.origin_x, .origin_y = view.grid.origin_y,
      .center_x = view.camera.center_x, .center_y = view.camera.center_y,
      .log2_pixel_size = view.camera.log2_pixel_size(height),
  };
}

std::vector<uint32_t> read_buffer(gpu::Context &context, const wgpu::Buffer &buffer,
                                  size_t count) {
  gpu::Tensor tensor{.buffer = buffer, .shape = gpu::Shape{count}, .type = gpu::kf32,
                     .bytes = count * 4};
  std::vector<uint32_t> out(count);
  gpu::Future future = gpu::toCPU(context, tensor, out);
  gpu::wait(context, future);
  return out;
}

class Harness {
public:
  explicit Harness(gpu::Context &context)
      : context_(context), kernel_(context),
        compositor_(context, wgpu::TextureFormat::RGBA8Unorm) {}

  // One full-image dispatch: the reference.
  std::vector<uint32_t> reference(const PaneRenderer::Request &r) {
    if (!scratch_.buffer || scratch_.bytes != size_t(r.width) * r.height * 4)
      scratch_ = gpu::createTensor(context_, gpu::Shape{size_t(r.width) * r.height}, gpu::kf32);
    auto encoder = context_.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    kernel_.begin(r.uniform, r.reference);
    kernel_.encode(pass, scratch_,
                   FractalKernel::Tile{.origin_x = r.origin_x, .origin_y = r.origin_y,
                                       .rect = {0, 0, int32_t(r.width), int32_t(r.height)}});
    pass.End();
    gpu::Future done = submit(context_, encoder.Finish());
    gpu::wait(context_, done);
    return read_buffer(context_, scratch_.buffer, size_t(r.width) * r.height);
  }

  // One app-style frame; returns the submit-to-done time in ms.
  double frame(PaneRenderer &pane, const PaneRenderer::Request &r, double budget,
               double *encoded = nullptr) {
    const auto start = Clock::now();
    pane.request(r, start);
    auto encoder = context_.device.CreateCommandEncoder();
    double pixels;
    {
      auto pass = encoder.BeginComputePass();
      pixels = pane.encode(pass, budget);
      pass.End();
    }
    pane.settle(Clock::now());
    if (encoded) *encoded = pixels;
    gpu::Future done = submit(context_, encoder.Finish());
    gpu::wait(context_, done);
    return elapsed_ms(start);
  }

  // Frames until the pane has nothing left to do.
  int finish(PaneRenderer &pane, const PaneRenderer::Request &r, double budget) {
    int frames = 0;
    do {
      frame(pane, r, budget);
      ++frames;
    } while (pane.busy() && frames < 100000);
    return frames;
  }

  std::vector<uint32_t> shown(const PaneRenderer &pane) {
    const auto s = pane.shown();
    if (!s) return {};
    return read_buffer(context_, s->buffer, size_t(s->width) * s->height);
  }

  // Composites one pane covering a w x h target and reads it back (RGBA8).
  std::vector<uint32_t> composite(const PaneRenderer &pane, uint32_t w, uint32_t h,
                                  uint32_t divisor) {
    wgpu::TextureDescriptor descriptor{};
    descriptor.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size = {w, h, 1};
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    wgpu::Texture texture = context_.device.CreateTexture(&descriptor);

    Compositor::Pane p;
    p.w = int32_t(w);
    p.h = int32_t(h);
    p.divisor = divisor;
    p.shown = pane.shown();
    auto encoder = context_.device.CreateCommandEncoder();
    compositor_.encode(encoder, texture.CreateView(), w, h, {p});

    const uint32_t row = (w * 4 + 255) & ~255u; // 256-byte bytesPerRow alignment
    gpu::Tensor staging = gpu::createTensor(context_, gpu::Shape{size_t(row / 4) * h}, gpu::kf32);
    wgpu::TexelCopyTextureInfo src{};
    src.texture = texture;
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = staging.buffer;
    dst.layout.bytesPerRow = row;
    dst.layout.rowsPerImage = h;
    const wgpu::Extent3D size{w, h, 1};
    encoder.CopyTextureToBuffer(&src, &dst, &size);
    gpu::Future done = submit(context_, encoder.Finish());
    gpu::wait(context_, done);
    std::vector<uint32_t> padded = read_buffer(context_, staging.buffer, size_t(row / 4) * h);
    std::vector<uint32_t> out(size_t(w) * h);
    for (uint32_t y = 0; y < h; ++y)
      std::memcpy(&out[size_t(y) * w], &padded[size_t(y) * (row / 4)], w * 4);
    return out;
  }

private:
  gpu::Context &context_;
  FractalKernel kernel_;
  Compositor compositor_;
  gpu::Tensor scratch_;
};

int failures = 0;

void report(const char *name, const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
            bool must_match = true) {
  size_t differ = 0;
  int max_delta = 0;
  if (a.size() != b.size()) {
    std::printf("  %-52s size mismatch (%zu vs %zu)\n", name, a.size(), b.size());
    if (must_match) ++failures;
    return;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i]) continue;
    ++differ;
    for (int c = 0; c < 4; ++c)
      max_delta = std::max(max_delta, std::abs(int((a[i] >> (8 * c)) & 255) -
                                               int((b[i] >> (8 * c)) & 255)));
  }
  const bool ok = !must_match || differ == 0;
  std::printf("  %-52s %8zu / %zu pixels differ (max channel delta %d)%s\n", name, differ,
              a.size(), max_delta, ok ? "" : "  FAIL");
  if (!ok) ++failures;
}

struct Case {
  const char *name;
  View view;
  uint32_t width, height;
};

} // namespace

int main(int argc, char **argv) {
  bool timing = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--timing") timing = true;
    else {
      std::fprintf(stderr, "usage: pipeline_test [--timing]\n");
      return 2;
    }
  }

  gpu::Context context = gpu::createContext();
  Harness harness(context);

  const Case cases[] = {
      {"home", View{.camera = Camera::at(-0.5, 0.0, 1.0)}, 1440, 1650},
      {"seahorse", View{.camera = Camera::at(-0.7453, 0.1127, 1e3), .iterations = 1000},
       720, 825},
      {"julia_outline", View{.camera = Camera::at(0.0, 0.0, 1.3),
                             .flags = kFlagJulia | kFlagOutline,
                             .julia_re = -0.8, .julia_im = 0.156}, 701, 533},
      {"deep_1e9_perturb", View{.camera = deep_camera(kDeepX, kDeepY, 9.0),
                                .iterations = 4000, .flags = kFlagPerturb}, 360, 413},
      {"deep_1e20_perturb_bla", View{.camera = deep_camera(kDeepX, kDeepY, 20.0),
                                     .iterations = 20000, .flags = kFlagPerturb}, 360, 413},
      {"deep_1e100_fx_outline",
       View{.camera = deep_camera(kMisiurewiczX, kMisiurewiczY, 100.0), .iterations = 5000,
            .flags = kFlagPerturb | kFlagOutline}, 333, 301},
      {"julia_1e30_perturb",
       View{.camera = deep_camera("1.5275031186435346107897464024449153375667459478117",
                                  "-0.0759121783522878670718141948260463483664221948479",
                                  30.0),
            .iterations = 2000, .flags = kFlagJulia | kFlagPerturb,
            .julia_re = -0.8, .julia_im = 0.156}, 300, 257},
  };

  std::mt19937 rng(1234);
  for (const Case &c : cases) {
    std::printf("%s (%ux%u)\n", c.name, c.width, c.height);
    const View start = with_reference(c.view, c.width, c.height);
    const PaneRenderer::Request base = request_for(start, c.width, c.height);
    const std::vector<uint32_t> expected = harness.reference(base);

    // 1. Progressive refinement: budgets far below one frame force many
    //    tiles on every level; the final image must equal the reference.
    for (double budget : {1e9, 20000.0, 3000.0}) {
      PaneRenderer pane(context);
      const int frames = harness.finish(pane, base, budget);
      char label[96];
      std::snprintf(label, sizeof(label), "progressive, budget %.0f px (%d frames)", budget, frames);
      report(label, harness.shown(pane), expected);
    }

    // 2. Pan reuse: integer pans through Camera::pan_pixels, shifted image
    //    + strips vs a full render of the panned view.
    {
      PaneRenderer pane(context);
      harness.finish(pane, base, 1e9);
      View view = start;
      const int pans[][2] = {{7, 0}, {0, -13}, {-40, 25}, {3, 3}, {-200, -1}, {1, 150}};
      for (const auto &pan : pans) {
        view.camera.pan_pixels(pan[0], pan[1], c.height);
        const PaneRenderer::Request r = request_for(view, c.width, c.height);
        pane.request(r, Clock::now());
        const double work = pane.remaining();
        const int frames = harness.finish(pane, r, 1e9);
        char label[96];
        std::snprintf(label, sizeof(label), "pan (%d, %d): computed %.1f%% of pixels, %d frame%s",
                      pan[0], pan[1], 100.0 * work / (double(c.width) * c.height), frames,
                      frames == 1 ? "" : "s");
        report(label, harness.shown(pane), harness.reference(r));
      }
      // The same final view rendered on a fresh grid (centre = view centre):
      // differs only by f32 rounding of the pixel coordinates.
      Camera fresh = view.camera;
      fresh.fold_pan();
      View refolded = view;
      refolded.camera = fresh;
      report("  (info) vs full render on a re-centred grid", harness.shown(pane),
             harness.reference(request_for(refolded, c.width, c.height)), false);
    }

    // 3. Pans in the middle of a progressive render, random sizes, then a
    //    sub-pixel drag (remainder carried) -- final image vs reference.
    {
      PaneRenderer pane(context);
      View view = start;
      PaneRenderer::Request r = request_for(view, c.width, c.height);
      std::uniform_int_distribution<int> step(-60, 60), frames(0, 4);
      for (int i = 0; i < 12; ++i) {
        for (int f = frames(rng); f > 0; --f) harness.frame(pane, r, 5000.0);
        view.camera.pan_pixels(step(rng), step(rng), c.height);
        r = request_for(view, c.width, c.height);
      }
      for (int i = 0; i < 9; ++i) view.camera.pan_pixels(0.37, -0.61, c.height);
      r = request_for(view, c.width, c.height);
      harness.finish(pane, r, 5000.0);
      report("random pans mid-refinement + sub-pixel drags", harness.shown(pane),
             harness.reference(r));
    }

    // 4. Compositor: presented pixels equal the image, 1x and half-res.
    {
      PaneRenderer pane(context);
      harness.finish(pane, base, 1e9);
      report("composite 1x == image", harness.composite(pane, c.width, c.height, 1), expected);
      std::vector<uint32_t> upscaled(size_t(c.width) * 2 * c.height * 2);
      for (uint32_t y = 0; y < 2 * c.height; ++y)
        for (uint32_t x = 0; x < 2 * c.width; ++x)
          upscaled[size_t(y) * 2 * c.width + x] = expected[size_t(y / 2) * c.width + x / 2];
      report("composite half-res == image upscaled 2x",
             harness.composite(pane, 2 * c.width, 2 * c.height, 2), upscaled);
    }

    // 5. Waiting for a reference: a view whose reference is not ready
    //    computes nothing and keeps the old image (pending), then renders
    //    normally once requested.
    {
      PaneRenderer pane(context);
      harness.finish(pane, base, 1e9);
      const auto before = pane.shown();
      View view = start;
      view.camera.zoom_at(0.3 * c.width, 0.6 * c.height, 1.7, c.width, c.height);
      view = with_reference(view, c.width, c.height);
      const PaneRenderer::Request r = request_for(view, c.width, c.height);
      bool idle = true;
      for (int i = 0; i < 3; ++i) {
        pane.wait_for(r, Clock::now());
        auto encoder = context.device.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        idle = idle && pane.encode(pass, 1e9) == 0.0;
        pass.End();
        pane.settle(Clock::now());
        gpu::Future done = submit(context, encoder.Finish());
        gpu::wait(context, done);
      }
      const auto during = pane.shown();
      const bool kept = during && before && during->buffer.Get() == before->buffer.Get() &&
                        pane.busy() && idle && during->scale < 1.0f;
      std::printf("  %-52s %s\n", "waiting: nothing computed, old image resampled",
                  kept ? "ok" : "FAIL");
      if (!kept) ++failures;
      harness.finish(pane, r, 1e9);
      report("after waiting: zoomed view == reference", harness.shown(pane),
             harness.reference(r));
    }
  }

  if (timing) {
    // Split view at the home views, both panes fully re-rendered every frame
    // (what palette animation does). Old path: per pane dispatch, read back,
    // wait (then SDL upload, not counted). New path: both panes + composition
    // in one submission, no readback. Interleaved, best of 15.
    std::printf("\ntiming: split view home + Julia dragon, 1440x1650 panes, full re-render\n");
    View m{.camera = Camera::at(-0.5, 0.0, 1.0)};
    View j{.camera = Camera::at(0.0, 0.0, 1.0), .flags = kFlagJulia, .julia_re = -0.8,
           .julia_im = 0.156};
    PaneRenderer pm(context), pj(context);
    double old_best = 1e30, new_best = 1e30;
    std::vector<uint32_t> host(size_t(1440) * 1650);
    gpu::Tensor framebuffer = gpu::createTensor(context, gpu::Shape{host.size()}, gpu::kf32);
    FractalKernel kernel(context);
    for (int i = 0; i < 17; ++i) {
      PaneRenderer::Request rm = request_for(m, 1440, 1650), rj = request_for(j, 1440, 1650);
      rm.uniform.color_phase = rj.uniform.color_phase = 0.01f * float(i + 1); // force full renders
      auto t0 = Clock::now();
      for (const auto *r : {&rm, &rj}) {
        auto encoder = context.device.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        kernel.begin(r->uniform);
        kernel.encode(pass, framebuffer, FractalKernel::Tile{.rect = {0, 0, 1440, 1650}});
        pass.End();
        gpu::Future dispatched = submit(context, encoder.Finish());
        gpu::Future downloaded = gpu::toCPU(context, framebuffer, host);
        gpu::wait(context, downloaded);
        gpu::wait(context, dispatched);
      }
      const double old_ms = elapsed_ms(t0);

      t0 = Clock::now();
      pm.request(rm, t0);
      pj.request(rj, t0);
      auto encoder = context.device.CreateCommandEncoder();
      {
        auto pass = encoder.BeginComputePass();
        pm.encode(pass, 1e12);
        pj.encode(pass, 1e12);
        pass.End();
      }
      pm.settle(Clock::now());
      pj.settle(Clock::now());
      wgpu::TextureDescriptor descriptor{};
      descriptor.usage = wgpu::TextureUsage::RenderAttachment;
      descriptor.dimension = wgpu::TextureDimension::e2D;
      descriptor.size = {2880, 1650, 1};
      descriptor.format = wgpu::TextureFormat::BGRA8Unorm;
      static wgpu::Texture target = context.device.CreateTexture(&descriptor);
      static Compositor compositor(context, wgpu::TextureFormat::BGRA8Unorm);
      Compositor::Pane a, b;
      a.w = b.w = 1440;
      a.h = b.h = 1650;
      b.x = 1440;
      a.shown = pm.shown();
      b.shown = pj.shown();
      a.divider_right = b.divider_left = true;
      compositor.encode(encoder, target.CreateView(), 2880, 1650, {a, b});
      gpu::Future done = submit(context, encoder.Finish());
      gpu::wait(context, done);
      const double new_ms = elapsed_ms(t0);
      if (i >= 2) {
        old_best = std::min(old_best, old_ms);
        new_best = std::min(new_best, new_ms);
      }
    }
    std::printf("  old path %.2f ms/frame (+ SDL texture uploads), new path %.2f ms/frame\n",
                old_best, new_best);

    // App-style deep-zoom panning: a full render, then a 10 px drag every
    // frame with the app's 12 ms budget and cost calibration.
    std::printf("\ntiming: deep_1e20 perturbation + BLA, 1440x1650 pane\n");
    const uint32_t w = 1440, h = 1650;
    View view = with_reference(View{.camera = deep_camera(kDeepX, kDeepY, 20.0),
                                    .iterations = 20000, .flags = kFlagPerturb},
                               w, h);
    PaneRenderer pane(context);
    PaneRenderer::Request r = request_for(view, w, h);
    auto t0 = Clock::now();
    harness.reference(r);
    std::printf("  full render (old path per frame, without readback): %.1f ms\n", elapsed_ms(t0));

    // Same budgeting and calibration as the app.
    auto calibrated_frame = [&](const PaneRenderer::Request &req) {
      pane.request(req, Clock::now());
      const double budget = share_budget({&pane})[0];
      double encoded = 0.0;
      const double ms = harness.frame(pane, req, budget, &encoded);
      calibrate({&pane}, {encoded}, ms);
      return ms;
    };
    t0 = Clock::now();
    int frames = 0, first_coarse = -1;
    double first_coarse_ms = 0.0;
    do {
      calibrated_frame(r);
      ++frames;
      if (first_coarse < 0 && pane.shown()) {
        first_coarse = frames;
        first_coarse_ms = elapsed_ms(t0);
      }
    } while (pane.busy());
    std::printf("  progressive from scratch: first image after %d frames / %.1f ms, "
                "full after %d frames / %.1f ms\n",
                first_coarse, first_coarse_ms, frames, elapsed_ms(t0));
    double worst = 0.0, total = 0.0;
    int current = 0;
    const int kFrames = 30;
    for (int i = 0; i < kFrames; ++i) {
      view.camera.pan_pixels(10, 4, h);
      r = request_for(view, w, h);
      const double ms = calibrated_frame(r);
      worst = std::max(worst, ms);
      total += ms;
      const auto s = pane.shown();
      if (s && s->offset_x == 0.0f && s->offset_y == 0.0f && s->scale == 1.0f && s->stride == 1)
        ++current;
    }
    std::printf("  dragging 10x4 px/frame: %.1f ms/frame avg, %.1f ms worst; %d/%d frames "
                "showed the exact current view at full resolution\n",
                total / kFrames, worst, current, kFrames);
  }

  std::printf("\n%s\n", failures ? "FAILED" : "all comparisons identical");
  return failures ? 1 : 0;
}
