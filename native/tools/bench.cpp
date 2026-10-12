// Headless benchmark + reference renderer for the fractal shader.
//
//   make bench && ./bench                 timings for every case
//   ./bench --png out/                    also write out/<case>.png
//   ./bench --case deep --reps 20         one case, more repetitions
//
// Each case renders one Retina split pane (1440x1650) through the same
// FractalKernel the app uses (one full-image tile). "compute" is dispatch + wait, "readback" is the
// GPU -> CPU copy, "frame" is the app's path (dispatch + readback, one wait).
// Times are the best of --reps after two warm-up runs, so other GPU work on
// the machine inflates them less.

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu.hpp"

#include "../src/pane_renderer.hpp"
#include "../src/viewport.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Case {
  const char *name;
  double center_x, center_y, zoom;
  uint32_t iterations;
  uint32_t flags;
  double julia_re = 0.0, julia_im = 0.0;
};

constexpr double kDeepX = -0.743643887037158704752191506114774;
constexpr double kDeepY = 0.131825904205311970493132056385139;

const Case kCases[] = {
    {"home", -0.5, 0.0, 1.0, 512, 0},
    {"home_outline", -0.5, 0.0, 1.0, 512, kFlagOutline},
    {"seahorse", -0.7453, 0.1127, 1e3, 1000, 0},
    {"interior_heavy", -0.16, 1.035, 40.0, 4000, 0},
    {"deep_1e9", kDeepX, kDeepY, 1e9, 4000, kFlagDouble},
    {"deep_1e12", kDeepX, kDeepY, 1e12, 6000, kFlagDouble},
    {"julia_dragon", 0.0, 0.0, 1.0, 512, kFlagJulia, -0.8, 0.156},
    {"julia_cantor", 0.0, 0.0, 0.463, 512, kFlagJulia | kFlagOutline, -2.5, 0.0},
};

ViewportUniform make_uniform(const Case &c, uint32_t width, uint32_t height) {
  const Camera cam{.center_x = c.center_x, .center_y = c.center_y, .zoom = c.zoom};
  const Camera::Extent e = Camera::extent(width, height);
  const DoubleSingle ox(cam.center_x), oy(cam.center_y);
  const DoubleSingle jx(c.julia_re), jy(c.julia_im);
  return ViewportUniform{
      .x_min = float(e.x_min), .x_max = float(e.x_max),
      .y_min = float(e.y_min), .y_max = float(e.y_max),
      .zoom = float(cam.zoom),
      .iteration_limit = c.iterations,
      .width = width, .height = height,
      .offset_x_hi = ox.hi, .offset_x_lo = ox.lo,
      .offset_y_hi = oy.hi, .offset_y_lo = oy.lo,
      .julia_x_hi = jx.hi, .julia_x_lo = jx.lo,
      .julia_y_hi = jy.hi, .julia_y_lo = jy.lo,
      .flags = c.flags,
      .color_phase = 0.0f,
  };
}

// Minimal RGBA8 PNG writer (zlib-compressed, no filtering).
bool write_png(const std::string &path, const std::vector<uint32_t> &pixels,
               uint32_t width, uint32_t height) {
  std::vector<unsigned char> raw;
  raw.reserve(size_t(height) * (1 + width * 4));
  for (uint32_t y = 0; y < height; ++y) {
    raw.push_back(0); // filter: none
    const auto *row = reinterpret_cast<const unsigned char *>(&pixels[size_t(y) * width]);
    raw.insert(raw.end(), row, row + width * 4);
  }
  uLongf packed_size = compressBound(raw.size());
  std::vector<unsigned char> packed(packed_size);
  if (compress2(packed.data(), &packed_size, raw.data(), raw.size(), 6) != Z_OK) return false;
  packed.resize(packed_size);

  FILE *file = std::fopen(path.c_str(), "wb");
  if (!file) return false;
  auto be32 = [&](uint32_t v) {
    const unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16),
                                (unsigned char)(v >> 8), (unsigned char)v};
    std::fwrite(b, 1, 4, file);
  };
  auto chunk = [&](const char *type, const unsigned char *data, uint32_t size) {
    be32(size);
    std::fwrite(type, 1, 4, file);
    if (size) std::fwrite(data, 1, size, file);
    uLong crc = crc32(0, reinterpret_cast<const Bytef *>(type), 4);
    if (size) crc = crc32(crc, data, size);
    be32(uint32_t(crc));
  };
  static const unsigned char kSignature[8] = {137, 'P', 'N', 'G', '\r', '\n', 26, '\n'};
  std::fwrite(kSignature, 1, 8, file);
  unsigned char header[13] = {};
  for (int i = 0; i < 4; ++i) {
    header[i] = (unsigned char)(width >> (24 - 8 * i));
    header[4 + i] = (unsigned char)(height >> (24 - 8 * i));
  }
  header[8] = 8; // bit depth
  header[9] = 6; // RGBA
  chunk("IHDR", header, 13);
  chunk("IDAT", packed.data(), uint32_t(packed.size()));
  chunk("IEND", nullptr, 0);
  return std::fclose(file) == 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string png_dir, only;
  int reps = 10;
  uint32_t width = 1440, height = 1650;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--png" && i + 1 < argc) png_dir = argv[++i];
    else if (arg == "--case" && i + 1 < argc) only = argv[++i];
    else if (arg == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%ux%u", &width, &height);
    else {
      std::fprintf(stderr, "usage: bench [--case NAME] [--reps N] [--size WxH] [--png DIR]\n");
      return 2;
    }
  }

  gpu::Context context = gpu::createContext();
  FractalKernel fractal(context);
  gpu::Tensor framebuffer =
      gpu::createTensor(context, gpu::Shape{size_t(width) * height}, gpu::kf32);
  std::vector<uint32_t> pixels(size_t(width) * height);

  std::printf("%ux%u, best of %d\n", width, height, reps);
  std::printf("%-16s %10s %10s %10s\n", "case", "compute", "readback", "frame");
  for (const Case &c : kCases) {
    if (!only.empty() && only != c.name) continue;
    const ViewportUniform uniform = make_uniform(c, width, height);
    const FractalKernel::Tile tile{.rect = {0, 0, int32_t(width), int32_t(height)}};
    auto dispatch = [&] {
      auto encoder = context.device.CreateCommandEncoder();
      auto pass = encoder.BeginComputePass();
      fractal.begin(uniform);
      fractal.encode(pass, framebuffer, tile);
      pass.End();
      return submit(context, encoder.Finish());
    };

    double compute = 1e30, readback = 1e30, frame = 1e30;
    for (int i = 0; i < reps + 2; ++i) {
      auto t0 = Clock::now();
      gpu::Future dispatched = dispatch();
      gpu::wait(context, dispatched);
      const double tc = elapsed_ms(t0);

      auto t1 = Clock::now();
      gpu::Future downloaded = gpu::toCPU(context, framebuffer, pixels);
      gpu::wait(context, downloaded);
      const double tr = elapsed_ms(t1);

      auto t2 = Clock::now();
      gpu::Future d2 = dispatch();
      gpu::Future r2 = gpu::toCPU(context, framebuffer, pixels);
      gpu::wait(context, r2);
      gpu::wait(context, d2);
      const double tf = elapsed_ms(t2);

      if (i >= 2) { // skip warm-up
        compute = std::min(compute, tc);
        readback = std::min(readback, tr);
        frame = std::min(frame, tf);
      }
    }
    std::printf("%-16s %8.2fms %8.2fms %8.2fms\n", c.name, compute, readback, frame);

    if (!png_dir.empty()) {
      const std::string path = png_dir + "/" + c.name + ".png";
      if (!write_png(path, pixels, width, height))
        std::fprintf(stderr, "could not write %s\n", path.c_str());
    }
  }
  return 0;
}
