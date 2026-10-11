// Native GPU Mandelbrot / Julia explorer: gpu.cpp (WebGPU via Dawn) compute
// shader + SDL2 window and input.
//
// Build (macOS / Linux, from the native/ directory, after staging Dawn with
// third_party/gpu.cpp/tools/build_dawn.sh):
//
//   clang++ -std=c++20 -O3 src/main.cpp -o fractal_explorer \
//     -Ithird_party/gpu.cpp -Ithird_party/gpu.cpp/third_party/dawn/include \
//     $(sdl2-config --cflags) \
//     -Lthird_party/gpu.cpp/third_party/dawn/lib -lwebgpu_dawn \
//     -Wl,-rpath,third_party/gpu.cpp/third_party/dawn/lib \
//     $(sdl2-config --libs)
//
// g++ works the same way (g++ 11+ for C++20). Do NOT add -ffast-math: the
// double -> (hi, lo) split relies on strict IEEE rounding.
//
// Controls:
//   Scroll wheel      zoom at cursor          Left drag     pan
//   Right click / J   Julia set for the point under the cursor (again: back)
//   + / -             iteration limit          P             precision: auto/f32/double-single
//   H                 toggle half-res render   C             animate palette
//   R                 reset view               Esc           quit

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

#include "gpu.hpp"

#include "fractal_shader.hpp"
#include "viewport.hpp"

namespace {

// ---------------------------------------------------------------------------
// GPU side: owns the framebuffer tensor and the compute kernel.
// ---------------------------------------------------------------------------

class FractalRenderer {
public:
  explicit FractalRenderer(gpu::Context &context)
      : context_(context), shader_(shaders::kFractal, gpu::Shape{16, 16, 1}) {
    shader_.label = "fractal";
  }

  // Dispatches one frame and reads it back. Pixels are packed RGBA8,
  // row-major, top row first.
  const std::vector<uint32_t> &render(const ViewportUniform &uniform) {
    if (uniform.width != width_ || uniform.height != height_) allocate(uniform);
    gpu::toGPU(context_, uniform, kernel_);
    gpu::Future dispatched = gpu::dispatchKernel(context_, kernel_);
    // The copy is queued behind the dispatch, so waiting on it waits for both.
    gpu::Future downloaded = gpu::toCPU(context_, framebuffer_, pixels_);
    gpu::wait(context_, downloaded);
    gpu::wait(context_, dispatched); // already complete; surfaces any error
    return pixels_;
  }

private:
  void allocate(const ViewportUniform &uniform) {
    width_ = uniform.width;
    height_ = uniform.height;
    const size_t count = size_t(width_) * height_;
    pixels_.assign(count, 0);
    // gpu.cpp has no u32 tensor type; kf32 has the same 4-byte element and
    // the shader declares the buffer as array<u32>.
    framebuffer_ = gpu::createTensor(context_, gpu::Shape{count}, gpu::kf32);
    kernel_ = gpu::createKernel(
        context_, shader_, gpu::Bindings{gpu::readWrite(framebuffer_)},
        gpu::ceilDiv(gpu::Shape{width_, height_, 1}, shader_.workgroupSize),
        uniform);
  }

  gpu::Context &context_;
  gpu::WGSL shader_;
  gpu::Tensor framebuffer_;
  gpu::Kernel kernel_;
  std::vector<uint32_t> pixels_;
  uint32_t width_ = 0, height_ = 0;
};

// ---------------------------------------------------------------------------
// Explorer state: cameras, mode and render settings.
// ---------------------------------------------------------------------------

enum class Precision { Auto, Single, DoubleSingle };

struct Explorer {
  Camera mandelbrot = initial_mandelbrot();
  Camera julia;
  bool julia_mode = false;
  double julia_re = -0.8, julia_im = 0.156;
  uint32_t iterations = 512;
  Precision precision = Precision::Auto;
  bool animate_colors = false;
  bool half_resolution = false;
  float color_phase = 0.0f;

  static Camera initial_mandelbrot() { return Camera{.center_x = -0.5}; }

  Camera &camera() { return julia_mode ? julia : mandelbrot; }
  const Camera &camera() const { return julia_mode ? julia : mandelbrot; }

  bool uses_double(uint32_t height) const {
    if (precision != Precision::Auto) return precision == Precision::DoubleSingle;
    // Switch once a pixel shrinks to a few f32 ulps of the coordinates.
    const Camera &cam = camera();
    const double magnitude =
        std::max({std::abs(cam.center_x), std::abs(cam.center_y), 1.0});
    return cam.pixel_size(height) < magnitude * 4e-6;
  }

  ViewportUniform uniform(uint32_t width, uint32_t height) const {
    const Camera &cam = camera();
    const Camera::Extent e = Camera::extent(width, height);
    const DoubleSingle ox(cam.center_x), oy(cam.center_y);
    const DoubleSingle jx(julia_re), jy(julia_im);
    uint32_t flags = 0;
    if (julia_mode) flags |= kFlagJulia;
    if (uses_double(height)) flags |= kFlagDouble;
    return ViewportUniform{
        .x_min = float(e.x_min), .x_max = float(e.x_max),
        .y_min = float(e.y_min), .y_max = float(e.y_max),
        .zoom = float(cam.zoom),
        .iteration_limit = iterations,
        .width = width, .height = height,
        .offset_x_hi = ox.hi, .offset_x_lo = ox.lo,
        .offset_y_hi = oy.hi, .offset_y_lo = oy.lo,
        .julia_x_hi = jx.hi, .julia_x_lo = jx.lo,
        .julia_y_hi = jy.hi, .julia_y_lo = jy.lo,
        .flags = flags,
        .color_phase = color_phase,
    };
  }
};

// ---------------------------------------------------------------------------
// SDL window / input / presentation.
// ---------------------------------------------------------------------------

struct SdlDeleter {
  void operator()(SDL_Window *w) const { SDL_DestroyWindow(w); }
  void operator()(SDL_Renderer *r) const { SDL_DestroyRenderer(r); }
  void operator()(SDL_Texture *t) const { SDL_DestroyTexture(t); }
};
template <typename T> using SdlPtr = std::unique_ptr<T, SdlDeleter>;

class App {
public:
  App() {
    window_.reset(SDL_CreateWindow(
        "Fractal Explorer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 800, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI));
    if (!window_) throw std::runtime_error(SDL_GetError());
    // No SDL_RENDERER_PRESENTVSYNC: the frame rate is unlocked.
    renderer_.reset(SDL_CreateRenderer(window_.get(), -1, SDL_RENDERER_ACCELERATED));
    if (!renderer_) throw std::runtime_error(SDL_GetError());
  }

  void run() {
    using Clock = std::chrono::steady_clock;
    auto last_frame = Clock::now(), last_title = last_frame;
    int frames = 0;

    while (running_) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) handle(event);

      const auto now = Clock::now();
      const double dt = std::chrono::duration<double>(now - last_frame).count();
      last_frame = now;
      if (state_.animate_colors)
        state_.color_phase = std::fmod(state_.color_phase + float(dt * 0.05), 1.0f);

      update_render_size();
      if (width_ == 0 || height_ == 0) { // minimised
        SDL_Delay(16);
        continue;
      }
      present(gpu_.render(state_.uniform(width_, height_)));

      ++frames;
      const double since_title = std::chrono::duration<double>(now - last_title).count();
      if (since_title >= 0.5) {
        update_title(frames / since_title);
        frames = 0;
        last_title = now;
      }
    }
  }

private:
  void handle(const SDL_Event &event) {
    switch (event.type) {
    case SDL_QUIT: running_ = false; break;
    case SDL_MOUSEWHEEL: {
      double px, py;
      cursor_pixels(px, py);
      const double factor = std::pow(1.15, double(event.wheel.preciseY));
      state_.camera().zoom_at(px, py, factor, width_, height_);
      break;
    }
    case SDL_MOUSEBUTTONDOWN:
      if (event.button.button == SDL_BUTTON_LEFT) dragging_ = true;
      if (event.button.button == SDL_BUTTON_RIGHT) toggle_julia();
      break;
    case SDL_MOUSEBUTTONUP:
      if (event.button.button == SDL_BUTTON_LEFT) dragging_ = false;
      break;
    case SDL_MOUSEMOTION:
      if (dragging_) {
        const double scale = pixels_per_point();
        state_.camera().pan_pixels(event.motion.xrel * scale,
                                   event.motion.yrel * scale, height_);
      }
      break;
    case SDL_KEYDOWN: handle_key(event.key.keysym.sym); break;
    }
  }

  void handle_key(SDL_Keycode key) {
    switch (key) {
    case SDLK_ESCAPE: running_ = false; break;
    case SDLK_j: toggle_julia(); break;
    case SDLK_EQUALS:
    case SDLK_PLUS:
    case SDLK_KP_PLUS:
      state_.iterations = std::min<uint32_t>(state_.iterations * 3 / 2, 1u << 20);
      break;
    case SDLK_MINUS:
    case SDLK_KP_MINUS:
      state_.iterations = std::max<uint32_t>(state_.iterations * 2 / 3, 32);
      break;
    case SDLK_p:
      state_.precision = Precision((int(state_.precision) + 1) % 3);
      break;
    case SDLK_h: state_.half_resolution = !state_.half_resolution; break;
    case SDLK_c: state_.animate_colors = !state_.animate_colors; break;
    case SDLK_r:
      state_.camera() = state_.julia_mode ? Camera{} : Explorer::initial_mandelbrot();
      break;
    }
  }

  // Mandelbrot -> Julia set of the point under the cursor; Julia -> back to
  // the Mandelbrot view exactly as it was left.
  void toggle_julia() {
    if (!state_.julia_mode) {
      double px, py;
      cursor_pixels(px, py);
      state_.mandelbrot.to_complex(px, py, width_, height_, state_.julia_re,
                                   state_.julia_im);
      state_.julia = Camera{};
    }
    state_.julia_mode = !state_.julia_mode;
  }

  double pixels_per_point() const {
    int window_w, window_h;
    SDL_GetWindowSize(window_.get(), &window_w, &window_h);
    return window_w > 0 ? double(width_) / window_w : 1.0;
  }

  void cursor_pixels(double &px, double &py) const {
    int x, y;
    SDL_GetMouseState(&x, &y);
    const double scale = pixels_per_point();
    px = x * scale;
    py = y * scale;
  }

  void update_render_size() {
    int output_w, output_h;
    SDL_GetRendererOutputSize(renderer_.get(), &output_w, &output_h);
    const int divisor = state_.half_resolution ? 2 : 1;
    width_ = uint32_t(std::max(output_w / divisor, 0));
    height_ = uint32_t(std::max(output_h / divisor, 0));
    if (width_ && height_ && (width_ != texture_w_ || height_ != texture_h_)) {
      texture_.reset(SDL_CreateTexture(renderer_.get(), SDL_PIXELFORMAT_RGBA32,
                                       SDL_TEXTUREACCESS_STREAMING, int(width_),
                                       int(height_)));
      if (!texture_) throw std::runtime_error(SDL_GetError());
      texture_w_ = width_;
      texture_h_ = height_;
    }
  }

  void present(const std::vector<uint32_t> &pixels) {
    SDL_UpdateTexture(texture_.get(), nullptr, pixels.data(),
                      int(width_ * sizeof(uint32_t)));
    SDL_RenderCopy(renderer_.get(), texture_.get(), nullptr, nullptr);
    SDL_RenderPresent(renderer_.get());
  }

  void update_title(double fps) {
    static constexpr const char *kPrecisionNames[] = {"auto", "f32", "double-single"};
    const bool dbl = state_.uses_double(height_);
    char title[256];
    std::snprintf(title, sizeof(title),
                  "%s | %.0f fps | zoom %.3g | %u iter | %s (%s) | %ux%u%s",
                  state_.julia_mode ? "Julia" : "Mandelbrot", fps,
                  state_.camera().zoom, state_.iterations,
                  dbl ? "double-single" : "f32",
                  kPrecisionNames[int(state_.precision)], width_, height_,
                  state_.half_resolution ? " (half-res)" : "");
    SDL_SetWindowTitle(window_.get(), title);
  }

  SdlPtr<SDL_Window> window_;
  SdlPtr<SDL_Renderer> renderer_;
  SdlPtr<SDL_Texture> texture_;
  uint32_t texture_w_ = 0, texture_h_ = 0;
  uint32_t width_ = 0, height_ = 0;

  gpu::Context context_ = gpu::createContext();
  FractalRenderer gpu_{context_};
  Explorer state_;
  bool running_ = true;
  bool dragging_ = false;
};

} // namespace

int main() {
  SDL_SetMainReady();
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }
  int status = 0;
  try {
    App app;
    app.run();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "fatal: %s\n", error.what());
    status = 1;
  }
  SDL_Quit();
  return status;
}
