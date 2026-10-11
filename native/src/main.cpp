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
// The window opens split: Mandelbrot on the left, the Julia set for the
// crosshair point on the right. Each pane has its own camera.
//
// Controls:
//   Scroll wheel      zoom at cursor (pane under cursor)
//   Left drag         pan (pane under cursor)
//   Mouse hover       Julia point follows the cursor over the Mandelbrot pane
//   Space             lock / unlock the Julia point (crosshair turns yellow)
//   Right drag        move the Julia point even while locked
//   Tab               split view <-> single view of the pane under the cursor
//   J                 single view: Julia for the point under cursor (again: back)
//   + / -             iteration limit          P    precision: auto/f32/double-single
//   D                 outline: draw the set itself (e.g. Julia Cantor dust)
//   K                 jump to a Cantor-set Julia (c = -2.5: dust on a line)
//   H                 toggle half-res render   C    animate palette
//   R                 reset pane under cursor  Esc  quit

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
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
// Explorer state: cameras, layout and render settings.
// ---------------------------------------------------------------------------

enum class Precision { Auto, Single, DoubleSingle };
enum class Fractal { Mandelbrot, Julia };

struct Explorer {
  Camera mandelbrot = initial(Fractal::Mandelbrot);
  Camera julia = initial(Fractal::Julia);
  bool split = true;
  bool follow_mouse = true; // split view: Julia point tracks the cursor
  Fractal single = Fractal::Mandelbrot; // single view: what fills the window
  double julia_re = -0.8, julia_im = 0.156;
  uint32_t iterations = 512;
  Precision precision = Precision::Auto;
  bool animate_colors = false;
  bool outline = false;
  bool half_resolution = false;
  float color_phase = 0.0f;

  static Camera initial(Fractal fractal) {
    switch (fractal) {
    case Fractal::Mandelbrot: return Camera{.center_x = -0.5};
    case Fractal::Julia: return Camera{};
    }
    return Camera{};
  }

  Camera &camera(Fractal fractal) {
    return fractal == Fractal::Julia ? julia : mandelbrot;
  }
  const Camera &camera(Fractal fractal) const {
    return fractal == Fractal::Julia ? julia : mandelbrot;
  }

  bool uses_double(Fractal fractal, uint32_t height) const {
    if (precision != Precision::Auto) return precision == Precision::DoubleSingle;
    // Switch once a pixel shrinks to a few f32 ulps of the coordinates.
    const Camera &cam = camera(fractal);
    const double magnitude =
        std::max({std::abs(cam.center_x), std::abs(cam.center_y), 1.0});
    return cam.pixel_size(height) < magnitude * 4e-6;
  }

  ViewportUniform uniform(Fractal fractal, uint32_t width, uint32_t height) const {
    const Camera &cam = camera(fractal);
    const Camera::Extent e = Camera::extent(width, height);
    const DoubleSingle ox(cam.center_x), oy(cam.center_y);
    const DoubleSingle jx(julia_re), jy(julia_im);
    uint32_t flags = 0;
    if (fractal == Fractal::Julia) flags |= kFlagJulia;
    if (uses_double(fractal, height)) flags |= kFlagDouble;
    if (outline) flags |= kFlagOutline;
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

// One on-screen view: a screen rectangle (output pixels), its GPU kernel and
// the texture its frames are uploaded to.
struct Pane {
  Pane(gpu::Context &context, Fractal kind) : kind(kind), gpu(context) {}

  Fractal kind;
  bool visible = false;
  SDL_Rect rect{};             // in renderer output pixels
  uint32_t width = 0, height = 0; // render resolution (rect / divisor)
  FractalRenderer gpu;
  SdlPtr<SDL_Texture> texture;
  uint32_t texture_w = 0, texture_h = 0;

  bool contains(double x, double y) const {
    return visible && x >= rect.x && x < rect.x + rect.w && y >= rect.y &&
           y < rect.y + rect.h;
  }
};

class App {
public:
  App() {
    window_.reset(SDL_CreateWindow(
        "Fractal Explorer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1600, 800, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI));
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

      if (!layout()) { // minimised
        SDL_Delay(16);
        continue;
      }
      draw();

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
  Pane &pane(Fractal kind) { return panes_[size_t(kind)]; }
  Pane &mandelbrot_pane() { return pane(Fractal::Mandelbrot); }
  Pane &julia_pane() { return pane(Fractal::Julia); }

  int divisor() const { return state_.half_resolution ? 2 : 1; }

  // Assigns each visible pane its screen rectangle. False when minimised.
  bool layout() {
    int output_w, output_h;
    SDL_GetRendererOutputSize(renderer_.get(), &output_w, &output_h);
    if (output_w <= 1 || output_h <= 0) return false;

    for (Pane &p : panes_) p.visible = false;
    if (state_.split) {
      const int left = output_w / 2;
      mandelbrot_pane().rect = {0, 0, left, output_h};
      julia_pane().rect = {left, 0, output_w - left, output_h};
      mandelbrot_pane().visible = julia_pane().visible = true;
    } else {
      pane(state_.single).rect = {0, 0, output_w, output_h};
      pane(state_.single).visible = true;
    }
    for (Pane &pane : panes_) {
      pane.width = uint32_t(std::max(pane.rect.w / divisor(), 1));
      pane.height = uint32_t(std::max(pane.rect.h / divisor(), 1));
    }
    return true;
  }

  void draw() {
    SDL_SetRenderDrawColor(renderer_.get(), 0, 0, 0, 255);
    SDL_RenderClear(renderer_.get());
    for (Pane &pane : panes_) {
      if (!pane.visible) continue;
      ensure_texture(pane);
      const auto &pixels =
          pane.gpu.render(state_.uniform(pane.kind, pane.width, pane.height));
      SDL_UpdateTexture(pane.texture.get(), nullptr, pixels.data(),
                        int(pane.width * sizeof(uint32_t)));
      SDL_RenderCopy(renderer_.get(), pane.texture.get(), nullptr, &pane.rect);
    }
    if (state_.split) {
      draw_julia_marker();
      const SDL_Rect divider{julia_pane().rect.x - 1, 0, 2, julia_pane().rect.h};
      SDL_SetRenderDrawColor(renderer_.get(), 20, 20, 20, 255);
      SDL_RenderFillRect(renderer_.get(), &divider);
    }
    SDL_RenderPresent(renderer_.get());
  }

  // Crosshair on the Mandelbrot pane at the Julia constant c.
  void draw_julia_marker() {
    Pane &m = mandelbrot_pane();
    double px, py;
    state_.mandelbrot.to_pixel(state_.julia_re, state_.julia_im, m.width,
                               m.height, px, py);
    const int x = m.rect.x + int(px * divisor());
    const int y = m.rect.y + int(py * divisor());
    if (!m.contains(x, y)) return;
    const int r = 14 * int(output_scale() + 0.5);
    if (state_.follow_mouse)
      SDL_SetRenderDrawColor(renderer_.get(), 255, 255, 255, 255);
    else
      SDL_SetRenderDrawColor(renderer_.get(), 255, 210, 0, 255);
    for (int o = -1; o <= 1; ++o) { // 3px thick
      SDL_RenderDrawLine(renderer_.get(), x - r, y + o, x + r, y + o);
      SDL_RenderDrawLine(renderer_.get(), x + o, y - r, x + o, y + r);
    }
  }

  void ensure_texture(Pane &pane) {
    if (pane.texture && pane.texture_w == pane.width && pane.texture_h == pane.height)
      return;
    pane.texture.reset(SDL_CreateTexture(renderer_.get(), SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING,
                                         int(pane.width), int(pane.height)));
    if (!pane.texture) throw std::runtime_error(SDL_GetError());
    pane.texture_w = pane.width;
    pane.texture_h = pane.height;
  }

  // ---- input --------------------------------------------------------------

  void handle(const SDL_Event &event) {
    switch (event.type) {
    case SDL_QUIT: running_ = false; break;
    case SDL_MOUSEWHEEL: {
      double x, y;
      cursor_output(x, y);
      if (Pane *pane = pane_at(x, y)) {
        double px, py;
        to_pane(*pane, x, y, px, py);
        const double factor = std::pow(1.15, double(event.wheel.preciseY));
        state_.camera(pane->kind).zoom_at(px, py, factor, pane->width, pane->height);
      }
      break;
    }
    case SDL_MOUSEBUTTONDOWN: {
      const double scale = output_scale();
      Pane *pane = pane_at(event.button.x * scale, event.button.y * scale);
      if (event.button.button == SDL_BUTTON_LEFT) drag_pane_ = pane;
      if (event.button.button == SDL_BUTTON_RIGHT) {
        if (!state_.split) {
          toggle_julia();
        } else if (pane && pane->kind == Fractal::Mandelbrot) {
          picking_ = true;
          pick_julia(event.button.x * scale, event.button.y * scale);
        }
      }
      break;
    }
    case SDL_MOUSEBUTTONUP:
      if (event.button.button == SDL_BUTTON_LEFT) drag_pane_ = nullptr;
      if (event.button.button == SDL_BUTTON_RIGHT) picking_ = false;
      break;
    case SDL_MOUSEMOTION: {
      const double scale = output_scale();
      if (drag_pane_) {
        const double step = scale / divisor();
        state_.camera(drag_pane_->kind)
            .pan_pixels(event.motion.xrel * step, event.motion.yrel * step,
                        drag_pane_->height);
      }
      // Hover-follow pauses while panning so the point stays put on the plane.
      const double x = event.motion.x * scale, y = event.motion.y * scale;
      const bool following = state_.split && state_.follow_mouse && !drag_pane_;
      if ((picking_ || following) && mandelbrot_pane().contains(x, y))
        pick_julia(x, y);
      break;
    }
    case SDL_KEYDOWN: handle_key(event.key.keysym.sym); break;
    }
  }

  void handle_key(SDL_Keycode key) {
    switch (key) {
    case SDLK_ESCAPE: running_ = false; break;
    case SDLK_TAB: {
      // Leaving split view keeps whichever pane the cursor is over.
      if (state_.split) {
        double x, y;
        cursor_output(x, y);
        if (Pane *pane = pane_at(x, y)) state_.single = pane->kind;
      }
      state_.split = !state_.split;
      drag_pane_ = nullptr;
      picking_ = false;
      break;
    }
    case SDLK_SPACE: state_.follow_mouse = !state_.follow_mouse; break;
    case SDLK_k: show_cantor_julia(); break;
    case SDLK_j:
      if (!state_.split) toggle_julia();
      break;
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
    case SDLK_d: state_.outline = !state_.outline; break;
    case SDLK_r: {
      double x, y;
      cursor_output(x, y);
      const Pane *pane = pane_at(x, y);
      const Fractal kind = pane ? pane->kind : state_.single;
      state_.camera(kind) = Explorer::initial(kind);
      break;
    }
    }
  }

  // Single view: Mandelbrot -> Julia set of the point under the cursor;
  // Julia -> back to the Mandelbrot view exactly as it was left.
  void toggle_julia() {
    if (state_.single == Fractal::Mandelbrot) {
      double x, y;
      cursor_output(x, y);
      pick_julia(x, y);
      state_.julia = Explorer::initial(Fractal::Julia);
    }
    state_.single = state_.single == Fractal::Julia ? Fractal::Mandelbrot
                                                    : Fractal::Julia;
  }

  // For real c < -2 the Julia set is a Cantor set on the real axis inside
  // [-beta, beta], beta = (1 + sqrt(1 - 4c)) / 2: each inverse step keeps two
  // sub-intervals and drops the middle, like the middle-thirds construction
  // (with unequal gaps). Locks the point and turns outline on so the dust,
  // which has no area, is visible.
  void show_cantor_julia() {
    constexpr double kC = -2.5;
    const double beta = (1.0 + std::sqrt(1.0 - 4.0 * kC)) / 2.0;
    state_.julia_re = kC;
    state_.julia_im = 0.0;
    state_.follow_mouse = false;
    state_.outline = true;
    picking_ = false;
    if (!state_.split) state_.single = Fractal::Julia;
    layout();

    // Frame [-beta, beta] with a 10% margin on each side.
    const Pane &j = julia_pane();
    const double aspect = double(j.width) / j.height;
    state_.julia = Camera{
        .zoom = 2.0 * Camera::kHalfHeight * aspect / (2.4 * beta)};

    // Make sure the crosshair at c is on screen in the Mandelbrot pane.
    Pane &m = mandelbrot_pane();
    double px, py;
    state_.mandelbrot.to_pixel(kC, 0.0, m.width, m.height, px, py);
    if (px < 0 || px >= m.width || py < 0 || py >= m.height) {
      const double m_aspect = double(m.width) / m.height;
      state_.mandelbrot = Camera{
          .center_x = -1.0, .zoom = 2.0 * Camera::kHalfHeight * m_aspect / 3.8};
    }
  }

  // Sets the Julia constant to the Mandelbrot coordinate at an output pixel.
  void pick_julia(double x, double y) {
    Pane &m = mandelbrot_pane();
    double px, py;
    to_pane(m, x, y, px, py);
    state_.mandelbrot.to_complex(px, py, m.width, m.height, state_.julia_re,
                                 state_.julia_im);
  }

  // Renderer output pixels per window point (2 on Retina displays).
  double output_scale() const {
    int window_w, window_h, output_w, output_h;
    SDL_GetWindowSize(window_.get(), &window_w, &window_h);
    SDL_GetRendererOutputSize(renderer_.get(), &output_w, &output_h);
    return window_w > 0 ? double(output_w) / window_w : 1.0;
  }

  void cursor_output(double &x, double &y) const {
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    const double scale = output_scale();
    x = mx * scale;
    y = my * scale;
  }

  Pane *pane_at(double x, double y) {
    for (Pane &pane : panes_)
      if (pane.contains(x, y)) return &pane;
    return nullptr;
  }

  // Output pixel -> pane-local render pixel.
  void to_pane(const Pane &pane, double x, double y, double &px, double &py) const {
    px = (x - pane.rect.x) / divisor();
    py = (y - pane.rect.y) / divisor();
  }

  void update_title(double fps) {
    static constexpr const char *kPrecisionNames[] = {"auto", "f32", "double-single"};
    auto precision = [&](Fractal kind) {
      return state_.uses_double(kind, pane(kind).height) ? "ds" : "f32";
    };
    char buffer[128];
    std::string title;
    auto append = [&](auto... args) {
      std::snprintf(buffer, sizeof(buffer), args...);
      title += buffer;
    };
    append("%.0f fps", fps);
    if (mandelbrot_pane().visible)
      append(" | Mandelbrot %.3gx %s", state_.mandelbrot.zoom,
             precision(Fractal::Mandelbrot));
    if (julia_pane().visible)
      append(" | Julia c = %.6f %+.6fi  %.3gx %s", state_.julia_re,
             state_.julia_im, state_.julia.zoom, precision(Fractal::Julia));
    if (state_.split) append("%s", state_.follow_mouse ? " (follows mouse)" : " (locked)");

    append(" | %u iter | precision %s%s%s", state_.iterations,
           kPrecisionNames[int(state_.precision)],
           state_.outline ? " | outline" : "",
           state_.half_resolution ? " | half-res" : "");
    SDL_SetWindowTitle(window_.get(), title.c_str());
  }

  SdlPtr<SDL_Window> window_;
  SdlPtr<SDL_Renderer> renderer_;

  gpu::Context context_ = gpu::createContext();
  // Indexed by Fractal.
  std::array<Pane, 2> panes_{Pane(context_, Fractal::Mandelbrot),
                             Pane(context_, Fractal::Julia)};
  Explorer state_;
  bool running_ = true;
  Pane *drag_pane_ = nullptr;
  bool picking_ = false;
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
