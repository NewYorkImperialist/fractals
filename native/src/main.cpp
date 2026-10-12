// Native GPU Mandelbrot / Julia explorer: gpu.cpp (WebGPU via Dawn) compute
// shader + SDL2 window and input. Frames go straight from the compute shader
// to a WebGPU surface on the window (no CPU round trip), and only when
// something changed: when idle the app sleeps waiting for events.
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
// plus MPFR / GMP (-lmpfr -lgmp; see the Makefile). g++ works the same way
// (g++ 11+ for C++20). Do NOT add -ffast-math: the double -> (hi, lo) split
// and the CPU-side FloatExp and BLA arithmetic rely on IEEE rounding and
// infinities.
//
// Deep zooms use perturbation against an arbitrary-precision reference orbit
// (perturbation.hpp), computed on a background thread; until a view's first
// usable reference is ready its pane keeps showing the previous image.
//
// Rendering is incremental (see pane_renderer.hpp): pans shift the last image
// and compute only the exposed strips, slow views refine coarse to fine in
// time slices while the previous image stays on screen, resampled. Interior
// pixels are cut short by an exact periodicity check (direct f32) and by
// interior block filling (all paths; see fractal_shader.hpp).
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
//   + / -             iteration limit          P    precision: auto/f32/perturbation
//   D                 outline: draw the set itself (e.g. Julia Cantor dust)
//   K                 jump to a Cantor-set Julia (c = -2.5: dust on a line)
//   H                 toggle half-res render   C    animate palette
//   R                 reset pane under cursor  Esc  quit

#define SDL_MAIN_HANDLED
#include <SDL.h>
#if defined(__APPLE__)
#include <SDL_metal.h>
#elif defined(__linux__)
#include <SDL_syswm.h>
#endif

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

#include "bigfloat.hpp"
#include "pane_renderer.hpp"
#include "perturbation.hpp"
#include "viewport.hpp"

namespace {

// ---------------------------------------------------------------------------
// Explorer state: cameras, layout and render settings.
// ---------------------------------------------------------------------------

enum class Precision { Auto, Single, Perturbation };
enum class Fractal { Mandelbrot, Julia };

struct Explorer {
  Camera mandelbrot = initial(Fractal::Mandelbrot);
  Camera julia = initial(Fractal::Julia);
  bool split = true;
  bool follow_mouse = true; // split view: Julia point tracks the cursor
  Fractal single = Fractal::Mandelbrot; // single view: what fills the window
  BigFloat julia_re = -0.8, julia_im = 0.156;
  uint32_t iterations = 512;
  Precision precision = Precision::Auto;
  bool animate_colors = false;
  bool outline = false;
  bool half_resolution = false;
  float color_phase = 0.0f;

  static Camera initial(Fractal fractal) {
    switch (fractal) {
    case Fractal::Mandelbrot: return Camera::at(-0.5, 0.0, 1.0);
    case Fractal::Julia: return Camera::at(0.0, 0.0, 1.0);
    }
    return Camera{};
  }

  Camera &camera(Fractal fractal) {
    return fractal == Fractal::Julia ? julia : mandelbrot;
  }
  const Camera &camera(Fractal fractal) const {
    return fractal == Fractal::Julia ? julia : mandelbrot;
  }

  bool uses_perturbation(Fractal fractal, uint32_t height) const {
    if (precision != Precision::Auto) return precision == Precision::Perturbation;
    return perturbation::needed(camera(fractal), height);
  }

  // The view on the camera's pixel grid (Camera::grid): view pixel (px, py)
  // is grid pixel (px + origin_x, py + origin_y).
  perturbation::View view(Fractal fractal, uint32_t width, uint32_t height) const {
    return perturbation::make_view(camera(fractal), fractal == Fractal::Julia, julia_re,
                                   julia_im, iterations, width, height);
  }
};

// ---------------------------------------------------------------------------
// WebGPU surface on the SDL window.
// ---------------------------------------------------------------------------

class Swapchain {
public:
  Swapchain(gpu::Context &context, SDL_Window *window)
      : context_(context), window_(window) {
    wgpu::SurfaceDescriptor descriptor{};
#if defined(__APPLE__)
    view_ = SDL_Metal_CreateView(window);
    if (!view_) throw std::runtime_error(SDL_GetError());
    wgpu::SurfaceSourceMetalLayer metal{};
    metal.layer = SDL_Metal_GetLayer(view_);
    descriptor.nextInChain = &metal;
#elif defined(__linux__)
    // Untested: Dawn's Vulkan backend on X11 or Wayland.
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(window, &info)) throw std::runtime_error(SDL_GetError());
#if defined(SDL_VIDEO_DRIVER_X11)
    wgpu::SurfaceSourceXlibWindow xlib{};
    if (info.subsystem == SDL_SYSWM_X11) {
      xlib.display = info.info.x11.display;
      xlib.window = uint64_t(info.info.x11.window);
      descriptor.nextInChain = &xlib;
    }
#endif
#if defined(SDL_VIDEO_DRIVER_WAYLAND)
    wgpu::SurfaceSourceWaylandSurface wayland{};
    if (info.subsystem == SDL_SYSWM_WAYLAND) {
      wayland.display = info.info.wl.display;
      wayland.surface = info.info.wl.surface;
      descriptor.nextInChain = &wayland;
    }
#endif
    if (!descriptor.nextInChain)
      throw std::runtime_error("unsupported SDL video driver (need X11 or Wayland)");
#else
#error "no WebGPU surface source for this platform"
#endif
    surface_ = context.instance.CreateSurface(&descriptor);
    if (!surface_) throw std::runtime_error("could not create WebGPU surface");

    wgpu::SurfaceCapabilities caps;
    surface_.GetCapabilities(context.adapter, &caps);
    if (caps.formatCount == 0) throw std::runtime_error("surface has no formats");
    // Plain 8-bit UNORM, like the old SDL texture: pixels go out unconverted.
    format_ = caps.formats[0];
    for (size_t i = 0; i < caps.formatCount; ++i)
      if (caps.formats[i] == wgpu::TextureFormat::BGRA8Unorm ||
          caps.formats[i] == wgpu::TextureFormat::RGBA8Unorm) {
        format_ = caps.formats[i];
        break;
      }
    // Unlocked frame rate where supported (frames are only drawn on demand).
    present_mode_ = wgpu::PresentMode::Fifo;
    for (wgpu::PresentMode preferred :
         {wgpu::PresentMode::Immediate, wgpu::PresentMode::Mailbox})
      for (size_t i = 0; i < caps.presentModeCount; ++i)
        if (caps.presentModes[i] == preferred &&
            present_mode_ == wgpu::PresentMode::Fifo)
          present_mode_ = preferred;
  }

  ~Swapchain() {
    surface_ = nullptr;
#if defined(__APPLE__)
    if (view_) SDL_Metal_DestroyView(view_);
#endif
  }

  Swapchain(const Swapchain &) = delete;
  Swapchain &operator=(const Swapchain &) = delete;

  wgpu::TextureFormat format() const { return format_; }
  wgpu::PresentMode present_mode() const { return present_mode_; }

  void size_in_pixels(int &width, int &height) const {
#if defined(__APPLE__)
    SDL_Metal_GetDrawableSize(window_, &width, &height);
#else
    SDL_GetWindowSizeInPixels(window_, &width, &height);
#endif
  }

  // The texture to draw this frame into; false if none is available.
  bool acquire(uint32_t width, uint32_t height, wgpu::Texture &texture) {
    if (width != width_ || height != height_) configure(width, height);
    for (int attempt = 0; attempt < 2; ++attempt) {
      wgpu::SurfaceTexture current;
      surface_.GetCurrentTexture(&current);
      switch (current.status) {
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
        texture = current.texture;
        return true;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
        texture = current.texture;
        width_ = 0; // reconfigure next frame
        return true;
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        configure(width, height);
        continue;
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        return false;
      default:
        throw std::runtime_error("could not get the surface texture");
      }
    }
    return false;
  }

  void present() { surface_.Present(); }

private:
  void configure(uint32_t width, uint32_t height) {
    wgpu::SurfaceConfiguration config{};
    config.device = context_.device;
    config.format = format_;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = width;
    config.height = height;
    config.presentMode = present_mode_;
    surface_.Configure(&config);
    width_ = width;
    height_ = height;
  }

  gpu::Context &context_;
  SDL_Window *window_;
#if defined(__APPLE__)
  SDL_MetalView view_ = nullptr;
#endif
  wgpu::Surface surface_;
  wgpu::TextureFormat format_ = wgpu::TextureFormat::Undefined;
  wgpu::PresentMode present_mode_ = wgpu::PresentMode::Fifo;
  uint32_t width_ = 0, height_ = 0;
};

// ---------------------------------------------------------------------------
// SDL window / input / frame loop.
// ---------------------------------------------------------------------------

struct SdlDeleter {
  void operator()(SDL_Window *w) const { SDL_DestroyWindow(w); }
};
template <typename T> using SdlPtr = std::unique_ptr<T, SdlDeleter>;

// One on-screen view: a screen rectangle (output pixels) and its renderer.
struct Pane {
  Pane(gpu::Context &context, Fractal kind) : kind(kind), renderer(context) {}

  Fractal kind;
  bool visible = false;
  SDL_Rect rect{};             // in output pixels
  uint32_t width = 0, height = 0; // render resolution (rect / divisor)
  PaneRenderer renderer;
  perturbation::ReferenceCache reference;
  uint32_t reference_id = 0;   // reference of the last perturbation request
  float color_phase = 0.0f;    // phase this pane renders with (see frame())
  double budget = 0.0;         // pixels it may compute this frame
  double encoded = 0.0;        // pixels in the frame in flight

  bool contains(double x, double y) const {
    return visible && x >= rect.x && x < rect.x + rect.w && y >= rect.y &&
           y < rect.y + rect.h;
  }
};

class App {
public:
  using Clock = std::chrono::steady_clock;

  App() {
    Uint32 flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
#if defined(__APPLE__)
    flags |= SDL_WINDOW_METAL;
#endif
    window_.reset(SDL_CreateWindow("Fractal Explorer", SDL_WINDOWPOS_CENTERED,
                                   SDL_WINDOWPOS_CENTERED, 1600, 800, flags));
    if (!window_) throw std::runtime_error(SDL_GetError());
    swapchain_ = std::make_unique<Swapchain>(context_, window_.get());
    compositor_ = std::make_unique<Compositor>(context_, swapchain_->format());
  }

  // Event driven: sleeps in SDL_WaitEventTimeout while nothing changes, polls while
  // a frame is on the GPU, and never has more than one frame in flight, so
  // input is handled while slow views compute in time slices.
  void run() {
    last_frame_ = last_title_ = Clock::now();
    bool idle = false;
    while (running_) {
      SDL_Event event;
      if (in_flight_) {
        if (SDL_WaitEventTimeout(&event, 1)) handle(event);
      } else if (idle) {
        // Not SDL_WaitEvent: SDL only turns SIGINT/SIGTERM into SDL_QUIT
        // when it pumps events, so wake up a few times a second -- or
        // often while a reference orbit is computed in the background.
        if (SDL_WaitEventTimeout(&event, references_busy() ? 4 : 250)) handle(event);
      }
      while (SDL_PollEvent(&event)) handle(event);

      if (in_flight_) {
        if (!gpu::poll(context_, frame_done_)) continue;
        finish_frame();
      }
      if (!running_) break;
      const bool was_idle = idle;
      idle = !frame();
      if (idle && !was_idle) update_title(true);
    }
    if (in_flight_) gpu::wait(context_, frame_done_);
  }

private:
  Pane &pane(Fractal kind) { return panes_[size_t(kind)]; }
  Pane &mandelbrot_pane() { return pane(Fractal::Mandelbrot); }
  Pane &julia_pane() { return pane(Fractal::Julia); }

  int divisor() const { return state_.half_resolution ? 2 : 1; }

  // Assigns each visible pane its screen rectangle. False when minimised.
  bool layout() {
    int output_w, output_h;
    swapchain_->size_in_pixels(output_w, output_h);
    if (output_w <= 1 || output_h <= 0) return false;
    output_w_ = uint32_t(output_w);
    output_h_ = uint32_t(output_h);

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

  bool references_busy() const {
    for (const Pane &p : panes_)
      if (p.visible && p.reference.busy()) return true;
    return false;
  }

  // Hands the pane's renderer its current view. Perturbation views use the
  // pane's cached reference (perhaps stale, but still correct: it is
  // replaced in the background); while no usable one exists the renderer
  // waits, keeping the previous image on screen.
  void request(Pane &pane, Clock::time_point now) {
    perturbation::View view = state_.view(pane.kind, pane.width, pane.height);
    std::shared_ptr<const perturbation::Reference> reference;
    const bool perturb = state_.uses_perturbation(pane.kind, pane.height);
    if (perturb) {
      reference = pane.reference.get(view);
      if (reference && reference->id != pane.reference_id) {
        // A new reference starts a new image anyway: move the grid anchor to
        // the view centre so pixel offsets from it stay small.
        pane.reference_id = reference->id;
        Camera &cam = state_.camera(pane.kind);
        if (cam.pan_x || cam.pan_y) {
          cam.fold_pan();
          view = state_.view(pane.kind, pane.width, pane.height);
        }
      }
    }
    PaneRenderer::Request r{
        .uniform = make_uniform(view, reference.get(), state_.outline, pane.color_phase),
        .reference = reference,
        .width = pane.width, .height = pane.height,
        .origin_x = view.grid.origin_x, .origin_y = view.grid.origin_y,
        .center_x = view.camera.center_x, .center_y = view.camera.center_y,
        .log2_pixel_size = view.camera.log2_pixel_size(pane.height),
    };
    if (perturb && !reference) pane.renderer.wait_for(r, now);
    else pane.renderer.request(r, now);
  }

  // Plans, encodes and presents one frame if anything changed. Compute for
  // all panes and the composition go into one submission. False when there
  // was nothing to do.
  bool frame() {
    const auto now = Clock::now();
    const double dt =
        std::min(std::chrono::duration<double>(now - last_frame_).count(), 0.1);
    last_frame_ = now;
    if (state_.animate_colors)
      state_.color_phase = std::fmod(state_.color_phase + float(dt * 0.05), 1.0f);

    if (!layout()) return false; // minimised

    std::vector<Pane *> active;
    for (Pane &pane : panes_) {
      if (!pane.visible) continue;
      // A new palette phase is picked up once the pane has caught up, so
      // animation never restarts a slow render that is still refining.
      if (!pane.renderer.busy()) pane.color_phase = state_.color_phase;
      request(pane, now);
      active.push_back(&pane);
    }

    std::vector<PaneRenderer *> renderers;
    for (Pane *p : active) renderers.push_back(&p->renderer);
    const std::vector<double> budgets = share_budget(renderers);
    for (size_t i = 0; i < active.size(); ++i) active[i]->budget = budgets[i];

    wgpu::CommandEncoder encoder = context_.device.CreateCommandEncoder();
    double encoded = 0.0;
    {
      wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
      for (Pane *p : active) {
        p->encoded = p->renderer.encode(pass, p->budget);
        encoded += p->encoded;
      }
      pass.End();
    }

    std::vector<Compositor::Pane> draws;
    for (Pane *p : active) {
      p->renderer.settle(now);
      draws.push_back(draw(*p));
    }
    bool changed = force_redraw_ || output_w_ != drawn_w_ || output_h_ != drawn_h_ ||
                   draws.size() != drawn_.size();
    for (size_t i = 0; !changed && i < draws.size(); ++i)
      changed = !Compositor::same(draws[i], drawn_[i]);
    if (encoded == 0.0 && !changed) return false;

    wgpu::Texture texture;
    const bool presenting = swapchain_->acquire(output_w_, output_h_, texture);
    if (presenting)
      compositor_->encode(encoder, texture.CreateView(), output_w_, output_h_, draws);
    frame_done_ = submit(context_, encoder.Finish());
    if (presenting) swapchain_->present();
    in_flight_ = true;
    submitted_ = Clock::now();

    force_redraw_ = !presenting;
    drawn_ = std::move(draws);
    drawn_w_ = output_w_;
    drawn_h_ = output_h_;
    return true;
  }

  // The frame on the GPU finished: update cost estimates and the title.
  void finish_frame() {
    in_flight_ = false;
    const auto now = Clock::now();
    last_frame_ms_ = std::chrono::duration<double, std::milli>(now - submitted_).count();

    std::vector<PaneRenderer *> renderers;
    std::vector<double> encoded;
    for (Pane &p : panes_) {
      renderers.push_back(&p.renderer);
      encoded.push_back(p.encoded);
    }
    calibrate(renderers, encoded, last_frame_ms_);
    for (Pane &p : panes_) p.encoded = 0.0;

    ++frames_;
    if (std::chrono::duration<double>(now - last_title_).count() >= 0.25)
      update_title(false);
  }

  Compositor::Pane draw(const Pane &pane) {
    Compositor::Pane d;
    d.x = pane.rect.x;
    d.y = pane.rect.y;
    d.w = pane.rect.w;
    d.h = pane.rect.h;
    d.divisor = uint32_t(divisor());
    d.shown = pane.renderer.shown();
    if (state_.split) {
      d.divider_right = pane.kind == Fractal::Mandelbrot;
      d.divider_left = pane.kind == Fractal::Julia;
      if (pane.kind == Fractal::Mandelbrot) julia_marker(d);
    }
    return d;
  }

  // Crosshair on the Mandelbrot pane at the Julia constant c.
  void julia_marker(Compositor::Pane &d) {
    Pane &m = mandelbrot_pane();
    double px, py;
    state_.mandelbrot.to_pixel(state_.julia_re, state_.julia_im, m.width,
                               m.height, px, py);
    const int x = m.rect.x + int(px * divisor());
    const int y = m.rect.y + int(py * divisor());
    if (!m.contains(x, y)) return;
    d.marker = true;
    d.marker_x = x;
    d.marker_y = y;
    d.marker_radius = 14 * int(output_scale() + 0.5);
    if (!state_.follow_mouse) {
      d.marker_rgb[0] = 255;
      d.marker_rgb[1] = 210;
      d.marker_rgb[2] = 0;
    }
  }

  // ---- input --------------------------------------------------------------

  void handle(const SDL_Event &event) {
    switch (event.type) {
    case SDL_QUIT: running_ = false; break;
    case SDL_WINDOWEVENT:
      if (event.window.event == SDL_WINDOWEVENT_EXPOSED) force_redraw_ = true;
      break;
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
    state_.julia_re = BigFloat(kC);
    state_.julia_im = BigFloat(0.0);
    state_.follow_mouse = false;
    state_.outline = true;
    picking_ = false;
    if (!state_.split) state_.single = Fractal::Julia;
    layout();

    // Frame [-beta, beta] with a 10% margin on each side.
    const Pane &j = julia_pane();
    const double aspect = double(j.width) / j.height;
    state_.julia = Camera::at(0.0, 0.0, 2.0 * Camera::kHalfHeight * aspect / (2.4 * beta));

    // Make sure the crosshair at c is on screen in the Mandelbrot pane.
    Pane &m = mandelbrot_pane();
    double px, py;
    state_.mandelbrot.to_pixel(state_.julia_re, state_.julia_im, m.width, m.height, px, py);
    if (px < 0 || px >= m.width || py < 0 || py >= m.height) {
      const double m_aspect = double(m.width) / m.height;
      state_.mandelbrot =
          Camera::at(-1.0, 0.0, 2.0 * Camera::kHalfHeight * m_aspect / 3.8);
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

  // Output pixels per window point (2 on Retina displays).
  double output_scale() const {
    int window_w, window_h, output_w, output_h;
    SDL_GetWindowSize(window_.get(), &window_w, &window_h);
    swapchain_->size_in_pixels(output_w, output_h);
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

  // While drawing: GPU time of the last frame and frames per second; when
  // idle the last frame's time.
  void update_title(bool idle) {
    static constexpr const char *kPrecisionNames[] = {"auto", "f32", "perturbation"};
    auto precision = [&](Fractal kind) -> std::string {
      if (!state_.uses_perturbation(kind, pane(kind).height)) return "f32";
      std::string name = state_.camera(kind).log2_zoom > perturbation::kFloatExpLog2Zoom
                             ? "perturb-fx" : "perturb";
      return pane(kind).reference.busy() ? name + " (ref...)" : name;
    };
    // %g, or a power of ten once the zoom leaves double range.
    auto zoom = [](const Camera &cam) {
      char text[32];
      if (cam.log2_zoom < 1000) std::snprintf(text, sizeof(text), "%.3g", cam.zoom());
      else std::snprintf(text, sizeof(text), "1e%.1f", cam.log2_zoom * std::log10(2.0));
      return std::string(text);
    };
    char buffer[128];
    std::string title;
    auto append = [&](auto... args) {
      std::snprintf(buffer, sizeof(buffer), args...);
      title += buffer;
    };
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - last_title_).count();
    if (idle)
      append("idle (last frame %.1f ms)", last_frame_ms_);
    else
      append("%.1f ms/frame %.0f fps", last_frame_ms_, frames_ / std::max(seconds, 1e-3));
    frames_ = 0;
    last_title_ = now;
    if (mandelbrot_pane().visible)
      append(" | Mandelbrot %sx %s", zoom(state_.mandelbrot).c_str(),
             precision(Fractal::Mandelbrot).c_str());
    if (julia_pane().visible)
      append(" | Julia c = %.6f %+.6fi  %sx %s", state_.julia_re.to_double(),
             state_.julia_im.to_double(), zoom(state_.julia).c_str(),
             precision(Fractal::Julia).c_str());
    if (state_.split) append("%s", state_.follow_mouse ? " (follows mouse)" : " (locked)");

    append(" | %u iter | precision %s%s%s", state_.iterations,
           kPrecisionNames[int(state_.precision)],
           state_.outline ? " | outline" : "",
           state_.half_resolution ? " | half-res" : "");
    for (const Pane &p : panes_)
      if (p.visible && p.renderer.busy()) {
        append("%s", " | refining");
        break;
      }
    SDL_SetWindowTitle(window_.get(), title.c_str());
  }

  SdlPtr<SDL_Window> window_;
  gpu::Context context_ = gpu::createContext();
  std::unique_ptr<Swapchain> swapchain_;
  std::unique_ptr<Compositor> compositor_;

  // Indexed by Fractal.
  std::array<Pane, 2> panes_{Pane(context_, Fractal::Mandelbrot),
                             Pane(context_, Fractal::Julia)};
  Explorer state_;
  bool running_ = true;
  Pane *drag_pane_ = nullptr;
  bool picking_ = false;

  uint32_t output_w_ = 0, output_h_ = 0;
  bool in_flight_ = false;
  gpu::Future frame_done_;
  Clock::time_point submitted_, last_frame_, last_title_;
  double last_frame_ms_ = 0.0;
  int frames_ = 0;
  // What is on screen, to skip frames that would draw the same.
  std::vector<Compositor::Pane> drawn_;
  uint32_t drawn_w_ = 0, drawn_h_ = 0;
  bool force_redraw_ = true;
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
