// Headless benchmark + reference renderer for the fractal shader.
//
//   make bench && ./bench                 timings for every case
//   ./bench --png out/                    also write out/<case>.png
//   ./bench --case deep --reps 20         one case, more repetitions
//   ./bench --mode perturb|direct         force a path (default: as the app)
//   ./bench --no-bla / --fx               perturbation without BLA / always Fx
//   ./bench --ref-offset 300              reference point 300 px off-centre
//   ./bench --view RE,IM,ZOOM,ITER[,FLAGS[,JRE,JIM]]   a custom view ("view";
//                                         FLAGS: 1 Julia, 2 perturbation, 4 outline;
//                                         any number of digits, ZOOM e.g. 1e500)
//
// Each case renders one Retina split pane (1440x1650) through the same
// FractalKernel the app uses (one full-image tile). "compute" is dispatch + wait, "readback" is the
// GPU -> CPU copy, "frame" is the app's path (dispatch + readback, one wait).
// Times are the best of --reps after two warm-up runs, so other GPU work on
// the machine inflates them less. "ref" is the CPU time to build the
// reference orbit and BLA table (perturbation only: once per view, and on a
// background thread in the app).

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

struct Case {
  std::string name;
  std::string center_x, center_y; // decimal, any number of digits
  double zoom_log10;
  uint32_t iterations;
  uint32_t flags; // kFlagJulia, kFlagOutline (kFlagPerturb: force perturbation)
  std::string julia_re = "0", julia_im = "0";
};

// Bench-only flag: render with the direct f32 path even if the app would not.
constexpr uint32_t kDirect = 1u << 31;

constexpr const char *kDeepX = "-0.743643887037158704752191506114774";
constexpr const char *kDeepY = "0.131825904205311970493132056385139";

// Misiurewicz point M(4,1): f^5(0) = f^4(0), solved by Newton in MPFR to
// ~1050 digits. Preperiodic points are on the boundary and the set is
// asymptotically self-similar there, so every zoom level shows the same
// three-armed spiral branching: a check that deep views stay sharp.
constexpr const char *kMisiurewiczX =
    "-0.101096363845622161025785445738622565463805442826253483876931177660780840740470584274821"
    "219810516779033404531908556741193971546144260911882355703907680301615758418143435454482306"
    "810666402941937826741189160781952644899539699533446682339312669900981755715042520653886699"
    "186134433396160061056277538514425960956070356822164327995613303692671476037779728405471268"
    "019807049595643431226130746218967378729105718156798011974862432778852122321403461414763630"
    "557687940308535051413491830290156250699019697103013983053924171441560695971404518403130646"
    "297572937526949235780834190080334599321631047277565102427518415567882860217117494166014994"
    "292168729820163412084468559854036611053169988354920408821721758551579669112773238736658541"
    "913437718460690637952961013445961198963342644412463211482083831589156200335794774948376981"
    "165488573041344217465208750805788439867684451431896769019091975933734307774014681012582243"
    "383655015596620385148128253616662669586629874466285477170335635740309213481784037921124430"
    "72779400152356332499276233708436342563966294999507";
constexpr const char *kMisiurewiczY =
    "0.9562865108091415007710960577299774358098333365105291700343143215005246590657167325269784"
    "107873398072043444724926469284366752406567465722656200815719741551313831228054884443810571"
    "677870203738395055477279309548311953190385170024877917670113105649462054553485859930615004"
    "427397447762045980085973915977603864276670744042253335354323566859404712327457083787857148"
    "248615873869675060108354066122688803113441095405220822267412663824851779098634770481811864"
    "806258738023969258382884081571925537012152280038588430181998328078474272633699174145410614"
    "739462589128789150553234349685929460773865017350529015660291220913396489775264958117114382"
    "643295654496157898274583055178847925814745420284296331306738636568497266813488050852556156"
    "070118798087306796335619488770524635469632864809712414972256317737930927791171028536023786"
    "748249296403835050376368416443718404941248192130019583547845448325690133411045103455350609"
    "521993487005760316821655656717224296679140427394529521728044627178041783201825478255681195"
    "65824835099851958604505404898824697099424962234930";

// Nucleus of the period-402 minibrot within 1e-50 of M(4,1) (ball period
// detection, then Newton): a deep minibrot, which makes every pixel near it
// rebase onto the reference again and again.
constexpr const char *kMinibrotX =
    "-0.101096363845622161025785445738622565463805442826251418502865344403047613157977642582377"
    "853304142482670075994575758364249335187076811898565419451231775903594164911536497612194078"
    "981083275294788214511962260738555456720073516197265952531530384733604079113357783428489188"
    "37261942108601461935671973994243294813433536020975";
constexpr const char *kMinibrotY =
    "0.9562865108091415007710960577299774358098333365105379538185955869202252349544520352804139"
    "894119243446383579464603819384281859988076641987584299374412411807715088991433187074790469"
    "947105688927018693520917931833916474462906581756228123557787018454004151634440565702981071"
    "89114496111923122826911685681587551774065718058776";

// Repelling fixed point beta = (1 + sqrt(1 - 4c)) / 2 of the dragon Julia
// set (c = -0.8 + 0.156i): on the Julia set, which is self-similar there.
constexpr const char *kJuliaBetaX =
    "1.527503118643534610789746402444915337566745947811708160339875197003201";
constexpr const char *kJuliaBetaY =
    "-0.075912178352287867071814194826348046366422194847978022539732593449187";

const Case kCases[] = {
    {"home", "-0.5", "0", 0.0, 512, 0},
    {"home_outline", "-0.5", "0", 0.0, 512, kFlagOutline},
    {"seahorse", "-0.7453", "0.1127", 3.0, 1000, kDirect}, // as the original bench
    {"interior_heavy", "-0.16", "1.035", 1.6020599913279623, 4000, 0}, // 40x
    {"deep_1e9", kDeepX, kDeepY, 9.0, 4000, 0},
    {"deep_1e12", kDeepX, kDeepY, 12.0, 6000, 0},
    {"julia_dragon", "0", "0", 0.0, 512, kFlagJulia, "-0.8", "0.156"},
    {"julia_cantor", "0", "0", -0.33441900898204685, 512, kFlagJulia | kFlagOutline, // 0.463x
     "-2.5", "0"},
    {"deep_1e20", kDeepX, kDeepY, 20.0, 20000, 0},
    {"deep_1e50", kMisiurewiczX, kMisiurewiczY, 50.0, 3000, 0},
    {"deep_1e100", kMisiurewiczX, kMisiurewiczY, 100.0, 5000, kFlagOutline},
    {"minibrot_1e100", kMinibrotX, kMinibrotY, 99.5, 20000, 0},
    {"julia_deep_1e60", kJuliaBetaX, kJuliaBetaY, 60.0, 2000, kFlagJulia, "-0.8", "0.156"},
    {"deep_1e300", kMisiurewiczX, kMisiurewiczY, 300.0, 12000, 0},
    {"deep_1e1000", kMisiurewiczX, kMisiurewiczY, 1000.0, 30000, kFlagOutline},
    // Interior-dominated views: a period-3 minibrot (outside the cardioid /
    // bulb tests) and a Julia set with a large attracting basin.
    {"minibrot", "-1.7685", "0", 1.4771212547196624, 8000, 0},             // 30x
    {"julia_rabbit", "0", "0", -0.09691001300805639, 2000, kFlagJulia, "-0.123", "0.745"}, // 0.8x
    // Stress cases for interior shortcuts: parabolic cusps and bulb
    // junctions, where escaping orbits linger next to a cycle.
    {"cusp", "0.25", "0", 2.4771212547196626, 4000, 0},                    // 300x
    {"junction", "-0.75", "0", 2.3010299956639813, 4000, kFlagOutline},    // 200x
    {"julia_cauliflower", "0", "0", 0.0, 4000, kFlagJulia, "0.25", "0"},
    {"deep_cusp", "-1.75", "0", 7.0, 8000, 0},
};

// log10 of a positive decimal such as "300", "0.463" or "1e5000" (beyond
// double range).
double parse_log10(const std::string &text) {
  const size_t e = text.find_first_of("eE");
  const double mantissa = std::atof(text.substr(0, e).c_str());
  const double exponent = e == std::string::npos ? 0.0 : std::atof(text.c_str() + e + 1);
  return std::log10(mantissa) + exponent;
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
  std::string png_dir, only, mode = "auto";
  int reps = 10;
  bool use_bla = true, force_fx = false;
  double ref_offset = 0.0; // move the reference point by this many pixels
  double zoom_log10 = NAN; // override every case's zoom (10^x)
  const char *center_x = nullptr, *center_y = nullptr; // override every case's centre
  std::vector<Case> cases(std::begin(kCases), std::end(kCases));
  uint32_t width = 1440, height = 1650;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--png" && i + 1 < argc) png_dir = argv[++i];
    else if (arg == "--case" && i + 1 < argc) only = argv[++i];
    else if (arg == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%ux%u", &width, &height);
    else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (arg == "--no-bla") use_bla = false;
    else if (arg == "--fx") force_fx = true;
    else if (arg == "--ref-offset" && i + 1 < argc) ref_offset = std::atof(argv[++i]);
    else if (arg == "--zoom" && i + 1 < argc) zoom_log10 = std::atof(argv[++i]);
    else if (arg == "--center" && i + 2 < argc) center_x = argv[++i], center_y = argv[++i];
    else if (arg == "--view" && i + 1 < argc) {
      std::vector<std::string> parts;
      for (std::string rest = argv[++i];;) {
        const size_t comma = rest.find(',');
        parts.push_back(rest.substr(0, comma));
        if (comma == std::string::npos) break;
        rest = rest.substr(comma + 1);
      }
      if (parts.size() < 4 || parts.size() == 6 || parts.size() > 7) {
        std::fprintf(stderr, "bad --view\n");
        return 2;
      }
      Case view{"view", parts[0], parts[1], parse_log10(parts[2]),
                uint32_t(std::atoi(parts[3].c_str())), 0};
      if (parts.size() > 4) view.flags = uint32_t(std::atoi(parts[4].c_str()));
      if (parts.size() > 6) view.julia_re = parts[5], view.julia_im = parts[6];
      cases = {view};
    }
    else {
      std::fprintf(stderr,
                   "usage: bench [--case NAME] [--reps N] [--size WxH] [--png DIR]\n"
                   "             [--mode auto|direct|perturb] [--no-bla] [--fx]\n"
                   "             [--ref-offset PIXELS] [--zoom LOG10] [--center X Y]\n"
                   "             [--view RE,IM,ZOOM,ITER[,FLAGS[,JRE,JIM]]]\n");
      return 2;
    }
  }

  gpu::Context context = gpu::createContext();
  FractalKernel fractal(context);
  gpu::Tensor framebuffer =
      gpu::createTensor(context, gpu::Shape{size_t(width) * height}, gpu::kf32);
  std::vector<uint32_t> pixels(size_t(width) * height);

  std::printf("%ux%u, best of %d\n", width, height, reps);
  std::printf("%-16s %10s %10s %10s %10s  %s\n", "case", "compute", "readback", "frame",
              "ref", "path");
  for (const Case &c : cases) {
    if (!only.empty() && only != c.name) continue;
    // Parse with enough digits for the case; the camera raises it with zoom.
    const double log2_zoom =
        (std::isnan(zoom_log10) ? c.zoom_log10 : zoom_log10) * std::log2(10.0);
    const long bits = Camera::bits_for(log2_zoom);
    Camera camera{.center_x = BigFloat(center_x ? center_x : c.center_x.c_str(), bits),
                  .center_y = BigFloat(center_y ? center_y : c.center_y.c_str(), bits),
                  .log2_zoom = log2_zoom};
    camera.update_precision();
    const BigFloat julia_re(c.julia_re.c_str(), bits), julia_im(c.julia_im.c_str(), bits);
    const perturbation::View view = perturbation::make_view(
        camera, (c.flags & kFlagJulia) != 0, julia_re, julia_im, c.iterations, width, height);

    const bool perturb =
        mode == "perturb" ||
        (mode == "auto" && !(c.flags & kDirect) &&
         ((c.flags & kFlagPerturb) || perturbation::needed(camera, height)));
    std::shared_ptr<const perturbation::Reference> ref;
    if (perturb) {
      perturbation::ReferenceKey key = perturbation::key_for(view);
      key.floatexp |= force_fx;
      if (ref_offset != 0.0) {
        // Same image expected from any reference: tests rebasing.
        const double step = ref_offset * 2.0 * Camera::kHalfHeight / height;
        key.x.add_scaled(step, -camera.log2_zoom);
        key.y.add_scaled(-0.5 * step, -camera.log2_zoom);
        if (!view.julia)
          key.bla_radius = FloatExp(2.0) * perturbation::view_radius(
                                               view, perturbation::offset_from(view, key));
      }
      ref = perturbation::compute_reference(key);
    }
    const ViewportUniform uniform =
        make_uniform(view, ref.get(), (c.flags & kFlagOutline) != 0, 0.0f, use_bla);
    const FractalKernel::Tile tile{.rect = {0, 0, int32_t(width), int32_t(height)}};
    auto dispatch = [&] {
      auto encoder = context.device.CreateCommandEncoder();
      auto pass = encoder.BeginComputePass();
      fractal.begin(uniform, ref);
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
    char ref_ms[24] = "-";
    std::string path = "f32";
    if (ref) {
      std::snprintf(ref_ms, sizeof(ref_ms), "%.2fms", ref->seconds * 1e3);
      path = ref->key.floatexp ? "perturb fx" : "perturb f32";
      if (!(uniform.flags & kFlagBla)) path += ", no BLA";
      path += ", orbit " + std::to_string(ref->orbit_length) + (ref->escaped ? " (escaped)" : "");
    }
    std::printf("%-16s %8.2fms %8.2fms %8.2fms %10s  %s\n", c.name.c_str(), compute, readback,
                frame, ref_ms, path.c_str());

    if (!png_dir.empty()) {
      const std::string path = png_dir + "/" + c.name + ".png";
      if (!write_png(path, pixels, width, height))
        std::fprintf(stderr, "could not write %s\n", path.c_str());
    }
  }
  return 0;
}
