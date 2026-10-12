#pragma once

// WGSL compute shader for the Mandelbrot / Julia explorer.
//
// gpu.cpp substitutes {{workgroupSize}} when the gpu::WGSL object is built.
// Bindings: 0 the pane image, 1 the view uniform, 2 the tile (dynamic offset):
// which region of the image a dispatch fills and how its pixels sit on the
// view's pixel grid, 3 and 4 the reference orbit and its BLA table
// (perturbation only; see perturbation.hpp).
//
// Tiles and progressive levels: a pixel's colour depends only on the view
// uniform and its grid position g = local pixel + tile.origin, so any tiling
// of the image, in any order, gives the same image, and an image panned by
// whole pixels can be shifted instead of recomputed. Grid pixels form three
// refinement levels: 0 = every 4th pixel in x and y, 1 = the rest of every
// 2nd, 2 = the rest. A dispatch computes the pixels of levels
// [level_lo, level_hi] inside [rect_min, rect_max).
//
// Two ways to iterate a pixel:
//
// * Direct f32: z_next = z^2 + c in single precision. Fastest, but a pixel
//   shrinks to f32 rounding of the coordinates at zoom ~1e5. The pixel's
//   coordinate is the grid anchor (a double-single hi + lo pair) plus its
//   small offset, added with an error-free transform so it is the correctly
//   rounded f32 of the exact sum. That transform only works if the compiler
//   keeps every rounding step, but Metal/Vulkan drivers may compile with
//   fast-math and fold (a + b) - a into b (verified on Apple M1 / Metal).
//   `keep()` XORs with a uniform that is always zero, an integer op the
//   optimiser cannot see through, so each rounded intermediate survives.
//
// * Perturbation: the CPU computes one reference orbit Z_n in arbitrary
//   precision (perturbation.hpp); each pixel only iterates its small offset
//   dz from it, dz_next = 2 Z dz + dz^2 + dc, which keeps its relative
//   precision however deep the zoom. dc is the pixel's offset from the grid
//   anchor plus the anchor's offset from the reference, so it too is a
//   function of the uniform and the grid position. Two refinements:
//   - Rebasing (Zhuoran): when |Z + dz| < |dz| the pixel is closer to the
//     critical point than to the reference, so it restarts on the orbit
//     that begins at z = 0 with dz = z. Also done when the reference orbit
//     runs out. One reference then works for the whole image, no glitches.
//   - BLA: a table of bilinear approximations dz -> A dz + B dc, each valid
//     while |dz| < r, lets a pixel skip 2^k iterations at once.
//   Deltas past ~1e-30 leave f32's exponent range, so deep views use "Fx"
//   numbers: an f32 mantissa (vec2 for complex numbers, max |component| in
//   [1, 2)) with a separate i32 exponent. Nothing in the perturbation code
//   relies on exact rounding, so fast-math compilers (Metal) are harmless.

namespace shaders {

inline constexpr const char *kFractal = R"WGSL(
struct Viewport {
  // Visible extent at zoom = 1, relative to the grid anchor (aspect-corrected).
  x_min: f32,
  x_max: f32,
  y_min: f32,
  y_max: f32,
  zoom: f32,
  iteration_limit: u32,
  width: u32,
  height: u32,
  // Direct path: grid anchor as double-single (hi, lo) pairs, Julia c.
  center_x_hi: f32,
  center_x_lo: f32,
  center_y_hi: f32,
  center_y_lo: f32,
  julia_x: f32,
  julia_y: f32,
  flags: u32,
  color_phase: f32,
  // Perturbation (Fx values: mantissa * 2^exp). 1 / zoom:
  inv_zoom: f32,
  inv_zoom_exp: i32,
  // Grid anchor minus the reference point (c, or z0 for Julia).
  ref_offset_x: f32,
  ref_offset_y: f32,
  ref_offset_exp: i32,
  // Julia: view's c minus the reference orbit's c (0 unless stale).
  julia_dc_x: f32,
  julia_dc_y: f32,
  julia_dc_exp: i32,
  // The primary orbit starts at index 0 of `orbit` and `bla`; rebasing
  // continues on the orbit from z = 0 (the same one for Mandelbrot).
  orbit_length: u32,
  rebase_base: u32,
  rebase_length: u32,
  rebase_bla_base: u32,
  reference_id: u32,  // host-side cache key only
  zero_bits: u32,     // always 0; see keep()
  _pad0: u32,
  _pad1: u32,
};

struct Tile {
  origin: vec2<i32>,    // grid position of image pixel (0, 0)
  rect_min: vec2<i32>,  // region, in image pixels
  rect_max: vec2<i32>,
  level_lo: u32,
  level_hi: u32,
};

// Reference orbit point: plain f32 (f32 mode, e = 0) or Fx. bla_r repeats
// the radius of the level-1 BLA entry starting here (log2 of it in Fx mode;
// 0 / -1e30 where there is none), so most failed lookups cost no extra load.
struct OrbitPoint {
  z: vec2<f32>,
  e: i32,
  bla_r: f32,
};

// dz -> a dz + b dc over 2^k iterations, valid while |dz|_1 < r. Exponents
// are 0 in f32 mode. Level k >= 1 of a segment holds ceil(steps / 2^k)
// entries; entry j starts at orbit index 1 + j 2^k.
struct Bla {
  a: vec2<f32>,
  b: vec2<f32>,
  a_exp: i32,
  b_exp: i32,
  r: f32,
  r_exp: i32,
};

@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<uniform> view: Viewport;
@group(0) @binding(2) var<uniform> tile: Tile;
@group(0) @binding(3) var<storage, read> orbit: array<OrbitPoint>;
@group(0) @binding(4) var<storage, read> bla: array<Bla>;

const FLAG_JULIA: u32 = 1u;
const FLAG_PERTURB: u32 = 2u;
const FLAG_OUTLINE: u32 = 4u;
const FLAG_FLOATEXP: u32 = 8u;
const FLAG_BLA: u32 = 16u;
const FLAG_INTERIOR: u32 = 32u;
// Large bailout radius (|z| > 16) keeps smooth colouring band-free.
const BAILOUT_SQ: f32 = 256.0;
const TAU: f32 = 6.28318530718;

// ---------------------------------------------------------------------------
// Double-single addition (direct path's pixel coordinate): vec2(hi, lo) with
// |lo| <= ulp(hi) / 2.
// ---------------------------------------------------------------------------

fn keep(x: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(x) ^ view.zero_bits);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = keep(a + b);
  let v = keep(s - a);
  return vec2<f32>(s, keep(a - keep(s - v)) + keep(b - v));
}

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = keep(a + b);
  return vec2<f32>(s, b - keep(s - a));
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  var s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  s.y += t.x;
  s = quick_two_sum(s.x, s.y);
  s.y += t.y;
  return quick_two_sum(s.x, s.y);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn l1(v: vec2<f32>) -> f32 {
  return abs(v.x) + abs(v.y);
}

// ---------------------------------------------------------------------------
// Fx: complex f32 mantissa with a shared i32 exponent. Zero has m = 0 and
// e = FX_ZERO, low enough that it never wins an exponent alignment.
// ---------------------------------------------------------------------------

struct Fx {
  m: vec2<f32>,
  e: i32,
};

const FX_ZERO: i32 = -16777216;

// 2^k for k in [-126, 127] (clamped), built from the exponent bits.
fn pow2(k: i32) -> f32 {
  return bitcast<f32>(u32(clamp(k, -126, 127) + 127) << 23u);
}

// Renormalise so the larger component is in [1, 2).
fn fx(m: vec2<f32>, e: i32) -> Fx {
  let a = max(abs(m.x), abs(m.y));
  let k = i32(bitcast<u32>(a) >> 23u) - 127;
  return Fx(m * pow2(-k), select(e + k, FX_ZERO, a == 0.0));
}

fn fx_add(a: Fx, b: Fx) -> Fx {
  let d = a.e - b.e;
  let a_big = d >= 0;
  // Shifts beyond 2^-126 only lose bits below f32 rounding.
  let small = select(a.m, b.m, a_big) * pow2(-abs(d));
  return fx(select(b.m, a.m, a_big) + small, max(a.e, b.e));
}

fn fx_mul(a: Fx, b: Fx) -> Fx {
  return fx(cmul(a.m, b.m), a.e + b.e);
}

fn fx_to_f32(a: Fx) -> vec2<f32> {
  return select(a.m * pow2(a.e), vec2<f32>(0.0), a.e < -126);
}

// |a|_1 < r * 2^r_exp for a normalised a.
fn fx_l1_less(a: Fx, r: f32, r_exp: i32) -> bool {
  return l1(a.m) * pow2(a.e - r_exp) < r;
}

// ---------------------------------------------------------------------------
// Escape-time iteration, z_next = z^2 + c. Returns (iterations, |z|^2,
// |dz|^2): with FLAG_OUTLINE the derivative dz is tracked alongside z
// (Julia: dz_next = 2 z dz, dz_0 = 1; Mandelbrot: dz_next = 2 z dz + 1,
// dz_0 = 0) for the distance estimate.
// ---------------------------------------------------------------------------

fn derivative_step(z: vec2<f32>, dz: vec2<f32>, dc: f32) -> vec2<f32> {
  return 2.0 * vec2<f32>(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) +
         vec2<f32>(dc, 0.0);
}

fn escape_single(z0: vec2<f32>, c: vec2<f32>, dz0: vec2<f32>, dc: f32) -> vec3<f32> {
  let outline = (view.flags & FLAG_OUTLINE) != 0u;
  var z = z0;
  var dz = dz0;
  var r2 = dot(z, z);
  var i = 0u;
  while (i < view.iteration_limit && r2 <= BAILOUT_SQ) {
    if (outline) {
      dz = derivative_step(z, dz, dc);
    }
    z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    r2 = dot(z, z);
    i += 1u;
  }
  return vec3<f32>(f32(i), r2, dot(dz, dz));
}

// ---------------------------------------------------------------------------
// Perturbation. The pixel's z = Z_n + dz_n, n indexing the current orbit
// segment. The derivative is tracked pre-scaled by the pixel size (d =
// dz/dc * pixel, or dz/dz0 * pixel for Julia) so it stays in range; it
// follows the full z, and through a BLA skip d -> a d + b d_step. Returns
// (iterations, |z|^2, |d|^2), like escape_single in pixel units.
// ---------------------------------------------------------------------------

// Longest valid BLA skip from orbit index n of a segment with `length` points
// whose table starts at `base`; levels are tried upwards while valid, since
// a merged entry's radius never exceeds its first half's. |dz|_1 =
// dz_l1 * 2^dz_exp. Returns (entry + 1, iterations), or (0, 0) for none.
fn bla_lookup(base: u32, length: u32, n: u32, dz_l1: f32, dz_exp: i32) -> vec2<u32> {
  var found = vec2<u32>(0u);
  let steps = length - 2u; // steps n -> n + 1 for n = 1 .. length - 2
  let s = n - 1u;
  if (n == 0u || s >= steps || steps < 2u) {
    return found;
  }
  var offset = base;
  var k = 1u;
  loop {
    let span = 1u << k;
    if ((s & (span - 1u)) != 0u) {
      break;
    }
    let index = offset + (s >> k);
    let entry = bla[index];
    if (!(dz_l1 * pow2(dz_exp - entry.r_exp) < entry.r)) {
      break;
    }
    found = vec2<u32>(index + 1u, min(span, steps - s));
    let count = (steps + span - 1u) >> k;
    if (count == 1u) {
      break;
    }
    offset += count;
    k += 1u;
  }
  return found;
}

fn perturb_f32(dz0: vec2<f32>, dc: vec2<f32>, d0: vec2<f32>, d_step: f32) -> vec3<f32> {
  let outline = (view.flags & FLAG_OUTLINE) != 0u;
  let use_bla = (view.flags & FLAG_BLA) != 0u;
  var base = 0u;
  var length = view.orbit_length;
  var bla_base = 0u;
  var n = 0u;
  var dz = dz0;
  var d = d0;
  var p = orbit[0];
  var big_z = p.z;
  var z = big_z + dz;
  var r2 = dot(z, z);
  var i = 0u;
  while (i < view.iteration_limit && r2 <= BAILOUT_SQ) {
    // One step is dz -> (2 Z + dz) dz + dc, a BLA skip a dz + b dc: share
    // the arithmetic so SIMD lanes that diverge here only split the lookup.
    // The next point is fetched first so its latency overlaps the maths
    // (n + 1 < length always holds here; see the rebase test below).
    let next = orbit[base + n + 1u];
    var a = 2.0 * big_z;
    var bdc = dc;
    var b_step = vec2<f32>(d_step, 0.0);
    var quadratic = 1.0;
    var steps = 1u;
    if (use_bla && l1(dz) < p.bla_r) {
      let skip = bla_lookup(bla_base, length, n, l1(dz), 0);
      if (skip.x != 0u) {
        let e = bla[skip.x - 1u];
        a = e.a;
        bdc = cmul(e.b, dc);
        b_step = e.b * d_step;
        quadratic = 0.0;
        steps = skip.y;
      }
    }
    // Derivative from the full z: 2 z = 2 Z + 2 dz for a step.
    if (outline) {
      d = cmul(a + 2.0 * quadratic * dz, d) + b_step;
    }
    dz = cmul(a + quadratic * dz, dz) + bdc;
    n += steps;
    i += steps;
    p = next;
    if (steps != 1u) {
      p = orbit[base + n];
    }
    big_z = p.z;
    z = big_z + dz;
    r2 = dot(z, z);
    if (l1(z) < l1(dz) || n + 1u >= length) {
      // Rebase onto the orbit from z = 0 (whose first point is 0).
      dz = z;
      n = 0u;
      base = view.rebase_base;
      length = view.rebase_length;
      bla_base = view.rebase_bla_base;
      p = OrbitPoint(vec2<f32>(0.0), 0, 0.0);
      big_z = p.z;
    }
  }
  return vec3<f32>(f32(min(i, view.iteration_limit)), r2, dot(d, d));
}

// Same iteration in Fx arithmetic, for deltas below f32 range.
fn perturb_fx(dz0: Fx, dc: Fx, d0: Fx, d_step: Fx) -> vec3<f32> {
  let outline = (view.flags & FLAG_OUTLINE) != 0u;
  let use_bla = (view.flags & FLAG_BLA) != 0u;
  var base = 0u;
  var length = view.orbit_length;
  var bla_base = 0u;
  var n = 0u;
  var dz = dz0;
  var d = d0;
  var p = orbit[0];
  var big_z = Fx(p.z, p.e);
  var z = fx_add(big_z, dz);
  var zf = fx_to_f32(z);
  var r2 = dot(zf, zf);
  var i = 0u;
  while (i < view.iteration_limit && r2 <= BAILOUT_SQ) {
    // As in perturb_f32: a step and a BLA skip share the arithmetic.
    var a = Fx(big_z.m, big_z.e + 1); // 2 Z
    var bdc = dc;
    var b_step = d_step;
    var quadratic = dz;
    var steps = 1u;
    if (use_bla && f32(dz.e) + log2(l1(dz.m)) < p.bla_r) {
      let skip = bla_lookup(bla_base, length, n, l1(dz.m), dz.e);
      if (skip.x != 0u) {
        let e = bla[skip.x - 1u];
        a = Fx(e.a, e.a_exp);
        let b = Fx(e.b, e.b_exp);
        bdc = fx_mul(b, dc);
        b_step = fx_mul(b, d_step);
        quadratic = Fx(vec2<f32>(0.0), FX_ZERO);
        steps = skip.y;
      }
    }
    // Derivative from the full z: 2 z = 2 Z + 2 dz for a step.
    if (outline) {
      d = fx_add(fx_mul(fx_add(a, Fx(quadratic.m, quadratic.e + 1)), d), b_step);
    }
    dz = fx_add(fx_mul(fx_add(a, quadratic), dz), bdc);
    n += steps;
    i += steps;
    p = orbit[base + n];
    big_z = Fx(p.z, p.e);
    z = fx_add(big_z, dz);
    zf = fx_to_f32(z);
    r2 = dot(zf, zf);
    if (fx_l1_less(z, l1(dz.m), dz.e) || n + 1u >= length) {
      dz = z;
      n = 0u;
      base = view.rebase_base;
      length = view.rebase_length;
      bla_base = view.rebase_bla_base;
      p = OrbitPoint(vec2<f32>(0.0), FX_ZERO, -1e30);
      big_z = Fx(p.z, p.e);
    }
  }
  let df = fx_to_f32(d);
  return vec3<f32>(f32(min(i, view.iteration_limit)), r2, dot(df, df));
}

// Main cardioid and period-2 bulb never escape: skip straight to the limit.
fn in_mandelbrot_interior(c: vec2<f32>) -> bool {
  let x = c.x - 0.25;
  let q = x * x + c.y * c.y;
  let bulb = (c.x + 1.0) * (c.x + 1.0) + c.y * c.y;
  return q * (q + x) <= 0.25 * c.y * c.y || bulb <= 0.0625;
}

fn shade(result: vec3<f32>) -> vec3<f32> {
  if (u32(result.x) >= view.iteration_limit) {
    return vec3<f32>(0.0);
  }
  // Normalised iteration count: continuous across escape bands.
  let mu = max(result.x + 1.0 - log2(0.5 * log(result.y)), 0.0);
  let t = sqrt(mu) * 0.12 + view.color_phase;
  let glow = 0.5 + 0.5 * cos(TAU * (t + vec3<f32>(0.0, 0.10, 0.20)));
  if ((view.flags & FLAG_OUTLINE) == 0u) {
    return glow;
  }
  // Distance estimate |z| ln|z| / |dz| to the set itself, in pixels. Points
  // of the set (including Cantor dust, which has no interior) light up.
  let distance = 0.5 * sqrt(result.y / result.z) * log(result.y);
  // Perturbation tracks the derivative in pixel units already.
  var pixel = 1.0;
  if ((view.flags & FLAG_PERTURB) == 0u) {
    pixel = (view.y_max - view.y_min) / (f32(view.height) * view.zoom);
  }
  // ~2-3 px wide so sub-pixel dust stays visible on high-DPI screens.
  let on_set = 1.0 - smoothstep(1.0, 3.0, distance / pixel);
  return mix(glow * 0.25, vec3<f32>(1.0), on_set);
}

// Perturbation for one pixel at zoom-1 plane offset `plane` from the grid
// anchor.
fn perturb_pixel(plane: vec2<f32>, julia: bool) -> vec3<f32> {
  let pixel_plane = (view.y_max - view.y_min) / f32(view.height);
  if ((view.flags & FLAG_FLOATEXP) != 0u) {
    let offset = fx(vec2<f32>(view.ref_offset_x, view.ref_offset_y), view.ref_offset_exp);
    let delta = fx_add(fx(plane * view.inv_zoom, view.inv_zoom_exp), offset);
    let pixel = fx(vec2<f32>(pixel_plane * view.inv_zoom, 0.0), view.inv_zoom_exp);
    let zero = Fx(vec2<f32>(0.0), FX_ZERO);
    if (julia) {
      let dc = fx(vec2<f32>(view.julia_dc_x, view.julia_dc_y), view.julia_dc_exp);
      return perturb_fx(delta, dc, pixel, zero);
    }
    return perturb_fx(zero, delta, zero, pixel);
  }
  let inv_zoom = fx_to_f32(Fx(vec2<f32>(view.inv_zoom, 0.0), view.inv_zoom_exp)).x;
  let delta = plane * inv_zoom +
              fx_to_f32(Fx(vec2<f32>(view.ref_offset_x, view.ref_offset_y), view.ref_offset_exp));
  let pixel = pixel_plane * inv_zoom;
  if (julia) {
    let dc = fx_to_f32(Fx(vec2<f32>(view.julia_dc_x, view.julia_dc_y), view.julia_dc_exp));
    return perturb_f32(delta, dc, vec2<f32>(pixel, 0.0), 0.0);
  }
  return perturb_f32(vec2<f32>(0.0), delta, vec2<f32>(0.0), pixel);
}

fn grid_level(g: vec2<i32>) -> u32 {
  if ((g.x & 3) == 0 && (g.y & 3) == 0) {
    return 0u;
  }
  if ((g.x & 1) == 0 && (g.y & 1) == 0) {
    return 1u;
  }
  return 2u;
}

// Invocation -> image pixel of the tile, or (-1, -1) when it has none. Must
// match tile_threads() on the host. A single level >= 1 is three of every
// 2s x 2s block (s = its stride), packed three invocations per block so no
// invocation idles; otherwise invocations step over the stride-s lattice.
fn tile_pixel(id: vec2<u32>) -> vec2<i32> {
  let stride = 4 >> tile.level_hi;
  let packed = tile.level_lo == tile.level_hi && tile.level_lo > 0u;
  let step = select(stride, 2 * stride, packed);
  // First lattice point at or before rect_min, on the grid.
  let g_min = tile.rect_min + tile.origin;
  let base = g_min - (g_min & vec2<i32>(step - 1)) - tile.origin;
  var p: vec2<i32>;
  if (packed) {
    let corner = base + vec2<i32>(i32(id.x / 3u), i32(id.y)) * step;
    let sub = id.x % 3u;
    p = corner + vec2<i32>(select(stride, 0, sub == 1u), select(0, stride, sub != 0u));
  } else {
    p = base + vec2<i32>(id.xy) * stride;
  }
  if (any(p < tile.rect_min) || any(p >= tile.rect_max)) {
    return vec2<i32>(-1);
  }
  let level = grid_level(p + tile.origin);
  if (level < tile.level_lo || level > tile.level_hi) {
    return vec2<i32>(-1);
  }
  return p;
}

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local = tile_pixel(gid.xy);
  if (local.x < 0) {
    return;
  }

  // Pixel centre -> zoom-1 plane offset (row 0 is the top edge, y_max).
  // Grid positions are small integers, exact in f32.
  let g = vec2<f32>(local + tile.origin);
  let uv = (g + 0.5) / vec2<f32>(f32(view.width), f32(view.height));
  let plane = vec2<f32>(mix(view.x_min, view.x_max, uv.x),
                        mix(view.y_max, view.y_min, uv.y));

  // Direct path: the small per-pixel delta is added to the double-single
  // anchor, giving the correctly rounded f32 coordinate.
  let offset = plane / view.zoom;
  let px = ds_add(vec2<f32>(view.center_x_hi, view.center_x_lo), vec2<f32>(offset.x, 0.0));
  let py = ds_add(vec2<f32>(view.center_y_hi, view.center_y_lo), vec2<f32>(offset.y, 0.0));
  let p = vec2<f32>(px.x, py.x);
  let julia = (view.flags & FLAG_JULIA) != 0u;

  var result: vec3<f32>;
  if ((view.flags & FLAG_PERTURB) != 0u) {
    if (!julia && (view.flags & FLAG_INTERIOR) != 0u && in_mandelbrot_interior(p)) {
      result = vec3<f32>(f32(view.iteration_limit), 0.0, 0.0);
    } else {
      result = perturb_pixel(plane, julia);
    }
  } else if (!julia && in_mandelbrot_interior(p)) {
    result = vec3<f32>(f32(view.iteration_limit), 0.0, 0.0);
  } else {
    // Derivative seeds: d/dz for Julia, d/dc for Mandelbrot.
    let dz0 = select(vec2<f32>(0.0), vec2<f32>(1.0, 0.0), julia);
    let dc = select(1.0, 0.0, julia);
    if (julia) {
      result = escape_single(p, vec2<f32>(view.julia_x, view.julia_y), dz0, dc);
    } else {
      result = escape_single(vec2<f32>(0.0), p, dz0, dc);
    }
  }

  pixels[u32(local.y) * view.width + u32(local.x)] =
      pack4x8unorm(vec4<f32>(shade(result), 1.0));
}
)WGSL";

} // namespace shaders
