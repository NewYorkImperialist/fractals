#pragma once

// WGSL compute shader for the Mandelbrot / Julia explorer.
//
// gpu.cpp substitutes {{workgroupSize}} when the gpu::WGSL object is built.
// Bindings: 0 the pane image, 1 the view uniform, 2 the tile (dynamic offset):
// which region of the image a dispatch fills and how its pixels sit on the
// view's pixel grid.
//
// Tiles and progressive levels: a pixel's colour depends only on the view
// uniform and its grid position g = local pixel + tile.origin, so any tiling
// of the image, in any order, gives the same image, and an image panned by
// whole pixels can be shifted instead of recomputed. Grid pixels form three
// refinement levels: 0 = every 4th pixel in x and y, 1 = the rest of every
// 2nd, 2 = the rest. A dispatch computes the pixels of levels
// [level_lo, level_hi] inside [rect_min, rect_max).
//
// Precision: WGSL has no f64, so deep zooms use "double-single" arithmetic —
// each real number is an unevaluated sum hi + lo of two f32s, giving ~48 bits
// of mantissa (zoom ~1e13 before pixelation vs ~1e5 for plain f32). The
// error-free transforms below only work if the compiler keeps every rounding
// step, but Metal/Vulkan drivers may compile with fast-math and fold
// (a + b) - a into b (verified on Apple M1 / Metal: the two_sum error term
// comes out exactly 0). `keep()` XORs with a uniform that is always zero, an
// integer op the optimiser cannot see through, so each rounded intermediate
// of the cancellation chain survives as written.
// two_prod uses fma() so it is exact whether or not multiply-adds are fused.

namespace shaders {

inline constexpr const char *kFractal = R"WGSL(
struct Viewport {
  // Visible extent at zoom = 1, relative to the offset (aspect-corrected).
  x_min: f32,
  x_max: f32,
  y_min: f32,
  y_max: f32,
  zoom: f32,
  iteration_limit: u32,
  width: u32,
  height: u32,
  // Viewport centre as double-single (hi, lo) pairs.
  offset_x_hi: f32,
  offset_x_lo: f32,
  offset_y_hi: f32,
  offset_y_lo: f32,
  // Julia constant c, also double-single.
  julia_x_hi: f32,
  julia_x_lo: f32,
  julia_y_hi: f32,
  julia_y_lo: f32,
  flags: u32,
  color_phase: f32,
  zero_bits: u32,  // always 0; see keep()
  _pad: u32,
};

struct Tile {
  origin: vec2<i32>,    // grid position of image pixel (0, 0)
  rect_min: vec2<i32>,  // region, in image pixels
  rect_max: vec2<i32>,
  level_lo: u32,
  level_hi: u32,
};

@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<uniform> view: Viewport;
@group(0) @binding(2) var<uniform> tile: Tile;

const FLAG_JULIA: u32 = 1u;
const FLAG_DOUBLE: u32 = 2u;
const FLAG_OUTLINE: u32 = 4u;
// Large bailout radius (|z| > 16) keeps smooth colouring band-free.
const BAILOUT_SQ: f32 = 256.0;
const TAU: f32 = 6.28318530718;

// ---------------------------------------------------------------------------
// Double-single arithmetic: vec2(hi, lo) with |lo| <= ulp(hi) / 2.
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

fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = keep(a * b);
  return vec2<f32>(p, fma(a, b, -p));
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  var s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  s.y += t.x;
  s = quick_two_sum(s.x, s.y);
  s.y += t.y;
  return quick_two_sum(s.x, s.y);
}

fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  var p = two_prod(a.x, b.x);
  p.y += a.x * b.y + a.y * b.x;
  return quick_two_sum(p.x, p.y);
}

fn ds_sqr(a: vec2<f32>) -> vec2<f32> {
  var p = two_prod(a.x, a.x);
  p.y += 2.0 * a.x * a.y;
  return quick_two_sum(p.x, p.y);
}

// ---------------------------------------------------------------------------
// Escape-time iteration, z_next = z^2 + c. Returns (iterations, |z|^2,
// |dz|^2): with FLAG_OUTLINE the derivative dz is tracked alongside z
// (Julia: dz_next = 2 z dz, dz_0 = 1; Mandelbrot: dz_next = 2 z dz + 1,
// dz_0 = 0) for the distance estimate. Only its magnitude matters, so f32 is
// enough even on the double-single path.
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

fn escape_double(z0x: vec2<f32>, z0y: vec2<f32>, cx: vec2<f32>, cy: vec2<f32>,
                 dz0: vec2<f32>, dc: f32) -> vec3<f32> {
  let outline = (view.flags & FLAG_OUTLINE) != 0u;
  var zx = z0x;
  var zy = z0y;
  var dz = dz0;
  var x2 = ds_sqr(zx);
  var y2 = ds_sqr(zy);
  var i = 0u;
  while (i < view.iteration_limit && x2.x + y2.x <= BAILOUT_SQ) {
    if (outline) {
      dz = derivative_step(vec2<f32>(zx.x, zy.x), dz, dc);
    }
    let xy = ds_mul(zx, zy);
    zy = ds_add(2.0 * xy, cy);  // scaling by 2 is exact
    zx = ds_add(ds_add(x2, -y2), cx);
    x2 = ds_sqr(zx);
    y2 = ds_sqr(zy);
    i += 1u;
  }
  return vec3<f32>(f32(i), x2.x + y2.x, dot(dz, dz));
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
  let pixel = (view.y_max - view.y_min) / (f32(view.height) * view.zoom);
  // ~2-3 px wide so sub-pixel dust stays visible on high-DPI screens.
  let on_set = 1.0 - smoothstep(1.0, 3.0, distance / pixel);
  return mix(glow * 0.25, vec3<f32>(1.0), on_set);
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
                        mix(view.y_max, view.y_min, uv.y)) / view.zoom;

  // The small per-pixel delta is added to the high-precision offset, so the
  // pixel coordinate keeps double-single precision even at extreme zoom.
  let px = ds_add(vec2<f32>(view.offset_x_hi, view.offset_x_lo), vec2<f32>(plane.x, 0.0));
  let py = ds_add(vec2<f32>(view.offset_y_hi, view.offset_y_lo), vec2<f32>(plane.y, 0.0));
  let jx = vec2<f32>(view.julia_x_hi, view.julia_x_lo);
  let jy = vec2<f32>(view.julia_y_hi, view.julia_y_lo);
  let julia = (view.flags & FLAG_JULIA) != 0u;
  // Derivative seeds: d/dz for Julia, d/dc for Mandelbrot.
  let dz0 = select(vec2<f32>(0.0), vec2<f32>(1.0, 0.0), julia);
  let dc = select(1.0, 0.0, julia);

  var result: vec3<f32>;
  if (!julia && in_mandelbrot_interior(vec2<f32>(px.x, py.x))) {
    result = vec3<f32>(f32(view.iteration_limit), 0.0, 0.0);
  } else if ((view.flags & FLAG_DOUBLE) != 0u) {
    if (julia) {
      result = escape_double(px, py, jx, jy, dz0, dc);
    } else {
      result = escape_double(vec2<f32>(0.0), vec2<f32>(0.0), px, py, dz0, dc);
    }
  } else {
    let p = vec2<f32>(px.x, py.x);
    if (julia) {
      result = escape_single(p, vec2<f32>(jx.x, jy.x), dz0, dc);
    } else {
      result = escape_single(vec2<f32>(0.0), p, dz0, dc);
    }
  }

  pixels[u32(local.y) * view.width + u32(local.x)] = pack4x8unorm(vec4<f32>(shade(result), 1.0));
}
)WGSL";

} // namespace shaders
