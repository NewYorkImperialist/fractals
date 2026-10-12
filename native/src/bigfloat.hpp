#pragma once

#include <mpfr.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// FloatExp: a double mantissa with a separate 64-bit exponent, so magnitudes
// far outside double range (deep-zoom pixel sizes, BLA coefficients) keep
// full double precision. value = m * 2^e with m == 0 or 0.5 <= |m| < 1.
// ---------------------------------------------------------------------------

struct FloatExp {
  static constexpr int64_t kZeroExp = INT64_MIN / 4; // exponent of 0

  double m = 0.0;
  int64_t e = kZeroExp;

  FloatExp() = default;
  FloatExp(double value) { // NOLINT: implicit, like a numeric type
    int exponent = 0;
    m = std::frexp(value, &exponent);
    e = m == 0.0 ? kZeroExp : exponent;
  }
  static FloatExp make(double mantissa, int64_t exponent) {
    int shift = 0;
    FloatExp r;
    r.m = std::frexp(mantissa, &shift);
    r.e = r.m == 0.0 ? kZeroExp : exponent + shift;
    return r;
  }

  bool is_zero() const { return m == 0.0; }
  // Saturates to 0 / inf outside double range.
  double to_double() const {
    if (is_zero()) return 0.0;
    return std::ldexp(m, int(std::clamp<int64_t>(e, -2000, 2000)));
  }
  double log2() const { return is_zero() ? -INFINITY : std::log2(std::abs(m)) + double(e); }

  FloatExp operator-() const { FloatExp r = *this; r.m = -r.m; return r; }
  friend FloatExp abs(FloatExp a) { a.m = std::abs(a.m); return a; }

  friend FloatExp operator*(const FloatExp &a, const FloatExp &b) {
    return make(a.m * b.m, a.e + b.e);
  }
  friend FloatExp operator/(const FloatExp &a, const FloatExp &b) {
    return make(a.m / b.m, a.e - b.e);
  }
  friend FloatExp operator+(const FloatExp &a, const FloatExp &b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;
    // Align to the larger exponent; a shift past 64 bits is below rounding.
    if (a.e >= b.e) {
      const int64_t d = a.e - b.e;
      return make(a.m + (d > 64 ? 0.0 : std::ldexp(b.m, -int(d))), a.e);
    }
    const int64_t d = b.e - a.e;
    return make(b.m + (d > 64 ? 0.0 : std::ldexp(a.m, -int(d))), b.e);
  }
  friend FloatExp operator-(const FloatExp &a, const FloatExp &b) { return a + -b; }
  FloatExp &operator+=(const FloatExp &b) { return *this = *this + b; }
  FloatExp &operator*=(const FloatExp &b) { return *this = *this * b; }

  friend bool operator<(const FloatExp &a, const FloatExp &b) {
    if ((a.m < 0) != (b.m < 0) || a.is_zero() || b.is_zero()) return a.m < b.m;
    // Same sign, both non-zero: compare exponents, then mantissas.
    if (a.e != b.e) return (a.e < b.e) == (a.m > 0);
    return a.m < b.m;
  }
  friend bool operator>(const FloatExp &a, const FloatExp &b) { return b < a; }
  friend FloatExp max(const FloatExp &a, const FloatExp &b) { return a < b ? b : a; }
  friend FloatExp min(const FloatExp &a, const FloatExp &b) { return a < b ? a : b; }

  friend FloatExp sqrt(const FloatExp &a) {
    if (a.is_zero()) return a;
    // Make the exponent even so it halves exactly.
    const bool odd = (a.e & 1) != 0;
    return make(std::sqrt(odd ? a.m * 2.0 : a.m), (odd ? a.e - 1 : a.e) / 2);
  }
};

// Complex number with FloatExp parts.
struct ComplexExp {
  FloatExp re, im;

  friend ComplexExp operator+(const ComplexExp &a, const ComplexExp &b) {
    return {a.re + b.re, a.im + b.im};
  }
  friend ComplexExp operator*(const ComplexExp &a, const ComplexExp &b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
  }
  friend ComplexExp operator*(const FloatExp &s, const ComplexExp &a) {
    return {s * a.re, s * a.im};
  }
  FloatExp norm() const { return re * re + im * im; }
  FloatExp abs() const { return sqrt(norm()); }
};

// ---------------------------------------------------------------------------
// BigFloat: value-semantics wrapper around an MPFR number. Precision is per
// value (bits of mantissa); results take the precision of the destination.
// ---------------------------------------------------------------------------

class BigFloat {
public:
  BigFloat(double value = 0.0, long bits = 64) { // NOLINT: implicit
    mpfr_init2(x_, bits);
    mpfr_set_d(x_, value, MPFR_RNDN);
  }
  // Decimal string, e.g. "-0.74364388703715870475219150611477"; the value is
  // rounded to `bits`.
  BigFloat(const char *decimal, long bits) {
    mpfr_init2(x_, bits);
    mpfr_set_str(x_, decimal, 10, MPFR_RNDN);
  }
  BigFloat(const BigFloat &other) {
    mpfr_init2(x_, mpfr_get_prec(other.x_));
    mpfr_set(x_, other.x_, MPFR_RNDN);
  }
  BigFloat(BigFloat &&other) noexcept {
    mpfr_init2(x_, MPFR_PREC_MIN);
    mpfr_swap(x_, other.x_);
  }
  BigFloat &operator=(const BigFloat &other) {
    if (this != &other) {
      mpfr_set_prec(x_, mpfr_get_prec(other.x_));
      mpfr_set(x_, other.x_, MPFR_RNDN);
    }
    return *this;
  }
  BigFloat &operator=(BigFloat &&other) noexcept {
    mpfr_swap(x_, other.x_);
    return *this;
  }
  BigFloat &operator=(double value) {
    mpfr_set_d(x_, value, MPFR_RNDN);
    return *this;
  }
  ~BigFloat() { mpfr_clear(x_); }

  mpfr_ptr get() { return x_; }
  mpfr_srcptr get() const { return x_; }

  long precision() const { return long(mpfr_get_prec(x_)); }
  // Raises (never lowers) the precision, keeping the value.
  void ensure_precision(long bits) {
    if (bits > precision()) mpfr_prec_round(x_, bits, MPFR_RNDN);
  }

  double to_double() const { return mpfr_get_d(x_, MPFR_RNDN); }
  FloatExp to_floatexp() const {
    long exponent = 0;
    const double m = mpfr_get_d_2exp(&exponent, x_, MPFR_RNDN);
    return FloatExp::make(m, exponent);
  }

  // this += value * 2^log2_scale, for scales far outside double range.
  void add_scaled(double value, double log2_scale) {
    const double whole = std::floor(log2_scale);
    mpfr_t term;
    mpfr_init2(term, 64);
    mpfr_set_d(term, value * std::exp2(log2_scale - whole), MPFR_RNDN);
    mpfr_mul_2si(term, term, long(whole), MPFR_RNDN);
    mpfr_add(x_, x_, term, MPFR_RNDN);
    mpfr_clear(term);
  }

  // (a - b) as a FloatExp, computed exactly before rounding.
  friend FloatExp difference(const BigFloat &a, const BigFloat &b) {
    mpfr_t d;
    mpfr_init2(d, std::max(a.precision(), b.precision()) + 2);
    mpfr_sub(d, a.x_, b.x_, MPFR_RNDN);
    long exponent = 0;
    const double m = mpfr_get_d_2exp(&exponent, d, MPFR_RNDN);
    mpfr_clear(d);
    return FloatExp::make(m, exponent);
  }

  friend bool operator==(const BigFloat &a, const BigFloat &b) {
    return mpfr_equal_p(a.x_, b.x_) != 0;
  }

  // Enough decimal digits to round-trip the current precision.
  std::string to_string() const {
    const int digits = int(double(precision()) * 0.30103) + 2;
    char *text = nullptr;
    mpfr_asprintf(&text, "%.*Rg", digits, x_);
    std::string result = text ? text : "";
    mpfr_free_str(text);
    return result;
  }

private:
  mpfr_t x_;
};
