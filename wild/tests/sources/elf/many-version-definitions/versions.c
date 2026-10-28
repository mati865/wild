// Preprocess into a linker script with 2500 versions, exceeding 64 KiB of version definitions.
// clang-format off
#define NODE(a, b, c, d) V##a##b##c##d {};
#define TEN(a, b, c) \
  NODE(a, b, c, 0) \
  NODE(a, b, c, 1) \
  NODE(a, b, c, 2) \
  NODE(a, b, c, 3) \
  NODE(a, b, c, 4) \
  NODE(a, b, c, 5) \
  NODE(a, b, c, 6) \
  NODE(a, b, c, 7) \
  NODE(a, b, c, 8) \
  NODE(a, b, c, 9)
#define HUNDRED(a, b) \
  TEN(a, b, 0) \
  TEN(a, b, 1) \
  TEN(a, b, 2) \
  TEN(a, b, 3) \
  TEN(a, b, 4) \
  TEN(a, b, 5) \
  TEN(a, b, 6) \
  TEN(a, b, 7) \
  TEN(a, b, 8) \
  TEN(a, b, 9)
#define THOUSAND(a) \
  HUNDRED(a, 0) \
  HUNDRED(a, 1) \
  HUNDRED(a, 2) \
  HUNDRED(a, 3) \
  HUNDRED(a, 4) \
  HUNDRED(a, 5) \
  HUNDRED(a, 6) \
  HUNDRED(a, 7) \
  HUNDRED(a, 8) \
  HUNDRED(a, 9)

VERSION { THOUSAND(0) THOUSAND(1) HUNDRED(2, 0) HUNDRED(2, 1) HUNDRED(2, 2) HUNDRED(2, 3) HUNDRED(2, 4) }
// clang-format on
