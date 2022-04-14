#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <limits.h>

#include <tmmintrin.h>
#include <wmmintrin.h>

static inline uint16_t find_escaped(const uint8_t input[16])
{
  const uint16_t e = 0x5555U;

  // Find backslashes
  __m128i backslash = _mm_cmpeq_epi8(_mm_set1_epi8('\\'), _mm_loadu_si128((__m128i*)input));
  uint16_t b = _mm_movemask_epi8(backslash);

  // Identify 'starts' - backslash characters not preceded by backslashes.
  uint16_t s = b & ~(b << 1);
  // Detect end of a odd-length sequence of backslashes starting on an even
  // offset.
  // Detail: ES gets all 'starts' that begin on even offsets
  uint16_t es = s & e;
  // Add B to ES, yielding carries on backslash sequences with event starts.
  uint16_t ec = b + es;
  // Filter out the backslashes from the previous addition, getting carries
  // only.
  uint16_t ece = ec & ~b;
  // Select only the end of sequences ending on an odd offset.
  uint16_t od1 = ece & ~e;

  // Detect end of a odd-length sequence of backslashes starting on an odd
  // offset details are as per the above sequence.
  uint16_t os = s & ~e;
  uint16_t oc = b + os;
  uint16_t oce = oc & ~b;
  uint16_t od2 = oce & e;

  // Merge results, yielding ends of all odd-length sequence of backslashes.
  uint16_t od = od1 | od2;

  return od;
}

static inline uint16_t find(const char input[16], char needle)
{
  __m128i result = _mm_cmpeq_epi8(_mm_set1_epi8(needle), _mm_loadu_si128((__m128i*)input));
  return _mm_movemask_epi8(result);
}

void print_mask(uint16_t mask) {
  printf("[ ");
  for(int i = 0, n = (sizeof(mask)*8)-1; i <= n; i++){
    char c = (mask &(1LL<<i))? '1' : '0';
    putchar(c);
  }
  printf(" ]\n");
}

static inline uint16_t lsb(uint16_t mask)
{
  return __builtin_ffs(mask);
}

// Perform a "cumulative bitwise xor," flipping bits each time a 1 is encountered.
//
// For example, prefix_xor(00100100) == 00011100
uint16_t prefix_xor(const uint16_t bitmask) {
  // There should be no such thing with a processing supporting avx2
  // but not clmul.
  __m128i all_ones = _mm_set1_epi8('\xFF');
  __m128i result = _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, bitmask), all_ones, 0);
  return _mm_cvtsi128_si64(result) & 0xFFFF;
}

int main(int argc, char *argv[])
{
  const char input[16] = { '"', 'f', 'o', 'o', ';', '"', 'b', 'a', 'r', ';', '"', '_', ';', '\0' };

  // Find unescaped portions
  uint16_t escaped = find_escaped(input);
  uint16_t strings = find(input, '"') & ~escaped;
  uint16_t comments = find(input, ';') & ~escaped;
  // Newlines only have meaning for comments whether or not they're escaped is
  // irrelevant so don't discard.
  uint16_t newlines = find(input, '_'); // Use '_' instead of '\n' for readability.

  printf("(using _ for LF during testing)\n");
  printf("input:      %.*s\n", sizeof(input), input);
  printf("escaped:  ");
  print_mask(escaped);
  printf("strings:  ");
  print_mask(strings);
  printf("comments: ");
  print_mask(comments);
  printf("newlines: ");
  print_mask(newlines);

  uint16_t ranges = 0u;

  do {
    uint16_t hi, lo, los = lsb(strings), loc = lsb(comments);
    if (!los && !loc) {
      break;
    } else if (los && los < loc) {
      lo = 1 << (los - 1);
      strings &= ~lo;
      hi = lsb(strings);
    } else {
      lo = 1 << (loc - 1);
      hi = lsb(newlines);
    }

    if (hi)
      hi = 1 << (hi - 1);
    else
      hi = 1 << ((sizeof(hi) * CHAR_BIT) - 1);
    ranges |= hi | lo;
    strings &= ~((hi << 1) - 1);
    comments &= ~((hi << 1) - 1);
    newlines &= ~((hi << 1) - 1);
  } while (1);

  printf("ranges:   ");
  print_mask(ranges);
  const uint16_t in_ranges = prefix_xor(ranges);
  printf("result:   ");
  print_mask(in_ranges);
  return 0;
}
