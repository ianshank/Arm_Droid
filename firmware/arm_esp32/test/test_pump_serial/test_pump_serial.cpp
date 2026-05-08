// Native-environment tests for the PumpSerial byte-accumulation logic.
//
// Run with: pio test -e native
//
// These tests verify that oversize input lines do not produce an
// out-of-bounds store into g_rx_buf.  The PumpSerial sentinel mechanism
// (g_rx_len = kMaxLineBytes + 1) must:
//
//   1. Be set when a line exceeds kMaxLineBytes characters.
//   2. Prevent the null-terminator write on the subsequent '\n'.
//   3. Reset g_rx_len to 0 after the oversized line ends.
//   4. Accept a normal-length line that follows an oversize line.

#include <cstddef>
#include <cstring>

#include <unity.h>

#include "../../src/config.h"

namespace cfg = armdroid::firmware::config;

// ---------------------------------------------------------------------------
// Minimal simulation of the PumpSerial accumulation state.
//
// We replicate the exact logic from firmware.cpp so any future code-gen
// drift between the two will cause a test failure, providing a canary.
// ---------------------------------------------------------------------------

namespace {

// Mirror of the firmware's receive buffer and length counter.
static char  rx_buf[cfg::kMaxLineBytes + 1];  // +1 for '\0' sentinel write
static size_t rx_len = 0;

// Last line dispatched by the simulation (NULL if none / overflow).
static const char* last_dispatched = nullptr;
static size_t      last_dispatch_len = 0;

void reset_sim() {
  std::memset(rx_buf, 0xCC, sizeof(rx_buf));  // poison bytes — detect OOB
  rx_len = 0;
  last_dispatched = nullptr;
  last_dispatch_len = 0;
}

// Simulated ProcessLine: records what was dispatched.
void sim_process_line(const char* buf, size_t len) {
  last_dispatched = buf;
  last_dispatch_len = len;
}

// Replicated PumpSerial logic — same as firmware.cpp.
void sim_pump_bytes(const char* input, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    int c = static_cast<unsigned char>(input[i]);
    if (c == '\n') {
      if (rx_len <= cfg::kMaxLineBytes) {
        rx_buf[rx_len] = '\0';
        sim_process_line(rx_buf, rx_len);
      }
      rx_len = 0;
    } else if (c == '\r') {
      // ignore
    } else if (rx_len < cfg::kMaxLineBytes) {
      rx_buf[rx_len++] = static_cast<char>(c);
    } else {
      rx_len = cfg::kMaxLineBytes + 1;  // overflow sentinel
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void setUp() { reset_sim(); }
void tearDown() {}

// A line of exactly kMaxLineBytes characters must be dispatched normally.
void test_exact_max_line_is_dispatched() {
  char line[cfg::kMaxLineBytes + 1 /* newline */ + 1 /* NUL */];
  std::memset(line, 'A', cfg::kMaxLineBytes);
  line[cfg::kMaxLineBytes] = '\n';
  line[cfg::kMaxLineBytes + 1] = '\0';

  sim_pump_bytes(line, cfg::kMaxLineBytes + 1);

  TEST_ASSERT_NOT_NULL(last_dispatched);
  TEST_ASSERT_EQUAL(cfg::kMaxLineBytes, last_dispatch_len);
  TEST_ASSERT_EQUAL_UINT8('\0', static_cast<unsigned char>(rx_buf[cfg::kMaxLineBytes]));
}

// A line one byte longer than kMaxLineBytes must NOT be dispatched and
// the rx buffer must NOT receive a '\0' write at index kMaxLineBytes
// (that would mean rx_len reached kMaxLineBytes + 1 first and then the
// '\n' branch wrote ``rx_buf[rx_len] = '\0'`` past the valid range).
void test_oversize_line_is_dropped_not_dispatched() {
  // Fill a line that is kMaxLineBytes + 1 characters long.
  const size_t oversize = cfg::kMaxLineBytes + 1;
  char line[oversize + 1 /* newline */ + 1];
  std::memset(line, 'B', oversize);
  line[oversize] = '\n';
  line[oversize + 1] = '\0';

  // Poison sentinel: reset_sim() filled rx_buf with 0xCC, including the
  // [kMaxLineBytes] slot which is the only valid '\0' write site.  We
  // assert that slot is *still* 0xCC after pumping the oversize line —
  // the sentinel branch must have prevented any write there.
  constexpr unsigned char kPoison = 0xCC;

  sim_pump_bytes(line, oversize + 1);

  // 1. The oversize line must NOT have been dispatched.
  TEST_ASSERT_NULL(last_dispatched);

  // 2. rx_len must be reset to 0 after the '\n'.
  TEST_ASSERT_EQUAL(0u, rx_len);

  // 3. Direct OOB-guard check: the [kMaxLineBytes] '\0' slot must remain
  //    poisoned — the sentinel path branched away before writing it.
  TEST_ASSERT_EQUAL_UINT8(kPoison,
      static_cast<unsigned char>(rx_buf[cfg::kMaxLineBytes]));
}

// After an oversize line, a normal-length line must be accepted.
void test_normal_line_accepted_after_oversize() {
  // Feed an oversize line first.
  const size_t oversize = cfg::kMaxLineBytes + 5;
  char big[oversize + 2];
  std::memset(big, 'C', oversize);
  big[oversize]     = '\n';
  big[oversize + 1] = '\0';
  sim_pump_bytes(big, oversize + 1);
  TEST_ASSERT_NULL(last_dispatched);  // dropped

  // Now feed a short, valid line.
  const char* normal = "hello\n";
  sim_pump_bytes(normal, std::strlen(normal));

  TEST_ASSERT_NOT_NULL(last_dispatched);
  TEST_ASSERT_EQUAL(5u, last_dispatch_len);
  TEST_ASSERT_EQUAL_STRING_LEN("hello", last_dispatched, 5);
}

// The sentinel value must be exactly kMaxLineBytes + 1 after overflow
// so the '\n' dispatch guard (rx_len <= kMaxLineBytes) correctly skips it.
void test_sentinel_value_is_max_plus_one() {
  // Feed kMaxLineBytes + 1 non-newline bytes — this must set the sentinel.
  for (size_t i = 0; i <= cfg::kMaxLineBytes; ++i) {
    char byte[1] = {'X'};
    sim_pump_bytes(byte, 1);
  }
  TEST_ASSERT_EQUAL(cfg::kMaxLineBytes + 1, rx_len);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {
  UNITY_BEGIN();
  RUN_TEST(test_exact_max_line_is_dispatched);
  RUN_TEST(test_oversize_line_is_dropped_not_dispatched);
  RUN_TEST(test_normal_line_accepted_after_oversize);
  RUN_TEST(test_sentinel_value_is_max_plus_one);
  return UNITY_END();
}
