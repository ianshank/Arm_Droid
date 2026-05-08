// Native-environment tests for the joint interpolator.
//
// Run with: pio test -e native
//
// The interpolator has no Arduino dependencies (header-only class in
// src/interpolator.h), so we exercise it on the host without
// hardware-in-the-loop. Synthetic timestamps drive interpolation
// deterministically.

#include <unity.h>

#include "../../src/config.h"
#include "../../src/interpolator.h"

using armdroid::firmware::Interpolator;
namespace cfg = armdroid::firmware::config;

void setUp() {}
void tearDown() {}

void test_init_holds_home_position() {
  float home[cfg::kNumJoints];
  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    home[i] = 0.1f * static_cast<float>(i + 1);
  }
  Interpolator interp;
  interp.Init(home);
  TEST_ASSERT_FALSE(interp.is_moving());
  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, home[i], interp.current()[i]);
  }
}

void test_segment_midpoint_is_halfway() {
  float home[cfg::kNumJoints] = {};
  Interpolator interp;
  interp.Init(home);

  float target[cfg::kNumJoints] = {};
  target[0] = 1.0f;
  target[1] = 0.5f;
  target[2] = -0.5f;
  interp.StageSegment(target, /*duration_ms=*/2000, /*now_ms=*/1000);

  // At t = start + 1000ms (halfway through 2000ms), positions are halved.
  interp.Tick(2000);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.5f, interp.current()[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.25f, interp.current()[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, -0.25f, interp.current()[2]);
  TEST_ASSERT_TRUE(interp.is_moving());
}

void test_segment_completion_pins_to_target() {
  float home[cfg::kNumJoints] = {};
  Interpolator interp;
  interp.Init(home);

  float target[cfg::kNumJoints] = {};
  target[0] = 1.0f;
  target[1] = 0.5f;
  interp.StageSegment(target, 1000, 0);

  interp.Tick(2000);  // well past end
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, interp.current()[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, interp.current()[1]);
  TEST_ASSERT_FALSE(interp.is_moving());
}

void test_freeze_pins_target_to_current() {
  float home[cfg::kNumJoints] = {};
  Interpolator interp;
  interp.Init(home);
  float target[cfg::kNumJoints] = {};
  target[0] = 1.0f;
  interp.StageSegment(target, 2000, 0);
  interp.Tick(1000);  // halfway
  float frozen = interp.current()[0];
  interp.Freeze();
  interp.Tick(5000);  // long after, should not move
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, frozen, interp.current()[0]);
  TEST_ASSERT_FALSE(interp.is_moving());
}

void test_velocity_estimate_matches_segment() {
  float home[cfg::kNumJoints] = {};
  Interpolator interp;
  interp.Init(home);
  float target[cfg::kNumJoints] = {};
  target[0] = 2.0f;
  interp.StageSegment(target, 1000, 0);  // 2 rad / 1 s = 2 rad/s
  interp.Tick(500);  // mid-segment so is_moving == true

  float vel[cfg::kNumJoints];
  interp.GetVelocities(vel);
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 2.0f, vel[0]);
  for (size_t i = 1; i < cfg::kNumJoints; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, vel[i]);
  }
}

void test_chained_segments_continue_from_current() {
  float home[cfg::kNumJoints] = {};
  Interpolator interp;
  interp.Init(home);
  float t1[cfg::kNumJoints] = {};
  t1[0] = 1.0f;
  interp.StageSegment(t1, 2000, 0);
  interp.Tick(1000);  // halfway, current[0] == 0.5

  float t2[cfg::kNumJoints] = {};
  t2[0] = 2.0f;
  interp.StageSegment(t2, 1000, 1000);  // start fresh from 0.5 -> 2.0
  interp.Tick(1500);  // halfway through new segment
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1.25f, interp.current()[0]);
}

int main(int /*argc*/, char** /*argv*/) {
  UNITY_BEGIN();
  RUN_TEST(test_init_holds_home_position);
  RUN_TEST(test_segment_midpoint_is_halfway);
  RUN_TEST(test_segment_completion_pins_to_target);
  RUN_TEST(test_freeze_pins_target_to_current);
  RUN_TEST(test_velocity_estimate_matches_segment);
  RUN_TEST(test_chained_segments_continue_from_current);
  return UNITY_END();
}
