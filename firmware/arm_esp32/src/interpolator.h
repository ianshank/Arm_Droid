// Joint interpolator: linearly interpolates between the current pose
// and a target pose over a host-specified duration.
//
// Header-only so the native test environment can include it without
// dragging in the Arduino servo libs.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "config.h"

namespace armdroid::firmware {

class Interpolator {
 public:
  Interpolator() = default;

  // Initialise both current and target to the home position.
  void Init(const float home[config::kNumJoints]) {
    for (size_t i = 0; i < config::kNumJoints; ++i) {
      current_[i] = home[i];
      segment_start_[i] = home[i];
      segment_target_[i] = home[i];
    }
    segment_start_ms_ = 0;
    segment_duration_ms_ = 0;
    is_moving_ = false;
  }

  // Stage a new segment. Caller is responsible for joint-limit checks;
  // this class trusts its inputs.
  void StageSegment(
      const float target[config::kNumJoints],
      uint32_t duration_ms,
      uint32_t now_ms) {
    for (size_t i = 0; i < config::kNumJoints; ++i) {
      segment_start_[i] = current_[i];
      segment_target_[i] = target[i];
    }
    segment_start_ms_ = now_ms;
    segment_duration_ms_ = duration_ms;
    is_moving_ = (duration_ms > 0);
  }

  // Advance the interpolator to wall-clock now_ms. Idempotent for
  // now_ms past the segment end.
  void Tick(uint32_t now_ms) {
    if (segment_duration_ms_ == 0) {
      is_moving_ = false;
      return;
    }
    uint32_t elapsed = now_ms - segment_start_ms_;
    if (elapsed >= segment_duration_ms_) {
      for (size_t i = 0; i < config::kNumJoints; ++i) {
        current_[i] = segment_target_[i];
      }
      is_moving_ = false;
      segment_duration_ms_ = 0;
      return;
    }
    float alpha =
        static_cast<float>(elapsed) / static_cast<float>(segment_duration_ms_);
    for (size_t i = 0; i < config::kNumJoints; ++i) {
      current_[i] =
          segment_start_[i] +
          alpha * (segment_target_[i] - segment_start_[i]);
    }
  }

  // Freeze the arm at the current interpolated pose. Used on e-stop.
  void Freeze() {
    for (size_t i = 0; i < config::kNumJoints; ++i) {
      segment_start_[i] = current_[i];
      segment_target_[i] = current_[i];
    }
    segment_duration_ms_ = 0;
    is_moving_ = false;
  }

  const float* current() const { return current_; }
  bool is_moving() const { return is_moving_; }

  // Velocity estimate: target-minus-start over duration. Constant
  // during a segment, zero between segments.
  void GetVelocities(float out[config::kNumJoints]) const {
    if (segment_duration_ms_ == 0 || !is_moving_) {
      for (size_t i = 0; i < config::kNumJoints; ++i) {
        out[i] = 0.0f;
      }
      return;
    }
    float dur_s = static_cast<float>(segment_duration_ms_) / 1000.0f;
    for (size_t i = 0; i < config::kNumJoints; ++i) {
      out[i] = (segment_target_[i] - segment_start_[i]) / dur_s;
    }
  }

 private:
  float current_[config::kNumJoints] = {};
  float segment_start_[config::kNumJoints] = {};
  float segment_target_[config::kNumJoints] = {};
  uint32_t segment_start_ms_ = 0;
  uint32_t segment_duration_ms_ = 0;
  bool is_moving_ = false;
};

}  // namespace armdroid::firmware
