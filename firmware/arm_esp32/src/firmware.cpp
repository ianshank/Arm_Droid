// armdroid arm controller firmware.
//
// Speaks the wire protocol described in PROTOCOL.md over the USB UART.
// Drives N hobby servos (count comes from the host-side codegen) via the
// ESP32Servo library. Implements host-mirroring validation rules (joint
// count, range, e-stop latch), linear interpolation between commanded
// poses, and a watchdog that auto-latches e-stop if the host falls silent.

#include <Arduino.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <math.h>

#include "config.h"
#include "interpolator.h"

namespace cfg = armdroid::firmware::config;
using armdroid::firmware::Interpolator;

namespace {

Servo g_servos[cfg::kNumJoints];
Interpolator g_interp;
bool g_estop_latched = false;
uint32_t g_last_cmd_ms = 0;
uint32_t g_last_heartbeat_ms = 0;
uint32_t g_last_interp_ms = 0;
uint32_t g_seq = 0;

char g_rx_buf[cfg::kMaxLineBytes + 1];
size_t g_rx_len = 0;

// Single shared JsonDocument reused by every fw->host emitter. The host
// receives one frame per line, so emitter calls do not nest; each
// EmitXxx() clears g_emit_doc before populating it. Reusing one doc
// avoids per-call heap churn on the ESP32, which matters because the
// state heartbeat fires at cfg::kHeartbeatHz and was previously
// allocating a fresh JsonDocument every tick.
//
// NOTE: We deliberately do NOT share this doc with the inbound
// deserialiser in ProcessLine() \u2014 the parsed-input doc holds string
// pointers (e.g. ``cmd``) that callers re-emit via EmitNak("unknown_cmd",
// cmd), which would alias into the same pool that EmitNak's clear() just
// reset.  Keeping parse/emit on separate docs avoids that hazard.
JsonDocument g_emit_doc;

constexpr float kHomePosition[cfg::kNumJoints] = {};  // zero-initialised

// Map joint i's value (radians for rotational joints, [0,1] for the
// gripper) to a servo pulse-width in microseconds.
int JointToPulseUs(size_t i, float value) {
  float minv = cfg::kJointMin[i];
  float maxv = cfg::kJointMax[i];
  if (value < minv) value = minv;
  if (value > maxv) value = maxv;
  float alpha = (value - minv) / (maxv - minv);
  int span = cfg::kPulseMaxUs[i] - cfg::kPulseMinUs[i];
  return cfg::kPulseMinUs[i] + static_cast<int>(alpha * static_cast<float>(span));
}

void WriteServos(const float* q) {
  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    int us = JointToPulseUs(i, q[i]);
    g_servos[i].writeMicroseconds(us);
  }
}

void EmitJson(JsonDocument& doc) {
  serializeJson(doc, Serial);
  Serial.write('\n');
}

void EmitAck(int id) {
  g_emit_doc.clear();
  g_emit_doc["t"] = "ack";
  g_emit_doc["id"] = id;
  EmitJson(g_emit_doc);
}

void EmitNak(int id, const char* err, const char* msg) {
  g_emit_doc.clear();
  g_emit_doc["t"] = "nak";
  g_emit_doc["id"] = id;
  g_emit_doc["err"] = err;
  g_emit_doc["msg"] = msg;
  EmitJson(g_emit_doc);
}

void EmitState() {
  g_emit_doc.clear();
  g_emit_doc["t"] = "state";
  g_emit_doc["seq"] = ++g_seq;
  g_emit_doc["ts"] = millis() / 1000.0;
  JsonArray q = g_emit_doc["q"].to<JsonArray>();
  JsonArray qd = g_emit_doc["qd"].to<JsonArray>();
  float vel[cfg::kNumJoints];
  g_interp.GetVelocities(vel);
  const float* cur = g_interp.current();
  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    q.add(cur[i]);
    qd.add(vel[i]);
  }
  g_emit_doc["mv"] = g_interp.is_moving();
  g_emit_doc["es"] = g_estop_latched;
  EmitJson(g_emit_doc);
}

void EmitBootEvent() {
  g_emit_doc.clear();
  g_emit_doc["t"] = "evt";
  g_emit_doc["kind"] = "boot";
  g_emit_doc["ver"] = cfg::kFirmwareVersion;
  g_emit_doc["ts"] = millis() / 1000.0;
  EmitJson(g_emit_doc);
}

void EmitFault(const char* code, const char* msg) {
  g_emit_doc.clear();
  g_emit_doc["t"] = "evt";
  g_emit_doc["kind"] = "fault";
  g_emit_doc["code"] = code;
  g_emit_doc["msg"] = msg;
  g_emit_doc["ts"] = millis() / 1000.0;
  EmitJson(g_emit_doc);
}

void LatchEstop() {
  if (!g_estop_latched) {
    g_interp.Freeze();
    g_estop_latched = true;
    EmitFault("estop", "watchdog or host request");
  }
}

void HandleSetJoints(int id, JsonObjectConst msg) {
  if (g_estop_latched) {
    EmitNak(id, "estop_latched", "cannot move during e-stop");
    return;
  }
  JsonArrayConst q = msg["q"];
  if (q.isNull()) {
    EmitNak(id, "bad_shape", "q missing or not list");
    return;
  }
  if (q.size() != cfg::kNumJoints) {
    EmitNak(id, "bad_joint_count", "joint count mismatch");
    return;
  }
  float target[cfg::kNumJoints];
  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    JsonVariantConst v = q[i];
    if (!v.is<float>() && !v.is<int>()) {
      EmitNak(id, "bad_shape", "non-numeric joint");
      return;
    }
    float fv = v.as<float>();
    if (!isfinite(fv)) {
      EmitNak(id, "out_of_range", "non-finite joint");
      return;
    }
    if (fv < cfg::kJointMin[i] || fv > cfg::kJointMax[i]) {
      EmitNak(id, "out_of_range", "joint exceeds firmware limit");
      return;
    }
    target[i] = fv;
  }
  uint32_t dur_ms = msg["dur_ms"].as<uint32_t>();
  if (dur_ms == 0) {
    EmitNak(id, "bad_shape", "dur_ms must be positive");
    return;
  }
  g_interp.StageSegment(target, dur_ms, millis());
  EmitAck(id);
}

void HandleEstop(int id) {
  g_interp.Freeze();
  g_estop_latched = true;
  EmitAck(id);
}

void HandleClearEstop(int id) {
  g_estop_latched = false;
  EmitAck(id);
}

void HandlePing(int id) { EmitAck(id); }

void HandleGetState(int id) {
  EmitAck(id);
  EmitState();
}

void DispatchCommand(JsonObjectConst msg) {
  int id = msg["id"].as<int>();
  const char* cmd = msg["cmd"].as<const char*>();
  if (cmd == nullptr) {
    EmitNak(id, "bad_shape", "cmd missing");
    return;
  }
  g_last_cmd_ms = millis();

  if (strcmp(cmd, "ping") == 0) {
    HandlePing(id);
  } else if (strcmp(cmd, "get_state") == 0) {
    HandleGetState(id);
  } else if (strcmp(cmd, "estop") == 0) {
    HandleEstop(id);
  } else if (strcmp(cmd, "clear_estop") == 0) {
    HandleClearEstop(id);
  } else if (strcmp(cmd, "set_joints") == 0) {
    HandleSetJoints(id, msg);
  } else {
    EmitNak(id, "unknown_cmd", cmd);
  }
}

void ProcessLine(const char* line, size_t len) {
  if (len == 0) return;
  if (len > cfg::kMaxLineBytes) {
    return;  // silently drop oversized lines
  }
  // Local doc for the inbound frame: must NOT alias g_emit_doc, because
  // string pointers harvested from this doc (e.g. ``cmd``) are passed to
  // EmitNak() which clears its own emit doc — sharing storage would
  // produce a use-after-clear hazard.
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, line, len);
  if (err) {
    EmitFault("bad_json", err.c_str());
    return;
  }
  if (!doc["t"].is<const char*>() ||
      strcmp(doc["t"].as<const char*>(), "cmd") != 0) {
    return;
  }
  DispatchCommand(doc.as<JsonObjectConst>());
}

void PumpSerial() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\n') {
      // Only dispatch when the line fits in the buffer.  When g_rx_len was
      // set to cfg::kMaxLineBytes+1 (the overflow sentinel), writing
      // g_rx_buf[g_rx_len] would be an out-of-bounds store.
      if (g_rx_len <= cfg::kMaxLineBytes) {
        g_rx_buf[g_rx_len] = '\0';
        ProcessLine(g_rx_buf, g_rx_len);
      }
      g_rx_len = 0;
    } else if (c == '\r') {
      // ignore — let \n do the framing
    } else if (g_rx_len < cfg::kMaxLineBytes) {
      g_rx_buf[g_rx_len++] = static_cast<char>(c);
    } else {
      g_rx_len = cfg::kMaxLineBytes + 1;  // sentinel: drop until next \n
    }
  }
}

void TickInterpolator(uint32_t now_ms) {
  if (now_ms - g_last_interp_ms < cfg::kInterpolatorPeriodMs) return;
  g_last_interp_ms = now_ms;
  g_interp.Tick(now_ms);
  WriteServos(g_interp.current());
}

void TickHeartbeat(uint32_t now_ms) {
  if (now_ms - g_last_heartbeat_ms < cfg::kHeartbeatPeriodMs) return;
  g_last_heartbeat_ms = now_ms;
  EmitState();
}

void TickWatchdog(uint32_t now_ms) {
  if (g_last_cmd_ms == 0) return;
  if (now_ms - g_last_cmd_ms > cfg::kWatchdogTimeoutMs) {
    LatchEstop();
  }
}

}  // namespace

void setup() {
  Serial.begin(cfg::kSerialBaud);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (size_t i = 0; i < cfg::kNumJoints; ++i) {
    g_servos[i].setPeriodHertz(50);
    g_servos[i].attach(
        cfg::kServoPins[i], cfg::kPulseMinUs[i], cfg::kPulseMaxUs[i]);
  }
  g_interp.Init(kHomePosition);
  WriteServos(g_interp.current());

  delay(100);
  EmitBootEvent();
}

void loop() {
  uint32_t now = millis();
  PumpSerial();
  TickInterpolator(now);
  TickHeartbeat(now);
  TickWatchdog(now);
}
