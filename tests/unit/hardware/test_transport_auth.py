"""Unit tests for transport/auth.py — HmacFramer and AuthTransport."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32.transport.auth import AuthTransport, HmacFramer

# ---------------------------------------------------------------------------
# HmacFramer — construction
# ---------------------------------------------------------------------------


def test_hmac_framer_rejects_short_key() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        HmacFramer(b"\x00" * 15)


def test_hmac_framer_accepts_min_key() -> None:
    framer = HmacFramer(b"\x01" * 16)
    assert framer is not None


def test_hmac_framer_from_hex_valid() -> None:
    hex_key = "deadbeef" * 4  # 32 hex chars = 16 bytes
    framer = HmacFramer.from_hex(hex_key)
    assert framer is not None


def test_hmac_framer_from_hex_invalid_chars() -> None:
    with pytest.raises(ValueError, match="not valid hex"):
        HmacFramer.from_hex("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")


def test_hmac_framer_from_hex_too_short() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        HmacFramer.from_hex("deadbeef")  # only 4 bytes


def test_hmac_framer_from_config_uses_key_hex(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.config.schema import TransportAuthConfig

    cfg = TransportAuthConfig(key_hex="aa" * 16, key_env_var="ARMDROID_HMAC_KEY", required=True)
    framer = HmacFramer.from_config(cfg)
    assert framer is not None


def test_hmac_framer_from_config_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.config.schema import TransportAuthConfig

    monkeypatch.setenv("MY_TEST_KEY", "bb" * 16)
    cfg = TransportAuthConfig(key_hex=None, key_env_var="MY_TEST_KEY", required=True)
    framer = HmacFramer.from_config(cfg)
    assert framer is not None


def test_hmac_framer_from_config_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.config.schema import TransportAuthConfig

    monkeypatch.delenv("ARMDROID_HMAC_KEY", raising=False)
    cfg = TransportAuthConfig(key_hex=None, key_env_var="ARMDROID_HMAC_KEY", required=True)
    with pytest.raises(ArmDriverError, match="ARMDROID_HMAC_KEY"):
        HmacFramer.from_config(cfg)


def test_hmac_framer_from_config_missing_not_required_returns_dummy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from armdroid.config.schema import TransportAuthConfig

    monkeypatch.delenv("ARMDROID_HMAC_KEY", raising=False)
    cfg = TransportAuthConfig(key_hex=None, key_env_var="ARMDROID_HMAC_KEY", required=False)
    framer = HmacFramer.from_config(cfg)
    assert framer is not None


# ---------------------------------------------------------------------------
# HmacFramer — sign / verify round-trip
# ---------------------------------------------------------------------------


def test_sign_adds_seq_and_sig() -> None:
    framer = HmacFramer(b"\x01" * 16)
    line = json.dumps({"t": "cmd", "id": 1, "cmd": "ping"}).encode("ascii") + b"\n"
    signed = framer.sign(line)
    parsed = json.loads(signed.decode("ascii").strip())
    assert "seq" in parsed
    assert "sig" in parsed
    assert len(parsed["sig"]) == 64  # 32 bytes = 64 hex chars
    assert parsed["seq"] == 1


def test_sign_increments_seq() -> None:
    framer = HmacFramer(b"\x01" * 16)
    base = json.dumps({"t": "cmd", "cmd": "ping"}).encode("ascii") + b"\n"
    for expected_seq in range(1, 4):
        signed = framer.sign(base)
        assert json.loads(signed)["seq"] == expected_seq


def test_verify_accepts_valid_signed_frame() -> None:
    key = b"\x42" * 16
    framer_tx = HmacFramer(key)
    framer_rx = HmacFramer(key)
    line = json.dumps({"t": "state", "q": [0.0] * 6, "qd": [0.0] * 6}).encode("ascii") + b"\n"
    signed = framer_tx.sign(line)
    verified = framer_rx.verify(signed)
    parsed = json.loads(verified.decode("ascii").strip())
    assert "sig" not in parsed
    assert "seq" in parsed


def test_verify_rejects_wrong_sig() -> None:
    key = b"\x01" * 16
    framer = HmacFramer(key)
    line = json.dumps({"t": "cmd", "seq": 1, "sig": "aa" * 32}).encode("ascii") + b"\n"
    with pytest.raises(ValueError, match="signature"):
        framer.verify(line)


def test_verify_rejects_replay() -> None:
    key = b"\x01" * 16
    framer_tx = HmacFramer(key)
    framer_rx = HmacFramer(key)
    base = json.dumps({"t": "cmd", "cmd": "ping"}).encode("ascii") + b"\n"
    signed = framer_tx.sign(base)
    framer_rx.verify(signed)  # first: ok
    signed2 = framer_tx.sign(base)
    framer_rx.verify(signed2)  # seq=2: ok
    with pytest.raises(ValueError, match=r"[Aa]nti.replay|seq"):
        # Replay signed (seq=1) after seq=2 accepted
        framer_rx.verify(signed)


def test_verify_rejects_missing_sig() -> None:
    framer = HmacFramer(b"\x01" * 16)
    line = json.dumps({"t": "cmd", "seq": 1}).encode("ascii") + b"\n"
    with pytest.raises(ValueError, match="sig"):
        framer.verify(line)


def test_verify_rejects_missing_seq() -> None:
    framer = HmacFramer(b"\x01" * 16)
    line = json.dumps({"t": "cmd", "sig": "aa" * 32}).encode("ascii") + b"\n"
    with pytest.raises(ValueError, match="seq"):
        framer.verify(line)


def test_sign_rejects_non_json() -> None:
    framer = HmacFramer(b"\x01" * 16)
    with pytest.raises(ValueError, match="non-JSON"):
        framer.sign(b"not json\n")


# ---------------------------------------------------------------------------
# AuthTransport
# ---------------------------------------------------------------------------


def _make_inner_transport(readline_value: bytes = b"") -> MagicMock:
    inner = MagicMock()
    inner.is_connected = True
    inner.connect = AsyncMock()
    inner.disconnect = AsyncMock()
    inner.write_line = AsyncMock()
    inner.readline = AsyncMock(return_value=readline_value)
    return inner


@pytest.mark.asyncio
async def test_auth_transport_signs_outgoing() -> None:
    key = b"\x01" * 16
    framer = HmacFramer(key)
    inner = _make_inner_transport()
    auth = AuthTransport(inner, framer)

    line = json.dumps({"t": "cmd", "cmd": "ping"}).encode("ascii") + b"\n"
    await auth.write_line(line)

    written: bytes = inner.write_line.call_args[0][0]
    parsed = json.loads(written.decode("ascii").strip())
    assert "sig" in parsed
    assert "seq" in parsed


@pytest.mark.asyncio
async def test_auth_transport_verifies_incoming() -> None:
    key = b"\x01" * 16
    framer_tx = HmacFramer(key)
    framer_rx = HmacFramer(key)

    incoming = json.dumps({"t": "state", "q": [0.0] * 6, "qd": [0.0] * 6}).encode("ascii") + b"\n"
    signed = framer_tx.sign(incoming)

    inner = _make_inner_transport(readline_value=signed)
    auth = AuthTransport(inner, framer_rx)
    result = await auth.readline()
    assert result != b""
    parsed = json.loads(result.decode("ascii").strip())
    assert "sig" not in parsed


@pytest.mark.asyncio
async def test_auth_transport_returns_empty_on_bad_sig() -> None:
    key = b"\x01" * 16
    framer = HmacFramer(key)
    bad_line = json.dumps({"t": "cmd", "seq": 1, "sig": "aa" * 32}).encode("ascii") + b"\n"
    inner = _make_inner_transport(readline_value=bad_line)
    auth = AuthTransport(inner, framer)
    result = await auth.readline()
    assert result == b""


@pytest.mark.asyncio
async def test_auth_transport_passes_through_empty_line() -> None:
    key = b"\x01" * 16
    framer = HmacFramer(key)
    inner = _make_inner_transport(readline_value=b"\n")
    auth = AuthTransport(inner, framer)
    result = await auth.readline()
    assert result == b"\n"


@pytest.mark.asyncio
async def test_auth_transport_delegates_connect_disconnect() -> None:
    key = b"\x01" * 16
    framer = HmacFramer(key)
    inner = _make_inner_transport()
    auth = AuthTransport(inner, framer)
    await auth.connect()
    inner.connect.assert_called_once()
    await auth.disconnect()
    inner.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_auth_transport_write_line_non_json_raises_arm_driver_error() -> None:
    """write_line with non-JSON data must raise ArmDriverError, not ValueError."""
    key = b"\x01" * 16
    framer = HmacFramer(key)
    inner = _make_inner_transport()
    auth = AuthTransport(inner, framer)

    with pytest.raises(ArmDriverError, match="sign"):
        await auth.write_line(b"not json at all\n")
    # Inner transport must NOT have been called.
    inner.write_line.assert_not_called()
