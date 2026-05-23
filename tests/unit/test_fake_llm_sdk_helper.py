"""Smoke tests for the lifted ``tests/helpers/fake_llm_sdk`` helper.

Pins the helper's shape so backends in Phase B (perception), Phase C
(planning), and Phase D (control VLA) can rely on the contract.
"""

from __future__ import annotations

import asyncio

from tests.helpers.fake_llm_sdk import (
    fake_anthropic_sdk,
    fake_gemini_sdk,
    fake_llm_sdk,
)


def test_anthropic_shape_exposes_client_class() -> None:
    sdk = fake_anthropic_sdk("hello")
    client = sdk.Anthropic(api_key="xx")
    resp = client.messages.create(model="m", max_tokens=10)
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "hello"
    assert client.messages.calls == [{"model": "m", "max_tokens": 10}]


def test_gemini_shape_exposes_client_class() -> None:
    sdk = fake_gemini_sdk("world")
    client = sdk.Client(api_key="xx")
    resp = client.messages.create(model="m")
    assert resp.content[0].text == "world"


def test_async_sibling_optional() -> None:
    sdk = fake_anthropic_sdk("ok", with_async=True)
    assert hasattr(sdk, "AsyncAnthropic")
    async_client = sdk.AsyncAnthropic(api_key="x")

    async def _run() -> str:
        r = await async_client.messages.create(model="m")
        return str(r.content[0].text)

    assert asyncio.run(_run()) == "ok"


def test_calls_isolated_per_client_instance() -> None:
    sdk = fake_anthropic_sdk("ok")
    c1 = sdk.Anthropic()
    c2 = sdk.Anthropic()
    c1.messages.create(model="m")
    assert c1.messages.calls != c2.messages.calls
    assert len(c2.messages.calls) == 0


def test_generic_fake_llm_sdk_custom_client_class() -> None:
    sdk = fake_llm_sdk("text", client_class="MyClient")
    assert hasattr(sdk, "MyClient")
