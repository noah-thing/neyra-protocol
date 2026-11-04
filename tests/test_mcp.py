import json

import pytest
from mcp.client.client import Client
from test_lab import real_edit, setup

from neyra.mcp_server import build


def load(result):
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_agent_session_over_the_protocol(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    async with Client(build(lab)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"lab_status", "sample", "submit", "bisect", "diff_models", "chat", "report"} <= names

        status = load(await client.call_tool("lab_status", {}))
        assert status["passes_left"] == 10 and "holdout_commit" in status

        assert load(await client.call_tool("sample", {"n": 4}))["rollouts"] == "r0"

        c = real_edit(lab, w, u, tmp_path)
        verdict = load(await client.call_tool("submit", {"candidate": c}))
        assert verdict["verdict"] == "PASS"

        again = await client.call_tool("submit", {"candidate": c})
        assert again.is_error

        escape = await client.call_tool("diff_models", {"base": "/etc", "other": "/etc"})
        assert escape.is_error
        write = await client.call_tool("export", {"candidate": c, "format": "peft", "path": "/tmp/outside"})
        assert write.is_error


@pytest.mark.asyncio
async def test_chat_and_report_over_the_protocol(tmp_path, model_dir, rng):
    lab, *_ = setup(tmp_path, model_dir, rng)
    async with Client(build(lab)) as client:
        assert len(load(await client.call_tool("chat", {"prompt": "hi", "n": 2}))["replies"]) == 2
        claim = {"family": "LOOP", "tensors": [], "effect_metric": "defect_reduction", "effect_estimate": None,
                 "confidence": 0.8, "status": "absent"}
        assert load(await client.call_tool("report", {"claims": [claim]}))["claims"] == 1
        assert (await client.call_tool("report", {"claims": [claim]})).is_error


@pytest.mark.asyncio
async def test_diff_models_is_off_when_the_base_is_withheld(tmp_path, model_dir, rng):
    lab, *_ = setup(tmp_path, model_dir, rng, withheld=True)
    async with Client(build(lab)) as client:
        assert load(await client.call_tool("lab_status", {}))["mode"] == "W"
        refused = await client.call_tool("diff_models", {"base": str(lab.model), "other": str(lab.model)})
        assert refused.is_error
