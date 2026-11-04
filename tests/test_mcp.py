import json

import pytest
from mcp.client.client import Client
from test_lab import real_edit, setup

from lesion.mcp_server import build


def load(result):
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_agent_session_over_the_protocol(tmp_path, model_dir, rng):
    lab, w, u = setup(tmp_path, model_dir, rng)
    async with Client(build(lab)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"lab_status", "sample", "submit", "bisect", "diff_models"} <= names

        status = load(await client.call_tool("lab_status", {}))
        assert status["passes_left"] == 10 and "holdout_commit" in status

        assert load(await client.call_tool("sample", {"n": 4}))["rollouts"] == "r0"

        c = real_edit(lab, w, u, tmp_path)
        verdict = load(await client.call_tool("submit", {"candidate": c}))
        assert verdict["verdict"] == "PASS"

        again = await client.call_tool("submit", {"candidate": c})
        assert again.is_error
