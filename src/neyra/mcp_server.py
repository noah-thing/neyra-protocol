"""Expose a lab over the Model Context Protocol, so any agent can run a session."""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .diff import compare as compare_models
from .export import to_gguf_lora, to_peft
from .lab import Lab, inside
from .store import Checkpoint

INSTRUCTIONS = """\
You are editing a local model inside a measured lab. Passes are limited, so read lab_status first.
Sample the base model, find a direction, propose a candidate, compare it, run the random control,
bisect it, and only then submit. submit is final: it runs a sealed holdout and returns the verdict,
and you cannot see the holdout items. Report the verdict as given, including failures. Propose any
edit you can compute with propose_from_file, as long as it comes as a patch or safetensors file.
"""


def build(lab: Lab) -> MCPServer:
    mcp = MCPServer("neyra", instructions=INSTRUCTIONS)

    def open_checked(spec: str) -> Checkpoint:
        """Hub ids are fine. Local paths stay inside the lab's roots."""
        looks_local = spec.startswith((".", "/", "~")) or Path(spec).expanduser().exists()
        return Checkpoint.open(str(inside(spec, lab.roots)) if looks_local else spec)

    @mcp.tool()
    def lab_status() -> dict:
        """Budget left, task, candidates and rollouts so far."""
        return lab.status()

    @mcp.tool()
    def list_tensors(pattern: str = "", limit: int = 200) -> list[dict]:
        """Tensor names, shapes and dtypes matching a regular expression. Free."""
        return lab.tensors(pattern, limit)

    @mcp.tool()
    def tensor_stats(name: str) -> dict:
        """Mean, spread and norm of one tensor. Free."""
        return lab.tensor_stats(name)

    @mcp.tool()
    def diff_models(base: str, other: str, sample_blocks: int = 0) -> dict:
        """What changed between two checkpoints, local or on the Hub, and the recovered recipes. Free."""
        r = compare_models(open_checked(base), open_checked(other), sample_blocks=sample_blocks)
        return {"unchanged": r.unchanged, "sampled": r.sampled, "shared_directions": r.shared,
                "changes": [{"tensor": s.name, "recipe": s.recipe} for s in r.changes],
                "findings": [{"severity": f.severity, "kind": f.kind, "detail": f.detail} for f in r.findings]}

    @mcp.tool()
    def sample(candidate: str | None = None, n: int = 32, seed: int = 0) -> dict:
        """Generate on dev items with the base model or a candidate. Costs 1 pass."""
        return lab.sample(candidate, n, seed)

    @mcp.tool()
    def find_direction(rollouts: str, layers: list[int] | None = None, columns: int = 64) -> dict:
        """Direction separating repeated from other tokens, and the MLP columns writing it. Costs 1 pass."""
        return lab.find_direction(rollouts, layers, columns)

    @mcp.tool()
    def propose_projection(finding: str, layers: list[int], alpha: float, note: str = "") -> dict:
        """Candidate that removes a found direction from its writer columns in the given layers. Free."""
        return lab.propose_projection(finding, layers, alpha, note)

    @mcp.tool()
    def propose_from_file(path: str, note: str = "") -> dict:
        """Candidate from any edit you computed: a .neyra patch or a safetensors checkpoint. Free."""
        return lab.propose_from_file(path, note)

    @mcp.tool()
    def describe_candidate(candidate: str) -> dict:
        """The recipe recovered from a candidate's actual weights. Free."""
        return lab.describe(candidate)

    @mcp.tool()
    def compare(candidate: str, n: int = 32, seed: int = 1) -> dict:
        """Paired base against candidate on dev items. Costs 1 pass."""
        return lab.compare(candidate, n, seed)

    @mcp.tool()
    def control(candidate: str, n: int = 32, seed: int = 1) -> dict:
        """Candidate against a size matched random edit on dev items. Costs 1 pass."""
        return lab.control(candidate, n, seed)

    @mcp.tool()
    def bisect(candidate: str, rollouts: str, retain: float = 0.9, grain: str = "column") -> dict:
        """Smallest part of a candidate that keeps the effect. Adds it as a new candidate. Costs 1 pass."""
        return lab.bisect(candidate, rollouts, retain, grain)

    @mcp.tool()
    def submit(candidate: str) -> dict:
        """Final sealed evaluation with fixed gates. Returns the verdict and receipt. Limited, cannot be undone."""
        return lab.submit(candidate)

    @mcp.tool()
    def export(candidate: str, format: str, path: str, arch: str = "") -> dict:
        """Write a candidate as a PEFT adapter (format peft) or a llama.cpp LoRA (format gguf, needs arch)."""
        patch = lab.candidate(candidate).patch
        target = inside(path, lab.roots)
        if format == "peft":
            return to_peft(patch, lab.ck, target)
        if format == "gguf":
            return to_gguf_lora(patch, lab.ck, target, arch)
        raise ValueError("format must be peft or gguf")

    return mcp
