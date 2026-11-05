import { test } from "node:test";
import assert from "node:assert/strict";
import { handle, MemoryLimiter } from "../src/handler.mjs";
import { MemoryStore } from "../src/store.mjs";
import { SPEC, validate } from "../src/schema.mjs";

const base = { schema: 1, neyra: "0.3.0", os: "macos" };
const reports = {
  verdict: () => ({ ...base, kind: "verdict", backend: "torch", family: "qwen", size: "<1B", verdict: "worked",
    cases: "50-99", gain_over_base: 15, gain_over_prompt: 5,
    quants: [{ format: "Q4_K_M", held: true }, { format: "Q8_0", held: true }] }),
  lint: () => ({ ...base, kind: "lint", trainer: "trl", family: "llama", size: "3-8B", rows: "1k-9.9k", blocked: true,
    problems: [{ id: "double_bos", level: "error" }, { id: "duplicates", level: "warning" }] }),
  train: () => ({ ...base, kind: "train", family: "qwen", size: "<1B", rows: "100-999", outcome: "trained",
    loss_improved: "yes", revert_exact: true, problems: [] }),
  hop: () => ({ ...base, kind: "hop", family: "gemma", size: "1-3B", hops: "3", ok: false, first_part: "template",
    first_cause: "other" }),
};
const good = reports.verdict;
const post = (body, headers = {}) => new Request("https://x.test/v1/report", {
  method: "POST", headers: { "content-type": "application/json", ...headers },
  body: typeof body === "string" ? body : JSON.stringify(body),
});
const setup = (limit = 100) => ({ store: new MemoryStore(), limiter: new MemoryLimiter(limit) });
const stats = async (env) => (await handle(new Request("https://x.test/v1/stats"), env)).json();

test("every kind of report is accepted and stored", async () => {
  const env = setup();
  for (const make of Object.values(reports)) assert.equal((await handle(post(make()), env)).status, 204);
  const a = await stats(env);
  assert.deepEqual(a.reports, { verdict: 1, lint: 1, train: 1, hop: 1 });
});

test("every extra, missing or mistyped field is rejected", async () => {
  const bad = [
    { ...good(), prompt: "hello" }, { ...good(), id: "abc" }, { ...good(), model: "acme/x" },
    (({ os, ...r }) => r)(good()), { ...good(), kind: "other" }, { ...good(), kind: undefined },
    { ...good(), family: "acme-llama" }, { ...good(), neyra: "0.3.0-dev+/home/bob" },
    { ...good(), gain_over_base: 12 }, { ...good(), gain_over_base: 15.0000001 }, { ...good(), gain_over_base: "15" },
    { ...good(), gain_over_prompt: 500 }, { ...good(), schema: 2 }, { ...good(), schema: "1" },
    { ...good(), quants: [{ format: "Q4_K_M", held: 1 }] },
    { ...good(), quants: [{ format: "model-Q4.gguf", held: true }] },
    { ...good(), quants: [{ format: "Q4_K_M", held: true, path: "/x" }] },
    { ...good(), quants: [{ format: "Q4_K_M", held: true }, { format: "Q4_K_M", held: false }] },
    { ...good(), quants: Array.from({ length: 9 }, () => ({ format: "other", held: true })) },
    { ...reports.lint(), problems: [{ id: "secret_file.txt", level: "error" }] },
    { ...reports.lint(), problems: [{ id: "mixed", level: "error", example: "row 3 text" }] },
    { ...reports.lint(), gain_over_base: 5 }, { ...reports.hop(), hops: "9" }, { ...reports.train(), rows: "12345" },
    [], null, "str", 5,
  ];
  for (const body of bad) {
    assert.equal(validate(JSON.parse(JSON.stringify(body ?? null))), null, JSON.stringify(body));
    assert.equal((await handle(post(JSON.stringify(body)), setup())).status, 400);
  }
  assert.equal((await handle(post("{not json"), setup())).status, 400);
});

test("the stored row holds only schema fields and no request data", async () => {
  const env = setup();
  await handle(post(good(), { "user-agent": "secret-agent", "cf-connecting-ip": "203.0.113.9", cookie: "a=b" }), env);
  const stored = JSON.stringify(await env.store.groups());
  for (const leak of ["secret-agent", "203.0.113.9", "a=b"]) assert.ok(!stored.includes(leak));
  const [row] = await env.store.groups();
  assert.deepEqual(Object.keys(row).sort(), [...Object.keys({ ...SPEC.common, ...SPEC.kinds.verdict }), "n"].sort());
});

test("the field lists are pinned", () => {
  assert.deepEqual(Object.keys(SPEC.common), ["schema", "kind", "neyra", "os"]);
  assert.deepEqual(Object.keys(SPEC.kinds.verdict), ["backend", "family", "size", "verdict", "cases", "gain_over_base",
    "gain_over_prompt", "quants"]);
  assert.deepEqual(Object.keys(SPEC.kinds.lint), ["trainer", "family", "size", "rows", "blocked", "problems"]);
  assert.deepEqual(Object.keys(SPEC.kinds.train), ["family", "size", "rows", "outcome", "loss_improved",
    "revert_exact", "problems"]);
  assert.deepEqual(Object.keys(SPEC.kinds.hop), ["family", "size", "hops", "ok", "first_part", "first_cause"]);
});

test("rate limit answers 429 per caller", async () => {
  const env = setup(2);
  const from = (ip) => post(good(), { "cf-connecting-ip": ip });
  assert.equal((await handle(from("1.1.1.1"), env)).status, 204);
  assert.equal((await handle(from("1.1.1.1"), env)).status, 204);
  assert.equal((await handle(from("1.1.1.1"), env)).status, 429);
  assert.equal((await handle(from("2.2.2.2"), env)).status, 204);
});

test("size, method, type and path limits", async () => {
  const env = setup();
  assert.equal((await handle(post("x".repeat(5000)), env)).status, 413);
  assert.equal((await handle(new Request("https://x.test/v1/report"), env)).status, 405);
  assert.equal((await handle(post(good(), { "content-type": "text/plain" }), env)).status, 415);
  assert.equal((await handle(new Request("https://x.test/other"), env)).status, 404);
});

test("the handler never writes to the console", async () => {
  const seen = [];
  const orig = { log: console.log, error: console.error, warn: console.warn, info: console.info };
  for (const k of Object.keys(orig)) console[k] = (...a) => seen.push(a);
  try {
    await handle(post(good(), { "cf-connecting-ip": "203.0.113.9" }), setup());
    await handle(post({ bad: 1 }), setup());
  } finally { Object.assign(console, orig); }
  assert.deepEqual(seen, []);
});

test("aggregate: verdict numbers", async () => {
  const env = setup();
  const add = async (over, times) => { for (let i = 0; i < times; i++) await handle(post({ ...good(), ...over }), env); };
  await add({ verdict: "worked", gain_over_base: 20, gain_over_prompt: 10 }, 3);
  await add({ verdict: "lost_in_quantization", gain_over_base: 10, gain_over_prompt: 5 }, 1);
  await add({ verdict: "prompt_enough", gain_over_base: 10, gain_over_prompt: 0, quants: [] }, 2);
  await add({ verdict: "did_not_work", family: "llama", size: "3-8B", gain_over_base: 0, gain_over_prompt: -5 }, 1);
  const a = await stats(env);
  const v = a.verdicts;
  assert.equal(v.runs, 7);
  assert.equal(v.beat_prompt.share, 0.571);
  assert.equal(v.prompt_would_have_done_it.share, 0.286);
  assert.equal(v.median_gain_over_base, 10);
  assert.equal(v.median_gain_over_prompt, 5);
  assert.deepEqual(v.quantization_lost_gain, { eligible: 4, lost: 1, share: 0.25 });
  assert.equal(a.verdicts_by_family.qwen.runs, 6);
  assert.deepEqual(a.verdicts_by_family.llama, { runs: 1, hidden: true });
  assert.equal(a.fine_tunes_checked, 7);
});

test("aggregate: problems caught before training, training and hop", async () => {
  const env = setup();
  const send = async (r, times) => { for (let i = 0; i < times; i++) await handle(post(r), env); };
  await send(reports.lint(), 3);
  await send({ ...reports.lint(), blocked: false, problems: [] }, 1);
  await send({ ...reports.train(), outcome: "blocked_by_lint", loss_improved: "unknown", revert_exact: false, problems: [{ id: "double_bos", level: "error" }] }, 1);
  await send(reports.train(), 2);
  await send({ ...reports.train(), loss_improved: "no", revert_exact: false }, 1);
  await send(reports.hop(), 5);
  const a = await stats(env);
  assert.equal(a.data_problems.checked, 8);
  assert.equal(a.data_problems.blocked_before_training, 4);
  assert.deepEqual(a.data_problems.by_type.double_bos, { runs: 4, as_error: 4 });
  assert.deepEqual(a.data_problems.by_type.duplicates, { runs: 3, as_error: 0 });
  assert.equal(a.training.runs, 4);
  assert.equal(a.training.loss_improved, 0.667);
  assert.equal(a.training.patch_reverted_exactly, 0.667);
  assert.deepEqual(a.hop, { runs: 5, found_a_break: 1, first_break_by_part: { template: 5 } });
  assert.equal(a.fine_tunes_checked, 4);
});
