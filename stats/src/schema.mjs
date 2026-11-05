// The only fields a report may hold. Keep in step with src/neyra/tune/share.py; a test compares the two.
// A field type is a list of allowed strings, "bool", "gain" (integer points, multiple of 5, -100..100),
// "version", or {list: {max, key, item}} for a short list of small records with a unique key.
export const SCHEMA = 1;
const FAMILIES = ["qwen", "llama", "gemma", "mistral", "deepseek", "phi", "smollm", "other"];
const SIZES = ["<1B", "1-3B", "3-8B", "8-30B", ">30B", "unknown"];
const ROWS = ["<100", "100-999", "1k-9.9k", "10k-99k", "100k+"];
const FORMATS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
  "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32", "other"];
const PROBLEMS = ["double_bos", "dropped", "empty", "end_masked", "foreign", "missing_bos", "missing_end", "prefix",
  "rejected", "system", "think", "truncated", "twice", "unlocated", "prompt", "duplicates", "dup_exact", "dup_near",
  "pad_eos", "template_gguf", "tokenizer_gguf", "leakage", "unreadable", "mixed", "other"];
const problems = { list: { max: 12, key: "id", item: { id: PROBLEMS, level: ["error", "warning", "note"] } } };
const PARTS = ["weights", "tokenizer", "template", "bos", "config", "generation"];
const CAUSES = ["dtype", "extra_tensors", "head", "leftover_adapter", "missing_tensors", "never_applied", "other_base",
  "partly_applied", "rounded_away", "weights_differ", "wrong_scale", "other"];

export const SPEC = {
  schema: SCHEMA,
  common: { schema: "schema", kind: "kind", neyra: "version", os: ["macos", "linux", "windows", "other"] },
  kinds: {
    verdict: {
      backend: ["torch", "mlx", "other"], family: FAMILIES, size: SIZES,
      verdict: ["worked", "worked_with_damage", "lost_in_quantization", "prompt_enough", "prompt_better",
        "did_not_work", "worse_than_base", "cannot_tell"],
      cases: ["10-29", "30-49", "50-99", "100-199", "200+"], gain_over_base: "gain", gain_over_prompt: "gain",
      quants: { list: { max: 8, key: "format", item: { format: FORMATS, held: "bool" } } },
    },
    lint: {
      trainer: ["trl", "unsloth", "axolotl", "mlx", "llama-factory"], family: FAMILIES, size: SIZES, rows: ROWS,
      blocked: "bool", problems,
    },
    train: {
      family: FAMILIES, size: SIZES, rows: ROWS, outcome: ["trained", "blocked_by_lint"],
      loss_improved: ["yes", "no", "unknown"], revert_exact: "bool", problems,
    },
    hop: {
      family: FAMILIES, size: SIZES, hops: ["2", "3", "4+"], ok: "bool", first_part: [...PARTS, "other", "none"],
      first_cause: [...CAUSES, "none"],
    },
  },
};

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const sameKeys = (o, keys) => {
  const have = Object.keys(o).sort();
  return have.length === keys.length && [...keys].sort().every((k, i) => k === have[i]);
};

function ok(type, v) {
  if (Array.isArray(type)) return type.includes(v);
  if (type === "bool") return typeof v === "boolean";
  if (type === "gain") return Number.isInteger(v) && v % 5 === 0 && v >= -100 && v <= 100;
  if (type === "version") return typeof v === "string" && /^\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(v);
  if (type === "schema") return v === SCHEMA;
  if (type === "kind") return typeof v === "string" && Object.hasOwn(SPEC.kinds, v);
  const { max, key, item } = type.list;
  if (!Array.isArray(v) || v.length > max) return false;
  const seen = new Set();
  for (const r of v) {
    if (!isObject(r) || !sameKeys(r, Object.keys(item)) || seen.has(r[key])) return false;
    seen.add(r[key]);
    if (!Object.entries(item).every(([k, t]) => ok(t, r[k]))) return false;
  }
  return true;
}

const clean = (type, v) => !Array.isArray(v) ? v : v.map((r) => Object.fromEntries(Object.keys(r).sort().map((k) => [k, r[k]])))
  .sort((a, b) => (a[type.list.key] < b[type.list.key] ? -1 : 1));

// Returns the report copied field by field in a fixed order, or null when anything is off.
export function validate(body) {
  if (!isObject(body) || !ok("kind", body.kind)) return null;
  const types = { ...SPEC.common, ...SPEC.kinds[body.kind] };
  if (!sameKeys(body, Object.keys(types))) return null;
  for (const [k, t] of Object.entries(types)) if (!ok(t, body[k])) return null;
  return Object.fromEntries(Object.keys(types).sort().map((k) => [k, typeof types[k] === "object" && !Array.isArray(types[k])
    ? clean(types[k], body[k]) : body[k]]));
}
