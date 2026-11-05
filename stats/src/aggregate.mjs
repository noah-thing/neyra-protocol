// The numbers the website shows. Input rows are groups of identical reports with a count `n`.
// Everything is computed from the validated, coarse fields; medians are medians of buckets of 5 points.
export const MIN_GROUP = 5;
const BEAT = ["worked", "worked_with_damage", "lost_in_quantization"]; // the tune beat the base and the prompt
const PROMPT_ENOUGH = ["prompt_enough", "prompt_better"]; // a prompt did the job

const sum = (rows, pred = () => true) => rows.filter(pred).reduce((s, r) => s + r.n, 0);
const share = (a, b) => (b ? Math.round((a / b) * 1000) / 1000 : null);

function median(rows, key) {
  const sorted = rows.map((r) => [r[key], r.n]).sort((a, b) => a[0] - b[0]);
  const total = sum(rows);
  if (!total) return null;
  const at = (rank) => {
    let seen = 0;
    for (const [v, n] of sorted) { seen += n; if (rank < seen) return v; }
  };
  return (at(Math.floor((total - 1) / 2)) + at(Math.floor(total / 2))) / 2;
}

function verdicts(rows) {
  const runs = sum(rows);
  if (runs < MIN_GROUP) return { runs, hidden: true };
  const won = sum(rows, (r) => BEAT.includes(r.verdict));
  const quantized = sum(rows, (r) => BEAT.includes(r.verdict) && r.quants.length > 0);
  const lost = sum(rows, (r) => r.verdict === "lost_in_quantization");
  const prompt = sum(rows, (r) => PROMPT_ENOUGH.includes(r.verdict));
  return {
    runs,
    beat_prompt: { runs: won, share: share(won, runs) },
    prompt_would_have_done_it: { runs: prompt, share: share(prompt, runs) },
    median_gain_over_base: median(rows, "gain_over_base"),
    median_gain_over_prompt: median(rows, "gain_over_prompt"),
    quantization_lost_gain: { eligible: quantized, lost, share: share(lost, quantized) },
  };
}

// Problems seen by `tune lint` and by the gate in front of `tune train`, counted once per run.
function problems(rows) {
  const checked = sum(rows);
  const by_type = {};
  for (const r of rows) for (const p of r.problems) {
    const t = (by_type[p.id] ||= { runs: 0, as_error: 0 });
    t.runs += r.n;
    if (p.level === "error") t.as_error += r.n;
  }
  const blocked = sum(rows, (r) => r.blocked ?? r.outcome === "blocked_by_lint");
  return {
    checked, blocked_before_training: blocked, share_blocked: share(blocked, checked),
    by_type: Object.fromEntries(Object.entries(by_type).sort()),
  };
}

const group = (rows, key) => {
  const out = {};
  for (const r of rows) (out[key(r)] ||= []).push(r);
  return Object.fromEntries(Object.entries(out).sort().map(([k, v]) => [k, verdicts(v)]));
};

export function aggregate(rows) {
  const of = (kind) => rows.filter((r) => r.kind === kind);
  const v = of("verdict"), lint = of("lint"), train = of("train"), hop = of("hop");
  const gated = [...lint, ...train];
  return {
    schema: 1,
    note: `Groups under ${MIN_GROUP} runs show only a count. Gains are points of score, in buckets of 5. ` +
      "'Prompt' means one few-shot prompt built from the user's own examples.",
    reports: { verdict: sum(v), lint: sum(lint), train: sum(train), hop: sum(hop) },
    fine_tunes_checked: sum(v) + sum(train),
    verdicts: verdicts(v),
    verdicts_by_family: group(v, (r) => r.family),
    verdicts_by_size: group(v, (r) => r.size),
    verdicts_by_family_size: group(v, (r) => `${r.family} ${r.size}`),
    data_problems: gated.length >= MIN_GROUP ? problems(gated) : { checked: sum(gated), hidden: true },
    training: {
      runs: sum(train),
      loss_improved: share(sum(train, (r) => r.loss_improved === "yes"), sum(train, (r) => r.loss_improved !== "unknown")),
      patch_reverted_exactly: share(sum(train, (r) => r.revert_exact), sum(train, (r) => r.outcome === "trained")),
    },
    hop: sum(hop) < MIN_GROUP ? { runs: sum(hop), hidden: true } : {
      runs: sum(hop), found_a_break: share(sum(hop, (r) => !r.ok), sum(hop)),
      first_break_by_part: Object.fromEntries(Object.entries(Object.groupBy(hop.filter((r) => !r.ok),
        (r) => r.first_part)).map(([k, v]) => [k, sum(v)]).sort()),
    },
  };
}
