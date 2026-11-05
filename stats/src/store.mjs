// A store holds validated reports and returns them grouped. Nothing else about a request reaches it.
// interface: add(report, month) -> Promise<void>;  groups() -> Promise<Array<report fields + count n>>

export class MemoryStore {
  constructor() { this.rows = new Map(); }

  async add(report) {
    const key = JSON.stringify(report);
    const hit = this.rows.get(key);
    if (hit) hit.n += 1; else this.rows.set(key, { report, n: 1 });
  }

  async groups() { return [...this.rows.values()].map(({ report, n }) => ({ ...report, n })); }
}

// Cloudflare D1. Identical reports share a row with a counter. The month is the only time kept.
export class D1Store {
  constructor(db) { this.db = db; }

  async add(report, month) {
    await this.db.prepare(
      `INSERT INTO reports (month, body, n) VALUES (?1, ?2, 1) ON CONFLICT (month, body) DO UPDATE SET n = n + 1`,
    ).bind(month, JSON.stringify(report)).run();
  }

  async groups() {
    const { results } = await this.db.prepare(`SELECT body, SUM(n) AS n FROM reports GROUP BY body`).all();
    return results.map((r) => ({ ...JSON.parse(r.body), n: r.n }));
  }
}
