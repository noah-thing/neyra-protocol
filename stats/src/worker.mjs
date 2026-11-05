// Cloudflare Workers entry. Bindings: DB (D1) and LIMITER (rate limiting), see wrangler.toml.
import { handle } from "./handler.mjs";
import { D1Store } from "./store.mjs";

export default {
  async fetch(request, env) {
    const limiter = { allow: async (key) => (await env.LIMITER.limit({ key })).success };
    return handle(request, { store: new D1Store(env.DB), limiter });
  },
};
