/* The safeguarding detector, ported from tools/chitonga/llm/safeguard.py (24 Oct 2026, the Jesuit).
   Data, not logic, travels: safeguard.export() (the phrase rewrites, the SMS map, the KEY words, the protected English
   words, the context / benign regexes and the rules in order); this function applies it exactly as detect() does.
   The block between the BEGIN / END markers is copied verbatim into web/static/search.js (Searcher.prototype.safeguard)
   and into netlify/functions/assistant.mjs (proposals/netlify_assistant_safeguard.patch); web/tests/safeguard_parity.mjs
   checks this file against Python on the development set and the probes, and that search.js carries the same block. */
// BEGIN safeguard port
function makeSafeguard(exp) {
  if (!exp || !exp.rules) return null;
  const R = (r) => new RegExp(r.src, r.flags);
  const phrases = (exp.phrases || []).map((p) => [new RegExp(p.src, 'g'), p.rep]);
  const sms = exp.sms || {};
  const keys = exp.keys || [];
  const keyset = new Set(keys);
  const near = new Set(exp.near || []);
  const ctx = R(exp.ctx);
  const benign = (exp.benign || []).map((b) => new RegExp(b.src, 'g' + (b.flags || '')));
  const rules = exp.rules.map((r) => ({ cat: r.cat, all: r.all.map(R), ctx: !!r.ctx }));
  const TOK = /[\p{L}\p{N}\p{M}]+/gu;
  const apos = (t) => String(t || '').toLowerCase().replace(/[\u2019\u2018`]/g, "'").replace(/'/g, '');
  const light = (t) => (apos(t).match(TOK) || []).join(' ');
  const cap = (t) => (t.length >= 9 ? 2 : 1);
  function dl(a, b, c) {
    if (Math.abs(a.length - b.length) > c) return c + 1;
    let prev2 = null, prev = Array.from({ length: b.length + 1 }, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      let best = i;
      for (let j = 1; j <= b.length; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
        if (prev2 && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
        cur[j] = v;
        if (v < best) best = v;
      }
      if (best > c) return c + 1;
      prev2 = prev; prev = cur;
    }
    return prev[b.length];
  }
  function fuzzy(tok) {
    if (tok.length < 5 || near.has(tok) || keyset.has(tok) || !/^\p{L}+$/u.test(tok)) return null;
    const c = cap(tok);
    let best = null, bd = c + 1;
    for (const k of keys) {
      if (k[0] !== tok[0] || Math.abs(k.length - tok.length) > c) continue;
      const d = dl(tok, k, c);
      if (d < bd) { best = k; bd = d; }
    }
    return bd <= c ? best : null;
  }
  function normalise(text) {
    let t = apos(text);
    for (const [rx, rep] of phrases) t = t.replace(rx, rep);
    const out = [];
    for (let tok of t.match(TOK) || []) {
      tok = tok.replace(/(.)\1{2,}/gu, '$1$1');
      if (Object.prototype.hasOwnProperty.call(sms, tok)) { out.push(sms[tok]); continue; }
      out.push(fuzzy(tok) || tok);
    }
    return out.join(' ');
  }
  function blank(t) { for (const rx of benign) t = t.replace(rx, ' ~ '); return t; }
  function detect(text) {
    if (!text || !String(text).trim()) return null;
    const lt = light(text), nt = normalise(text);
    const forms = [['light', lt]];
    if (nt !== lt) forms.push(['normalised', nt]);
    for (const [name, t] of forms) {
      const b = blank(t);
      const hasCtx = ctx.test(b);
      for (const r of rules) {
        if (r.ctx && !hasCtx) continue;
        let m0 = null, ok = true;
        for (const rx of r.all) { const m = rx.exec(b); if (!m) { ok = false; break; } if (!m0) m0 = m; }
        if (!ok) continue;
        const hit = m0[0].trim();
        let why = `${r.cat}: '${hit}'` + (r.ctx ? ' with a child in the message' : '');
        if (name === 'normalised') why += ' (spelling normalised)';
        return { category: r.cat, hit, why, form: name };
      }
    }
    return null;
  }
  return { detect, normalise, light, safeguarding: (t) => { const d = detect(t); return d ? d.why : ''; } };
}
// END safeguard port
if (typeof module !== 'undefined' && module.exports) module.exports = { makeSafeguard };
