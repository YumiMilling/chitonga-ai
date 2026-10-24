// Parity of the safeguarding JS port with Python (24 Oct 2026, the Jesuit). K0, no network.
//   node tools/chitonga/harness/web/tests/safeguard_parity.mjs
// 1 Python writes safeguard.export() and its own verdicts on the development set (tests/safeguard/dev_v1.jsonl) and the
//   export probes; 2 web/safeguard_port.js gives the same verdict (category and reason) on every one; 3 the port block in
//   web/static/search.js is byte-identical to the one in safeguard_port.js.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, '..');
const HARNESS = path.resolve(WEB, '..');
const LLM = path.resolve(HARNESS, '..', 'llm');
const require = createRequire(import.meta.url);
const { makeSafeguard } = require(path.join(WEB, 'safeguard_port.js'));

const py = `
import json, sys
sys.path.insert(0, ${JSON.stringify(LLM)})
import safeguard as S
rows = [json.loads(x) for x in open(${JSON.stringify(path.join(HARNESS, 'tests', 'safeguard', 'dev_v1.jsonl'))}, encoding='utf-8') if x.strip()]
texts = [r['text'] for r in rows] + S.PROBES
out = {'export': S.export(), 'cases': [{'t': t, 'why': S.safeguarding(t), 'norm': S.normalise(t)} for t in texts]}
sys.stdout.write(json.dumps(out, ensure_ascii=False))
`;
const data = JSON.parse(execFileSync('python', ['-X', 'utf8', '-c', py], { encoding: 'utf8', maxBuffer: 64 << 20 }));
const sg = makeSafeguard(data.export);
let bad = 0;
for (const c of data.cases) {
  const why = sg.safeguarding(c.t);
  const norm = sg.normalise(c.t);
  if (why !== c.why || norm !== c.norm) {
    bad++;
    console.log('MISMATCH', JSON.stringify(c.t), '\n  py:', c.why, '|', c.norm, '\n  js:', why, '|', norm);
  }
}
const block = (s) => { const a = s.indexOf('// BEGIN safeguard port'); const b = s.indexOf('// END safeguard port'); return a >= 0 && b > a ? s.slice(a, b) : null; };
const port = block(fs.readFileSync(path.join(WEB, 'safeguard_port.js'), 'utf8'));
const inSearch = block(fs.readFileSync(path.join(WEB, 'static', 'search.js'), 'utf8'));
const flat = (x) => x.replace(/\r\n/g, '\n').split('\n').map((l) => l.trim()).join('\n');   // search.js indents it
const sameBlock = port !== null && inSearch !== null && flat(port) === flat(inSearch);
if (!sameBlock) { bad++; console.log('search.js does not carry the same port block as safeguard_port.js'); }
console.log(`safeguard parity: ${data.cases.length - (bad - (sameBlock ? 0 : 1))}/${data.cases.length} cases agree; search.js block ${sameBlock ? 'identical' : 'DIFFERS'}`);
process.exit(bad ? 1 : 0);
