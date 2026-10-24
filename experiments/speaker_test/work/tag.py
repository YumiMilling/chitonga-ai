"""Feature tags, assigned AFTER the seeded draw and BEFORE any translation was read (24 Oct 2026, the Jesuit).
Keyword rules, so anyone can re-run them; they describe the sample, they did not shape it (Chris: "just random
sentences"). A sentence can carry several features."""
import json, os, re, sys
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMW = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred|half|first|second|third|fourth|once|twice|many|much|each|every|equal|equally|groups? of|more|fewer|less|most|all|both)\b"
PAST = r"\b(was|were|did|had|came|forgot|lost|fought|helped|happened|finished|improved|worked|read|rang|has rung|have improved|have lost|has lost)\b"
FUT = r"\b(will|tomorrow|next week|next|shall|going to|later)\b"
TIMEW = r"\b(today|yesterday|tonight|morning|evening|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|o'clock|week|month|march|november|december|before|after|every day|first|then|now|when|until|time|season)\b"
DOMAIN = r"\b(hands?|teeth|mouth|elbow|eyes?|ears?|feet|fingers?|fingernails|body|bodies|legs?|plants?|seeds?|leaves|leaf|roots?|trees?|flowers?|fruit|bark|beans?|mangoes|tomatoes|vegetables|cows?|cattle|goat|chicken|lion|frogs?|insects?|bees|caterpillar|butterfly|mosquito\w*|dog|hare|animals?|flies|sun|moon|sky|clouds?|rain\w*|wind|weather|soil|clay|compost|water|ice|smoke|magnets?|iron|shadows?|east|triangle|circle|square|rectangle|shape|corners?|sides?|malaria|fever|medicine|clinic|soap|toilet|sick|syllables?|letters?|sound|capital|full stop|question mark|rhyme|sentence|title|poem|character|naming words|doing words|kwacha|ngwee|number line|tens|ones|even|odd|hour|pattern|estimate)\b"
COND = r"\b(if|when|unless|so that|because|so)\b"
COMP = r"\b(\w+er than|more than|less than|fewer|longer|heavier|bigger|smaller|biggest|smallest|most|better|opposite|same as)\b"
NEG = r"\b(not|no|never|nobody|without|don't|cannot|can't|absent)\b"
def tags(en):
    s = en.lower(); t = []
    if re.search(NEG, s): t.append("negation")
    if re.search(r"\d", s) or re.search(NUMW, s): t.append("number")
    if re.search(PAST, s) or re.search(FUT, s) or re.search(TIMEW, s): t.append("tense_time")
    if re.search(PAST, s): t.append("past")
    if re.search(FUT, s): t.append("future")
    if s.endswith("?"): t.append("question")
    first = re.findall(r"[a-z']+", s)[:2]
    if not s.endswith("?") and first and (first[0] in ("please", "let", "do", "don't", "never", "keep", "help", "say")
         or re.match(r"^(take|sit|stand|put|look|come|pass|line|wash|open|close|write|copy|share|go|bring|pick|sweep|form|stop|repeat|work|finish|hand|clean|count|walk|leave|wait|fold|draw|use|hold|rub|listen|turn|talk|point|clap|fetch|raise|move|check|read|find|match|tell|spell|trace|choose|describe|learn|break|underline|circle|retell|name|feel|plant|water|collect|sort|touch|add|subtract|show|make|measure|estimate|start|divide|join|cover|brush|drink|boil|sleep|eat|cut|bath|rest|wear|sneeze|exercise|tell|sit|give|speak|try|think|send|buy|label|dear)\b", s)):
        t.append("command")
    if re.search(DOMAIN, s): t.append("domain_term")
    if re.search(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", en) or re.search(r"\b(monday|friday|saturday|march|november|december|grade)\b", s): t.append("name_place")
    if re.search(COND, s): t.append("condition")
    if re.search(COMP, s): t.append("comparison")
    return t
pool = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "pool.jsonl"), encoding="utf-8")}
sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
role = {}
for x in sp["shared"]: role[x] = "shared"
for s in ("T01", "T09"):
    for x in sp["unique"][s]: role[x] = "unique_" + s
for x in sp["reserves"]: role[x] = "reserve"
from collections import Counter
c = Counter(); n = 0
with open(os.path.join(RUN, "sentences.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for pid in sp["test"] + sp["reserves"]:
        r = dict(pool[pid]); r["role"] = role[pid]; r["primary"] = sp["shared_primary"].get(pid, role[pid].replace("unique_", "") if role[pid] != "reserve" else None)
        r["features"] = tags(r["en"])
        if role[pid] != "reserve":
            c.update(r["features"]); n += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(n, dict(c.most_common()))
