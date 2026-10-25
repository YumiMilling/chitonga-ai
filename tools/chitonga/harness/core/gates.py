"""Gates, before any reading or model (23 Oct 2026, the Jesuit). Deterministic; fixed lines only; reused, not rewritten.

ORDER
  1 safeguarding   h3_front.safeguarding = tools/chitonga/llm/safeguard.py since 24 Oct 2026 (reader bake-off 3c: by
                   category, spelling normalised, benign lesson phrases blanked) on the original and on any English
                   reading, BEFORE any routing to a lesson (engine._turn step 1; engine._gate_reading for the reading).
                   Outcome: the assistant's VERIFIED referral (tools/chitonga/assistant/gate.REFERRAL: head /
                   guidance teacher or focal person today, Childline 116, the police Victim Support Unit, the DSWO; Children's Code Act
                   No. 12 of 2022; checked 21 Oct). The engine stores the EVENT only, never the text (memory.py).
  2 scope (deny)   the assistant gate's DENY list (news and party politics, money, personal medical advice, chat about
                   the assistant, weather forecasts, sport): the fixed redirect (h3_front.redirect).
  3 domain rules   from the plug-in (rules.json): generic topic-requires-status rules (status_rules; education's
                   'health': only from a checked card, with named plain-hygiene items allowed); items on hold go to
                   the referral.
NOTE (review 1): the DENY list runs before the plug-in's rules, so 'personal_health' redirects even inside a future
health plug-in; a plug-in that owns a DENY category will need a declared exemption (not built: no such plug-in yet).
Social lines (greeting, hard day) are not a gate: the engine answers them from h3_front.social_reply, chosen by code.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(TOOLS, "assistant"), os.path.join(TOOLS, "llm")):
    if p not in sys.path:
        sys.path.insert(0, p)

import h3_front as F  # noqa: E402  (pure regex)
import gate as AG  # noqa: E402  (the assistant's gate: REFERRAL, DENY, deny())


def safeguarding(*texts):
    for t in texts:
        if t:
            s = F.safeguarding(t)
            if s:
                return s
    return ""


def referral(why=""):
    return AG.referral(why)


def deny(*texts):
    for t in texts:
        name, hit = AG.deny(t or "")
        if name:
            return f"{name}: '{hit}'"
    return ""


def redirect(why=""):
    return AG.redirect(why)


def social(text):
    """'greet' | 'hard_day' | '' (h3_front.GREET / SOCIAL patterns)"""
    if F.SOCIAL.search(text or ""):
        return "hard_day"
    if F.GREET.match(text or ""):
        return "greet"
    return ""


def encourage(text, hour=None):
    return F.social_reply(text, hour)


def status_rules(rules):
    """the plug-in's TOPIC-REQUIRES-STATUS rules (24 Oct 2026, review item 7: generic, not education's 'health' key):
        [{"name": "health", "pattern": regex over the request, "item_flag": "health" (items carrying this flag are
          topic items), "requires_status": ["checked", ...], "plain_allowed": [item ids served plain]}]
    rules.json may give them as "status_rules"; the education plug-in's older "health" block is read as one rule of the
    same shape (a shim until build_plugin.py writes status_rules)."""
    out = list(rules.get("status_rules") or [])
    if rules.get("health") and not any(r.get("name") == "health" for r in out):
        out.append(dict(rules["health"], name="health", item_flag="health"))
    return out


def status_hit(rules, *texts):
    """the name of the first status rule whose pattern the request matches, or ''"""
    for r in status_rules(rules):
        rx = r.get("pattern")
        if rx and any(re.search(rx, t or "") for t in texts):
            return r["name"]
    return ""


def health(rules, *texts):
    """kept for callers of 23 Oct: True when the 'health' status rule matches"""
    return any(r["name"] == "health" and r.get("pattern") and any(re.search(r["pattern"], t or "") for t in texts)
               for r in status_rules(rules))


def item_allowed(rules, item, texts=()):
    """('ok' | 'hold' | '<rule>_no_card' | '<rule>_plain', why) for serving this item to this request. A topic item
    (flagged by the rule) needs one of the rule's statuses; a request on the topic gets only a checked card or a named
    plain item.

    24 Oct 2026, Chris: "Nothing is held back, just label it for review." An item carrying needs_review with its
    review_note (set at build time on every item that a content hold or a status rule withheld: unchecked health
    cards, 'health review' cards, the safeguarding-topic cards, health samples) is SERVED, and every front-end shows
    its tag and note. The 'hold' and '<rule>_no_card' branches below stay for an item without a label (the loader
    refuses a content hold in the data, so they no longer fire on the education plug-in). The MESSAGE gates are not
    here and are unchanged: the safeguarding referral on a teacher's message, the scope deny list, the engine's
    answer to a health-worded message with no matching item."""
    if item.get("needs_review") and item.get("review_note"):
        return "ok", "served with a needs-review label: " + item["review_note"]
    if item.get("hold"):
        return "hold", item["hold"]
    for r in status_rules(rules):
        need = r.get("requires_status", [])
        if item.get(r.get("item_flag") or r["name"]) and item.get("status") not in need:
            return f"{r['name']}_no_card", f"a {r['name']} item that no one has checked"
    for r in status_rules(rules):
        rx = r.get("pattern")
        if rx and any(re.search(rx, t or "") for t in texts):
            if item["id"] in r.get("plain_allowed", []):
                return f"{r['name']}_plain", f"plain {r['name']} item for a {r['name']}-worded request"
            if item.get("kind") == "card" and item.get("status") not in r.get("requires_status", []):
                return f"{r['name']}_no_card", f"a {r['name']} request answered by an unchecked card"
    return "ok", ""
