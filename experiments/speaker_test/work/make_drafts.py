"""WhatsApp drafts for the speaker test, desk format (24 Oct 2026, the Jesuit). NOTHING is sent from here.

*cd first line; about 8 lines a message; "our translator", never "synthetic"; English then our Tonga; ok / small /
wrong with the pay; message 1 explains the task. A message is written only when every line in it is translated
(served.jsonl); re-run as the translation goes on. The key (message, line -> item or control) is
drafts/KEY_lines.json: it holds item ids, no Tonga of ours that is not already in the message.
    python -X utf8 make_drafts.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
D = os.path.join(RUN, "drafts")
NAME = {"T01": "Father", "T09": "T09"}
INTRO = ("*cd {name}, a new task: we are testing our Tonga translator on lines a Grade 1 to 4 teacher says or writes. "
         "I will send about 100 short lines over the next days, about 8 in each message: the English, then our translator's "
         "Tonga. For each line please reply:\n"
         "ok (K1) if the Tonga is right, as you would accept it in class;\n"
         "small: your line (K2) if it needs a small fix;\n"
         "wrong: your line (K2) if the meaning is wrong.\n"
         "'You' in the English can mean one child or the whole class: either is fine unless the line says. "
         "Your answers show us exactly where the translator makes mistakes. Up to K16 a message.\n")
HEAD = "*cd {name}, translator test, message {k} of {n}: ok (K1), or small: your line (K2), or wrong: your line (K2). Up to K{cap}.\n"


def main():
    sp = json.load(open(os.path.join(RUN, "split.json"), encoding="utf-8"))
    sv = {}
    p = os.path.join(RUN, "served.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                sv[r["id"]] = r
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RUN, "pool.jsonl"), encoding="utf-8")}
    os.makedirs(D, exist_ok=True)
    key = {}
    made = []
    for spk in ("T01", "T09"):
        ctl = {c["cid"]: c for c in sp["plan"][spk]["controls"]}
        msgs = sp["plan"][spk]["messages"]
        for k, m in enumerate(msgs, 1):
            lines, kl, ready = [], [], True
            for i, x in enumerate(m, 1):
                if x in ctl:
                    c = ctl[x]
                    if c["kind"] == "repeat":
                        en, src = pool[c["of"]]["en"], c["of"]
                        toi = sv.get(c["of"], {}).get("toi")
                    else:
                        en, src = pool[c["en_from"]]["en"], c["toi_from"]
                        toi = sv.get(c["toi_from"], {}).get("toi")
                    kl.append({"line": i, "control": x, "kind": c["kind"], "en_id": c.get("of") or c.get("en_from"),
                               "toi_id": src})
                else:
                    en, toi = pool[x]["en"], sv.get(x, {}).get("toi")
                    kl.append({"line": i, "item": x, "shared": x in sp["shared"]})
                if not toi:
                    ready = False
                    break
                lines.append(f"{i}. {en} = {toi}")
            if not ready:
                continue
            name = NAME[spk]
            head = INTRO.format(name=name) if k == 1 else HEAD.format(name=name, k=k, n=len(msgs), cap=2 * len(m))
            txt = head + "\n".join(lines) + "\n"
            if k == 1:
                txt += "\nExample reply: 1 ok, 2 small: <your line>, 3 wrong: <your line>. Thank you!\n"
            fn = os.path.join(D, f"{spk}_msg{k:02d}.txt")
            open(fn, "w", encoding="utf-8", newline="\n").write(txt)
            key[f"{spk}_msg{k:02d}"] = kl
            made.append(fn)
    json.dump(key, open(os.path.join(D, "KEY_lines.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(made), "drafts:", ", ".join(os.path.basename(f) for f in made))


if __name__ == "__main__":
    main()
