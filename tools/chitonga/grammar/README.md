# Chitonga grammar engine (from Hopgood, 2nd ed., ZEPH 1992)

`tonga.py` (verbs) and `concords.py` (noun-side agreement) turn the rules of Hopgood's *A Practical Introduction to
Chitonga* into code. Every rule cites its lesson and page. `verify.py` tests them.

    python -X utf8 tools/chitonga/grammar/verify.py
    python tools/chitonga/grammar/tonga.py leta        # all tenses of one verb

## What it does
- Verbs, 14 forms: present (-la-), -a- tense, remote past (-ka-), futures (-zoo-, -yoo-), negative (ta- ... -i),
  negative future (-koo-), perfect (-ide/-ede, irregulars as printed), negative perfect, negative pluperfect (tee-),
  negative future perfect (-koozoo-), past continuous and pluperfect (ka- + concord), subjunctive (-e).
- Sound rules: n+l=nd, n+y=nj, nasal assimilation, ng/nj before vowels (L22-24); ta+u=to, ta+a=ta, ta+i=tii (L18);
  ka+u=ko (L67); a+i=e in possessives (wa+isu=wesu, L5); mu- to mw- before a vowel (L4); the i of im-/in-/iŋ- nouns
  drops after a particle (L7).
- Nouns: the whole Table of Concords (Appendix B) as data: subject and object concords, genitive, possessives,
  demonstratives (4 types), adjectives, numerals, pronouns, with dialect variants kept as alternatives.

## Tests (11 Oct 2026)
| test | result |
|---|---|
| printed paradigms in the notes (consistency: rules were written from these) | 113/113 |
| INDEPENDENT: verb forms with marked joins in the drill sentences | **172/172** |
| INDEPENDENT: noun + possessive phrases in the drills (class from the noun's prefix) | **40/42** |

## Open questions for a speaker (the book is not consistent, or a form is unclear)
- Class 6 possessive: *a-ngu* (Ma-twi a-ngu) beside *aa-ko*, *aa-nu* in other drills; the table prints *aa-*.
- Negative pluperfect: the book prints *Teewakabwene* for both "thou hadst not seen" and "he had not seen" (p290).
- Misprints listed in vision_notes/appendix_b_concords.json ("unreadable"), e.g. *nbubo* (class 14), *nduluiya*.
- Forms given only as examples, not full paradigms: future perfect (L68), subjunctive (L70), remote negative future.

A form the engine produces is a PREDICTION until a speaker confirms it.
