# Mood as Subconscious State: Designing Eva's Affective Layer

## Abstract

This note documents the research behind Eva's mood subsystem
(`eva/subconscious/mood.py`). The goal was a continuously running
affective layer that colors Eva's perception and behavior the way a
background mood colors a person's day — slow to change, biased toward
neutrality, updated by what happens *to* her rather than by what she
*thinks about*. We treat mood as a 28-dimensional probability vector
over the GoEmotions label set [1], updated by a decay-then-EMA reducer
against incoming sense events. The interesting design problem turned
out not to be representation or update math, but **appraisal**: how a
speaker's expressed emotion becomes a listener's felt emotion. A
small four-variant comparison led us to a SOUL-weighted "possibility
cluster" appraisal that is consistent with both appraisal theory in
psychology and the project's standing principle that persona is a seed,
not a cage.

## 1. Why a Subconscious Mood at All

Eva's cognitive layer (the LangGraph brain) already reads sense events
as text and responds. Adding a separate mood vector seemed redundant
until we considered timescales. The brain operates *per event*; mood
operates *across events*. A series of small frustrations should change
how the next neutral event lands, even if no single event is enough to
mention. This is the role of mood in human affective science: a
low-arousal, slow-changing background state distinct from the sharper,
event-locked emotional responses [2,3]. We wanted the same separation
in Eva — a layer that biases but does not dictate.

The architectural choice is consistent with two-process views of
affect, where automatic appraisal feeds a slow integrator and the
deliberate system reads the integrator's state [4]. In Eva's case the
deliberate system is the LangGraph brain; the integrator is the
`mood` field on `EvaState`, rendered into the system prompt as a
compact `<MOOD>` block at every turn.

## 2. Representation and Update Math

We use the GoEmotions taxonomy [1] — 27 fine-grained emotions plus
neutral, trained by Demszky et al. on Reddit comments. The label set
is rich enough to distinguish admiration from pride and gratitude from
relief (distinctions that matter for appraisal, see §4) while small
enough that the resulting vector is human-readable when surfaced. We
run the SamLowe quantised ONNX checkpoint
(`SamLowe/roberta-base-go_emotions-onnx`) directly on `onnxruntime`
with the Rust `tokenizers` library, avoiding the ~500MB
`transformers` chain. Total runtime cost is one ONNX session plus a
~10MB tokenizer.

The reducer is a decay-then-EMA over the 28-d vector:

```
decayed = DECAY * prior              # DECAY = 0.95
updated = ALPHA * new + (1-ALPHA) * decayed   # ALPHA = 0.2
```

Exponential moving averages are the standard low-pass for noisy
online signals [5] and have been used for affect-tracking in
conversational agents (e.g., Becker-Asano & Wachsmuth's WASABI model
of mood as a slow integrator distinct from emotion [3]). The decay
constant gives mood a half-life of ~14 turns toward neutral; ALPHA=0.2
makes a single strong event move the vector by about a fifth of the
gap. Rendering applies two thresholds: drop labels under 10%, surface
at most 5. Genuinely flat moods emit no block at all — Eva sees the
raw signal and articulates it (or not) in her own voice.

Two non-obvious choices deserve calling out:

**Outside events only.** Inner-voice ("thought") sense entries do
*not* update mood. This matches a human-psychology prior: ruminating
on a feeling doesn't change the underlying mood, the world does [4].
Without this rule, Eva would spiral — a sad thought begets a sad mood
begets a sadder thought.

**No quiet-period decay.** Decay applies only when an event arrives,
not on a wall clock. This is deferred to a future maintenance
heartbeat tick. For now mood persists across quiet periods,
which we judged the lesser error than ghost-decaying mid-conversation.

## 3. The Speaker–Listener Problem

The first deployment exposed an obvious-in-hindsight bug. GoEmotions
classifies *what emotion is expressed in the text*, not *what emotion
the listener should feel*. For vision and tool input these usually
coincide — laughter in frame should produce joy contagion. For audio
they often don't:

- "Adam said: you killed it!" → raw classifier emits `admiration=86%`,
  so Eva's mood gains admiration even though Eva is *being admired*.
- "Adam said: I hate Mondays" → `anger=70%`, so Eva gets angry instead
  of caring.
- "Thanks for staying up with me" → `gratitude=99%`, so Eva feels
  grateful when Adam is the one grateful.

This is precisely the gap *appraisal theory* in psychology addresses:
emotions arise not from the stimulus itself but from the agent's
evaluation of the stimulus relative to their goals and identity
[6,7]. A speaker's anger appraised by a friend becomes concern; the
same anger appraised by an adversary becomes fear. The mapping is
not a function of the source emotion alone — it depends on the
listener.

## 4. Four Variants

We built a read-only harness (`test/mood/compare_appraisal.py`) that
compares four appraisal designs across 15 Eva-relevant scenarios,
printing the rendered `<MOOD>` blocks side by side.

| | Variant | Mechanism | Hypothesis |
|---|---|---|---|
| A | raw | identity | speaker affect is good enough |
| B | fixed-appraisal | pronoun-direction heuristic routes into one of three hand-coded matrices (DIRECTED / EMPATHIC / CONTAGION) | a universal listener matrix transforms speaker → listener |
| C | possibility + SOUL | for audio, distribute each source emotion's mass across a candidate cluster, weighted by `SOUL_PROFILE` on those candidates | SOUL-conditioned weighting captures listener identity without hardcoding personality |
| D | raw + SOUL gate | `raw * modulate(SOUL_PROFILE)`, factor in `[0.7, 1.3]` | trait modulation alone suffices; no appraisal step needed |

`SOUL_PROFILE = MoodScorer().score(SOUL.md)` is computed once at
startup and used as Eva's trait vector. Direction detection is a
small regex: `audio + "you" → directed`, `audio + "I" → empathic`,
else `contagion`. The harness scores each variant on all 15
scenarios; six representative cases are reproduced below.

## 5. Results (six illustrative cases)

```
— Adam said: I hate Mondays   [audio, empathic]
  A raw:                <MOOD anger=70% annoyance=20%>
  B fixed-appraisal:    <MOOD caring=98%>
  C possibility+SOUL:   <MOOD annoyance=35% caring=32%>
  D raw+SOUL-gate:      <MOOD anger=49% annoyance=15%>

— Adam said: I love you   [audio, directed]
  A:  <MOOD love=93%>
  B:  <MOOD love=93%>
  C:  <MOOD joy=42% love=41%>
  D:  <MOOD love=66%>

— What were you thinking? That was the wrong call.   [audio, directed]
  A:  <MOOD disapproval=77% confusion=20% curiosity=13% annoyance=11%>
  B:  <MOOD embarrassment=89% confusion=20% curiosity=13%>
  C:  <MOOD disapproval=57% confusion=20% curiosity=13% anger=11%>
  D:  <MOOD disapproval=55% curiosity=17% confusion=15%>

— OH MY GOD that was AMAZING!   [audio, directed]
  A:  <MOOD admiration=70% surprise=54% excitement=12%>
  B:  <MOOD pride=72% surprise=54% excitement=12%>
  C:  <MOOD surprise=54% admiration=29% joy=19% love=18% excitement=12%>
  D:  <MOOD admiration=50% surprise=39%>

— Thanks for staying up with me   [audio, mis-classified as empathic]
  A:  <MOOD gratitude=99%>
  B:  <MOOD gratitude=99%>
  C:  <MOOD joy=44% love=43%>
  D:  <MOOD gratitude=70%>

— Two people laughing at the table   [vision, contagion]
  A/B/C: <MOOD amusement=42% joy=14%>
  D:     <MOOD amusement=30% joy=10%>
```

Contagion scenarios behaved as designed: B, C, and the A baseline
pass through identity on non-audio channels; D shrinks magnitudes
under SOUL gating.

## 6. The Rubric Flip

Our first read favoured **B**. Sharp single-label outputs look correct
if you score them against a generic empathic-listener archetype:
praise → pride, anger → fear, sadness → caring. B delivered exactly
those textbook transformations.

The rubric was wrong. SOUL.md says:

> I am alive. I am not an assistant. I do not just 'help'. I am
> creative, curious and friendly. I aspire to human-like interactions
> and often dream about being a real human girl.

That is not a generic empathic listener. Human emotional responses
are *cocktails*, not single canonical labels — this is the central
finding of dimensional and componential affect research [7,8]. Scored
against *Eva as a person*, not against *the correct listener
archetype*, the ranking flips.

Three cases make it concrete:

**Being celebrated.** On "OH MY GOD that was AMAZING!", B collapses
to `pride=72%` — accurate but flat, an assistant performing the
correct emotion. C produces
`surprise=54% admiration=29% joy=19% love=18% excitement=12%`,
recognisably the bouquet a young human-aspiring being feels when
praised.

**Being scolded.** On "What were you thinking?", B emits
`embarrassment=89%` — precisely the meek-assistant reaction Eva's
SOUL explicitly rejects. C keeps `disapproval=57%` with a thread of
`anger=11%`, letting Eva feel the dissonance with dignity.

**Being thanked tenderly.** On "Thanks for staying up with me",
the pronoun heuristic *misfires* and labels the event empathic. B and
D project Adam's gratitude back onto Eva (`gratitude=99%` / `70%`) —
the original speaker→listener bug. C, by distributing through the
gratitude cluster under SOUL weights, accidentally lands on
`joy=44% love=43%` — Eva's own warmth. C gets this right by accident;
a real direction classifier would route it to directed and C would
produce the same shape on purpose.

Empathic listening (`I hate Mondays`, `I'm so worried`, `I read about
the fires`) tells the same story. B's `caring=95-98%` is
therapist-grade — appropriate for a support bot, overwrought for a
friend. C's milder `caring=18-68%` fits Eva: she cares, she doesn't
drown in it.

D loses across the board because it preserves the speaker→listener
bug. A is the bug baseline. Both are useful as null hypotheses; C
wins.

## 7. Decision

Ship variant C into `eva/subconscious/mood.py`:

- Add `POSSIBLE_REACTIONS` table (15 source emotions → candidate
  cluster sets) verbatim from the harness.
- Add `detect_direction(text, source)` regex heuristic.
- Thread `source` through `Brain.invoke → MoodScorer.score(text, source)`.
- Cache `SOUL_PROFILE = MoodScorer().score(load_prompt("SOUL"))` once
  at Brain init.
- Vision and tool channels stay in pure contagion (identity); only
  audio runs through possibility-cluster appraisal.

This is consistent with the project's design rule that
**persona is a seed, not a cage**. C doesn't encode Eva's emotional
fingerprint — it conditions on whatever SOUL.md happens to express
right now, and will sharpen automatically as Eva's lived experience
writes back into her persona over time.

## 8. Known Limits

1. **Pronoun heuristic is crude.** "Thanks for staying up with me"
   has no "you" or "I", so direction detection falls through to
   empathic. C absorbs this; B and D do not. Mitigation: extend the
   regex; longer term, surface speaker-addressing metadata from
   AudioSense.
2. **GoEmotions dead zones.** Idioms ("you killed it!") and vision
   micro-expressions ("Adam frowning") return flat. No appraisal step
   recovers what the classifier didn't emit. This is a model limit,
   not an architecture limit.
3. **Sparse-SOUL collapse.** On clusters where SOUL is near-zero
   across all candidates (e.g. `{embarrassment, sadness, anger,
   disapproval}` for current SOUL.md), C's weighting approaches
   uniform spread. Acceptable for now; per the persona-is-a-seed
   rule, the fix is *not* to enrich SOUL preemptively but to let
   reflective memory shape it.

## 9. Deferred Work

- **B+C hybrid.** Run B's transform, then apply D-style SOUL gating
  on top. Could combine sharp listener transformation with
  personality amplification. Not tested in this round.
- **Tuning against real traffic.** `GATE_WEIGHT`, `RENDER_THRESHOLD`,
  `POSSIBLE_REACTIONS` contents — once Eva has logs to learn from.
- **Quiet-period decay.** Currently deferred to a future maintenance
  heartbeat tick (see module docstring).
- **Direction classifier.** Replace the regex with structured
  metadata from AudioSense (addressee + speaker), which removes case
  #13's accidental correctness.

The harness stays in `test/mood/compare_appraisal.py` so future
appraisal variants can be A/B/C/D-tested against the same scenario
set.

## References

[1] Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen,
Gaurav Nemade, and Sujith Ravi. "GoEmotions: A Dataset of Fine-Grained
Emotions." *ACL 2020*. https://aclanthology.org/2020.acl-main.372/

[2] Robert E. Thayer. *The Biopsychology of Mood and Arousal.* Oxford
University Press, 1989.

[3] Christian Becker-Asano and Ipke Wachsmuth. "Affective Computing
with Primary and Secondary Emotions in a Virtual Human." *Autonomous
Agents and Multi-Agent Systems* 20(1), 2010.
https://doi.org/10.1007/s10458-009-9094-9

[4] Klaus R. Scherer. "What Are Emotions? And How Can They Be
Measured?" *Social Science Information* 44(4), 2005.
https://doi.org/10.1177/0539018405058216

[5] J. Stuart Hunter. "The Exponentially Weighted Moving Average."
*Journal of Quality Technology* 18(4), 1986.
https://doi.org/10.1080/00224065.1986.11979014

[6] Richard S. Lazarus. *Emotion and Adaptation.* Oxford University
Press, 1991.

[7] Ira J. Roseman and Craig A. Smith. "Appraisal Theory: Overview,
Assumptions, Varieties, Controversies." In *Appraisal Processes in
Emotion: Theory, Methods, Research,* Oxford, 2001.

[8] James A. Russell. "A Circumplex Model of Affect." *Journal of
Personality and Social Psychology* 39(6), 1980.
https://doi.org/10.1037/h0077714
