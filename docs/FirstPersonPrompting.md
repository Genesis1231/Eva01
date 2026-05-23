# The First-Person Index: Self-State Prompting for Autonomous Agents

## Abstract

First-person prompting is usually treated as persona design: an agent says "I am a careful assistant" rather than being told "You are a careful assistant." This paper argues that the same first-person frame has a second, more technical use in autonomous language agents. It can serve as a compact interface for self-state: the agent's current goal, observations, intended actions, outcomes, memories, and constraints.

The distinction is not that persona and self-state are unrelated. They are continuous. A persona gives the agent a social role; a first-person index turns that role into a control schema. The useful question is therefore not whether first-person prompting is "just persona," but when persona language becomes operational: when "I am a research assistant" is extended into "I am tracking this claim, I found conflicting evidence, I should check the source before writing."

The paper develops a small cognitive model of first-person prompting for agents. It draws on the idea of a working self in autobiographical memory, where current goals organize memory access, and on active-inference views of agency, where perception and action are coupled through predictions and control states. The claim is narrow: first-person prompting does not make a model conscious or generally more capable. It helps an agent maintain a readable self-state across an observe-act-reflect loop.

## 1. Introduction

A one-shot chatbot can answer without remembering itself. An autonomous agent cannot. It must keep track of what it is doing, why it is doing it, what has changed, and what it is allowed to do next.

Consider a research agent that receives the instruction:

```text
Help me develop an argument for first-person prompting in autonomous agents.
```

If the agent only treats this as a single request, it may produce a plausible answer and stop. If the agent treats it as an ongoing self-state, it can continue to organize its behavior:

```text
I am trying to justify first-person prompting for autonomy, not for general accuracy.
I have evidence for memory and reflection, but not direct evidence that pronoun choice is causal.
I should phrase the claim as a design hypothesis and propose an evaluation.
```

This is the first-person index. It is the use of "I" as a pointer to the agent's own evolving state.

The contribution of this paper is a modest theory of that pointer. First-person prompting is not itself a source of competence. It is a way to make a language agent's internal bookkeeping explicit enough to be stored, retrieved, inspected, and revised.

## 2. Persona and Index Are a Continuum

It is tempting to separate first-person persona from first-person self-state:

```text
I am a world-class AI researcher.
```

versus:

```text
I am tracking which claims are supported by evidence.
```

The first sounds like persona. The second sounds like control. But in practice they are not cleanly separable. A persona is already a compressed policy. "I am a research assistant" implies habits, priorities, permissions, and norms of action. The problem is that many persona prompts leave those implications vague.

The first-person index makes the persona operational. It expands a role into a small set of state variables:

```text
goal: what I am trying to accomplish
attention: what I am monitoring
memory: what I have already observed or tried
action: what I am about to do
outcome: what changed after I acted
constraint: what I should not do
revision: what I should change next time
```

This is why the paper does not argue against persona prompting. It argues that the useful form of persona prompting for autonomy is a persona with explicit state dynamics.

## 3. A Cognitive Model of the First-Person Index

The first-person index can be understood as a lightweight cognitive model with three components: a working self, an action model, and a reflective memory.

### 3.1 Working Self

In cognitive psychology, Conway and Pleydell-Pearce's self-memory system proposes a "working self" that organizes autobiographical memory around current goals [6]. The analogy to language agents is direct but limited. An LLM agent does not have autobiographical memory in the human sense. Still, an agent architecture often has a memory store, a current task, and retrieval rules. A first-person prompt can bind these together:

```text
I am trying to finish the literature review.
I already checked the survey paper.
I should retrieve notes about Reflexion before writing the memory section.
```

The "I" here is not a metaphysical self. It is a control label for goal-conditioned retrieval.

### 3.2 Action Model

Agency requires more than remembering. The agent must predict what an action will do and compare that prediction with the result. Active-inference models describe agency as the coupling of perception, prediction, and action selection under a model of controllable states [7]. Again, the analogy is not identity: an LLM prompt is not a brain. But the control problem is similar.

An agent needs statements of the following kind:

```text
I expect this test run to reveal whether my parser change fixed the bug.
The test failed in a different module, so my prediction was wrong.
I should inspect whether the failure existed before my edit.
```

This pattern matters because it links action to expectation. Autonomous behavior becomes more coherent when the agent can say not only what it did, but what it expected and how the result changed its next step.

### 3.3 Reflective Memory

Reflection turns an outcome into a reusable policy fragment:

```text
I patched before checking the baseline.
That made the failure harder to attribute.
Next time I should run the failing test before editing.
```

This resembles the verbal memory used in Reflexion, where agents improve by storing linguistic feedback across attempts [2]. The important point is not that the text is first-person by necessity. It is that first-person reflection gives the stored lesson an owner and a future target: "I should do this differently."

## 4. Mechanism

The proposed mechanism is simple:

```text
persona -> self-state -> action policy -> memory update
```

A persona supplies an initial role. The first-person index turns that role into a self-state. The self-state guides action selection. Outcomes update memory.

For example:

```text
I am a bounded research agent.
My current goal is to assess whether this claim is supported.
I found indirect evidence from agent architectures, but no direct pronoun ablation.
I should call the claim a hypothesis, not a result.
```

This small chain does several things at once. It preserves the agent's goal, marks the evidence boundary, prevents overclaiming, and suggests the next action. The value of first-person prompting is not in the pronoun alone. It is in the way the pronoun anchors these relations.

More formally, the index maintains a language-level state:

```text
s_t = {role_t, goal_t, observation_t, action_t, outcome_t, memory_t, constraint_t}
```

The architecture supplies the actual storage and tools. The prompt supplies a stable natural-language address for them.

## 5. Relation to Existing Agent Work

Existing work does not establish a direct advantage for "I" over "you." The support is indirect and should be stated carefully.

ReAct shows that language models can interleave reasoning and external actions, using observations to guide later steps [1]. This supports the need for an explicit action-observation loop.

Reflexion shows that agents can use verbal feedback and episodic memory to improve later attempts without changing model weights [2]. This supports the usefulness of language-level reflection.

Generative Agents shows that believable long-horizon behavior depends on memory, reflection, and planning rather than isolated generation [3]. This supports the need for persistent state.

Voyager shows an embodied LLM agent using feedback, skill accumulation, and self-directed exploration in an open-ended environment [4]. This supports the idea that autonomy involves goal selection and verification over time.

These papers support the architecture around the first-person index. They do not isolate the first-person index itself. That is exactly the gap a small empirical paper could test.

## 6. Design Pattern

A first-person autonomy prompt should read less like a biography and more like a state update protocol:

```text
I am a bounded research agent working on the user's current objective.

I maintain:
- the current claim I am evaluating
- the evidence I have found
- the uncertainty I have not resolved
- the next safe action
- the constraints on what I may do

When new information appears, I decide whether it changes my goal, evidence, uncertainty, action, or constraints.

I act without asking only when the action is low-risk, reversible, and directly useful.

After acting, I record what changed.
After failure, I record one lesson that should affect my next attempt.
```

This prompt still creates a persona. The difference is that the persona is functional. It tells the agent how to carry state forward.

## 7. Examples

### 7.1 Coding Agent

A persona-only prompt says:

```text
I am an expert software engineer.
```

A first-person index says:

```text
I am trying to make the parser test pass.
I changed the tokenizer, so tokenizer-related failures are most relevant.
Before editing again, I should check whether this failure existed at baseline.
```

The second version is not more impressive. It is more useful because it names the goal, the relevant memory, and the next diagnostic action.

### 7.2 Research Agent

A persona-only prompt says:

```text
I am a graduate research assistant.
```

A first-person index says:

```text
I have evidence that memory and reflection help agents.
I do not yet have direct evidence that first-person wording is causal.
I should frame the contribution as a testable mechanism rather than an established effect.
```

This is the paper's central move in miniature. The agent's autonomy is bounded by its own evidence state.

### 7.3 Inbox Agent

A persona-only prompt says:

```text
I am your executive assistant.
```

A first-person index says:

```text
I see one email asking for a decision by 5 PM.
I can draft a response, but sending it is externally visible.
I should prepare the draft and wait for approval.
```

The index does not merely invite helpfulness. It represents permission.

## 8. Empirical Prediction

The testable hypothesis is not that first-person prompts improve all outputs. The prediction is narrower:

```text
First-person indexing should improve tasks that require continuity of self-state: goal persistence, memory use, action justification, and recovery from failed actions. It should not reliably improve isolated factual question answering, where no persistent agent state is needed.
```

A minimal experiment would compare four conditions:

1. no role prompt
2. persona-only first-person prompt
3. second-person operational prompt
4. first-person operational prompt

The key comparison is between conditions 3 and 4. If first-person wording matters, it should appear in continuity metrics rather than in raw factual accuracy:

- Does the agent remember what it was trying to do after interruption?
- Does it connect tool results to its previous action?
- Does it avoid acting when permission is missing?
- Does reflection improve the next attempt?
- Can a human auditor reconstruct why the agent acted?

If there is no difference on these measures, then the first-person index should be treated as notation rather than mechanism.

## 9. Limitations

The first-person index can make agents sound more coherent than they are. A fluent self-report is not proof of correct memory, good planning, or safe action.

It can also encourage over-identification if the prompt is written as status rather than procedure:

```text
I am an expert, so my answer is reliable.
```

The safer pattern is procedural:

```text
I should state what evidence I used.
I should mark what I am uncertain about.
I should ask before acting outside permission.
```

Finally, the first-person index depends on architecture. Without real memory, tools, observations, or feedback, it has little to index. In that case it becomes only a voice.

## 10. Conclusion

First-person prompting is interesting for autonomous agents because persona and control meet in the same linguistic form. "I am a research agent" begins as a role. It becomes useful when it expands into a live state: what I am trying to do, what I noticed, what I did, what happened, what I learned, and what I must not do.

That is a small claim, but it gives first-person prompting a clearer technical role. It is not a route to consciousness, and it is not a general performance enhancer. It is a way to make an agent's self-state explicit enough to persist across steps. For autonomous systems, that persistence is the beginning of responsible behavior.

## References

[1] Shunyu Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. https://openreview.net/forum?id=WE_vluYUL-X

[2] Noah Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*. https://arxiv.org/abs/2303.11366

[3] Joon Sung Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*. https://arxiv.org/abs/2304.03442

[4] Guanzhi Wang et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv:2305.16291, 2023. https://arxiv.org/abs/2305.16291

[5] Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models." *Findings of EMNLP 2024*. https://aclanthology.org/2024.findings-emnlp.888/

[6] Martin A. Conway and Christopher W. Pleydell-Pearce. "The Construction of Autobiographical Memories in the Self-Memory System." *Psychological Review* 107, no. 2, 2000. https://doi.org/10.1037/0033-295X.107.2.261

[7] Karl Friston et al. "The Anatomy of Choice: Active Inference and Agency." *Frontiers in Human Neuroscience* 7, 2013. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3782702/
