VoE + ToM Memory Loop — Consolidated SPEC (v 0.2)

⸻

1. Purpose

Create a conversational agent that (1) predicts the user’s next move, (2) detects Violation-of-Expectation (VoE) when it is wrong, (3) distils every surprise into a short fact, and (4) injects only the relevant facts back into future turns.

⸻

2. Conceptual Stack

Layer	Responsibility
Prediction (ToM)	Make a blind guess about the next user utterance and list “needed info” keywords.
VoE Logic	Compare last prediction with actual user text; if different, derive & store a fact. (No LLM call.)
Fact Store	Per-user, append-only ring buffer (max N facts). Facts have id · text · tags · created_at.
Retrieval	Fuzzy/keyword match between needed-info list and fact tags/text; return ≤ K facts.
Response	Craft visible reply, using chat history + retrieved facts, and output the next prediction for the following turn.


⸻

3. Data Model

Fact:
  id          UUID
  text        string          # distilled surprise or preference
  tags        list[string]    # 1-5 noun phrases
  created_at  timestamp

Memory:  dict[user_id] -> list[Fact]   # newest-last, capped at MAX_FACTS


⸻

4. Per-Turn Control Flow (two LLM calls)

①   user_msg arrives  →  history += (user, user_msg)

②   VoE CHECK (local)
     if mismatch(last_prediction, user_msg):
         fact = distil_fact(last_prediction, user_msg)
         memory[user].append(fact)

③   PREDICTOR  (LLM call #1, sees history only)
     outputs:
         PREDICTION_NEXT
         NEEDED_INFO [kw1, kw2, …]

④   RETRIEVE  (local)
     facts = fuzzy_match(memory[user], NEEDED_INFO)

⑤   RESPONDER  (LLM call #2, sees history + facts)
     outputs:
         REPLY_TEXT       # spoken to user
         NEXT_PREDICTION  # cached for next VoE check

⑥   send_to_user(REPLY_TEXT)
     history += (assistant, REPLY_TEXT)
     last_prediction = NEXT_PREDICTION


⸻

5. Pseudocode Implementation Skeleton

def handle_turn(user, user_msg):
    history[user].append(("user", user_msg))

    # -- 2  VoE -------------------------------------------------
    if last_prediction[user] and mismatch(last_prediction[user], user_msg):
        memory[user].append(distil_fact(last_prediction[user], user_msg))

    # -- 3  Predictor ------------------------------------------
    needed_info, pred = LLM_predictor(history[user])     # CALL 1

    # -- 4  Retrieval ------------------------------------------
    facts = retrieve(memory[user], needed_info)

    # -- 5  Responder ------------------------------------------
    reply, next_pred = LLM_responder(history[user], facts)  # CALL 2

    send_to_user(reply)
    history[user].append(("assistant", reply))
    last_prediction[user] = next_pred


⸻

6. Key Algorithms

mismatch(a, b) → Boolean
  • Use fuzzy ratio < THRESHOLD or intent classifier.

distil_fact(pred, actual) → string
  • LLM prompt: “Write one sentence describing what you learned from the difference.”

retrieve(mem, needed_info[]) → list[Fact]
  • For each fact, score = max{fuzzy(f.text, kw), fuzzy(tag, kw)}.
  • Return top K by score, newest-first.

⸻

7. Testing Plan (minimal)
	•	Hypothesis H₁: Predictor + VoE loop raises prediction-accuracy ≥ 20 % vs. baseline.
	•	Metric: turn-level match rate (similarity ≥ 0.6).
	•	A/B: 60 convos per arm, χ² test. Pass if p < 0.05 and improvement ≥ 20 %.

⸻

8. Config Knobs

Name	Default	Note
MAX_FACTS per user	200	Ring-buffer trim oldest.
FUZZY_THRESHOLD	60	Below ⇒ VoE triggered.
TOP_K facts	10	Injected into responder prompt.


⸻

This spec now bundles: architecture, data model, turn-loop pseudocode, algorithm sketches, and test criteria—all the essentials to build and evaluate the Predictor + Responder VoE system.