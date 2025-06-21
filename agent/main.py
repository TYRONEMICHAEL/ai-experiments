import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from livekit import rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, ChatRole
from livekit.agents.log import logger
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()


# ---------------------------------------------------------------------------
#  (1)  ────  Data Model & Memory  ──────────────────────────────
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    id: str
    text: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


# In-memory store: { user_id -> [Fact, …] }
# This is GLOBAL and persists across agent reconnections
_GLOBAL_MEMORY: dict[str, List[Fact]] = defaultdict(list)
_MAX_FACTS_PER_USER = 200  # Ring-buffer cap
_MIN_FUZZY_SCORE = 60.0  # 0-100 for retrieval
_FACTS_FILE = "agent_facts.json"  # File to persist facts


def _fuzzy_score(a: str, b: str) -> float:
    """0-100 similarity score."""
    return fuzz.ratio(a.lower(), b.lower())


def _save_facts_to_file():
    """Save facts to a JSON file for persistence across process restarts."""
    try:
        facts_data = {}
        for user_id, facts in _GLOBAL_MEMORY.items():
            facts_data[user_id] = [
                {
                    "id": f.id,
                    "text": f.text,
                    "created_at": f.created_at.isoformat(),
                    "tags": f.tags,
                }
                for f in facts
            ]
        
        with open(_FACTS_FILE, 'w') as f:
            json.dump(facts_data, f, indent=2)
        logger.info(f"Saved {sum(len(facts) for facts in _GLOBAL_MEMORY.values())} facts to {_FACTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save facts to file: {e}")


def _load_facts_from_file():
    """Load facts from JSON file on startup."""
    try:
        if os.path.exists(_FACTS_FILE):
            with open(_FACTS_FILE, 'r') as f:
                facts_data = json.load(f)
            
            for user_id, fact_list in facts_data.items():
                _GLOBAL_MEMORY[user_id] = [
                    Fact(
                        id=f["id"],
                        text=f["text"],
                        created_at=datetime.fromisoformat(f["created_at"]),
                        tags=f["tags"],
                    )
                    for f in fact_list
                ]
            
            total_facts = sum(len(facts) for facts in _GLOBAL_MEMORY.values())
            logger.info(f"Loaded {total_facts} facts from {_FACTS_FILE}")
        else:
            logger.info(f"No facts file found at {_FACTS_FILE} - starting fresh")
    except Exception as e:
        logger.error(f"Failed to load facts from file: {e}")


# Load facts on module import
_load_facts_from_file()


# ---------------------------------------------------------------------------
#  (2)  ────  Agent Class & Core Logic  ──────────────────────────────
# ---------------------------------------------------------------------------


class ToMAgent(VoicePipelineAgent):
    def __init__(
        self,
        voe_enabled: bool,
        room: rtc.Room,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.room = room
        self.voe_enabled = voe_enabled
        self.last_prediction: Optional[str] = None
        self.is_speaking = False
        self.first_user_interaction = True  # Track if this is the first user interaction

        if self.voe_enabled:
            logger.info("VoE/ToM logic is ENABLED.")
            # Use the GLOBAL memory store instead of per-instance
            self.memory = _GLOBAL_MEMORY
            # Debug: Log what's in memory at startup
            total_facts = sum(len(facts) for facts in self.memory.values())
            logger.info(f"AGENT STARTUP: Found {total_facts} total facts in memory")
            for user_id, facts in self.memory.items():
                logger.info(f"  User '{user_id}': {len(facts)} facts")
                for i, fact in enumerate(facts):
                    logger.info(f"    Fact {i+1}: {fact.text}")
        else:
            logger.info("VoE/ToM logic is DISABLED (baseline mode).")

    # -- VoE/ToM Logic Steps (from SPEC.md) --

    async def _voe_check(self, user_id: str, user_msg: str):
        """(Step 2) VoE CHECK: If prediction was wrong, distill a new fact."""
        if self.last_prediction and user_msg:
            score = _fuzzy_score(self.last_prediction, user_msg)
            logger.info(f"VoE CHECK: prediction='{self.last_prediction}', actual='{user_msg}', score={score:.2f}")
            if score < 50.0:
                logger.info(f"VoE TRIGGERED! Score {score:.2f} < 50.0")
                fact_text = await self._distil_fact(self.last_prediction, user_msg)
                await self._store_fact(user_id, fact_text)
                logger.info(f"VoE: Created and stored new fact: {fact_text}")
            else:
                logger.info(f"VoE: No mismatch detected, score {score:.2f} >= 50.0")
        else:
            logger.info(f"VoE CHECK: Skipped - last_prediction='{self.last_prediction}', user_msg='{user_msg}'")

    async def _predictor(self) -> Tuple[str, List[str]]:
        """(Step 3) PREDICTOR: Guess next user reply and needed info."""
        predictor_ctx = ChatContext(messages=self.chat_ctx.messages.copy())
        predictor_ctx.messages.append(
            ChatMessage(
                role="system",
                content=(
                    "Based on the conversation so far, what is your best guess for the "
                    "user's next reply? Also, provide 1-3 keywords of information that "
                    "would be most useful to formulate your own response.\n\n"
                    "You MUST respond with valid JSON in this exact format:\n"
                    '{"next_prediction": "your prediction here", "needed_info": ["keyword1", "keyword2"]}\n\n'
                    "Do not include any other text, just the JSON."
                ),
            )
        )

        try:
            stream = self.llm.chat(chat_ctx=predictor_ctx)
            json_str = ""
            async for c in stream:
                if c.choices and c.choices[0].delta.content:
                    json_str += c.choices[0].delta.content
            
            json_str = json_str.strip()
            logger.info(f"Predictor raw response: {json_str}")
            
            if not json_str:
                logger.error("Predictor returned empty response")
                return "I'm not sure what you'll say next.", ["general"]
            
            # Try to parse JSON
            data = json.loads(json_str)
            prediction = data.get("next_prediction", "")
            needed_info = data.get("needed_info", [])
            
            if not prediction:
                prediction = "I'm not sure what you'll say next."
            if not needed_info:
                needed_info = ["general"]
                
            logger.info(f"Predictor output: pred='{prediction}', needed={needed_info}")
            return prediction, needed_info
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in predictor: {e}, raw content: '{json_str}'")
            return "I'm not sure what you'll say next.", ["general"]
        except Exception as e:
            logger.error(f"Error in predictor stage: {e}", exc_info=True)
            return "I'm not sure what you'll say next.", ["general"]

    def _retrieve(self, user_id: str, needed_info: List[str]) -> List[Fact]:
        """(Step 4) RETRIEVE: Get top K facts based on needed_info keywords."""
        lst = self.memory.get(user_id, [])
        if not lst or not needed_info:
            return []

        ranked: List[Tuple[float, Fact]] = []
        for f in lst:
            score_text = max(
                (_fuzzy_score(kw, f.text) for kw in needed_info), default=0
            )
            score_tag = (
                max(
                    (_fuzzy_score(kw, t) for kw in needed_info for t in f.tags),
                    default=0,
                )
                if f.tags
                else 0
            )
            score = max(score_text, score_tag)
            if score >= _MIN_FUZZY_SCORE:
                ranked.append((score, f))

        ranked.sort(key=lambda s: (s[0], s[1].created_at), reverse=True)
        return [f for _, f in ranked[:10]]

    async def _responder_with_facts(self, facts: List[Fact], is_initial_greeting: bool = False) -> str:
        """(Step 5) RESPONDER: Generate reply using chat history + retrieved facts."""
        responder_ctx = ChatContext(messages=self.chat_ctx.messages.copy())

        if facts:
            fact_content = " | ".join(f.text for f in facts)
            responder_ctx.messages.insert(
                0,
                ChatMessage(
                    role="system", content=f"[Relevant facts from memory] {fact_content}"
                ),
            )

        if is_initial_greeting:
            # Special prompt for conversation start when we have facts
            system_prompt = (
                "You are a voice assistant with memory. You have been provided with facts from previous conversations. "
                "This is the start of a new conversation session. Greet the user warmly and subtly show you remember them "
                "by referencing something from your memory (e.g., asking a follow-up question about a past topic, "
                "mentioning something they were interested in). Keep it natural and conversational. "
                "Then respond to their current message."
            )
        else:
            # Normal conversation prompt
            system_prompt = (
                "You are a voice assistant with a memory. You have been provided with relevant facts from previous conversations. "
                "If there are any facts, use them to subtly show you remember the user (e.g., asking a follow-up question about a past topic). "
                "Then, respond to their most recent message based on the full conversation history. "
                "Keep your response conversational and engaging."
            )

        responder_ctx.messages.append(
            ChatMessage(role="system", content=system_prompt)
        )

        stream = self.llm.chat(chat_ctx=responder_ctx)
        reply_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                reply_content += chunk.choices[0].delta.content

        logger.info(f"Responder generated: {reply_content}")
        return reply_content.strip()

    # -- Baseline (non-VoE) Logic --

    async def _baseline_responder(self) -> str:
        """A simple, single-LLM-call response for baseline measurement."""
        ctx = ChatContext(messages=self.chat_ctx.messages.copy())
        ctx.messages.append(
            ChatMessage(
                role="system",
                content="You are a helpful voice assistant. Generate a spoken reply to the user.",
            )
        )
        return await self._llm_to_string(ctx)

    # -- Helper Methods --

    async def _store_fact(self, user_id: str, text: str, tags: Optional[List[str]] = None):
        """Stores a new fact in memory for a specific user."""
        fact = Fact(id=str(uuid.uuid4()), text=text, tags=tags or ["voe"])
        lst = self.memory[user_id]
        lst.append(fact)
        logger.info(f"STORED FACT for user {user_id}: {text}")
        logger.info(f"Total facts for user {user_id}: {len(lst)}")
        if len(lst) > _MAX_FACTS_PER_USER:
            self.memory[user_id] = lst[-_MAX_FACTS_PER_USER:]
            logger.info(f"Trimmed facts to {_MAX_FACTS_PER_USER}")

        # Save to file for persistence
        _save_facts_to_file()
        
        await self._update_participant_attributes(user_id)

    async def _update_participant_attributes(self, user_id: str):
        """Updates the local participant attributes with the latest facts for the UI."""
        facts = self.memory.get(user_id, [])
        logger.info(f"UI UPDATE: Updating participant attributes with {len(facts)} facts")
        facts_list = [
            {
                "id": f.id,
                "text": f.text,
                "created_at": f.created_at.isoformat(),
                "tags": f.tags,
            }
            for f in facts
        ]

        facts_json = json.dumps(facts_list)
        logger.info(f"UI UPDATE: Facts JSON ({len(facts_json)} chars): {facts_json[:200]}...")

        try:
            # Try the correct method for updating attributes
            attributes = {"voe_facts": facts_json}
            await self.room.local_participant.set_attributes(attributes)
            logger.info(f"UI UPDATE: Successfully sent {len(facts_list)} facts to UI using set_attributes")
            # Verify the attributes were set
            current_attrs = self.room.local_participant.attributes
            logger.info(f"UI UPDATE: Current participant attributes keys: {list(current_attrs.keys())}")
            if "voe_facts" in current_attrs:
                logger.info(f"UI UPDATE: voe_facts attribute is present, length: {len(current_attrs['voe_facts'])}")
            else:
                logger.warning("UI UPDATE: voe_facts attribute not found in current attributes")
        except Exception as e:
            logger.error(f"UI UPDATE: Failed with set_attributes: {e}")
            try:
                # Fallback: try update_attributes
                await self.room.local_participant.update_attributes({"voe_facts": facts_json})
                logger.info(f"UI UPDATE: Successfully sent {len(facts_list)} facts to UI using update_attributes")
            except Exception as e2:
                logger.error(f"UI UPDATE: Failed with update_attributes too: {e2}")
                # Try direct metadata update
                try:
                    self.room.local_participant.attributes["voe_facts"] = facts_json
                    logger.info(f"UI UPDATE: Successfully set {len(facts_list)} facts via direct attribute assignment")
                except Exception as e3:
                    logger.error(f"UI UPDATE: All methods failed: {e3}")

    async def _distil_fact(self, prediction: str, actual: str) -> str:
        """Generate a detailed fact from the mismatch between prediction and reality."""
        distil_ctx = ChatContext()
        distil_ctx.messages.append(
            ChatMessage(
                role="system",
                content=(
                    "You are analyzing a conversation where an AI assistant made a prediction about what a user would say next, "
                    "but the user said something different. Your job is to extract a detailed, specific fact about the user "
                    "that can be used to improve future conversations.\n\n"
                    "Create a comprehensive fact that includes:\n"
                    "- What the user is interested in or cares about\n"
                    "- Their preferences, opinions, or attitudes\n"
                    "- Relevant context about their situation or background\n"
                    "- Specific details that would help personalize future interactions\n"
                    "- Any emotional tone or communication style preferences\n\n"
                    "Make the fact detailed and actionable. Instead of 'user likes music', write something like "
                    "'user is passionate about jazz music, particularly enjoys Miles Davis and John Coltrane, "
                    "and is currently learning to play saxophone'.\n\n"
                    "Write exactly one detailed sentence that captures the most important insight about the user."
                ),
            )
        )
        distil_ctx.messages.append(
            ChatMessage(
                role="user",
                content=f"PREDICTED: {prediction}\nACTUAL: {actual}\n\nWhat detailed fact about the user can you extract from this mismatch?"
            )
        )

        try:
            stream = self.llm.chat(chat_ctx=distil_ctx)
            fact_text = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    fact_text += chunk.choices[0].delta.content
            
            fact_text = fact_text.strip()
            logger.info(f"Distilled detailed fact: {fact_text}")
            return fact_text
        except Exception as e:
            logger.error(f"Error distilling fact: {e}", exc_info=True)
            return f"User said '{actual}' when expected '{prediction}' - unexpected response pattern"

    async def _llm_to_string(self, context: ChatContext) -> str:
        """Helper to get a string response from the LLM."""
        try:
            stream = self.llm.chat(chat_ctx=context)
            return "".join(
                [
                    c.choices[0].delta.content
                    async for c in stream
                    if c.choices and c.choices[0].delta.content
                ]
            )
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}", exc_info=True)
            return "I'm having a little trouble thinking right now."


# ---------------------------------------------------------------------------
#  (3)  ────  Agent Entrypoint & Setup  ─────────────────────────────────────
# ---------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    """
    This is the main entrypoint for the agent. It sets up the agent,
    memory, and event handlers, and then starts the conversation.
    """
    voe_enabled = os.getenv("VOE_ENABLED", "true").lower() == "true"

    # Create initial context with facts if they exist
    user_id = "default_user"
    existing_facts = _GLOBAL_MEMORY.get(user_id, [])
    
    if voe_enabled and existing_facts:
        logger.info(f"AGENT CREATION: Injecting {len(existing_facts)} facts into system prompt")
        fact_content = " | ".join(f.text for f in existing_facts)
        initial_ctx = ChatContext(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        f"You are a voice assistant with memory of previous conversations. "
                        f"Here are facts about the user from past interactions: {fact_content}\n\n"
                        f"Use this information to personalize your responses and show that you remember the user. "
                        f"When the user asks what you know about them, list these facts naturally. "
                        f"Be conversational and reference relevant facts in your responses."
                    )
                )
            ]
        )
    else:
        initial_ctx = ChatContext()
        if voe_enabled:
            logger.info("AGENT CREATION: No existing facts found")

    agent = ToMAgent(
        voe_enabled=voe_enabled,
        room=ctx.room,
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(model="sonic-2"),
        chat_ctx=initial_ctx,
    )

    loop = asyncio.get_running_loop()

    @agent.on("agent_is_speaking")
    def on_agent_is_speaking(speaking: bool):
        agent.is_speaking = speaking

    @agent.on("user_speech_committed")
    def on_user_speech_committed(transcript: ChatMessage):
        agent.chat_ctx.messages.append(transcript)
        asyncio.create_task(_reply_to_user(agent, transcript.content))

    async def _reply_to_user(agent: ToMAgent, user_msg: str):
        user_id = "default_user"  # In a real app, get from participant
        
        logger.info(f"_reply_to_user called: user_msg='{user_msg}', last_prediction='{agent.last_prediction}'")
        
        if agent.voe_enabled:
            # Do VoE check if we have a previous prediction
            if agent.last_prediction:
                await agent._voe_check(user_id, user_msg)
            
            # Always use predictor flow after first interaction, but on first interaction use baseline for response
            if agent.first_user_interaction:
                # First interaction - use baseline for response but still make prediction for next turn
                logger.info("FIRST INTERACTION - using baseline (facts already in system) but making prediction")
                reply = await agent._baseline_responder()
                # Make prediction for next turn
                prediction, needed_info = await agent._predictor()
                if prediction and prediction.strip():
                    agent.last_prediction = prediction.strip()
                    logger.info(f"FIRST INTERACTION: STORED PREDICTION for next turn: '{agent.last_prediction}'")
                else:
                    agent.last_prediction = "I expect a general response."
                    logger.warning("FIRST INTERACTION: Predictor returned empty/invalid prediction, using fallback")
                agent.first_user_interaction = False
            else:
                # Normal conversation flow with predictor
                logger.info("NORMAL CONVERSATION FLOW - using predictor")
                prediction, needed_info = await agent._predictor()
                facts = agent._retrieve(user_id, needed_info)
                reply = await agent._responder_with_facts(facts)
                # Store prediction for next VoE check - ensure it's not empty
                if prediction and prediction.strip():
                    agent.last_prediction = prediction.strip()
                    logger.info(f"STORED PREDICTION for next turn: '{agent.last_prediction}'")
                else:
                    agent.last_prediction = "I expect a general response."
                    logger.warning("Predictor returned empty/invalid prediction, using fallback")
        else:
            reply = await agent._baseline_responder()
            agent.last_prediction = ""
        
        await agent.say(reply, allow_interruptions=True)
        agent.chat_ctx.messages.append(ChatMessage(role="assistant", content=reply))

    await ctx.connect()

    voices = []
    cartesia_voices: List[dict[str, Any]] = ctx.proc.userdata.get("cartesia_voices", [])
    for voice in cartesia_voices:
        voices.append({"id": voice["id"], "name": voice["name"]})
    voices.sort(key=lambda x: x["name"])
    await ctx.room.local_participant.set_attributes({"voices": json.dumps(voices)})

    agent.start(ctx.room)
    
    # Update UI with existing facts on startup
    if voe_enabled:
        await agent._update_participant_attributes("default_user")
    
    await agent.say("Hi there, how are you doing today?", allow_interruptions=True)


def prewarm(proc: JobProcess):
    """Preload models when a process starts to speed up the first interaction."""
    proc.userdata["vad"] = silero.VAD.load()

    # Fetch available Cartesia voices
    try:
        headers = {
            "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.cartesia.ai/voices", headers=headers)
        if response.status_code == 200:
            proc.userdata["cartesia_voices"] = response.json().get("voices", [])
        else:
            logger.warning(
                f"Failed to fetch Cartesia voices: {response.status_code} - {response.text}"
            )
            proc.userdata["cartesia_voices"] = []
    except Exception as e:
        logger.error(f"Error fetching Cartesia voices: {e}", exc_info=True)
        proc.userdata["cartesia_voices"] = []


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
