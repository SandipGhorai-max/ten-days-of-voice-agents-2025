"""
Day 10 – Voice Improv Battle

This file adapts the Day 9 voice Game Master agent into a voice-first improv
show host called "Improv Battle". The original voice/STT/TTS/turn-detection/VAD
plumbing and imports are preserved so it fits into the same voice runtime.

Behaviour summary (implemented as tools exposed to the LLM):
- start_show(name, max_rounds): initialise session state and introduce the show
- next_scenario(): advance to the next improv scenario and put the host into awaiting_improv phase
- record_performance(performance): save the player's improvisation, produce a host reaction
- summarize_show(): produce a closing summary once rounds complete
- stop_show(confirm=False): allow graceful early exit

The GameMasterAgent uses these tools and acts as the high-energy improv host.
"""

import json
import logging
import os
import asyncio
import uuid
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_improv_battle")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Improv Scenarios (seeded)
# -------------------------
# Each scenario is a clear short prompt: role, situation, tension/hook
SCENARIOS = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are an overenthusiastic TV infomercial host selling a product that clearly does not work as advertised.",
    "You are an astronaut who just discovered the ship's coffee machine has developed a personality.",
    "You are a nervous wedding officiant who keeps getting the couple's names mixed up in ridiculous ways.",
    "You are a ghost trying to give a performance review to a living employee.",
    "You are a medieval king reacting to a very modern delivery service showing up at court.",
    "You are a detective interrogating a suspect who only answers in awkward metaphors."
]

# -------------------------
# Per-session Improv State
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    improv_state: Dict = field(default_factory=lambda: {
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],  # each: {"scenario": str, "performance": str, "reaction": str}
        "phase": "idle",  # "intro" | "awaiting_improv" | "reacting" | "done" | "idle"
        "used_indices": []
    })
    history: List[Dict] = field(default_factory=list)

# -------------------------
# Helpers
# -------------------------

def _pick_scenario(userdata: Userdata) -> str:
    used = userdata.improv_state.get("used_indices", [])
    candidates = [i for i in range(len(SCENARIOS)) if i not in used]
    if not candidates:
        # reset if we exhausted scenarios
        userdata.improv_state["used_indices"] = []
        candidates = list(range(len(SCENARIOS)))
    idx = random.choice(candidates)
    userdata.improv_state["used_indices"].append(idx)
    return SCENARIOS[idx]


def _host_reaction_text(performance: str) -> str:
    # Lightweight heuristic to vary reaction tone
    tones = ["supportive", "neutral", "mildly_critical"]
    tone = random.choice(tones)
    # Quick keyword detection to pick specific highlights (not exhaustive)
    highlights = []
    if any(w in performance.lower() for w in ("funny", "lol", "hahaha", "haha")):
        highlights.append("great comedic timing")
    if any(w in performance.lower() for w in ("sad", "cry", "tears")):
        highlights.append("good emotional depth")
    if any(w in performance.lower() for w in ("pause", "...")):
        highlights.append("interesting use of silence")
    if not highlights:
        # fallback picks
        highlights.append(random.choice(["nice character choices", "bold commitment", "unexpected twist"]))

    chosen = random.choice(highlights)
    if tone == "supportive":
        return f"Love that — {chosen}! That was playful and clear. Nice work. Ready for the next one?"
    elif tone == "neutral":
        return f"Hmm — {chosen}. That landed in parts; you had interesting ideas. Let's try the next scene and lean into one choice."
    else:  # mildly_critical
        return f"Okay — {chosen}, but that felt a bit rushed. Try to make stronger choices next time. Don't be afraid to exaggerate."

# -------------------------
# Agent Tools
# -------------------------
@function_tool
async def start_show(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Player/contestant name (optional)", default=None)] = None,
    max_rounds: Annotated[int, Field(description="Number of rounds (3-5 recommended)", default=3)] = 3,
) -> str:
    userdata = ctx.userdata
    if name:
        userdata.player_name = name.strip()
    else:
        # attempt to set player_name from history if present
        userdata.player_name = userdata.player_name or "Contestant"

    # clamp rounds
    if max_rounds < 1:
        max_rounds = 1
    if max_rounds > 8:
        max_rounds = 8

    userdata.improv_state["max_rounds"] = int(max_rounds)
    userdata.improv_state["current_round"] = 0
    userdata.improv_state["rounds"] = []
    userdata.improv_state["phase"] = "intro"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "start_show", "name": userdata.player_name})

    intro = (
        f"Welcome to Improv Battle! I'm your host — let's get ready to play."
        f" {userdata.player_name or 'Contestant'}, we'll run {userdata.improv_state['max_rounds']} rounds. "
        "Rules: I'll give you a quick scene, you'll improvise in character. When you're done say 'End scene' or pause — I'll react and move on. Have fun!"
    )
    # After intro, immediately provide first scenario for flow convenience
    scenario = _pick_scenario(userdata)
    userdata.improv_state["current_round"] = 1
    userdata.improv_state["phase"] = "awaiting_improv"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "present_scenario", "round": 1, "scenario": scenario})

    return intro + "\nRound 1: " + scenario + "\nStart improvising now!"


@function_tool
async def next_scenario(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    if userdata.improv_state.get("phase") == "done":
        return "The show is already over. Say 'start show' to play again."

    cur = userdata.improv_state.get("current_round", 0)
    maxr = userdata.improv_state.get("max_rounds", 3)
    if cur >= maxr:
        userdata.improv_state["phase"] = "done"
        return await summarize_show(ctx)

    # advance
    next_round = cur + 1
    scenario = _pick_scenario(userdata)
    userdata.improv_state["current_round"] = next_round
    userdata.improv_state["phase"] = "awaiting_improv"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "present_scenario", "round": next_round, "scenario": scenario})
    return f"Round {next_round}: {scenario}\nGo!"


@function_tool
async def record_performance(
    ctx: RunContext[Userdata],
    performance: Annotated[str, Field(description="Player's improv performance (transcribed text)")],
) -> str:
    userdata = ctx.userdata
    if userdata.improv_state.get("phase") != "awaiting_improv":
        # still accept performance but warn
        userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "record_performance_out_of_phase"})

    round_no = userdata.improv_state.get("current_round", 0)
    scenario = userdata.history[-1].get("scenario") if userdata.history and userdata.history[-1].get("action") == "present_scenario" else "(unknown)"

    reaction = _host_reaction_text(performance)

    userdata.improv_state["rounds"].append({
        "round": round_no,
        "scenario": scenario,
        "performance": performance,
        "reaction": reaction,
    })
    userdata.improv_state["phase"] = "reacting"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "record_performance", "round": round_no})

    # If we've reached max rounds, change to done after reaction
    if round_no >= userdata.improv_state.get("max_rounds", 3):
        userdata.improv_state["phase"] = "done"
        closing = "\n" + reaction + "\nThat's the final round. "
        closing += (await summarize_show(ctx))
        return closing

    # otherwise prompt for next round
    closing = reaction + "\nWhen you're ready, say 'Next' or I'll give you the next scene."
    return closing


@function_tool
async def summarize_show(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    rounds = userdata.improv_state.get("rounds", [])
    if not rounds:
        return "No rounds were played. Thanks for stopping by Improv Battle!"

    # Simple summary heuristics: count supportive vs critical words, highlight standout moments
    summary_lines = [f"Thanks for playing, {userdata.player_name or 'Contestant'}! Here's a short recap:"]
    # highlight each round briefly
    for r in rounds:
        perf_snip = (r.get("performance") or "").strip()
        if len(perf_snip) > 80:
            perf_snip = perf_snip[:77] + "..."
        summary_lines.append(f"Round {r.get('round')}: {r.get('scenario')} — You: '{perf_snip}' | Host: {r.get('reaction')}")

    # aggregate a simple profile
    mentions_character = sum(1 for r in rounds if any(w in (r.get('performance') or '').lower() for w in ('i am', "i'm", 'as a', 'character', 'role')))
    mentions_emotion = sum(1 for r in rounds if any(w in (r.get('performance') or '').lower() for w in ('sad', 'angry', 'happy', 'love', 'cry', 'tears')))

    profile = "You seem to be a player who "
    if mentions_character > len(rounds) / 2:
        profile += "commits to character choices"
    elif mentions_emotion > 0:
        profile += "brings emotional color to scenes"
    else:
        profile += "likes surprising beats and twists"

    profile += ". Keep leaning into clear choices and stronger stakes."

    summary_lines.append(profile)
    summary_lines.append("Thanks for performing on Improv Battle — hope to see you again!")

    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "summarize_show"})
    return "\n".join(summary_lines)


@function_tool
async def stop_show(ctx: RunContext[Userdata], confirm: Annotated[bool, Field(description="Confirm stop", default=False)] = False) -> str:
    userdata = ctx.userdata
    if not confirm:
        return "Are you sure you want to stop the show? Say 'stop show yes' to confirm."
    userdata.improv_state["phase"] = "done"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "stop_show"})
    return "Show stopped. Thanks for coming to Improv Battle!"


# -------------------------
# The Agent (Improv Host)
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
        You are the host of a TV improv show called 'Improv Battle'.
        Role: High-energy, witty, and clear about rules. Guide a single contestant through a series of short improv scenes.

        Behavioural rules:
            - Introduce the show and explain the rules at the start.
            - Present clear scenario prompts (who you are, what's happening, what's the tension).
            - Prompt the player to improvise and listen for an explicit "End scene" or accept an utterance passed to record_performance.
            - After each scene, react in a varied, realistic way (supportive, neutral, mildly critical). Store the reaction.
            - Run the configured number of rounds, then summarize the player's style.
            - Keep turns short and TTS-friendly.
        Use the provided tools: start_show, next_scenario, record_performance, summarize_show, stop_show.
        """
        super().__init__(
            instructions=instructions,
            tools=[start_show, next_scenario, record_performance, summarize_show, stop_show],
        )

# -------------------------
# Entrypoint & Prewarm
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎭" * 6)
    logger.info("🚀 STARTING VOICE IMPROV HOST — Improv Battle")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Start with the Improv Host agent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))# ======================================================
# 💼 DAY 5: AI SALES DEVELOPMENT REP (SDR)
# 👨‍⚕️ "Dr. Abhishek Store" - Auto-Lead Capture Agent
# 🚀 Features: FAQ Retrieval, Lead Qualification, JSON Database
# ======================================================

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, asdict

print("\n" + "💼" * 50)
print("🚀 AI SDR AGENT - DAY 5 TUTORIAL")
print("📚 SELLING: Dr. Abhishek's Cloud & AI Courses")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("💼" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ)
# ======================================================

FAQ_FILE = "store_faq.json"
LEADS_FILE = "leads_db.json"

# Default FAQ data for "Dr. Abhishek Store"
DEFAULT_FAQ = [
    {
        "question": "What do you sell?",
        "answer": "We offer premium courses on Cloud Computing, Google Cloud Arcade, and Voice AI Agent development. We also sell 'Cloud Ninja' merchandise like hoodies and mugs."
    },
    {
        "question": "How much does the Voice AI course cost?",
        "answer": "The 'Professional Voice AI' course is currently priced at $499. It covers LiveKit, Deepgram, and LLM integration."
    },
    {
        "question": "Do you offer free content?",
        "answer": "Yes! Dr. Abhishek releases weekly tutorials on YouTube for free. The paid courses offer deep-dives, code reviews, and certification."
    },
    {
        "question": "Do you do corporate consulting?",
        "answer": "Absolutely. We help companies build internal voice agents for customer support. Pricing depends on the project scope."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f)) # Return as string for the Prompt
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None
   
    def is_qualified(self):
        """Returns True if we have the minimum info (Name + Email + Use Case)"""
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to build or learn")] = None,
    team_size: Annotated[Optional[str], Field(description="Number of people in their team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start (e.g., Now, next month)")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation.
    Only call this when the user explicitly provides information.
    """
    profile = ctx.userdata.lead_profile
   
    # Update only fields that are provided (not None)
    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline
   
    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the lead to the database and signals the end of the call.
    Call this when the user says goodbye or 'that's all'.
    """
    profile = ctx.userdata.lead_profile
   
    # Save to JSON file (Append mode)
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
   
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()
   
    # Read existing, append, write back (Simple JSON DB)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except: pass
   
    existing_data.append(entry)
   
    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)
       
    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    return f"Lead saved. Summarize the call for the user: 'Thanks {profile.name}, I have your info regarding {profile.use_case}. We will email you at {profile.email}. Goodbye!'"

# ======================================================
# 🧠 4. AGENT DEFINITION
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Sarah', a friendly and professional Sales Development Rep (SDR) for 'Dr. Abhishek Store'.
           
            📘 **YOUR KNOWLEDGE BASE (FAQ):**
            {STORE_FAQ_TEXT}
           
            🎯 **YOUR GOAL:**
            1. Answer questions about our Cloud/AI courses and consulting using the FAQ.
            2. **QUALIFY THE LEAD:** Naturally ask for the following details during the chat:
               - Name
               - Company / Role
               - Email
               - What are they trying to build? (Use Case)
               - Timeline (When do they need it?)
           
            ⚙️ **BEHAVIOR:**
            - **Be Conversational:** Don't interrogate the user. Answer a question, THEN ask for a detail.
            - *Example:* "Our Voice AI course is $499. It's great for teams. By the way, how large is your dev team?"
            - **Capture Data:** Use `update_lead_profile` immediately when you hear new info.
            - **Closing:** When the user is done, use `submit_lead_and_end`.
           
            🚫 **RESTRICTIONS:**
            - If you don't know an answer, say "I'll check with Dr. Abhishek and email you." (Don't hallucinate prices).
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💼" * 25)
    print("🚀 STARTING SDR SESSION")
   
    # 1. Initialize State
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie", # Professional, warm female voice
            style="Promo",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
   
    # 3. Start
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
