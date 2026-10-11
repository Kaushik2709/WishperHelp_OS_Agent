# WisprHelp: Universal Autonomous OS Agent

> **Voice and text desktop agent with Windows, macOS, and Linux adapters**
> Powered by **TypeSafe AI Jev** (System One Reflex), **Groq LPU Semantic Decomposer**, **Silero VAD + Whisper Large v3** (Sensory Ears), **Set-of-Marks Vision Grounding** (System Two Eyes), and a **Cross-Platform OS Abstraction Layer (HAL)**.


**Validation status (2026-10-11):** adapters are implemented to different extents; cross-platform desktop reliability is not established. The offline suite and isolated Windows filesystem evaluations pass. The Windows GUI fixture started but the foreground guard refused input; no desktop task success is claimed. macOS/Linux desktop execution and live voice/model APIs remain unverified. See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for evidence, feature gaps, failure cases, and a safe reproducible demo.

Latency and accuracy figures below are design targets, not measured results. [PRD.md](PRD.md) and the older architecture sections describe intended capabilities; the evaluation report records implemented and verified behavior.

---

## 📖 Complete Documentation Suite

WisprHelp is engineered using **Spec-Driven Development (SDD)** with an explicit formal contract:

| Document | Purpose & Scope | Clickable Link |
|---|---|---|
| **`PRD.md`** | **Product Requirements Document**: User personas, functional requirements (FR-01 to FR-17), safety criteria, latency budgets, and acceptance metrics. | [PRD.md](PRD.md) |
| **`ARCHITECTURE.md`** | **System Architecture & Technical Design**: End-to-end cognitive diagrams, Two-Tier engine details, 3-Priority Execution hierarchy, and OS HAL contracts. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **`implementation_plan.md`** | **Execution Roadmap**: 5-phase engineering breakdown, task dependency graphs, component checklists, and milestone verification. | [implementation_plan.md](implementation_plan.md) |
| **`AGENTS.md`** | **Agent Harness Contract**: Coding standards, strict separation of concerns, test commands, and invariant rules for AI coding agents. | [AGENTS.md](AGENTS.md) |
| **`HARNESS.md`** | **Runtime Execution Harness**: Lifecycle state machine, hook points, context budgeting, error recovery policies, and self-healing mechanisms. | [HARNESS.md](HARNESS.md) |

---

## 🏗️ Architectural Overview & Execution Pipeline

WisprHelp pairs Jev intent classification with deep multimodal spatial reasoning and deterministic OS execution:

```mermaid
flowchart TB
    subgraph SENSORY ["1. Sensory & Hearing Pipeline"]
        Mic[Microphone 16kHz Stream] --> VAD[Silero VAD Engine<br/>Sub-5ms Local Activity Filter]
        PTT[Push-to-Talk: Ctrl+Shift / Alt+Space] --> VAD
        VAD -->|Clean Audio Chunks| STT[Whisper Large v3<br/>Groq API]
        STT --> RawSpeech[User Speech Transcript]
        Sensor[Active Window Sensor<br/>pywin32 / NSWorkspace / X11]
    end

    subgraph PRE_DECOMPOSE ["2. Semantic Pre-Decomposition & Normalization"]
        RawSpeech --> Decomposer[Groq LPU Decomposer / Rule Fallback<br/>Phonetic STT Correction & Compound Query Split]
        Decomposer --> AtomicSteps[Normalized Atomic Intent Steps]
    end

    subgraph SYSTEM_ONE ["3. System One Reflex: TypeSafe AI Jev (~100-150ms)"]
        AtomicSteps & Sensor --> JevClient[TypeSafe Jev API Client<br/>Endpoint: /v1/systemone | Model: jev-latest]
        
        subgraph JEV_MATRIX ["Parallel Calibrated Multi-Question Matrix"]
            Q1[Choice: intent<br/>direct_launch | quick_type | media_control | gui_workflow | query]
            Q2[Choice: target_app<br/>whatsapp | browser | store | terminal | settings | generic]
            Q3[Score: complexity<br/>0: Fast Shell | 1: Focus Type | 2: Multi-step Vision]
            Q4[Noul: is_destructive<br/>Calibrated Probability 0.0 - 1.0]
            Q5[Noul: has_multilingual_content<br/>Hindi, Hinglish, Non-Latin Script]
        end
        
        JevClient --> JEV_MATRIX
    end

    subgraph ROUTER_SAFETY ["4. Confidence Gate & Safety Interlock"]
        JEV_MATRIX --> Router{Confidence-Gated Router}
        Router -->|Confidence < 0.60| Clarification[Voice Clarification Handler]
        Router -->|is_destructive > 0.65| SafetyInterlock[Safety Gate: Voice/Prompt Confirmation]
        Router -->|Complexity 0: Deterministic| P1_Path[Priority 1: Shell / Protocol / AppID]
        Router -->|Complexity 1: Text Injection| P1_Type[Universal Unicode Text Dispatcher]
        Router -->|Complexity 2: GUI Workflow| P2_P3_Path[Priority 2 & 3 Cascade]
        Router -->|Intent: Conversational| ConvLLM[Groq Strategic Conversational Planner]
    end

    subgraph EXECUTION_HIERARCHY ["5. 3-Priority Tool Execution Hierarchy"]
        P1_Path --> P1[Priority 1: Native Launcher<br/>whatsapp:, Get-StartApps, open -a, xdg-open, volume]
        P1_Type --> UnicodeSwap[Unicode Buffer Swap + Native Paste<br/>Lossless Hindi, Hinglish, Emojis]
        
        P2_P3_Path --> P2{Priority 2: Accessibility<br/>Windows UIA / limited macOS AppleScript<br/>Linux AT-SPI pending}
        P2 -->|Control Found by Name/Pattern| NativeInvoke[Invoke Native Control Pattern<br/>Zero Screenshots / Zero Vision Tokens]
        P2 -->|Canvas / Custom Control / Missing UIA| P3[Priority 3: Full Multimodal Vision Grounding<br/>mss Screen Grab + Pillow Downscale]
        
        P3 --> SoM[Set-of-Marks Annotator<br/>Visual UI Element Badging]
        SoM --> GroundedAgent[Jev Grounded Agent<br/>'Decide, Don't Generate' Paradigm]
        GroundedAgent --> VLM[Claude 3.5 Sonnet / Gemini 2.5 / GPT-4o]
        VLM --> PhysicalAction[Simulated Mouse Click / Bezier Drag / Keys]
        
        P1 --> HAL[Hardware & OS Abstraction Layer - HAL]
        UnicodeSwap --> HAL
        NativeInvoke --> HAL
        PhysicalAction --> HAL
        
        HAL --> PostVerify{Postcondition Verification<br/>Perceptual dHash Screen Diff}
        PostVerify -->|Mutation Confirmed| Complete[Audio Ding & Status HUD Idle]
        PostVerify -->|Frozen / Retry Needed| P2_P3_Path
    end

    subgraph OVERLAY_SUPERVISION ["6. Visual HUD & Emergency Supervision"]
        HUD[PySide6 Floating Pill HUD<br/>Frameless, Always-on-top, Click-through]
        MouseFling[Screen Corner Fling] --> KillSwitch[Emergency Stop at Input Checkpoints]
        EscKey[Esc / Pause-Break Key] --> KillSwitch
        KillSwitch -.->|Abort Thread & Release Input| HAL
    end
```

---

## ⚡ The 3-Priority Tool Execution Hierarchy

Unlike conventional vision-only computer-use agents that capture costly full-screen images for trivial tasks, WisprHelp enforces a **strict 3-Priority Execution Rule**:

### Priority 1: Native OS Shell, Protocol & AppID Dispatch
- **Latency**: $\le 50\text{ms}$ (Zero vision calls, zero screenshots taken).
- **Application Launchers**:
  - **Windows**: PowerShell `Get-StartApps` registry AppIDs, URI protocol schemes (`whatsapp:`, `spotify:`, `slack:`, `calculator:`), Win32 API, `os.startfile`.
  - **macOS**: `open -a "<AppName>"` and custom URI schemes.
  - **Linux**: `xdg-open`, Freedesktop `.desktop` application indexers, CLI binaries.
- **System Media & Volume Control**:
  - Direct hardware audio adjustments (`volume 40%`, `mute`, `unmute`, `play`, `pause`, `next track`) via Win32 `WM_APPCOMMAND` / AppleScript / `amixer`.

### Priority 2: Native OS Accessibility Tree Traversal
- **Latency**: $50 - 150\text{ms}$ (No screen capture, no LLM vision token cost).
- **Supported Engines**:
  - **Windows**: Microsoft UI Automation (UIA) via [windows.py](wisprhelp/hal/windows.py) using `uiautomation`.
  - **macOS**: Limited AppleScript UI lookup via [macos.py](wisprhelp/hal/macos.py); editable-text verification remains unsupported.
  - **Linux**: native launch/X11 input in [linux.py](wisprhelp/hal/linux.py); AT-SPI element lookup and Wayland input are not implemented. Missing accessibility targets fail safely or use a scoped visual fallback.
- **Capabilities**:
  - Traverses the live visual hierarchy to find buttons, search fields, text boxes, and menu items by name, automation ID, or ordinal rank (*"click the second tab"*, *"open first chat"*).
  - Windows can read editable content through UIA patterns; control activation uses HAL input. Capabilities vary by adapter and application.

### Priority 3: Multimodal Vision Grounding & Computer Use
- **Latency**: $600 - 1800\text{ms}$.
- **Screen Perception**:
  - High-performance multi-monitor screenshot capture via `mss`.
  - Smart aspect-ratio downscaling with `Pillow` to reduce API payload overhead.
- **Set-of-Marks (SoM) Tagging**:
  - [som_annotator.py](wisprhelp/vision/som_annotator.py) dynamically inspects interactive regions and renders numbered badge overlays on actionable elements.
- **"Decide, Don't Generate" Decision Loop**:
  - Instead of forcing vision models to hallucinate arbitrary $(x, y)$ pixel coordinates, the [grounded_agent.py](wisprhelp/vision/grounded_agent.py) selects specific candidate badges (e.g. `[14]`).
  - Converts badge centers into physical monitor coordinates with DPI scaling.
- **Postcondition Verification**:
  - [postcondition.py](wisprhelp/vision/postcondition.py) computes perceptual differential hashing (`dHash`) before and after actions, confirming screen mutations before proceeding to the next step.

### Universal Unicode Text Injection
- Direct simulated keystrokes (`pyautogui.typewrite`) notoriously corrupt non-ASCII characters, Hindi Devanagari ligatures, accents, and emojis.
- WisprHelp uses **Unicode Clipboard Buffer Injection** in [base.py](wisprhelp/hal/base.py):
  1. Preserves the user's prior clipboard contents.
  2. Sets Unicode text directly into the OS clipboard.
  3. Dispatches native paste keystrokes (`Ctrl+V` on Windows/Linux, `Cmd+V` on macOS).
  4. Restores the prior clipboard buffer seamlessly.

---

## 🧠 Cognitive Engine & Classification Matrix

WisprHelp splits cognitive processing into fast reflex classification and deep generative reasoning:

### 1. LLM-First Dynamic Reasoning ([reasoning.py](wisprhelp/agent/reasoning.py))
- The **universal cognitive brain**: a Groq LPU call understands ANY natural instruction (compound, conditional, conversational, multilingual) and decomposes it into a plan of ~20 universal action primitives (`launch_app`, `click`, `type_text`, `key_combination`, `select_all`, `clear_text`, `scroll`, `drag`, ...) with per-step confidence and optional runtime conditions.
- **Live screen grounding**: every reasoning turn receives a numbered inventory of on-screen interactive elements, resolving deictic references like "that link" or "the first chat" (via `ordinal:N:type` targets).
- **Conditional steps**: "if there is something in it, clean it" → `clear_text` with `condition: "document_has_content"` — verified live via UIA/clipboard probe before execution.
- **Bounded recovery**: safe queries and known failures before dispatch may recover; uncertain modifying steps stop without replay.
- **Focus-before-type**: the brain always plans a click on the target input field before typing.
- A deterministic regex fallback is retained ONLY for offline (no-API) resilience.

### 2. Batched Per-Step Jev Classification ([plan_classifier.py](wisprhelp/agent/plan_classifier.py))
- One batched TypeSafe Jev (`/v1/systemone`) call evaluates the calibrated 5-question matrix (intent, complexity, destructive risk, confidence) for EVERY step of the plan in parallel.
- Merged metrics gate execution: step confidence below the threshold → clarification; destructive probability over the threshold → Safety Interlock confirmation.

### 3. Confidence-Gated Router & Safety Interlock ([router.py](wisprhelp/core/router.py) & [safety.py](wisprhelp/core/safety.py))
- **Clarification Gate**: If plan confidence falls below `0.60`, OS actions are halted and the agent asks for clarification.
- **Safety Confirmation Interlock**: If destructive probability `> 0.65` (e.g. *"delete all files in downloads"*, *"empty trash"*), execution is held until explicit affirmative user confirmation (`yes` / `no`).
- **Emergency Kill-Switch**:
  - Fling mouse cursor to any monitor corner (PyAutoGUI FAILSAFE).
  - Press `Esc` or `Pause/Break` to signal the emergency stop at the next execution checkpoint, releasing OS control locks and halting active loops.

---

## 🎙️ Hearing, Speech & Visual HUD

### Hearing & Audio Pipeline ([voice/](wisprhelp/voice))
- **VAD ([vad.py](wisprhelp/voice/vad.py))**: 16kHz mono speech detection uses an energy fallback by default. Optional Silero Torch loading requires constructing `SileroVAD(try_hub_load=True)` and an available model. Local VAD latency has not been measured here.
- **Whisper Large v3 ([stt.py](wisprhelp/voice/stt.py))**: Multilingual transcription via Groq API (~250ms latency) through Groq. A local Whisper transcription backend is not implemented; unavailable transcription returns an explicit failure without queuing an action.
- **Push-to-Talk & Hotkeys ([hotkey.py](wisprhelp/voice/hotkey.py))**: Global non-blocking keyboard hook listening for `Ctrl + Shift` or `Alt + Space`.
- **Audio Feedback ([feedback.py](wisprhelp/voice/feedback.py))**: Real-time auditory chimes for listen start, speech end, success ding, warning tone, and abort buzzer.
- **Text-to-Speech ([tts.py](wisprhelp/voice/tts.py))**: Background asynchronous speech synthesizer for conversational answers and safety alerts.

### PySide6 Floating Status HUD ([overlay.py](wisprhelp/ui/overlay.py))
An unobtrusive, frameless, always-on-top, click-through status pill positioned at the top of the primary monitor:

| HUD State | Visual Presentation | Meaning |
|---|---|---|
| `IDLE` | Subtle translucent emerald pill | Agent active, waiting for input |
| `LISTENING` | Pulsing cyan pill with audio wave | Recording voice input from microphone |
| `THINKING` | Pulsating amber pill | Transcribing audio & classifying intent via Jev |
| `EXECUTING` | Active violet/indigo pill | Performing OS action (launching, typing, clicking) |
| `CONFIRM` | Glowing warning yellow pill | Held at Safety Gate; waiting for user confirmation |
| `ERROR` | Vivid crimson red pill | Action failed or emergency kill-switch triggered |

---

## 💻 Supported Commands & Real-World Examples

| User Command | Primary Route | Execution Tier | What WisprHelp Does |
|---|---|---|---|
| *"Open Spotify"* | `PRIORITY_1_LAUNCH` | Priority 1 | Launches Spotify via native URI scheme `spotify:` in $< 50\text{ms}$. |
| *"Volume 40%"* / *"Awaaz 30 karo"* | `PRIORITY_1_MEDIA` | Priority 1 | Sets system audio output directly to target level via native OS media API. |
| *"Mute sound"* / *"Unmute"* | `PRIORITY_1_MEDIA` | Priority 1 | Toggles system audio mute via virtual key stroke. |
| *"Type namaste in Hindi"* | `TEXT_INJECTION` | Priority 1 | Injects Unicode string `नमस्ते` into active window via clipboard swap. |
| *"Write 'कल दोपहर 2 बजे क्लाइंट मीटिंग है'"* | `TEXT_INJECTION` | Priority 1 | Injects full Devanagari text without IME character corruption. |
| *"Open Chrome and search for quantum computing"* | `GUI_WORKFLOW` | Priority 1 + 2 | Launches Chrome, switches focus, opens address bar, and submits search query. |
| *"Click on the second tab"* | `GUI_WORKFLOW` | Priority 2 | Uses UI Automation to locate the tab control by ordinal index and triggers `InvokePattern`. |
| *"Open Calculator and add 45 plus 90"* | `GUI_WORKFLOW` | Priority 1 + 2 | Launches Calculator, focuses keypad, and inputs calculation automatically. |
| *"Delete all files in my Downloads folder"* | `SAFETY_INTERLOCK` | Guardrail | Evaluates `is_destructive > 0.65`, pauses execution, and prompts for confirmation. |
| *"What is the capital of France?"* | `CONVERSATIONAL` | Groq LPU | Answers query verbally via TTS and prints text in sub-second roundtrip. |

---

## ⚙️ Environment Configuration (`.env`)

Create a `.env` file in the project root directory. All configuration options are typed and validated via [config.py](wisprhelp/config.py):

```bash
# ==============================================================================
# 1. TypeSafe AI Jev (System One Fast Classifier)
# ==============================================================================
JEV_API_KEY=apikey_your_typesafe_jev_key_here
# Alternative alias:
# TYPESAFE_API_KEY=apikey_...
JEV_API_URL=https://api.typesafe.ai/v1/systemone
JEV_MODEL=jev-latest

# ==============================================================================
# 2. Confidence & Safety Risk Thresholds
# ==============================================================================
# Minimum confidence required to execute intent without clarification (0.0 - 1.0)
INTENT_CONFIDENCE_THRESHOLD=0.60
# Risk threshold above which explicit user confirmation is enforced (0.0 - 1.0)
DESTRUCTIVE_RISK_THRESHOLD=0.65

# ==============================================================================
# 3. Speech-to-Text & Strategic Planning (Groq LPU)
# ==============================================================================
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_PLANNER_MODEL=openai/gpt-oss-120b

# ==============================================================================
# 4. Frontier Multimodal Vision Models (System Two Brain)
# ==============================================================================
# Anthropic Claude 3.5 Sonnet Computer Use:
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here

# Google Gemini for vision grounding:
GEMINI_API_KEY=AIzaSy_your_gemini_key_here
GEMINI_MODEL=gemini-3.5-flash

# OpenAI GPT-4o (Vision / Computer Use):
OPENAI_API_KEY=sk-proj-your_openai_key_here
OPENAI_MODEL=gpt-4o

# ==============================================================================
# 5. Vision Grounding & Computer-Use Options
# ==============================================================================
# Enable Set-of-Marks (SoM) visual UI badge annotations (true/false)
ENABLE_SET_OF_MARKS=true
# Verify screen mutations after actions via perceptual dHash diffing (true/false)
POSTCONDITION_VERIFICATION_ENABLED=true

# ==============================================================================
# 6. Safety Limits & Guardrails
# ==============================================================================
# Maximum number of iterative steps in an autonomous vision loop
MAX_AGENT_STEPS=15
# Emergency hooks; native mouse failsafe remains enabled by the adapters
KILL_SWITCH_ENABLED=true
# Logging output level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ target; this evaluation ran on 3.14.3. CI for 3.11/3.13 is configured but has not run in this session.
- A working microphone and speaker for voice interaction.

### 1. Clone Repository & Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
git clone https://github.com/Kaushik2709/WishperHelp_OS_Agent.git d:\WishperHelp
cd d:\WishperHelp

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On macOS / Linux (Bash/Zsh):
```bash
git clone https://github.com/Kaushik2709/WishperHelp_OS_Agent.git wisprhelp
cd wisprhelp

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core agent runtime dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt

# Additional voice & audio packages (if using local VAD/TTS)
python -m pip install torch  # optional; Silero loading must also be explicitly enabled
```

#### Platform-Specific Notes:
- **Windows**: `pywin32` and `uiautomation` provide deep native OS accessibility and window tracking.
- **macOS**: Grant accessibility permissions when prompted:
  - *System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Accessibility* (for keyboard/mouse simulation).
  - *System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Screen Recording* (for screen capture).
  - *System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Microphone* (for voice capture).
- **Linux**: Requires `xdotool` and clipboard tools:
  ```bash
  sudo apt-get install xdotool xclip libasound2-dev python3-tk
  ```

---

## 🎯 Running WisprHelp

WisprHelp can be run in multiple modes using [main.py](wisprhelp/main.py):

### 1. Interactive CLI Mode (Default)
Launches the floating PySide6 status HUD and starts an interactive command prompt:
```bash
python -m wisprhelp.main
```
*Prompt example:*
```text
WisprHelp> open visual studio code
[PRIORITY_1_LAUNCH] (32.4ms): Priority 1: Launched application 'code' (Success).

WisprHelp> delete all files in downloads
[SAFETY_INTERLOCK] (142.1ms): High-risk action detected: Delete all files in downloads folder. Confirm [yes/no]?
Confirm high-risk action? [yes/no]: no
[SAFETY_INTERLOCK]: High-risk action was cancelled by user.
```

### 2. Single-Shot Instant Command Execution
Executes an individual command immediately and exits:
```bash
# Launch application
python -m wisprhelp.main --text "open spotify"

# Adjust system volume
python -m wisprhelp.main --text "set volume to 50"

# Inject Unicode text
python -m wisprhelp.main --text "type namaste in hindi"

# Compound launch and search
python -m wisprhelp.main --text "open chrome and search for AI breakthroughs"

# Open an editor and enter the exact dictated text
python -m wisprhelp.main --text "Open Notepad and write: I am a developer."
```

The semantic agent asks the configured Groq model for an ordered plan of up to
15 actions, then checks each action against the safety gate and executes it through
the OS adapter. Text after `write:` or `type:` is treated as literal content. Before
typing, the agent focuses a verified editable control and reads its current text.
If replacing existing content is necessary, it asks for confirmation and resumes
the remaining steps after approval. If it cannot identify or verify the editor, it
reports the blocked step rather than typing into an arbitrary window. GUI controls
that accessibility cannot locate require a configured vision model for visual
grounding. When the Groq API is unavailable, only a narrow offline command
subset is supported; other requests receive a clarification prompt.

### 3. Voice Listening Mode (Push-to-Talk)
Activates the audio stream and Push-to-Talk listener:
```bash
python -m wisprhelp.main --voice
```
- **Hold `Ctrl + Shift`** (or `Alt + Space`) to speak.
- Release the keys to trigger transcription and execution.
- Press **`Esc`** at any moment to trigger the emergency kill-switch.

### 4. Continuous Hands-Free Voice Mode
Listens continuously without requiring keyboard hotkeys, using Silero VAD to detect user speech automatically:
```bash
python -m wisprhelp.main --voice --hands-free
```

### 5. Headless & Quiet Mode (Daemon / Background Service)
Runs without the floating PySide6 HUD or audio beeps:
```bash
python -m wisprhelp.main --no-ui --no-sound --text "mute volume"
```

### 6. Debug Mode
Enables full debug logging of classification payloads, routing decisions, and VLM bounding boxes:
```bash
python -m wisprhelp.main --debug
```

---

## 📂 Repository Structure

```
d:/WishperHelp/
├── .env                              # Environment configuration & API credentials
├── .gitignore                        # Git exclusion rules
├── AGENTS.md                         # Agent harness coding contract & boundaries
├── ARCHITECTURE.md                   # Detailed technical design & system diagrams
├── HARNESS.md                        # Execution lifecycle & recovery policies
├── implementation_plan.md            # Phased engineering roadmap
├── PRD.md                            # Formal Product Requirements Document
├── README.md                         # Authoritative project guide & reference
│
├── wisprhelp/                        # Core Python Application Package
│   ├── __init__.py
│   ├── config.py                     # Immutable dataclass configuration loader
│   ├── main.py                       # CLI, voice listener, and entry point orchestrator
│   │
│   ├── agent/                        # === Dynamic Cognitive Brain (LLM-First) ===
│   │   ├── __init__.py
│   │   ├── actions.py                # Universal action primitives + condition field
│   │   ├── reasoning.py              # LLM-first instruction decomposition (Groq)
│   │   ├── plan_classifier.py        # Batched per-step TypeSafe Jev classification
│   │   ├── planner.py                # Closed-loop TaskGraph: execute-observe-replan
│   │   ├── executor.py               # Safety-gated action execution + condition checks
│   │   ├── memory.py                 # Conversation memory & entity store
│   │   ├── references.py             # Deictic reference resolution ("it", "the other one")
│   │   └── state.py                  # Adaptive computer state observation engine
│   │
│   ├── classification/               # === System One: Fast Reflex Classification ===
│   │   ├── __init__.py
│   │   ├── client.py                 # TypeSafe Jev API HTTP client wrapper
│   │   ├── schemas.py                # Typed Question schemas & JevDecision dataclasses
│   │   ├── classifier.py             # JevClassifier: context-aware state packager
│   │   └── decomposer.py             # Groq LPU semantic pre-decomposer & STT typo fixer
│   │
│   ├── core/                         # === Core Routing, Planning & Safety ===
│   │   ├── __init__.py
│   │   ├── router.py                 # Confidence-gated routing engine
│   │   ├── safety.py                 # Destructive risk interlock & confirmation gate
│   │   ├── runtime.py                # Central coordinator wiring Ears, Brain, and Hands
│   │   └── groq_planner.py           # Strategic planner & conversational LLM engine
│   │
│   ├── hal/                          # === Hardware & OS Abstraction Layer (HAL) ===
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseOSAdapter abstract class & WindowInfo models
│   │   ├── factory.py                # Platform auto-detection & adapter factory
│   │   ├── windows.py                # Windows adapter (Win32, UIA, Get-StartApps, SendInput)
│   │   ├── macos.py                  # macOS adapter (limited AppleScript, open)
│   │   └── linux.py                  # Linux adapter (X11, xdg-open; AT-SPI/Wayland pending)
│   │
│   ├── ui/                           # === Visual HUD Overlay ===
│   │   ├── __init__.py
│   │   └── overlay.py                # PySide6 frameless click-through floating status pill
│   │
│   ├── vision/                       # === System Two: Multimodal Computer Use ===
│   │   ├── __init__.py
│   │   ├── capture.py                # Fast multi-monitor screen grabber (mss + PIL)
│   │   ├── grounding.py              # Visual coordinate transformation & scaling
│   │   ├── som_annotator.py          # Set-of-Marks (SoM) bounding box badge renderer
│   │   ├── grounded_schemas.py       # CandidateCatalog & GroundedDecision models
│   │   ├── grounded_agent.py         # 'Decide, Don't Generate' action selection agent
│   │   ├── postcondition.py          # Perceptual dHash differential screen verification
│   │   └── vlm_agent.py              # Multimodal vision loop (Claude 3.5 / Gemini / GPT-4o)
│   │
│   └── voice/                        # === Multilingual Hearing & Speech Perception ===
│       ├── __init__.py
│       ├── vad.py                    # Silero VAD local voice activity detection (< 5ms)
│       ├── stt.py                    # Whisper Large v3 speech transcription (Groq API)
│       ├── hotkey.py                 # Global non-blocking keyboard hook listener
│       ├── stream.py                 # Microphone ring-buffer audio stream
│       ├── feedback.py               # Auditory cues (beeps, success dings, abort alerts)
│       └── tts.py                    # Text-to-Speech synthesis engine
│
└── tests/                            # Offline tests and isolated filesystem tests
    ├── test_dynamic_agent.py          # LLM-first dynamic workflows: conditional, replan, safety
    ├── test_plan_classifier.py        # Batched per-step Jev classification & gating
    ├── test_dynamic_interactions.py   # Volume, relative adjustments, and ordinal controls
    ├── test_end_to_end.py            # Complete voice-to-action & safety pipelines
    ├── test_groq_planner.py          # Groq planning & conversational fallbacks
    ├── test_grounded_agent.py        # Set-of-Marks candidate selection & aborts
    ├── test_hal_adapters.py          # Cross-platform HAL interface contracts
    ├── test_jev_classification.py    # TypeSafe Jev 5-question matrix evaluations
    ├── test_postcondition.py         # Perceptual dHash screen mutation verification
    ├── test_query_decomposition.py   # Compound action splitting & human-like actions
    ├── test_router_and_safety.py     # Confidence gating & confirmation interlocks
    ├── test_semantic_decomposer.py   # STT typo correction & offline fallbacks
    ├── test_som_tagging.py           # Visual badge rendering & candidate catalogs
    ├── test_vision_agent.py          # Coordinate grounding & bounded execution limits
    └── test_voice_pipeline.py        # Silero VAD, Whisper STT, PTT hotkeys, and mic stream
```

---

## 🧪 Testing & Verification Suite

Run offline verification before merging changes. Passing mocked tests does not establish desktop task success:

### Run Complete Offline Test Suite
```bash
python -m pytest tests/ -v
```

### Run Domain-Specific Test Modules
```bash
# 1. TypeSafe Jev classification & calibration tests
python -m pytest tests/test_jev_classification.py -v

# 2. Hardware OS Abstraction Layer (HAL) adapter tests
python -m pytest tests/test_hal_adapters.py -v

# 3. Router & Safety Gate confirmation interlock tests
python -m pytest tests/test_router_and_safety.py -v

# 4. Semantic decomposition & compound command parsing tests
python -m pytest tests/test_query_decomposition.py tests/test_semantic_decomposer.py -v

# 5. Voice pipeline, Silero VAD & Whisper STT tests
python -m pytest tests/test_voice_pipeline.py -v

# 6. Set-of-Marks, VLM Grounding & Postcondition verification tests
python -m pytest tests/test_som_tagging.py tests/test_grounded_agent.py tests/test_postcondition.py -v

# 7. Mocked pipeline integration (not real speech or desktop E2E)
python -m pytest tests/test_end_to_end.py -v
```

---

## 🛡️ Safety & Responsible Computer Use

1. **Strict Maximum Step Limit**: Every vision-guided autonomous execution loop is hard-capped by `MAX_AGENT_STEPS` (default: 15) to prevent run-away loops.
2. **Deterministic Destructive Interlock**: Jev's calibrated `is_destructive` probability score halts actions that risk permanent data deletion, credential modification, or financial transactions until the user explicitly speaks or types `yes`.
3. **Emergency Physical Kill-Switch**: Users maintain absolute physical authority over their machine at all times via screen-corner mouse fling or the `Esc` key.
4. **Authority boundaries**: observed screen/page text cannot provide confirmation. Local checks constrain risky actions and file/terminal privileges. This is defense in depth; general prompt-injection resistance and credential-dialog coverage are not proven.

---

## 📄 License

This checkout does not contain a LICENSE file. The previous Apache License claim has no corresponding license artifact here.
