# WisprHelp: Universal Autonomous OS Agent

> **Hands-Free, Zero-Touch Computer-Using AI Assistant for Windows, macOS, and Linux**  
> Powered by **TypeSafe AI Jev** (System One Reflex), **Silero VAD + Whisper Large v3** (Ears), and **Frontier Multimodal Vision** (System Two Brain).

---

## 📖 Complete Project Documentation Suite

This project follows **Spec-Driven Development (SDD)** with a formal Agent Harness documentation suite:

| Document | Purpose & Description | Link |
|---|---|---|
| **`PRD.md`** | **Product Requirements Document**: Business goals, user personas, functional specifications (VAD, Ears, Brain, 3-Priority Hands, Safety), and success metrics. | [PRD.md](file:///d:/WishperHelp/PRD.md) |
| **`ARCHITECTURE.md`** | **System Architecture**: High-level diagrams, Two-Tier Cognitive Engine, 3-Priority Tool Execution Hierarchy, Cross-Platform HAL, and PySide6 HUD. | [ARCHITECTURE.md](file:///d:/WishperHelp/ARCHITECTURE.md) |
| **`IMPLEMENTATION_PLAN.md`** | **Execution Plan**: Step-by-step 5-phase engineering roadmap, directory structure, task breakdown, and verification milestones. | [IMPLEMENTATION_PLAN.md](file:///d:/WishperHelp/IMPLEMENTATION_PLAN.md) |
| **`AGENTS.md`** | **Agent Harness Contract**: Technical rules, coding standards, environment boundaries, test commands, and constraints for AI coding agents. | [AGENTS.md](file:///d:/WishperHelp/AGENTS.md) |
| **`HARNESS.md`** | **Runtime Harness Specification**: Execution lifecycle, hook points, state management, context budgeting, and error recovery policies. | [HARNESS.md](file:///d:/WishperHelp/HARNESS.md) |

---

## ⚡ Key Highlights

### 1. Two-Tier Cognitive Architecture with 3-Priority Execution
- **System One (TypeSafe Jev)**: Rapid classification (~100-150ms) across 5 parallel dimensions (`intent`, `target_app`, `complexity`, `is_destructive`, `has_multilingual_content`) with calibrated probabilities.
- **3-Priority Execution Hierarchy**:
  1. **Priority 1 (Direct OS Shell / Protocol / AppID)**: Instant zero-vision app launching (`whatsapp:`, `Get-StartApps`, `open -a`, `xdg-open`).
  2. **Priority 2 (Native Accessibility Tree)**: Fast control finding via Windows UIA (`uiautomation`/`pywinauto`), macOS AX, and Linux AT-SPI without image inference.
  3. **Priority 3 (Multimodal Vision Grounding)**: Full screen capture (`mss` + `PIL` downscaled on demand) + OmniParser / Claude 3.5 Computer Use / Gemini 2.5 Flash for canvas & custom apps.

### 2. Universal Computer-Using Agent (CUA)
- Operates like a human sitting in front of the screen.
- Works across **ANY desktop application**: Chrome, Excel, VS Code, Terminal, Slack, WhatsApp, File Explorer, System Settings, or custom enterprise software.

### 3. Speech & Hearing Pipeline
- **Silero VAD**: Sub-5ms local voice activity detection filtering room noise and silence.
- **Whisper Large v3**: Multilingual transcription (English, Hindi, Hinglish, 90+ languages).

### 4. Cross-Platform OS Abstraction Layer (HAL)
- Unified interface with native adapters for:
  - **Windows**: Win32, UI Automation, `Get-StartApps`, URI schemes, SendInput.
  - **macOS**: AppleScript, Cocoa `NSWorkspace`, Quartz, `open`.
  - **Linux**: X11/Wayland, `xdg-open`, AT-SPI, `xdotool`.

### 5. Multilingual Unicode Injection & PySide6 HUD
- Direct typing of Hindi (Devanagari: `नमस्ते`), Hinglish, English, and emojis into any focused window without font corruption.
- Click-through PySide6 floating status pill showing real-time agent state.

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
```bash
# Clone or navigate to the directory
cd d:/WishperHelp

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install typesafe-sdk pyautogui mss pillow pyperclip requests python-dotenv pydantic pytest PySide6 uiautomation
```

### 2. Configure API Keys (`.env`)
```bash
# TypeSafe AI Jev (System One Classifier)
JEV_API_KEY=apikey_...

# Multimodal Vision Agent (System Two)
ANTHROPIC_API_KEY=sk-ant-...
# or
GEMINI_API_KEY=AIzaSy...

# Multilingual Speech-to-Text (Ear)
GROQ_API_KEY=gsk_...
```

### 3. Run Verification Tests
```bash
python -m pytest tests/ -v
```

### 4. Run WisprHelp
```bash
# 1. Interactive Assistant Loop (with PySide6 HUD)
python wisprhelp/main.py

# 2. Single Quick Command Execution (instant sub-second CLI)
python wisprhelp/main.py --text "mute volume"
python wisprhelp/main.py --text "open notepad"
python wisprhelp/main.py --text "टाइप करो नमस्ते दुनिया"

# 3. Voice Listening Mode (Push-to-Talk via Alt+Space)
python wisprhelp/main.py --voice

# 4. Headless / Server Mode (no GUI HUD, no audio beeps)
python wisprhelp/main.py --no-ui --no-sound
```
