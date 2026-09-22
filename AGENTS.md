# Agent Harness Guidelines & Technical Contract (`AGENTS.md`)

This document is the authoritative technical contract for AI coding agents modifying, testing, or maintaining the **WisprHelp** codebase.

---

## 1. Project Overview & Boundaries

- **Repository**: `WisprHelp` — Autonomous Universal Computer-Using Agent (CUA).
- **Core Technology Stack**:
  - Python 3.10+ (tested on Python 3.11–3.14).
  - Classification: **TypeSafe AI Jev** (`jev-latest` / `jev-1.13.0`) via `typesafe-sdk` or standard HTTPS.
  - Hearing & Speech: **Silero VAD** (local sub-5ms VAD) $\rightarrow$ **Whisper Large v3** via Groq API or `faster-whisper`.
  - Screen Perception: `mss`, `Pillow` (downscaled), Microsoft OmniParser, and Anthropic Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash.
  - OS Automation: 
    - Windows: `pywin32`, `uiautomation` / `pywinauto`, `Get-StartApps` + AppID, `os.startfile`.
    - macOS: Cocoa `NSWorkspace`, Quartz, AppleScript / `osascript`, `open -a`.
    - Linux: `xdotool`, `xclip` / `wl-clipboard`, `xdg-open`, AT-SPI.
    - Universal: `pyautogui`, `pyperclip` (Unicode clipboard swap).
  - Overlay: `PySide6` (click-through floating pill HUD).

### Hard Boundaries: What Agents Must NEVER Do
1. **Never commit secrets or API keys**: Keep `.env` out of version control. Read keys strictly via `wisprhelp.config.Config`.
2. **Never hardcode Windows-only commands in core logic**: All OS interactions must go through the `wisprhelp.hal.base.BaseOSAdapter` interface.
3. **Never bypass the Safety Interlock**: Never disable the `is_destructive` confirmation gate or the emergency mouse/Esc kill-switch.
4. **Never run unbounded agent loops**: Every autonomous loop must enforce a maximum step limit (default: 15 steps).

---

## 2. Environment & Dependencies

### Virtual Environment Setup
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Required Dependencies
```bash
pip install typesafe-sdk pyautogui mss pillow pyperclip requests python-dotenv pydantic pytest PySide6 uiautomation
```

### Environment Variables (`.env`)
```bash
# TypeSafe AI Jev (System One Classifier)
JEV_API_KEY=apikey_...
# or
TYPESAFE_API_KEY=apikey_...

# Multimodal Vision Agent (System Two)
ANTHROPIC_API_KEY=sk-ant-...
# or
GEMINI_API_KEY=AIzaSy...

# Speech-to-Text (Ear)
GROQ_API_KEY=gsk_...
```

---

## 3. Code Standards & Architectural Rules

### 3.1 Strict Separation of Concerns
- **`wisprhelp.classification`**: Strictly deals with TypeSafe Jev schemas, API calls, and returning calibrated `JevClassificationResult`. Never invoke mouse/keyboard actions here.
- **`wisprhelp.hal`**: Strictly deals with native OS system calls, window management, clipboard manipulation, and simulated hardware inputs. Never put LLM reasoning or prompt text here.
- **`wisprhelp.core.router`**: Evaluates Jev's output against confidence thresholds and cascades through the 3-priority execution hierarchy.
- **`wisprhelp.voice`**: Manages Silero VAD audio frame filtering, hotkey listeners, and Whisper transcription.
- **`wisprhelp.vision`**: Deals with screen frame acquisition (`mss`), image downscaling (`PIL`), coordinate transformations, and VLM computer-use loops.

### 3.2 3-Priority Execution Rule
When executing desktop interactions:
1. **Priority 1 (Shell / Protocol / AppID)**: Use native launchers first. Never take a screenshot just to launch an app.
2. **Priority 2 (Native Accessibility)**: Try OS Accessibility Tree (UIA/AX/AT-SPI) to find button/input coordinates or invoke them directly before calling vision models.
3. **Priority 3 (Vision Screen Grab)**: Only capture screenshots and invoke VLM when Priority 1 and 2 cannot locate the target.

### 3.3 Universal Unicode Typing Rule
Never use `pyautogui.typewrite(text)` for arbitrary user speech. Direct simulated keystrokes corrupt Hindi Devanagari, emojis, and non-ASCII text. Always use:
```python
os_adapter.inject_unicode_text(text)
```
which swaps the clipboard buffer and pastes using native OS key events.

---

## 4. Verification & Testing Contracts

Before marking any task as complete, an agent MUST run the verification suite:

```bash
# Run all automated unit tests
python -m pytest tests/ -v

# Run only Jev classification tests
python -m pytest tests/test_jev_classification.py -v

# Run OS HAL adapter tests
python -m pytest tests/test_hal_adapters.py -v
```
