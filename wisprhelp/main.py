"""WisprHelp Main Application Runner & Interactive Assistant Entry Point.

Wires the Ear (Silero VAD + Whisper), System One Brain (TypeSafe Jev),
System Two Eyes (MSS + VLM), 3-Priority Hands (OS HAL), and PySide6 HUD.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure project root is in sys.path when invoked directly as python wisprhelp/main.py
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force UTF-8 standard streams on Windows to prevent charmap/CP1252 print crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from wisprhelp.classification.classifier import JevClassifier
from wisprhelp.config import config
from wisprhelp.core.router import CommandRouter
from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.core.safety import SafetyGate
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.ui.overlay import HUDState, StatusOverlayHUD
from wisprhelp.vision.vlm_agent import VLMAgent
from wisprhelp.voice.feedback import AudioFeedback
from wisprhelp.voice.hotkey import PushToTalkListener
from wisprhelp.voice.stt import WhisperSTT
from wisprhelp.voice.vad import SileroVAD
from wisprhelp.voice.tts import TextToSpeech
from wisprhelp.voice.stream import MicrophoneStream

# Configure UTF-8 safe console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wisprhelp")


def print_banner() -> None:
    banner = r"""
===================================================================
     _    _ _                 _   _      _       
    | |  | (_)               | | | |    | |      
    | |  | |_ ___ _ __  _ __ | |_| | ___| |_ __  
    | |/\| | / __| '_ \| '__||  _  |/ _ \ | '_ \ 
    \  /\  / \__ \ |_) | |   | | | |  __/ | |_) |
     \/  \/|_|___/ .__/|_|   \_| |_/\___|_| .__/ 
                 | |                      | |    
                 |_|                      |_|    
   Universal Autonomous Computer-Using Agent (CUA)
===================================================================
"""
    print(banner)
    print(f" OS Platform       : {platform.system()} ({platform.release()})")
    print(f" TypeSafe Jev Key  : {'Configured (Active)' if config.jev_api_key else 'Missing (Offline heuristic)'}")
    print(f" Groq Whisper Key  : {'Configured' if config.groq_api_key else 'Missing (Mock mode)'}")
    print(f" Vision VLM Key    : {'Configured' if (config.anthropic_api_key or config.gemini_api_key) else 'Missing (Offline sim)'}")
    print(f" Hotkey Trigger    : Ctrl + Shift (Push-to-Talk)")
    print(f" Emergency Abort   : Esc or Pause/Break")
    print("===================================================================\n")


def build_runtime(enable_gui: bool = True, enable_sound: bool = True) -> tuple[RuntimeCoordinator, StatusOverlayHUD, AudioFeedback, VLMAgent]:
    """Constructs the fully integrated WisprHelp runtime graph."""
    os_adapter = get_os_adapter()
    classifier = JevClassifier()
    router = CommandRouter()
    safety_gate = SafetyGate()
    vlm_agent = VLMAgent(os_adapter=os_adapter)
    hud = StatusOverlayHUD(enable_gui=enable_gui)
    feedback = AudioFeedback(enabled=enable_sound)

    coordinator = RuntimeCoordinator(
        classifier=classifier,
        router=router,
        safety_gate=safety_gate,
        os_adapter=os_adapter,
        vlm_agent=vlm_agent,
        hud=hud,
        feedback=feedback,
    )
    return coordinator, hud, feedback, vlm_agent


def run_interactive_cli(coordinator: RuntimeCoordinator, hud: StatusOverlayHUD) -> None:
    """Runs interactive command line prompt loop."""
    print("Type a command (or 'exit' to quit). For safety confirmations, answer 'yes' or 'no'.\n")
    hud.start()

    while True:
        try:
            user_input = input("WisprHelp> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting WisprHelp. Goodbye!")
                break

            # Process command
            res = coordinator.process_command(user_input)
            print(f"[{res.route.value.upper()}] ({res.latency_ms:.1f}ms): {res.message}")

            # If held at safety gate, prompt user immediately
            if res.route.value == "safety_interlock" and coordinator.safety_gate.has_pending():
                confirm = input("\nConfirm high-risk action? [yes/no]: ").strip()
                confirm_res = coordinator.process_command(user_input, confirmation_response=confirm)
                print(f"[{confirm_res.route.value.upper()}]: {confirm_res.message}")

        except (KeyboardInterrupt, EOFError):
            print("\nShutting down...")
            break

    hud.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="WisprHelp Autonomous Universal Computer-Using Agent")
    parser.add_argument("--text", type=str, help="Single command to execute immediately")
    parser.add_argument("--voice", action="store_true", help="Start in voice listening mode")
    parser.add_argument("--hands-free", action="store_true", help="Enable continuous hands-free voice detection instead of push-to-talk")
    parser.add_argument("--no-ui", action="store_true", help="Disable floating PySide6 HUD")
    parser.add_argument("--no-sound", action="store_true", help="Disable auditory feedback beeps")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    coordinator, hud, feedback, vlm_agent = build_runtime(
        enable_gui=not args.no_ui,
        enable_sound=not args.no_sound,
    )

    tts = TextToSpeech(enabled=not args.no_sound)
    stt = WhisperSTT()

    # Emergency kill-switch abort callback
    def on_emergency_abort():
        logger.critical("EMERGENCY ABORT HOOK TRIGGERED!")
        vlm_agent.abort()
        hud.set_state(HUDState.ERROR, "EMERGENCY ABORT")
        feedback.play_abort()
        tts.speak_async("Execution aborted.")

    mic_stream: Optional[MicrophoneStream] = None

    def on_ptt_press():
        feedback.play_listen_start()
        hud.set_state(HUDState.LISTENING, "Listening (PTT)...")
        if mic_stream:
            mic_stream.set_gate(True)

    def on_ptt_release():
        feedback.play_speech_end()
        if mic_stream:
            flushed = mic_stream.set_gate(False)
            if not flushed:
                # User released too fast or no audio was accumulated
                hud.set_state(HUDState.LISTENING, "Listening for voice...")
            else:
                hud.set_state(HUDState.THINKING, "Transcribing speech...")

    hotkey_listener = PushToTalkListener(
        on_press_callback=on_ptt_press,
        on_release_callback=on_ptt_release,
        on_abort_callback=on_emergency_abort,
    )
    hotkey_listener.start()

    # Handle graceful Ctrl+C
    def sigint_handler(sig, frame):
        print("\nSIGINT received. Cleaning up...")
        if mic_stream:
            mic_stream.stop()
        hotkey_listener.stop()
        hud.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # 1. Single Command Mode
    if args.text:
        hud.start()
        res = coordinator.process_command(args.text)
        print(f"\nResult [{res.route.value}]: {res.message} (Latency: {res.latency_ms:.1f}ms)")
        hud.stop()
        hotkey_listener.stop()
        sys.exit(0)

    # 2. Voice Mode
    if args.voice:
        ptt_mode = not args.hands_free
        if ptt_mode:
            print("Voice listening mode active (Push-to-Talk).")
            print("Hold Ctrl+Shift (or Alt+Shift / Alt+Space) to talk, release to execute.")
        else:
            print("Voice listening mode active (Continuous Hands-Free).")
            print("Speak directly into your microphone.")
        print("Press Esc to abort at any time. Press Ctrl+C to stop.\n")

        hud.start()
        hud.set_state(HUDState.LISTENING, "Listening for voice...")

        execution_lock = threading.Lock()

        def on_speech(audio_data: bytes):
            # Guard against overlapping triggers while an action is executing
            if not execution_lock.acquire(blocking=False):
                logger.warning("Another voice action is actively running. Ignoring overlapping utterance.")
                return

            try:
                hud.set_state(HUDState.THINKING, "Transcribing speech...")
                transcript = stt.transcribe(audio_data)
                if not transcript or not transcript.strip():
                    hud.set_state(HUDState.LISTENING, "Listening for voice...")
                    return
                print(f"\n[Voice Heard]: '{transcript}'")
                res = coordinator.process_command(transcript)
                print(f"[{res.route.value.upper()}] ({res.latency_ms:.1f}ms): {res.message}")
                if res.route.value == "conversational_query":
                    tts.speak_async(res.message)
                elif res.route.value == "safety_interlock":
                    tts.speak_async("High risk action detected. Please confirm with yes or no.")
                elif res.success:
                    if "Composed and typed text" in res.message:
                        tts.speak_async("I've written that into your active document.")
                    elif "Searched for" in res.message:
                        tts.speak_async("Searching the web for you.")
                    elif "Navigated to" in res.message:
                        tts.speak_async("Opening website.")
                    elif "Opened new browser tab" in res.message:
                        tts.speak_async("New tab opened.")
                    elif "Closed current browser tab" in res.message:
                        tts.speak_async("Tab closed.")
                    elif "Launched application" in res.message:
                        app_name = res.message.split("'")[1] if "'" in res.message else "app"
                        tts.speak_async(f"Opening {app_name}.")
                elif not res.success:
                    tts.speak_async("Could not complete action.")
            except Exception as exc:
                logger.error("Error in voice command execution: %s", exc)
                hud.set_state(HUDState.ERROR, "Error")
                time.sleep(1.0)
            finally:
                execution_lock.release()
                if hud.current_state not in (HUDState.EXECUTING, HUDState.CONFIRM):
                    hud.set_state(HUDState.LISTENING, "Listening for voice...")

        mic_stream = MicrophoneStream(on_speech_captured=on_speech, push_to_talk_mode=ptt_mode)
        mic_stream.start()

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            if mic_stream:
                mic_stream.stop()
            hud.stop()
            hotkey_listener.stop()
        return

    # 3. Default: Interactive CLI prompt
    run_interactive_cli(coordinator, hud)
    hotkey_listener.stop()


if __name__ == "__main__":
    main()
