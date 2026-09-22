"""Unit Tests for WisprHelp Phase 4 Voice Pipeline.

Tests Silero VAD state machine, Whisper STT packaging, hotkey listener,
and non-blocking auditory cues.
"""

import math
import struct
import wave
import io
import pytest

from wisprhelp.voice.vad import SileroVAD, VADState
from wisprhelp.voice.stt import WhisperSTT
from wisprhelp.voice.hotkey import PushToTalkListener
from wisprhelp.voice.feedback import AudioFeedback


def generate_pcm_sine(duration_sec: float, frequency: float = 440.0, sample_rate: int = 16000, amplitude: float = 16000.0) -> bytes:
    """Generates synthetic 16-bit mono PCM sine wave audio bytes."""
    num_samples = int(duration_sec * sample_rate)
    samples = []
    for i in range(num_samples):
        val = int(amplitude * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
        samples.append(val)
    return struct.pack(f"<{num_samples}h", *samples)


def generate_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generates pure silence PCM bytes."""
    num_samples = int(duration_sec * sample_rate)
    return b"\x00" * (num_samples * 2)


def test_silero_vad_silence_keeps_listening():
    vad = SileroVAD(chunk_size=512, silence_duration_sec=0.2)
    silence_chunk = generate_silence(512 / 16000)

    # 5 frames of silence
    for _ in range(5):
        result = vad.process_chunk(silence_chunk)
        assert result is None
        assert vad.state == VADState.LISTENING


def test_silero_vad_speech_start_and_end():
    vad = SileroVAD(chunk_size=512, silence_duration_sec=0.1, speech_threshold=0.2)
    silence_chunk = generate_silence(512 / 16000)
    speech_chunk = generate_pcm_sine(512 / 16000, frequency=300.0, amplitude=25000.0)

    # 1. Start with silence
    assert vad.process_chunk(silence_chunk) is None
    assert vad.state == VADState.LISTENING

    # 2. Inject speech chunk
    assert vad.process_chunk(speech_chunk) is None
    assert vad.state == VADState.SPEAKING

    # 3. Continue speech
    for _ in range(3):
        assert vad.process_chunk(speech_chunk) is None
        assert vad.state == VADState.SPEAKING

    # 4. Inject silence until speech ends (0.1s / 0.032s = ~4 frames)
    finished_audio = None
    for _ in range(5):
        res = vad.process_chunk(silence_chunk)
        if res is not None:
            finished_audio = res
            break

    assert finished_audio is not None
    assert len(finished_audio) > 0
    assert vad.state == VADState.IDLE


def test_silero_vad_max_utterance_safety():
    # 0.2s max utterance limit
    vad = SileroVAD(chunk_size=512, max_utterance_sec=0.2, silence_duration_sec=2.0)
    speech_chunk = generate_pcm_sine(512 / 16000, amplitude=25000.0)

    finished_audio = None
    # Feed more frames than max utterance
    for _ in range(15):
        res = vad.process_chunk(speech_chunk)
        if res is not None:
            finished_audio = res
            break

    assert finished_audio is not None
    assert vad.state == VADState.IDLE


def test_whisper_stt_pcm_to_wav_conversion():
    pcm_data = generate_pcm_sine(0.1, frequency=440.0)
    wav_bytes = WhisperSTT.pcm_to_wav_bytes(pcm_data, sample_rate=16000)

    assert len(wav_bytes) > len(pcm_data)
    # Validate WAV header
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        frames = wf.readframes(wf.getnframes())
        assert frames == pcm_data


def test_whisper_stt_empty_and_fallback():
    stt = WhisperSTT(api_key=None)
    assert stt.transcribe(b"") == ""

    # Silence/near-zero RMS returns empty immediately
    silence = generate_silence(0.5)
    assert stt.transcribe(silence) == ""

    # When no key or offline with audible audio, returns empty string cleanly without crashing
    pcm_data = generate_pcm_sine(0.05)
    result = stt.transcribe(pcm_data)
    assert isinstance(result, str)


def test_whisper_stt_sanitize_hallucinations():
    """Verify that common Whisper silence hallucinations and prompt echoes are cleanly discarded."""
    # Hallucinations should return empty string
    assert WhisperSTT.sanitize_transcription("Hinglish, Hinglish, bilingual conversation") == ""
    assert WhisperSTT.sanitize_transcription("English, Hindi, Hinglish, bilingual conversation.") == ""
    assert WhisperSTT.sanitize_transcription("bilingual conversation") == ""
    assert WhisperSTT.sanitize_transcription("Thank you for watching.") == ""
    assert WhisperSTT.sanitize_transcription("Subtitles by Amara.org") == ""
    assert WhisperSTT.sanitize_transcription("...") == ""

    # Phonetic normalization for programming extensions (.by -> .py)
    assert WhisperSTT.sanitize_transcription("create a file called main.by") == "create a file called main.py"
    assert WhisperSTT.sanitize_transcription("make test dot by") == "make test.py"
    assert WhisperSTT.sanitize_transcription("In VS Code, create a file called app.by") == "In VS Code, create a file called app.py"


def test_hotkey_listener_key_normalization():
    listener = PushToTalkListener()

    class MockKey:
        def __init__(self, char=None):
            self.char = char

    assert listener._normalize_key(MockKey("A")) == "a"
    assert listener._normalize_key(MockKey("1")) == "1"


def test_hotkey_listener_emergency_abort_callback():
    abort_called = []

    def on_abort():
        abort_called.append(True)

    listener = PushToTalkListener(on_abort_callback=on_abort)

    # Mock pynput key
    from pynput import keyboard
    listener._handle_key_press(keyboard.Key.esc)

    assert len(abort_called) == 1
    assert abort_called[0] is True


def test_hotkey_listener_ctrl_shift_ptt_trigger():
    """Verify that Ctrl + Shift activates Push-to-Talk and release deactivates it."""
    from pynput import keyboard
    press_events = []
    release_events = []

    listener = PushToTalkListener(
        on_press_callback=lambda: press_events.append("pressed"),
        on_release_callback=lambda: release_events.append("released"),
    )

    # Press Ctrl, then Shift -> should trigger PTT
    listener._handle_key_press(keyboard.Key.ctrl_l)
    assert len(press_events) == 0
    listener._handle_key_press(keyboard.Key.shift_l)
    assert len(press_events) == 1
    assert listener._is_active is True

    # Release Shift -> should release PTT
    listener._handle_key_release(keyboard.Key.shift_l)
    assert len(release_events) == 1
    assert listener._is_active is False


def test_hotkey_listener_alt_combos_ptt_trigger():
    """Verify that Alt + Shift and Alt + Space also activate Push-to-Talk."""
    from pynput import keyboard
    press_events = []
    release_events = []

    listener = PushToTalkListener(
        on_press_callback=lambda: press_events.append("pressed"),
        on_release_callback=lambda: release_events.append("released"),
    )

    # 1. Alt + Shift
    listener._handle_key_press(keyboard.Key.alt_l)
    listener._handle_key_press(keyboard.Key.shift_r)
    assert len(press_events) == 1
    listener._handle_key_release(keyboard.Key.shift_r)
    assert len(release_events) == 1

    # 2. Alt + Space
    listener._handle_key_press(keyboard.Key.space)
    assert len(press_events) == 2
    listener._handle_key_release(keyboard.Key.space)
    assert len(release_events) == 2


def test_audio_feedback_safe_execution():
    feedback = AudioFeedback(enabled=True)
    # Call all cues to ensure no exceptions are raised
    feedback.play_listen_start()
    feedback.play_speech_end()
    feedback.play_success()
    feedback.play_warning()
    feedback.play_abort()


def test_text_to_speech_safe_execution():
    from wisprhelp.voice.tts import TextToSpeech
    tts = TextToSpeech(enabled=True)
    # Test safe execution
    tts.speak("WisprHelp ready")
    tts.speak_async("Background test")


def test_microphone_stream_gating():
    from wisprhelp.voice.stream import MicrophoneStream
    captured = []

    def on_speech(data):
        captured.append(data)

    stream = MicrophoneStream(on_speech_captured=on_speech, push_to_talk_mode=True)
    assert stream._gate_open is False

    # Open gate for PTT
    stream.set_gate(True)
    assert stream._gate_open is True

    # Close gate
    stream.set_gate(False)
    assert stream._gate_open is False


def test_microphone_stream_ptt_audio_accumulation():
    import time
    from wisprhelp.voice.stream import MicrophoneStream
    captured = []

    def on_speech(data):
        captured.append(data)

    stream = MicrophoneStream(on_speech_captured=on_speech, push_to_talk_mode=True)
    stream.set_gate(True)

    # Simulate 0.5 seconds of incoming audio chunks during PTT (16 chunks of 1024 bytes)
    chunk = b"\x01\x00" * 512
    for _ in range(16):
        stream._ptt_buffer.append(chunk)

    # Release PTT
    flushed = stream.set_gate(False)
    assert flushed is not None
    assert len(flushed) == 16 * 1024

    # Wait briefly for background thread dispatch
    time.sleep(0.1)
    assert len(captured) == 1
    assert captured[0] == flushed
