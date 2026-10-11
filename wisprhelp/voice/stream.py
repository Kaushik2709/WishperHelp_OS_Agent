"""Live Microphone Audio Streaming Pipeline.

Captures real-time 16kHz mono 16-bit PCM audio from the default input device
and feeds chunks through SileroVAD to detect user speech boundaries.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from wisprhelp.voice.vad import SileroVAD, VADState

logger = logging.getLogger(__name__)


class MicrophoneStream:
    """Continuous microphone capture pipeline integrating sounddevice with SileroVAD."""

    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        on_speech_captured: Optional[Callable[[bytes], None]] = None,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms window
        push_to_talk_mode: bool = False,
    ) -> None:
        self.vad = vad or SileroVAD(sample_rate=sample_rate, chunk_size=chunk_size)
        self.on_speech_captured = on_speech_captured
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.push_to_talk_mode = push_to_talk_mode

        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._stream = None
        self._is_running = False
        self._gate_open = not push_to_talk_mode
        self._is_ptt_pressed = False
        self._worker_thread: Optional[threading.Thread] = None
        self._ptt_buffer: list[bytes] = []
        self._lock = threading.Lock()

    def flush(self) -> Optional[bytes]:
        """Flushes and finalizes any accumulated speech in the VAD or PTT buffer."""
        with self._lock:
            if len(self._ptt_buffer) > 0:
                audio = b"".join(self._ptt_buffer)
                self._ptt_buffer.clear()
                self.vad.reset()
                return audio
            if len(self.vad._active_buffer) > 0:
                audio = self.vad._finalize_speech()
                self.vad.reset()
                return audio
            if len(self.vad._pre_buffer) > 0:
                audio = b"".join(self.vad._pre_buffer)
                self.vad.reset()
                return audio
            self.vad.reset()
            return None

    def set_gate(self, open_gate: bool) -> Optional[bytes]:
        """Opens or closes audio ingestion for push-to-talk. Returns flushed speech when closed."""
        with self._lock:
            self._gate_open = open_gate
            if open_gate:
                # PTT key pressed: mark active, open gate, clear buffers for clean speech capture
                self._is_ptt_pressed = True
                self._ptt_buffer.clear()
                self.vad.reset()
                return None
            else:
                # PTT key released: mark inactive, close gate if in ptt mode
                self._is_ptt_pressed = False
                if self.push_to_talk_mode:
                    self._gate_open = False

                audio_to_send = None
                if len(self._ptt_buffer) > 0:
                    audio_to_send = b"".join(self._ptt_buffer)
                    self._ptt_buffer.clear()
                elif len(self.vad._active_buffer) > 0:
                    audio_to_send = self.vad._finalize_speech()
                elif len(self.vad._pre_buffer) > 0:
                    audio_to_send = b"".join(self.vad._pre_buffer)

                self.vad.reset()

                # Require at least 0.25s of audio (~8000 bytes at 16kHz 16-bit mono)
                min_bytes = int(self.sample_rate * 2 * 0.25)
                if audio_to_send and len(audio_to_send) >= min_bytes:
                    logger.info("Push-to-Talk completed. Captured %d audio bytes for STT.", len(audio_to_send))
                    if self.on_speech_captured:
                        # Asynchronous execution in daemon thread prevents blocking the caller
                        threading.Thread(
                            target=self._safe_dispatch_speech,
                            args=(audio_to_send,),
                            daemon=True,
                        ).start()
                    return audio_to_send
                else:
                    logger.debug("Push-to-Talk audio empty or too short. Ignored.")
                    return None

    def _safe_dispatch_speech(self, audio_data: bytes) -> None:
        """Safely invokes speech callback in background thread."""
        try:
            if self.on_speech_captured:
                self.on_speech_captured(audio_data)
        except Exception as exc:
            logger.error("Error in speech captured callback: %s", exc)

    def start(self) -> bool:
        """Starts microphone recording stream and VAD worker thread."""
        if self._is_running:
            return True

        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.debug("Audio stream status: %s", status)
                chunk = bytes(indata)
                with self._lock:
                    if self._gate_open:
                        self._ptt_buffer.append(chunk)
                        try:
                            self._audio_queue.put_nowait(chunk)
                        except queue.Full:
                            pass

            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            )
            self._stream.start()
            self._is_running = True

            try:
                default_input = sd.query_devices(kind="input")
                logger.info(
                    "Capturing from default input device: '%s' (max in: %.0f%%)",
                    default_input.get("name", "unknown"),
                    float(default_input.get("max_input_channels", 0)),
                )
            except Exception:
                pass

            self._worker_thread = threading.Thread(target=self._vad_worker, daemon=True)
            self._worker_thread.start()
            logger.info("MicrophoneStream recording started at %d Hz.", self.sample_rate)
            return True
        except Exception as exc:
            logger.warning("Could not open microphone audio stream: %s", exc)
            self._is_running = False
            return False

    def _vad_worker(self) -> None:
        """Background worker consuming audio chunks and feeding SileroVAD."""
        while self._is_running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # In Push-to-Talk mode or while PTT key is held, suppress VAD auto-dispatching
            if self.push_to_talk_mode or self._is_ptt_pressed:
                continue

            speech_buffer = self.vad.process_chunk(chunk)
            if speech_buffer and self.on_speech_captured:
                logger.info("Speech utterance finalized (%d bytes). Dispatching to callback.", len(speech_buffer))
                try:
                    self.on_speech_captured(speech_buffer)
                except Exception as exc:
                    logger.error("Error in on_speech_captured callback: %s", exc)

    def stop(self) -> None:
        """Stops the audio stream and worker."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.vad.reset()
        logger.info("MicrophoneStream stopped.")
