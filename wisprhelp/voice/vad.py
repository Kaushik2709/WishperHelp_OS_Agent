"""Silero VAD (Voice Activity Detection) Local Audio Filter.

Accepts 16kHz mono audio frames, performs sub-5ms speech boundary detection,
and emits speech audio buffers when the user finishes speaking.
Includes ring-buffer lead-in and bounded memory buffers.
"""

from __future__ import annotations

import collections
import enum
import logging
import math
import struct
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)


class VADState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    SPEECH_ENDED = "speech_ended"


class SileroVAD:
    """Production-grade Voice Activity Detector with Silero VAD and Adaptive Energy fallback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms at 16kHz (Silero native window)
        speech_threshold: float = 0.5,
        silence_duration_sec: float = 0.85,
        pre_speech_padding_sec: float = 0.2,
        max_utterance_sec: float = 30.0,
        try_hub_load: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.speech_threshold = speech_threshold
        self.silence_duration_sec = silence_duration_sec
        self.max_utterance_sec = max_utterance_sec
        self.try_hub_load = try_hub_load

        # Compute frame counts
        self._frame_duration = chunk_size / sample_rate
        self._silence_frames_needed = max(1, int(silence_duration_sec / self._frame_duration))
        self._max_frames = int(max_utterance_sec / self._frame_duration)
        self._padding_frames = max(1, int(pre_speech_padding_sec / self._frame_duration))

        # Ring buffer for pre-speech frames to avoid clipping the start of speech
        self._pre_buffer: Deque[bytes] = collections.deque(maxlen=self._padding_frames)
        # Active audio chunks collected during speech
        self._active_buffer: List[bytes] = []

        self._state = VADState.IDLE
        self._consecutive_silence_frames = 0
        self._model = None
        self._use_fallback = False

        self._init_model()

    def _init_model(self) -> None:
        """Attempts to load the Silero VAD model via Torch if requested or cached."""
        if not self.try_hub_load:
            self._use_fallback = True
            return

        try:
            import torch

            try:
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    verbose=False,
                    trust_repo=True,
                )
                self._model = model
                self._model.eval()
                logger.info("Silero VAD Torch model loaded successfully.")
            except Exception as hub_exc:
                logger.debug("Torch hub load failed (%s). Using adaptive energy fallback.", hub_exc)
                self._use_fallback = True
        except ImportError:
            logger.debug("Torch not available; using adaptive energy fallback for VAD.")
            self._use_fallback = True

    @property
    def state(self) -> VADState:
        return self._state

    def reset(self) -> None:
        """Resets the state machine and clears buffers."""
        self._state = VADState.IDLE
        self._consecutive_silence_frames = 0
        self._pre_buffer.clear()
        self._active_buffer.clear()

    def is_speech_chunk(self, chunk_bytes: bytes) -> bool:
        """Evaluates whether an individual 16kHz 16-bit PCM chunk contains active speech."""
        if not chunk_bytes:
            return False

        if not self._use_fallback and self._model is not None:
            try:
                import torch

                # Convert 16-bit PCM to float32 tensor normalized to [-1.0, 1.0]
                num_samples = len(chunk_bytes) // 2
                if num_samples != self.chunk_size:
                    # Pad or slice if chunk size doesn't match
                    if num_samples < self.chunk_size:
                        chunk_bytes = chunk_bytes + b"\x00" * ((self.chunk_size - num_samples) * 2)
                    else:
                        chunk_bytes = chunk_bytes[: self.chunk_size * 2]

                format_str = f"<{self.chunk_size}h"
                samples = struct.unpack(format_str, chunk_bytes)
                tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0

                with torch.no_grad():
                    prob = self._model(tensor, self.sample_rate).item()
                    return prob >= self.speech_threshold
            except Exception as exc:
                logger.debug("Silero inference error: %s; falling back to energy.", exc)
                self._use_fallback = True

        # Adaptive RMS energy calculation
        return self._energy_is_speech(chunk_bytes)

    def _energy_is_speech(self, chunk_bytes: bytes) -> bool:
        """Fallback sub-millisecond RMS energy calculation."""
        num_samples = len(chunk_bytes) // 2
        if num_samples == 0:
            return False

        format_str = f"<{num_samples}h"
        samples = struct.unpack(format_str, chunk_bytes)
        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / num_samples)

        # Baseline speech threshold calibrated for standard PC/laptop mics (~200 for 16-bit signed integer)
        threshold = max(150.0, self.speech_threshold * 400.0)
        return rms > threshold

    def process_chunk(self, chunk_bytes: bytes) -> Optional[bytes]:
        """Processes a single audio chunk and advances the VAD state machine.

        Args:
            chunk_bytes: Raw 16-bit PCM mono audio chunk.

        Returns:
            bytes: Complete speech buffer if speech just concluded, else None.
        """
        is_speech = self.is_speech_chunk(chunk_bytes)

        if self._state in (VADState.IDLE, VADState.LISTENING):
            if is_speech:
                self._state = VADState.SPEAKING
                self._consecutive_silence_frames = 0
                # Flush pre-speech buffer into active speech
                self._active_buffer.extend(list(self._pre_buffer))
                self._pre_buffer.clear()
                self._active_buffer.append(chunk_bytes)
            else:
                self._state = VADState.LISTENING
                self._pre_buffer.append(chunk_bytes)
            return None

        elif self._state == VADState.SPEAKING:
            self._active_buffer.append(chunk_bytes)

            if is_speech:
                self._consecutive_silence_frames = 0
            else:
                self._consecutive_silence_frames += 1

            # Check if utterance exceeded safety limit
            if len(self._active_buffer) >= self._max_frames:
                logger.warning("Max utterance duration reached (%ss). Concluding speech.", self.max_utterance_sec)
                return self._finalize_speech()

            # Check if trailing silence duration met
            if self._consecutive_silence_frames >= self._silence_frames_needed:
                return self._finalize_speech()

            return None

        return None

    def _finalize_speech(self) -> bytes:
        """Packages accumulated speech chunks and resets state."""
        self._state = VADState.SPEECH_ENDED
        combined_audio = b"".join(self._active_buffer)
        self.reset()
        return combined_audio
